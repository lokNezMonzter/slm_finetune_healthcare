import os
import json
import torch
import wandb
import numpy as np
from dotenv import load_dotenv
from datasets import load_from_disk
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import get_chat_template, train_on_responses_only

load_dotenv()

# Load and slice the train set
train_set = load_from_disk("/mnt/datasets/medgemma_extracts/train").select(range(2500))
test_set = load_from_disk("/mnt/datasets/medgemma_extracts/test")

print("\n📥 successfully loaded train and test sets from disk")
print(f"#️⃣ records in train_set: {len(train_set)}")
print(f"#️⃣ records in test_set: {len(test_set)}")

JSON_SCHEMA = {
    "summary": "A concise, 1-2 sentence abstractive summary of the clinical scenario.",
    "clinical_reasoning": "A step-by-step logical breakdown of the diagnoses, treatments, or clinical decisions made in the text. Explain WHY certain relationships exist. Keep short and brief but to the point.",
    "relationships": [
        {
            "subject": "Source entity (e.g., Patient, Drug, Symptom)",
            "predicate": "Use STANDARD POSITIVE relationships (e.g., HAS_HISTORY, SHOWS_SYMPTOM, DIAGNOSED_WITH, PRESCRIBED). Do not use negated verbs like 'DENIES' or 'LACKS'.",
            "object": "Target entity",
            "polarity": "positive OR negative (Use 'negative' if the patient denies the history or lacks the symptom)",
            "certainty": "confirmed, suspected, OR hedged",
            "evidence": "The exact verbatim text snippet that proves this relationship."
        }
    ],
    "keywords": ["List", "of", "important", "clinical", "NER", "terms"]
}

# Using the Dynamic Quantized version for superior JSON logic retention
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
    use_gradient_checkpointing=True
)

# Targeting all linear layers to prevent Catastrophic Forgetting
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None
)

# Load llama-3 chat template
tokenizer = get_chat_template(tokenizer, chat_template="llama-3")


def format_prompts(examples):
    instructions = examples["raw_medical_text"]
    outputs = examples["data"]
    texts = []

    # Load the JSON schema as a formatted json string
    schema_string = json.dumps(JSON_SCHEMA, indent=4)

    for instruction, output in zip(instructions, outputs):
        messages = [
            {
                "role": "system",
                "content": f"""
                You are an expert clinical informatician. Extract data strictly into this JSON schema:\n
                
                {schema_string}

                --------------------------
                CRITICAL FORMATTING RULES
                --------------------------
                1. Use ONLY double quotes (") for all JSON keys and string values.
                2. If clinical terms contain apostrophes (e.g., patient's, Alzheimer's), leave them as raw characters inside the double-quoted strings.
                3. Your entire response MUST start strictly with the '{{' character and end strictly with the '}}' character.
                4. Output raw JSON only. No explanations, no markdown formatting, no code blocks.\n\n"""
            },
            {
                "role": "user",
                "content": f"""CONTEXT:\n{instruction}"""
            },
            {
                "role": "assistant",
                "content": output
            }
        ]
        texts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        )
    return {"text": texts}

train_set_formatted = train_set.map(format_prompts, batched=True)
test_set_formatted = test_set.map(format_prompts, batched=True)

print("\n✅ successfully formatted train and test set for fine tuning")

# Calculate the token length of the train set
token_counts = [len(tokenizer.tokenize(text)) for text in train_set_formatted["text"]]
p95 = np.percentile(token_counts, 95)

print(f"\nℹ️ 95% of your prompts are shorter than {p95} tokens.")

# Initialize and configure SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_set_formatted,
    eval_dataset=test_set_formatted,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=4096,
        dataset_num_proc=2,

        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        optim="adamw_8bit",
        bf16=torch.cuda.is_bf16_supported(),

        num_train_epochs=2,
        max_steps=-1, # set to -1 to use epochs

        learning_rate=1e-4,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,

        logging_steps=10,
        report_to="wandb",
        seed=3407,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        per_device_eval_batch_size=1,
        eval_accumulation_steps=8,
        eval_strategy="steps",
        eval_steps=50,

        save_strategy="steps",
        save_steps=50,

        output_dir="/workspace/checkpoints/llama_checkpoints",
        save_total_limit = 3,
    )
)
print("\n✅ trainer for fine tuning configured")

# Masking completion - this prevents the model from learning the instructions
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n"
)
print("\n✅ masking complete")

# Initialize wandb for logging
wandb.init(
    project=os.environ.get("WANDB_PROJECT"),
    name="llama-3.2-3B-Instruct",
    tags=["llama3.2", "4bit", "qlora"],
    config=trainer.args
)
print("\n✅ WANDB initialized")
print("\n💡 getting ready to fine tune...\n")

# Define the exact path from your SFTConfig
output_dir = "/workspace/checkpoints/llama_checkpoints"

# Probe the disk for existing checkpoints
has_checkpoints = False
if os.path.exists(output_dir):
    # Look for any folder that starts with "checkpoint-"
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if len(checkpoints) > 0:
        has_checkpoints = True

try:
    if has_checkpoints:
        print(f"📦 Found existing checkpoint in {output_dir}. Resuming training...")
        trainer_stats = trainer.train(resume_from_checkpoint=True)
    else:
        print("🚀 No checkpoints found. Starting fresh 2.5k training run...")
        trainer_stats = trainer.train()
except Exception as e:
    print(f"⚠️ CRITICAL CRASH: {e}")
else:
    # Local save
    model.save_pretrained("/mnt/models/llama-3.2-3B-unsloth-instruct-ft")
    tokenizer.save_pretrained("/mnt/models/llama-3.2-3B-unsloth-instruct-ft")

    # Push to HuggingFace
    try:
        hf_repo_name = "loknezmonzter/llama-3.2-3B-unsloth-instruct-ft"

        # Merge LoRA adapters to base model
        # Creates a single, unified 16-bit precision model (fine tuned)
        model.push_to_hub_merged(
            hf_repo_name,
            tokenizer,
            save_method="merged_16bit",
            token=os.environ.get("HF_TOKEN")
        )
        print("📤 pushed to HF Hub!")
    except Exception as e:
        print(f"❌ error {e} while pushing to HF Hub!")
finally:
    # 1. Force WandB to upload and close
    if wandb.run is not None:
        wandb.finish()
    
    # 2. Clear VRAM for the next attempt
    torch.cuda.empty_cache()
    print("✅ Background job cleaned up. VRAM released.")