import os
import sys
import json
import torch
import wandb
import argparse
from dotenv import load_dotenv
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training, 
    AutoPeftModelForCausalLM
)

# Need to resolve relative paths; add this before custom imports
import sys
sys.path.insert(0, "/workspace")


# Important constants
MODEL_NAME = "microsoft/Phi-4-mini-instruct"
HF_REPO_NAME = "loknezmonzter/phi-4-mini-instruct-ft-medgemma-extracts"
DATASET_PATH = "/mnt/huggingface/data/medgemma_extracts"
BASE_DIR = "/mnt/huggingface/models"
FINE_TUNED_MODEL = "phi-4-mini-instruct-ft-medgemma-extracts"
MAX_SEQ_LENGTH = 4096
NUM_TRAIN_EPOCHS = 3 
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 7e-5
SEED = 3407

# Initalize parser for cli arguments
parser = argparse.ArgumentParser(
    description="Is the code re-running a failed run or a fresh one"
)

# Add argument - check if script is for failed run or a new run
# Boolean - True if passed, False otherwise
parser.add_argument(
    "--new",
    action="store_true",
    help="Is this a failed run (False) or a new run (True)"
)

parser.add_argument(
    "--wandb",
    type=str,
    required=True,
    help="Configure the wandb run dynamically"
)

# Parse arguments
args = parser.parse_args()

# Run setup before anything else
# Import fails without initializing relative path to /workspace
from scripts.utils.setup import run_setup
run_setup(args.new, model=FINE_TUNED_MODEL)

# Load environment variables
load_dotenv()

# Exact JSON schema - convert to JSON string
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
SCHEMA_STRING = json.dumps(JSON_SCHEMA, indent=4)

# Finalized system prompt
SYSTEM_PROMPT = (
    "You are an expert clinical informatician. "
    f"Extract data strictly into this JSON schema:\n\n{SCHEMA_STRING}\n\n"
    "CRITICAL RULES:\n"
    "1. Use ONLY double quotes for all JSON keys and string values.\n"
    "2. Response MUST start with '{' and end with '}'.\n"
    "3. Output raw JSON only — no markdown, no code blocks.\n"
    "4. Provide values for ALL keys in the schema.\n"
    "5. Apostrophes in clinical terms (e.g., patient's) are allowed inside double-quoted strings.\n"
    "6. Extract at max 10 most clinically significant relationships only. "
    "Prioritize: diagnosis > treatment > symptoms > history.\n"
)
print("\n✅ system prompt ready")

# Load and configure tokenizer padding
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Must pad from right side for training 
tokenizer.padding_side = "right"

print("\n✅ tokenizer initialized and configured\n")

# Create bits and bytes configuration for n-bit fine tuning
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load the model with bits and bytes config
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2",
    dtype=torch.bfloat16
)

# Sync model config with tokenizer
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id

print("\n✅ model initialized and configured\n")

# Required for 4-bit training
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False
model.config.pretraining_tp = 1 # keep 1 for single GPU training (no distribution)

# LoRA configuration
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "qkv_proj",      # Fused Q+K+V (replaces q_proj, k_proj, v_proj)
        "o_proj",        # Attention output 
        "gate_up_proj",  # Fused gate+up MLP (replaces gate_proj, up_proj)
        "down_proj",     # MLP down
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("\n✅ LoRA adapters applied\n")

# Only load the train set for now
# Use a slice of train set for dry run
dataset = load_from_disk(f"{DATASET_PATH}/train")

def format_example(example):
    """
    Format columns into standard Hugging Face conversation dictionaries.
    The output dict MUST expose a single structural column names "messages"
    """
    # Ground truth from medgemma extractions
    target_json = json.dumps(example["data"], indent=2)

     # Build prompt string with chat template (includes assistant header)
    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"CONTEXT:\n{example['raw_medical_text']}"}
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True  # Adds the assistant role header (e.g., <|assistant|>)
    )

    # Completion is just the JSON output + EOS
    completion_text = target_json + tokenizer.eos_token

    return {
        "prompt": prompt_text,      
        "completion": completion_text  
    }

# Randomize the train dataset 
# Split entire train set into smaller train and test
train = dataset.shuffle(seed=42) 
split = train.train_test_split(test_size=0.1, seed=42)

# Use 1650 random records for training and 200 random records for evaluation
train_split = split["train"].select(range(2500))
eval_split = split["test"].select(range(250))

# Format into chat template
train = train_split.map(format_example, desc="Formatting train")
eval = eval_split.map(format_example, desc="Formatting eval")

print("\n✅ successfully prepared train and test sets")
print(f"#️⃣ rows in train: {len(train)}")
print(f"#️⃣ rows in eval: {len(eval)}\n")

wandb.init(
    project="phi-4-mini-instruct-ft",
    name=f"ft-2500-rows-{args.wandb}",
    config={
        "model": "phi4-mini-instruct",
        "r": 32,
        "lora_alpha": 64,
        "learning_rate": 7e-5,
        "epochs": 3,
        "batch_size": 4,
        "grad_accum": 8,
        "train_records": 2500,
        "eval_records": 250,
    }
)
print("\n✅ wandb initialized for logging and monitoring\n")           

def calculate_warmup_steps():
    effective_batch_size = BATCH_SIZE * GRAD_ACCUM_STEPS
    steps_per_epoch = len(train) // effective_batch_size
    total_steps = steps_per_epoch * NUM_TRAIN_EPOCHS
    warmup_steps = round(total_steps * 0.03)
    return warmup_steps

# Training configuration for fine tuning
training_args = SFTConfig(
    output_dir=f"{BASE_DIR}/{FINE_TUNED_MODEL}/checkpoints",

    # --- Set step count for train --- #
    num_train_epochs=3,
    max_steps=-1, # ensures enough optimizer steps to give a clear picture of dry run

    # --- Batch configuration --- #
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    # --- Optimizations --- #
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=calculate_warmup_steps(),
    weight_decay=0.01,
    max_grad_norm=1.0,
    optim="paged_adamw_8bit",

    # --- Precision control --- #
    bf16=True,
    tf32=True,

    # --- Masking --- #
    completion_only_loss=True,
    packing=False,

    # --- Checkpointing --- #
    save_steps=38,
    eval_steps=38,
    save_strategy="steps",
    eval_strategy="steps",
    save_total_limit=6,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # --- Logging and reporting --- #
    logging_steps=5,
    report_to="wandb",

    seed=SEED
)
print("\n✅ SFTConfig ready for trainer\n")

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=eval,
    processing_class=tokenizer,
    peft_config=lora_config
)
print("\n✅ SFTTrainer ready\n")

# Probe the disk for existing checkpoints
has_checkpoints = False
checkpoints_dir = f"{BASE_DIR}/{FINE_TUNED_MODEL}/checkpoints"
if os.path.exists(checkpoints_dir):
    # Look for any folder that starts with "checkpoint-"
    checkpoints = [d for d in os.listdir(checkpoints_dir) if d.startswith("checkpoint-")]
    if len(checkpoints) > 0:
        has_checkpoints = True

try:
    if has_checkpoints:
        print(f"📦 Found existing checkpoint in {checkpoints_dir}. Resuming training...")
        trainer_stats = trainer.train(resume_from_checkpoint=True)
    else:
        print("🚀 No checkpoints found. Starting fresh 2.5k training run...")
        trainer_stats = trainer.train()

except Exception as e:
    print(f"⚠️ CRITICAL CRASH: {e}")

else:
    # Find best checkpoint from trainer state
    trainer_state_path = os.path.join(
        training_args.output_dir,
        "trainer_state.json"
    )

    with open(trainer_state_path, "r") as f:
        trainer_state = json.load(f)

    # Get the best checkpoint from the json config
    best_checkpoint = trainer_state.get("best_model_checkpoint")
    merged_dir = f"{BASE_DIR}/{FINE_TUNED_MODEL}/merged"

    # Load best checkpoint and merge LoRA into base model
    merged_model = AutoPeftModelForCausalLM.from_pretrained(
        best_checkpoint,      # loads best checkpoint automatically
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    merged_model = merged_model.merge_and_unload()
    print("✅ LoRA merged into base model")

    os.makedirs(merged_dir, exist_ok=True)
    merged_model.save_pretrained(
        merged_dir,
        safe_serialization = True,
    )
    tokenizer.save_pretrained(merged_dir)
    print(f"✅ Merged model saved locally to {merged_dir}")

    try:
        # Push merged model to HF Hub
        merged_model.push_to_hub(
            HF_REPO_NAME,
            token=os.environ.get("HF_TOKEN"),
            private=False,                # set False if public repo
            safe_serialization=True,      # saves as .safetensors not .bin
        )

        # Push tokenizer separately 
        tokenizer.push_to_hub(
            HF_REPO_NAME,
            token=os.environ.get("HF_TOKEN"),
            private=False,
        )

        print(f"📤 pushed to HF Hub: {HF_REPO_NAME}")
    except Exception as e:
        print(f"❌ error {e} while pushing to HF Hub!")

finally:
    # 1. Force WandB to upload and close
    if wandb.run is not None:
        wandb.finish()
    
    # 2. Clear VRAM for the next attempt
    torch.cuda.empty_cache()
    print("✅ Background job cleaned up. VRAM released.")

print(f"🏁 Fine tuning of {MODEL_NAME} successfully complete!")
