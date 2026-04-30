#!/usr/bin/env python
# coding: utf-8

# # Project Description
# This is under construction and will be updated soon

# ## Initial Setup
# 
# - Load `Qwen3.5-4B` from **unsloth** and initialize the weights
# - Configure the **LoRA** adapters for the model
# - Load and explore the dataset `lavita/MedQuad` from **HuggingFace**

# In[ ]:


from unsloth import FastVisionModel

# Initialize model weights and tokenizer
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/qwen3.5-4B",
    load_in_4bit=False, # Qwen3.5 does not support 4-bit quantization; QLoRA not possible; use only LoRA 
    use_gradient_checkpointing=True
)

model = FastVisionModel.get_peft_model(
    model,
    fine_tune_vision_layers=False,
    fine_tune_language_layers=True,
    fine_tune_attention_modules=True,
    finetune_mlp_modules=True,

    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",

    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None
)


# In[ ]:


from datasets import load_dataset

dataset = load_dataset('lavita/MedQuAD')
dataset = dataset['train']
dataset,dataset[0]


# ## Dataset Processing - Prepare For Fine Tuning
# 
# #### Prompt Formatting
# Build the data dictionary so that every fine tuning prompt passed to LLM is in the format expected by it (**ChatML**). The chat template is applied to the tokenizer so that the tokenizer object is permenantly modified. This allows the tokenizer to know how to use Qwen specific tags `<|im_start|>` and `<|im_end|>`
# 
# The ChatML template must be bound to the tokenizer first before mapping the dataset.
# 
# #### Train On Responses Only
# This is a technique that ensures **Completion-Only Training** for the benchmarking protocol. It ensures the model's loss function only optimizes for generating the medical answer rather than wasting computational power learning how to memorize the user's questions
# 
# #### Dataset Split
# The dataset needs to be split into 90-10 ratio where 90% is split for training and the remaining 10% for testing. The data processing steps need to be applied to both the train and test splits. If the processing is only applied on the train split, the `SFTTrainer` will be looking for specific ChatML tags in the test set when calculating the **validation loss** and since the tags are not present, incorrect validation loss will be reported 

# In[ ]:


from unsloth.chat_templates import get_chat_template
# Bind the template to tokenizer
tokenizer = get_chat_template(
    tokenizer,
    chat_template='qwen3-instruct'
)

print(f"\033[1;31mrows before cleaning: {len(dataset)}\033[1;31m")

def remove_empty_rows(example):
    # Check if the question or answer is none
    if example['question'] is None or example['answer'] is None:
        return False

    # Check if they are just empty strings or whitespace
    if len(str(example['question']).strip()) == 0 or len(str(example['answer']).strip()) == 0:
        return False

    return True

clean_dataset = dataset.filter(remove_empty_rows)

print(f"\033[1;31mrows after cleaning: {len(clean_dataset)}\033[1;31m")

# Split the cleaned dataset
split_dataset = clean_dataset.train_test_split(test_size=0.1, seed=42)

# Define the formatting function
def format_medquad_prompts(examples):
    texts = []
    for question, answer in zip(examples['question'], examples['answer']):
        messages = [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': answer}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    return {'text': texts}

print("\033[1;31mapplying ChatML formatting...\033[1;31m")

formatted_dataset = split_dataset.map(format_medquad_prompts, batched=True)


# In[ ]:


formatted_dataset, formatted_dataset['train'], formatted_dataset['test']


# In[ ]:


formatted_dataset['train'][0], formatted_dataset['test'][0]


# ## Fine Tuning Configuration

# In[ ]:


from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

trainer = SFTTrainer(
    model = model,
    tokenizer=tokenizer,
    train_dataset = formatted_dataset['train'],
    eval_dataset = formatted_dataset['test'],
    args=SFTConfig(
        dataset_text_field='text',

        # Architecture configuration
        max_seq_length=2048,
        dataset_num_proc= 2, # Uses multiple CPU cores to speed up data loading

        # Memory and speed control
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim='adamw_8bit',
        bf16=True, # Activates A4000 hardware acceleration

        # Duration control
        num_train_epochs=3,

        # Learning rate and scheduling
        learning_rate=2e-4,
        weight_decay=0.01,
        lr_scheduler_type='cosine',
        warmup_ratio=0.1,

        # Logging 
        logging_steps=10,
        report_to="wandb",
        seed=3407,

        # Auto-revert to best weights
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        save_total_limit=2,

        # Ensure checkpointing to prevent catastrophic failure
        output_dir="medquad_checkpoints",
        per_device_eval_batch_size=1,
        eval_accumulation_steps=10,

        # Syncing Eval and Save to happen every 100 steps for frequent backups
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100
    )
)

trainer = train_on_responses_only(
    trainer,
    instruction_part='<|im_start|>user\n',
    response_part='<|im_start|>assistant\n'
)


# In[ ]:


import os
import wandb

# Initialize wandb to track performance
wandb.init(
    project=os.environ['WANDB_PROJECT'],
    name='qwen-4b-medquad',
    tags=['qwen', '16-bit', 'medquad'],
    config=trainer.args
)

# START TRAINING
print("\033[1;31mstarting training loop...\033[1;31m")
trainer_stats = trainer.train()

# LOCAL SAVE
local_save_path = "models/qwen-3.5-4b-medquad-lora"
print(f"\033[1;31msaving locally to path {local_save_path}\033[1;31m")

model.save_pretrained(local_save_path)
tokenizer.save_pretrained(local_save_path)

# PUSH TO HUGGING FACE
hf_token = os.environ.get("HF_API_KEY")

try:
    hf_repo_name = "loknezmonzter/qwen-3.5-4b-medquad-lora"
    print("\033[1;31mpushing to huggingface repo {hf_repo_name}\033[1;31m")

    # Merge LoRA adapters to base model
    # Creates a single, unified 16-bit precision model (fine tuned)
    model.push_to_hub_merged(
        hf_repo_name,
        tokenizer,
        save_method="merged_16bit",
        token=hf_token
    )

    print("\033[1;31mcloud sync complete!\033[1;31m")
except Exception as e:
    print(f"\033[1;31m[ERR]error while pushing: {e}\033[1;31m")


# In[ ]:




