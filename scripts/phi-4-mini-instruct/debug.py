import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = "microsoft/Phi-4-mini-instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config,
    device_map="auto", attn_implementation="flash_attention_2", dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.config.pad_token_id = tokenizer.pad_token_id

model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

print(f"1. After loading + kbit prep: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB peak")

lora_config = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.0, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
)

model = get_peft_model(model, lora_config)
torch.cuda.reset_peak_memory_stats()

print(f"2. After LoRA: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

print(f"Gradient checkpointing enabled: {model.is_gradient_checkpointing}")

# Simulate a single training step
model.train()
dummy_input = tokenizer("test " * 2048, return_tensors="pt").to("cuda")
dummy_labels = dummy_input["input_ids"].clone()

out = model(**dummy_input, labels=dummy_labels)
loss = out.loss
loss.backward()

print(f"3. After 1 forward+backward (2048 tokens): {torch.cuda.max_memory_allocated() / 1e9:.2f} GB peak")

torch.cuda.empty_cache()

# Now try 5120 tokens
torch.cuda.reset_peak_memory_stats()
dummy_input = tokenizer("test " * 4096, return_tensors="pt").to("cuda")
dummy_labels = dummy_input["input_ids"].clone()

out = model(**dummy_input, labels=dummy_labels)
loss = out.loss
loss.backward()

print(f"4. After 1 forward+backward (5120 tokens): {torch.cuda.max_memory_allocated() / 1e9:.2f} GB peak")