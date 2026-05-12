import torch
from unsloth import FastLanguageModel

print("\n=== 🛠️  STAGE 1: HARDWARE & DRIVER CHECK 🛠️  ===")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name:        {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM:      {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Flash Attention: {torch.backends.cuda.flash_sdp_enabled()}")
else:
    print("❌ CRITICAL: PyTorch cannot see the GPU.")

print("\n=== 🚀 STAGE 2: UNSLOTH & 4-BIT QUANTIZATION 🚀 ===")
print("Downloading a lightweight 1B parameter model to test memory allocation...")

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-1B-Instruct",
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True
    )
    print("\n=== 🧠 STAGE 3: INFERENCE PIPELINE 🧠 ===")
    print("Testing forward pass and token generation...")
    
    # Prepare model for inference (disables gradient calculation for speed)
    FastLanguageModel.for_inference(model)

    # A quick medical prompt to test the text-generation pipeline
    inputs = tokenizer(
        ["Explain the primary use case of an MRI scan in two sentences:"], 
        return_tensors="pt"
    ).to("cuda")

    # Generate response
    outputs = model.generate(**inputs, max_new_tokens=1024, use_cache=True)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    print("\n--- Model Output ---")
    print(response[0])
    print("--------------------")
    print("\n🎉 SUCCESS: ALL SYSTEMS GO! Your A6000 is fully operational.")

except Exception as e:
    print(f"\n❌ TEST FAILED. Error details:\n{e}")
    