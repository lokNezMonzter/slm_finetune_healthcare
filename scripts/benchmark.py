# NOTE: This code needs to be modified based on the model and the dataset being used for benchmark
import wandb
import torch
import re
import os
import gc
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset
from unsloth import FastVisionModel
from unsloth.chat_templates import get_chat_template
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils import printf

load_dotenv()

# ==================================
# 1. SETTINGS
# ==================================
benchmark_run = "qwen3.5-4B-medquad-benchmark"
max_samples = 100

models_dict = {
    "qwen-3.5-4B-base": "unsloth/qwen3.5-4B",
    "qwen-3.5-4B-fine-tuned": "loknezmonzter/qwen-3.5-4b-medquad-lora"
}

# TODO: Re-run with both datasets for fine tuned
# datasets = ["gsm8k", "medqa"]
datasets = ["medqa"]

# ==================================
# 2. EXTRACTION
# ==================================
def extract_gsm8k_answer(text):
    match = re.search(r'####\s*([\$]?[-?\d\.\,]+)', text)
    if match:
        return match.group(1).replace("$", "").replace(",", "").strip()
    return None

# 1. THE EXTRACTION FUNCTION
def extract_medqa_answer(text):
    # 1. Strip the thinking block if it exists to avoid matching option lists
    clean_text = text.split("</think>")[-1] if "</think>" in text else text
    
    # 2. Look for the explicit final answer format
    match = re.search(r'(?:final answer|answer|choice|option)(?:\s*is)?[\s:]*(ANSWER:\s*[A-D])\b', clean_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # 3. Fallback: If no explicit answer, find the last standalone A, B, C, or D 
    # but only in the last 150 characters (the conclusion)
    suffix = clean_text[-150:]
    matches = re.findall(r'\b([A-D])\b', suffix)
    if matches:
        return matches[-1].upper()
        
    return None

def process_medqa_dataset(dataset_name, dataset):
    val_map = {"A": "0", "B": "1", "C": "2", "D": "3", "0": "0", "1": "1", "2": "2", "3": "3"}

    if dataset_name == "gsm8k":
        pass

    if dataset_name == "medqa":
        options_str = "\n".join([f"{k}) {v}" for k, v in item["options"].items()])
        prompt_content = f"{item['question']}\n{options_str}\n"

        messages = [
            {
                "role": "system",
                "content": """You are a concise medical evaluator. 
                Think step by step and provide the final answer. The final answer must always be in the format: ANSWER: [LETTER] for example: ANSWER: C"""
            },
            {
                "role": "user",
                "content": prompt_content
            }
        ]


# ==================================
# 3. EVALUATION
# ==================================
# Initialize wandb for logging and live reporting
wandb.init(
    project=os.environ["WANDB_PROJECT"],
    name=benchmark_run,
    tags=["qwen", "medquad"]
)

# 4-BIT BITSANDBYTES CONFIG
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


for model_name, model_path in models_dict.items():
    printf(f"loading {model_name} model from {model_path}...")

    # Load model (uses Unsloth)
    # model, tokenizer = FastVisionModel.from_pretrained(
    #     model_name=model_path,
    #     max_seq_length=4096,
    #     load_in_4bit=False
    # )
    # FastVisionModel.for_inference(model)

    # Load model from HuggingFace
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Load tokenizer and apply qwen3 chat template
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")

    for dataset_name in datasets:
        dataset = None
        printf(f"benchmarking {model_name} on dataset {dataset_name}")

        if dataset_name == "gsm8k":
            dataset = load_dataset("gsm8k", "main", split=f"test[:{max_samples}]")

        if dataset_name == "medqa":
            dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split=f"test[:{max_samples}]")

        # Initialize W&B Table for qualitative analysis
        table = wandb.Table(
            columns=["MODEL", "DATASET", "QUESTION", "EXPECTED", "PREDICTED", "IS_CORRECT", "RAW_OUTPUT"]
        )

        if dataset == None:
            printf("error loading dataset", type="err")

        correct=0
        for idx, item in enumerate(tqdm(dataset)):

            # Prompt Formatting
            if dataset_name == "gsm8k":
                question = item["question"] + "\nThink step by step. End with: #### [number]"
                ground_truth = extract_gsm8k_answer(item["answer"])
                
            if dataset_name == "medqa":
                options = "\n".join([f"{k}) {v}" for k, v in item["options"].items()])
                question = f"{item['question']}\n{options}\nThink step by step. End with: Answer: [Letter]"
                ground_truth = item["answer_idx"]

            # Inference Pipeline
            messages = [
                {
                    "role": "user",
                    "content": question
                }
            ]

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False, 
                add_generation_prompt=True
            )
            inputs = tokenizer(text=[prompt], return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=1024)
            raw_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split("assistant\n")[-1]

            # Extraction Pipeline
            pred = extract_gsm8k_answer(raw_output) if dataset_name == "gsm8k" else extract_medqa_answer(raw_output)
            
            is_correct = (str(pred) == str(ground_truth))
            if is_correct:
                correct += 1

            # LIVE W&B LOGGING: Line charts (From original.py)
            wandb.log({
                f"{model_name}_{dataset_name}_Rolling_Accuracy": correct / (idx + 1),
                f"{model_name}_{dataset_name}_Step": idx
            })

            # Append to table
            table.add_data(model_name, dataset_name, question, str(ground_truth), str(pred), is_correct, raw_output)
        
        # FINAL SUMMARY & UPLOAD
        final_accuracy = correct / len(dataset)
        wandb.run.summary[f"{model_name}_{dataset_name}_Final_Accuracy"] = final_accuracy
        
        # Upload the completed table to W&B
        wandb.log({f"{model_name}_{dataset_name}_Qualitative_Results": table})
        printf(f"--> Final Accuracy ({model_name} on {dataset_name}): {final_accuracy:.2%}")

    # ==========================================
    # CRITICAL VRAM CLEANUP (From current.py)
    # ==========================================
    printf(f"\n[CLEANUP] Unloading {model_name} from GPU...")
    del model
    del tokenizer
    gc.collect() # Python Garbage Collector
    torch.cuda.empty_cache() # PyTorch VRAM Release
    printf("VRAM cleared. Proceeding to next model.")

wandb.finish()