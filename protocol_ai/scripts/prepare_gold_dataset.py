import json
import time
import os
from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
# Replace 'your_username' with your actual Hugging Face handle
HF_REPO_ID = "loknezmonzter/ProtocolAI-AACT" 
HF_TOKEN = os.environ.get("HF_TOKEN") # Ensure this is set in your VM

WORKSPACE_DIR = "/workspace/data/protocol_ai"

RAW_DISTILLED_FILE = f"{WORKSPACE_DIR}/aact_distilled.jsonl"
PROCESSED_DISTILLED_FILE = f"{WORKSPACE_DIR}/aact_distilled_sanitized.jsonl"
SILVER_JUDGED_FILE = f"{WORKSPACE_DIR}/aact_silver_judged_sft.jsonl"
SILVER_FILE = f"{WORKSPACE_DIR}/aact_silver_sft.jsonl"
RAW_PROMPTS_FILE = f"{WORKSPACE_DIR}/aact_intermediate_prompts_sft.jsonl"
GOLD_SHAREGPT_FILE = f"{WORKSPACE_DIR}/aact_gold_sharegpt.jsonl"


def format_gold_dataset():
    print("Phase 1: Formatting Gold Dataset for Unsloth...")
    start_time = time.time()
    
    # Audit Trackers
    kept_count = 0
    dropped_hallucination = 0
    dropped_low_score = 0
    dropped_structural = 0
    
    with open(SILVER_JUDGED_FILE, 'r', encoding='utf-8') as infile, \
         open(GOLD_SHAREGPT_FILE, 'w', encoding='utf-8') as outfile:
         
        for line in infile:
            record = json.loads(line.strip())
            judge = record.get("judge_evaluation", {})
            
            # 1. RUTHLESS QA FILTER: Apply >=4 Score and ZERO Hallucination rules
            is_hallucinated = judge.get("hallucination_detected", True) # Default to True if missing to be safe
            score = judge.get("medical_accuracy_score", 0)
            
            if is_hallucinated:
                dropped_hallucination += 1
                continue
                
            if score < 4:
                dropped_low_score += 1
                continue
            
            data = record.get("data", {})
            source_text = data.get("protocol", record.get("text", record.get("prompt", "")))
            relationships = data.get("relationships", [])
            
            # Ensure the required keys actually exist
            if not source_text or not relationships:
                dropped_structural += 1
                continue
                
            # 2. CONVERSATIONAL FORMATTING & SCHEMA ALIGNMENT
            # Notice we wrap relationships in the exact dict expected by the system prompt
            target_json_object = {"relationships": relationships}
            
            messages = {
                "messages": [
                    {"role": "user", "content": source_text},
                    {"role": "assistant", "content": json.dumps(target_json_object, ensure_ascii=False)}
                ]
            }
            
            outfile.write(json.dumps(messages, ensure_ascii=False) + "\n")
            kept_count += 1

    print("\n" + "="*50)
    print("\nFORMATTING COMPLETE...")
    print(f"[-] Dropped (Hallucinations): {dropped_hallucination}")
    print(f"[-] Dropped (Score < 4):      {dropped_low_score}")
    print(f"[-] Dropped (Missing Data):   {dropped_structural}")
    print(f"\nFinal Gold Records:        {kept_count}")
    print(f"Time: {time.time() - start_time:.2f} seconds")
    print("="*50 + "\n")

def push_to_huggingface():
    print(f"\nPhase 2: Pushing Pipeline Revisions to Hugging Face ({HF_REPO_ID})...")
    
    uploads = [
        {"file": RAW_DISTILLED_FILE, "revision": "distilled_raw", "desc": "Raw 27B Outputs"},
        {"file": PROCESSED_DISTILLED_FILE, "revision": "distilled_processed", "desc": "Sanitized & Deduplicated"},
        {"file": SILVER_JUDGED_FILE, "revision": "silver_judged", "desc": "Silver dataset after evaluation by judge llm"},
        {"file": SILVER_FILE, "revision": "aact_silver", "desc": "Prepared silver dataset"},
        {"file": RAW_PROMPTS_FILE, "revision": "aact_silver", "desc": "AACT raw dataset with prompts"},
        {"file": GOLD_SHAREGPT_FILE, "revision": "main", "desc": "Final Gold ShareGPT Dataset"}
    ]
    
    for upload in uploads:
        file_path = upload["file"]
        rev = upload["revision"]
        
        if not os.path.exists(file_path):
            print(f"WARNING: Could not find {file_path}. Skipping this upload.")
            continue
            
        print(f"Uploading {upload['desc']} to branch: '{rev}'...")
        
        dataset = Dataset.from_json(file_path)
        dataset.push_to_hub(
            repo_id=HF_REPO_ID,
            revision=rev,
            token=HF_TOKEN,
            private=True 
        )
        print(f"Successfully pushed to {rev}!\n")

if __name__ == "__main__":
    format_gold_dataset()
    push_to_huggingface()