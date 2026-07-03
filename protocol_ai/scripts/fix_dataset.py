import json
import os
import re
import time

# ==========================================
# PATHS (Adjust these to match your VM)
# ==========================================
# 1. The file containing ONLY {"prompt": "..."}
RAW_PROMPTS_FILE = "/workspace/data/protocol_ai/aact_prompts.jsonl" 

# 2. Your sanitized distilled dataset (Missing the prompt)
SANITIZED_DISTILLED_FILE = "/workspace/data/protocol_ai/aact_distilled_sanitized.jsonl"

# 3. The output file that you will feed into the LLM Judge
READY_FOR_JUDGE_FILE = "/workspace/data/protocol_ai/silver_aact_evaluated.jsonl"

def prepare_dataset_for_judge():
    print("🚀 Phase 1: Indexing Raw Prompts via Regex...")
    start_time = time.time()
    
    raw_prompt_db = {}
    
    if not os.path.exists(RAW_PROMPTS_FILE):
        print(f"❌ CRITICAL ERROR: Cannot find {RAW_PROMPTS_FILE}")
        return

    # Regex to extract "NCT01224353" from the raw prompt string
    nct_pattern = re.compile(r"Registry ID:\s*(NCT\d+)")

    with open(RAW_PROMPTS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                prompt_text = record.get("prompt", "")
                
                match = nct_pattern.search(prompt_text)
                if match:
                    # Store the ENTIRE prompt string in the dictionary, keyed by NCT ID
                    raw_prompt_db[match.group(1)] = prompt_text
            except json.JSONDecodeError:
                continue
                
    print(f"✅ Indexed {len(raw_prompt_db)} original prompt blocks.")
    
    print("🚀 Phase 2: Fusing Prompt to Distilled Dataset (ID, PROMPT, DATA, STATUS)...")
    merged_count = 0
    dropped_missing_prompt = 0
    
    with open(SANITIZED_DISTILLED_FILE, 'r', encoding='utf-8') as infile, \
         open(READY_FOR_JUDGE_FILE, 'w', encoding='utf-8') as outfile:
         
        for line in infile:
            try:
                record = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
                
            record_id = record.get("id")
            
            # Look up the corresponding raw prompt using the ID
            raw_prompt_string = raw_prompt_db.get(record_id)
            
            if not raw_prompt_string:
                dropped_missing_prompt += 1
                continue
                
            # CONSTRUCT EXACT REQUESTED SCHEMA: ID, PROMPT, DATA, STATUS
            final_row = {
                "id": record_id,
                "prompt": raw_prompt_string,
                "data": record.get("data", {}),
                "status": record.get("status", "success")
            }
            
            # WRITE TO DISK
            outfile.write(json.dumps(final_row, ensure_ascii=False) + "\n")
            merged_count += 1

    print("\n" + "="*50)
    print("✅ PRE-JUDGE DATASET PREPARATION COMPLETE")
    print(f"[+] Successfully Merged:      {merged_count} records")
    print(f"[-] Dropped (No Match Found): {dropped_missing_prompt} records")
    print(f"💾 Output saved to: {READY_FOR_JUDGE_FILE}")
    print(f"⏱️ Execution Time: {time.time() - start_time:.2f} seconds")
    print("="*50 + "\n")

if __name__ == "__main__":
    prepare_dataset_for_judge()