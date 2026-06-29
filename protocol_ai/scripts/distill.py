import os
import json
import asyncio
import time
import re
from datetime import timedelta
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential,
    retry_if_exception_type
)

# User defined modules
from validate import ClinicalExtraction, validate_record
from parser import parse_json

load_dotenv()

# Configuration
VLLM_API_BASE = "http://172.17.0.1:8000/v1" 
MODEL_ID = "unsloth/medgemma-27b-text-it-bnb-4bit"
CONCURRENT_REQUESTS = 48 
MAX_TOTAL_TOKENS = 4096

# Updated paths for ProtocolAI
INPUT_PROMPTS_FILE = "/workspace/data/protocol_ai/aact_intermediate_prompts.jsonl"
OUTPUT_FILE = "/workspace/data/protocol_ai/aact_distilled.jsonl"


def load_existing_results(filepath):
    completed = set()
    failed = []
    if not os.path.exists(filepath):
        return completed, failed

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                record_id = record.get("id")
                if record.get("status") == "success" and record_id:
                    completed.add(record_id)
                elif record_id:
                    failed.append(record)
            except json.JSONDecodeError:
                print(f"⚠️ Corrupted line {line_num} in existing file, skipping")
                continue
    return completed, failed


def filter_dataset(dataset, completed_ids):
    original_count = len(dataset)
    filtered = [r for r in dataset if r["id"] not in completed_ids]
    skipped = original_count - len(filtered)
    if skipped:
        print(f"⏩ Skipping {skipped} already-completed records")
    return filtered


@retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        reraise=True
)
async def call_vllm_with_retry(**kwargs):
    return await client.chat.completions.create(**kwargs)


async def process_single_record(semaphore, record_id, prompt_text):
    async with semaphore:    
        try:
            response = await call_vllm_with_retry(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text} # Passing the pre-compiled markdown prompt directly
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=MAX_TOTAL_TOKENS
            )
            
            raw_content = response.choices[0].message.content
            parsed_data = parse_json(raw_content)

            if parsed_data is None:
                print(f"❌ Parse failed: {record_id}", flush=True)
                return {"id": record_id, "status": "error", "error": "json_parse_failed"}
            else:
                is_valid, clean_data, status_msg = validate_record(parsed_data, record_id)
                if is_valid:
                    print(f"✅ Success: {record_id}", flush=True)
                    return {"id": record_id, "status": "success", "data": clean_data}
                else:
                    print(f"❌ Schema fail: {record_id} | {status_msg[:100]}", flush=True)
                    return {
                        "id": record_id,
                        "status": "schema_error",
                        "data": parsed_data,
                        "error": status_msg
                    }
                
        except Exception as e:
            print(f"❌ Failed: {record_id} | Error: {str(e)[:100]}", flush=True)
            return {"id": record_id, "status": "error", "error": str(e)}


async def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    print("\n🔍 Locating raw AACT intermediate prompts...")

    dataset = []
    # Load the local prompt file and extract NCT_ID via regex
    with open(INPUT_PROMPTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            prompt_text = record["prompt"]
            
            # Deterministic ID Extraction
            id_match = re.search(r"Registry ID:\s*(NCT\d+)", prompt_text)
            if id_match:
                dataset.append({
                    "id": id_match.group(1),
                    "text": prompt_text
                })
            else:
                print("⚠️ Warning: Could not extract NCT ID from a prompt. Skipping.")

    # Resume logic
    completed_ids, failed_records = load_existing_results(OUTPUT_FILE)
    dataset = filter_dataset(dataset, completed_ids)

    if not dataset:
        print("✅ All records already processed!")
        return

    print(f"▶️ Processing {len(dataset)} records...", flush=True)
    
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    
    tasks = [
        process_single_record(semaphore, record["id"], record["text"]) for record in dataset
    ]
    
    processed_count = 0
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for task in asyncio.as_completed(tasks):
            result = await task
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno()) 

            processed_count += 1
            if processed_count % 100 == 0:
                print(f"📝 Persisted {processed_count}/{len(dataset)} records")

    print(f"🏁 Finished processing the records. Saved to {OUTPUT_FILE}")

# Initialize vLLM client
client = AsyncOpenAI(
    base_url=VLLM_API_BASE,
    api_key="vllm-local",
    timeout=1800.0
)
print(f"\n✅ vLLM server initialized with model {MODEL_ID}")

# ProtocolAI JSON Schema Configuration
JSON_SCHEMA = {
    "protocol": "The Official Title string of the trial.",
    "reasoning": "Step-by-step trace of how you mapped the unstructured text to relationships. Identify handling of missing variables.",
    "relationships": [
        {
            "source": "Entity Name String (e.g., Trial Name, Drug Name, Diagnosis, or specific Patient Criteria)",
            "type": "Must be exactly one of the strict ENUM predicates.",
            "target": "Target Entity String"
        }
    ]
}
SCHEMA_STRING = json.dumps(JSON_SCHEMA, indent=4)

# ProtocolAI System Prompt
SYSTEM_PROMPT = (
    "You are an expert clinical informatician constructing a medical knowledge graph. "
    f"Extract data strictly into this JSON schema:\n\n{SCHEMA_STRING}\n\n"
    "CRITICAL RULES:\n"
    "1. You MUST generate the 'reasoning' key sequentially BEFORE the 'relationships' array.\n"
    "2. If an entity or relationship is missing from the text, use literal null. Do not hallucinate variables.\n"
    "3. Use ONLY double quotes for all JSON keys and string values.\n"
    "4. Output raw JSON only — no markdown, no code blocks.\n"
    "5. Extract at max 15 most clinically significant relationships only.\n\n"
    "ENUM CONSTRAINTS — USE EXACTLY THESE STRINGS FOR 'type', NO SYNONYMS:\n"
    "- 'type': MUST be exactly one of: 'HAS_INDICATION', 'TESTS_INTERVENTION', 'USES_CONTROL', 'MEASURES_ENDPOINT', 'TARGETS_BIOMARKER', 'REQUIRES_CRITERION', 'EXCLUDES_CRITERION'\n"
)


if __name__ == "__main__":
    start_time = time.perf_counter()

    try:
        asyncio.run(main())
    finally:
        end_time = time.perf_counter()
        total_seconds = end_time - start_time
        readable_time = str(timedelta(seconds=int(total_seconds)))
        
        print("\n" + "="*40)
        print(f"BATCH RUN COMPLETE")
        print(f"Total Time: {readable_time}")
        print(f"Total Seconds: {total_seconds:.2f}")
        print("="*40)