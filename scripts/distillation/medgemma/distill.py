import os
import json
import asyncio
import time
import random
import functools
from datetime import timedelta
from openai import AsyncOpenAI
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential,
    retry_if_exception_type
)

# User defined modules
from validate_schema import ClinicalExtraction, validate_record
from parser import parse_json

load_dotenv()

# Configuration
VLLM_API_BASE = "http://172.17.0.1:8000/v1" # Using the Docker bridge IP we established
MODEL_ID = "unsloth/medgemma-27b-text-it-bnb-4bit"
CONCURRENT_REQUESTS = 32 # vLLM will batch these together on the A6000
MAX_TOTAL_TOKENS = 4096
OUTPUT_FILE = "/workspace/data/distilled/pmc_patients/pmc_patients_distilled.jsonl"


def load_existing_results(filepath):
    """
    Returns (completed_ids, failed_records)
    """
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
    """
    Remove all records that were processed successfully
    """
    original_count = len(dataset)
    filtered = [r for r in dataset if r["id"] not in completed_ids]
    skipped = original_count - len(filtered)

    if skipped:
        print(f"⏩ Skipping {skipped} already-completed records")
    
    return filtered

# Function to deal with vLLM OOM, timeout or server failures
@retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        reraise=True
)
async def call_vllm_with_retry(**kwargs):
    return await client.chat.completions.create(**kwargs)


async def process_single_record(semaphore, record_id, clinical_text):
    """
    Processes a single record, controlled by a semaphore.
    Prevents overwhelming the queue
    """
    async with semaphore:    
        try:
            response = await call_vllm_with_retry(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Extract structured clinical data from:\n\n{clinical_text}"}
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

    random.seed(42)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    print("\n🔍 Locating raw PMC-Patients dataset...")

    file_path = hf_hub_download(
        repo_id="zhengyun21/PMC-Patients", 
        filename="PMC-Patients-V2.json", 
        repo_type="dataset"
    )

    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    random.shuffle(raw_data)
    target_dataset = raw_data[:15000]

    dataset = []
    for i, record in enumerate(target_dataset):
        dataset.append({
            "id": record.get("patient_uid", str(i)),
            "text": record["patient"]
        })

    # Resume logic - load existing progress and de-duplicate
    completed_ids, failed_records = load_existing_results(OUTPUT_FILE)
    dataset = filter_dataset(dataset, completed_ids)

    if not dataset:
        print("✅ All records already processed!")
        return

    print(f"▶️ Processing {len(dataset)} records...", flush=True)
    
    # Semaphore restricts us to sending CONCURRENT_REQUESTS at a time
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    
    # Create the task queue
    tasks = [
        process_single_record(semaphore, record["id"], record["text"]) for record in dataset
    ]
    
    # Process and save incrementally to avoid losing data if it crashes
    processed_count = 0
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for task in asyncio.as_completed(tasks):
            result = await task
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush() # Force write to disk immediately
            os.fsync(f.fileno())  # Force OS-level flush to disk

            processed_count += 1
            if processed_count % 100 == 0:
                print(f"📝 Persisted {processed_count}/{len(dataset)} records")

    print(f"🏁 Finished processing the records. Saved to {OUTPUT_FILE}")

# Initialize vLLM client; vLLM uses OpenAI wrappers
client = AsyncOpenAI(
    base_url=VLLM_API_BASE,
    api_key="vllm-local",
    timeout=1800.0
)
print(f"\n✅ vLLM server initialized with model {MODEL_ID}")

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
    "Prioritize: diagnosis > treatment > symptoms > history.\n\n"
    "ENUM CONSTRAINTS — USE EXACTLY THESE STRINGS, NO SYNONYMS:\n"
    "- polarity: MUST be exactly 'positive' or 'negative'\n"
    "- certainty: MUST be exactly 'confirmed', 'suspected', or 'hedged'\n"
    "- predicate: MUST be exactly one of: 'HAS_HISTORY', 'SHOWS_SYMPTOM', 'DIAGNOSED_WITH', 'PRESCRIBED'\n"
    "Do NOT use 'presumed', 'likely', 'probable', 'definite', 'possible', or any other variation.\n"
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
        