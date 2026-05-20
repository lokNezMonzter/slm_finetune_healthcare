import os
import json
import asyncio
import time
import random
from datetime import timedelta
from openai import AsyncOpenAI
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from scripts.utils2.colors import printf

load_dotenv()

# Configuration
VLLM_API_BASE = "http://172.17.0.1:8000/v1" # Using the Docker bridge IP we established
MODEL_ID = "unsloth/medgemma-27b-text-it-bnb-4bit"
CONCURRENT_REQUESTS = 6 # vLLM will batch these together on the A6000
MAX_TOTAL_TOKENS = 4096

client = AsyncOpenAI(
    base_url=VLLM_API_BASE,
    api_key="vllm-local",
    timeout=1800.0
)

# [INSERT YOUR UPDATED SYSTEM PROMPT HERE]
SYSTEM_PROMPT = """You are an expert clinical informatician. Your task is to extract highly structured knowledge from raw clinical text to train smaller models.

IMPORTANT INSTRUCTIONS!! DO NOT IGNORE!!
=========================================

- You must analyze the text and return a strict JSON object following this EXACT schema
- The 'clinical_resoning' key in the schema MUST be strictly limited to 2 or 3 sentences. DO NOT write long paragraphs.

EXACT SCHEMA:
-------------

{
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

Output ONLY valid JSON. Do not include markdown formatting like ```json."""


async def process_single_record(semaphore, record_id, clinical_text):
    """Processes a single record, controlled by a semaphore to prevent overwhelming the queue."""
    async with semaphore:
        try:
           
            response = await client.chat.completions.create(
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
            parsed_data = json.loads(raw_content)
        
        except Exception as e:
            print(f"❌ Failed: {record_id} | Error: {str(e)[:100]}", flush=True)
            return {"record_id": record_id, "status": "error", "error": str(e)}
        
        else:
            print(f"✅ Success: {record_id}", flush=True)
            return {"id": record_id, "status": "success", "data": parsed_data}

async def main():
    
    printf("Locating raw PMC-Patients dataset...")
    # dataset = load_dataset("zhengyun21/PMC-Patients", split="train")

    file_path = hf_hub_download(
        repo_id="zhengyun21/PMC-Patients", 
        filename="PMC-Patients-V2.json", 
        repo_type="dataset"
    )

    # Exclude the first 1,000 records you already processed
    # total_records = len(dataset)
    # remaining_records = dataset.select(range(total_records))

    # Shuffle the remaining pool 
    # shuffled_pool = remaining_records.shuffle(seed=42)

    # Grab your random 1,000 subset
    # target_dataset = shuffled_pool.select(range(5555))

    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    remaining_pool = raw_data[1000:]

    random.seed(42)
    random.shuffle(remaining_pool)

    target_dataset = remaining_pool[:4567]

    dataset = []
    for i, record in enumerate(target_dataset):
        dataset.append({
            "id": record.get("patient_uid", str(i)),
            "text": record["patient"]
        })

    print(f"Starting batch extraction of {len(dataset)} records...", flush=True)
    
    # Semaphore restricts us to sending CONCURRENT_REQUESTS at a time
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    
    # Create the task queue
    tasks = [process_single_record(semaphore, record["id"], record["text"]) for record in dataset]
    
    # Process and save incrementally to avoid losing data if it crashes
    with open("distillation_results.jsonl", "a") as f:
        for task in asyncio.as_completed(tasks):
            result = await task
            f.write(json.dumps(result) + "\n")
            f.flush() # Force write to disk immediately

    printf(f"Finished processing the records. Saved to distillation_results.jsonl")


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
        