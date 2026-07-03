import json
import time
import asyncio
import os
import sys
import logging
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AsyncEvaluation")

load_dotenv()

# ==========================================
# DEPLOYMENT CONFIGURATION
# ==========================================
API_KEY = "vllm-local"
BASE_URL = "http://172.17.0.1:8000/v1"
MODEL_NAME = "kosbu/Llama-3.3-70B-Instruct-AWQ" 
CONCURRENCY_LIMIT = 16 # Hard limit for 48GB VRAM (A6000)

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=1800.0)

# ==========================================
# PATHS (Updated for ProtocolAI Pipeline)
# ==========================================
INPUT_FILE = "/workspace/data/protocol_ai/03_silver_aact.jsonl"
OUTPUT_FILE = "/workspace/data/protocol_ai/04_silver_aact_evaluated.jsonl"
ERROR_OUTPUT_FILE = "/workspace/data/protocol_ai/aact_judge_errors.jsonl"

# ==========================================
# EVALUATION PROMPTS
# ==========================================
SYSTEM_PROMPT = """You are an expert Clinical Data Extraction Judge.
Your absolute primary task is to cross-reference the provided EXTRACTED RELATIONSHIPS strictly against the provided SOURCE TRIAL TEXT.

The model extracted knowledge graph relationships using ONLY these exact ENUMs:
- HAS_INDICATION (The condition being studied)
- TESTS_INTERVENTION (The experimental drug/device)
- USES_CONTROL (The placebo or active comparator)
- MEASURES_ENDPOINT (The clinical outcome metric)
- REQUIRES_CRITERION (General demographic/health inclusion rules)
- EXCLUDES_CRITERION (All exclusion rules)
- TARGETS_BIOMARKER (Specific molecular/genetic targeting)

You must verify the following:
- If the extracted data and the relationships are medically accurate, correctly categorized, and grounded explicitly in the source text.
- If the extracted protocol title is actually present in the context.
- If the raw context explicitly supports and justifies the AI's clinical reasoning trace.

Identify any hallucinations (invented data, inferred metrics, or repetitive loops).

Output STRICTLY in the following JSON format:
{
  "medical_accuracy_score": <int between 1 and 5>,
  "hallucination_detected": <boolean>,
  "hallucination_details": "<string describing the error, or null if perfect>",
  "reasoning": "<string briefly explaining your logic>"
}"""

# ==========================================
# ASYNC WORKER FUNCTION
# ==========================================
async def evaluate_record(record_data, sem, pbar, success_file, error_file, file_lock):
    async with sem:
        record_id = record_data.get("id", "UNKNOWN")
        try:
            # Safely grab the text (handling potential nested prompt structures)
            source_text = record_data.get("text")
            extracted_payload = record_data.get("data", {})
            
            USER_PROMPT = f"""
            SOURCE TRIAL TEXT:
            {source_text}
            
            EXTRACTED DATA (To Evaluate):
            {json.dumps(extracted_payload, indent=2)}
            """
            
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT}
                ],
                # Force JSON format if supported by the vLLM engine
                response_format={"type": "json_object"},
                temperature=0.1 # Low temp for deterministic judging
            )
            
            raw_content = response.choices[0].message.content
            
            # SAFEGUARD: Strip markdown wrappers if the model ignores the json_object directive
            clean_content = raw_content.replace("```json", "").replace("```", "").strip()
            evaluation_json = json.loads(clean_content)
            
            # CORE INJECTION: Append the 70B evaluation to the 27B's data row
            record_data["judge_evaluation"] = evaluation_json
            
            async with file_lock:
                success_file.write(json.dumps(record_data, ensure_ascii=False) + "\n")
                success_file.flush()

            # LIVE TERMINAL OUTPUT: Update the progress bar with the latest result
            score = evaluation_json.get("medical_accuracy_score", "N/A")
            hallucinated = evaluation_json.get("hallucination_detected", False)
            pbar.set_postfix(ID=record_id, Score=score, Hallucinated=hallucinated)
                
        except Exception as e:
            error_payload = {
                "id": record_id,
                "error": str(e),
                "record_data": record_data
            }
            async with file_lock:
                error_file.write(json.dumps(error_payload, ensure_ascii=False) + "\n")
                error_file.flush()
        finally:
            pbar.update(1)

# ==========================================
# ORCHESTRATION
# ==========================================
async def main():
    
    if not os.path.exists(INPUT_FILE):
        logger.error(f"Input file not found: {INPUT_FILE}")
        sys.exit(1)

    logger.info(f"Loading data from {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        all_records = [json.loads(line) for line in f]
        
    # ---------------------------------------------------------
    # SAFEGUARD: Resume Logic (Do not evaluate the same row twice)
    # ---------------------------------------------------------
    processed_ids = set()
    for file_path in [OUTPUT_FILE, ERROR_OUTPUT_FILE]:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        processed_ids.add(json.loads(line).get("id"))
                    except json.JSONDecodeError:
                        continue
                        
    records_to_process = [r for r in all_records if r.get("id") not in processed_ids]
    
    total_records = len(records_to_process)
    skipped = len(all_records) - total_records
    
    if skipped > 0:
        logger.info(f"Resuming... Skipped {skipped} already processed records.")
    
    if total_records == 0:
        logger.info("All records have been evaluated. Exiting.")
        return

    logger.info(f"Starting async evaluation for {total_records} records using {MODEL_NAME}")

    # START THE CLOCK
    start_time = time.time()

    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    file_lock = asyncio.Lock()

    # Open output files in append mode
    with open(OUTPUT_FILE, "a", encoding="utf-8") as success_file, \
         open(ERROR_OUTPUT_FILE, "a", encoding="utf-8") as error_file:
        
        with tqdm(total=total_records, desc="Evaluating records") as pbar:
            tasks = [
                evaluate_record(record, sem, pbar, success_file, error_file, file_lock)
                for record in records_to_process
            ]
            await asyncio.gather(*tasks)
            
    # STOP THE CLOCK
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    logger.info("Evaluation Phase Complete.")
    logger.info(f"Total Execution Time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

if __name__ == "__main__":
    # Handle older Python event loop closing issues gracefully
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())