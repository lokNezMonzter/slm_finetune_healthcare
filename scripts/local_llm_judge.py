import json
import asyncio
import os
import sys
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
from custom_logger import setup_logger

# Load env variables and setup logging
load_dotenv()
logger = setup_logger("AsyncEvaluation")

# ==========================================
# DEPLOYMENT CONFIGURATION
# ==========================================
MODE = "LOCAL" # Change to "DEEPSEEK" to use the API

if MODE == "LOCAL":
    # 1. Local Llama-3.3-70B-AWQ Settings for A6000 (48GB VRAM)
    API_KEY = "vllm-local"
    BASE_URL = "http://172.17.0.1:8000/v1"
    MODEL_NAME = "kosbu/Llama-3.3-70B-Instruct-AWQ" # Adjust to match your exact vLLM launch string
    CONCURRENCY_LIMIT = 4 # Hard limit to prevent A6000 OOM with 16-bit KV Cache
else:
    # 2. DeepSeek API Settings
    API_KEY = os.getenv("DEEPSEEK_API_KEY")
    BASE_URL = "https://api.deepseek.com"
    MODEL_NAME = "deepseek-chat" # Must use deepseek-chat. Reasoner does not support JSON mode.
    CONCURRENCY_LIMIT = 20 # DeepSeek API handles high concurrency

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=1800.0)

# ==========================================
# FILE PATHS
# ==========================================
INPUT_FILE = f"/workspace/data/distillation_results_cleaned_deduped.jsonl"
OUTPUT_FILE = f"/workspace/outputs/llm_judge_results.jsonl"
ERROR_OUTPUT_FILE = f"/workspace/outputs/llm_judge_errors.jsonl"

# ==========================================
# OBJECTIVE JUDGE PROMPT
# ==========================================
JUDGE_SYSTEM_PROMPT = """
You are a ruthless Clinical Audit AI. You do not grade on a curve. You do not give the benefit of the doubt.

You must use SUBTRACTIVE SCORING. Start every category at 10 and apply strict penalties based on the rules below. If a penalty applies, you MUST subtract the points.

### DEDUCTION RULES:
1. Summary Relevance (Start at 10):
   - Deduct 3 points if it misses the primary diagnosis, key treatment, or final outcome.
   - Deduct 2 points if it includes unnecessary chronological fluff.
2. Reasoning Quality (Start at 10):
   - Deduct 4 points if the reasoning merely summarizes the text instead of explaining the clinical 'Why' (Chain of Thought).
   - Deduct 3 points if the logic jumps to a conclusion without citing the underlying labs/symptoms from the text.
3. Relationship Validity (Start at 10):
   - Deduct 5 points if a subject-predicate-object triplet is medically false or hallucinated.
   - Deduct 2 points if a triplet is too vague to be useful for a student model.
4. Keyword Quality (Start at 10):
   - Deduct 3 points if it includes generic words (e.g., "patient", "fever", "hospital") instead of highly specific clinical entities (e.g., specific drugs, exact lab markers).

CRITICAL INSTRUCTION: You are STRICTLY FORBIDDEN from inventing new penalties, altering the point values, or deducting points for reasons not listed above. If a flaw does not perfectly match a rule above, you MUST NOT deduct points.

### EVALUATION STEPS:
1. Compare [RAW TEXT] to [EXTRACTION] line-by-line.
2. Draft your `judge_justification` first. You MUST explicitly list the exact penalties you are applying for each category (e.g., "Keywords contain generic terms like 'patient': -3 points. Reasoning lacks 'why': -4 points.").
3. Calculate the final scores by subtracting your stated penalties from 10. Calculate `overall_score` as the integer average of the 4 categories.

OUTPUT FORMAT:
Output ONLY a valid JSON object. 
{
    "judge_justification": "<List your exact arithmetic penalties here first>",
    "summary_score": 0,
    "reasoning_score": 0,
    "relationships_score": 0,
    "keywords_score": 0,
    "hallucination_detected": false,
    "overall_score": 0
}
"""

async def evaluate_record(record, sem, pbar, success_file, error_file, file_lock):
    """Processes a single record through the Async API with concurrency control"""
    async with sem:
        record_id = str(record.get("id", "unknown_id"))
        record_data = record.get("data", {})

        student_extraction = {
            "summary": record_data.get("summary", ""),
            "clinical_reasoning": record_data.get("clinical_reasoning", ""),
            "relationships": record_data.get("relationships", ""),
            "keywords": record_data.get("keywords", "")
        }

        # Truncate raw text to prevent context blowout
        raw_text = str(record.get("raw_medical_text", ""))

        user_content = f"[RAW TEXT]\n{raw_text}\n\n[EXTRACTION]\n{json.dumps(student_extraction)}"

        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0.0,
                max_tokens=500, # Use max_tokens for standard/chat models
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content.strip()}
                ]
            )
            
            # Parse result and append to original record
            judge_result = json.loads(response.choices[0].message.content)
            record["judge_evaluation"] = judge_result
            
            # Write to success file immediately
            async with file_lock:
                success_file.write(json.dumps(record) + "\n")
                success_file.flush()

        except Exception as e:
            # Capture failure logic
            error_payload = {
                "id": record_id,
                "error": str(e),
                "record_data": record_data
            }
            async with file_lock:
                error_file.write(json.dumps(error_payload) + "\n")
                error_file.flush()
        finally:
            pbar.update(1)

async def main():
    if not os.path.exists(INPUT_FILE):
        logger.error(f"Input file not found: {INPUT_FILE}")
        sys.exit(1)

    logger.info(f"Loading data from {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    
    total_records = len(records)
    logger.info(f"Starting async evaluation for {total_records} records using {MODE} ({MODEL_NAME})")

    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    file_lock = asyncio.Lock()

    # Open output files in append mode so progress isn't lost if the script is interrupted
    with open(OUTPUT_FILE, "a", encoding="utf-8") as success_file, \
         open(ERROR_OUTPUT_FILE, "a", encoding="utf-8") as error_file:
        
        with tqdm(total=total_records, desc=f"Evaluating records") as pbar:
            tasks = [
                evaluate_record(record, sem, pbar, success_file, error_file, file_lock)
                for record in records
            ]
            await asyncio.gather(*tasks)

    logger.info(f"Evaluation complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())