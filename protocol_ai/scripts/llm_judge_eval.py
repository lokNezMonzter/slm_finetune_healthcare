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
CONCURRENCY_LIMIT = 16  # Hard limit for 48GB VRAM (A6000)

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=1800.0)

# ==========================================
# RUN MODE
# ==========================================
# True: sanity mode — writes to sanity_*.jsonl, capped at SANITY_LIMIT records.
# False: full production run over all remaining records.
SANITY_MODE = False
SANITY_LIMIT = 100

# ==========================================
# PATHS
# ==========================================
INPUT_FILE = "/workspace/data/protocol_ai/03_silver_aact_sft.jsonl"

if SANITY_MODE:
    OUTPUT_FILE = "/workspace/data/protocol_ai/sanity_evaluated.jsonl"
    ERROR_OUTPUT_FILE = "/workspace/data/protocol_ai/sanity_errors.jsonl"
else:
    OUTPUT_FILE = "/workspace/data/protocol_ai/04_silver_aact_evaluated.jsonl"
    ERROR_OUTPUT_FILE = "/workspace/data/protocol_ai/aact_judge_errors.jsonl"

# ==========================================
# SYSTEM PROMPT
# Expanded ontology_violation examples cover the four failure patterns
# the previous sanity run missed. Score / hallucination_detected are NOT
# requested from the model — they are derived deterministically in Python.
# ==========================================
SYSTEM_PROMPT = """You are a Clinical Data Extraction Judge. For each extracted relationship, decide if it is (a) literally supported by the SOURCE TRIAL TEXT and (b) ontologically valid.

IGNORE your own medical knowledge. A relationship can be medically true and still be HALLUCINATED if it is not literally in the source. Judge textual faithfulness and ontology only.

ONTOLOGY — target must match:
- HAS_INDICATION → disease/condition/syndrome (not drug/vaccine)
- TESTS_INTERVENTION → drug/device/vaccine/procedure being tested
- USES_CONTROL → placebo/comparator/standard of care
- MEASURES_ENDPOINT → clinical metric or measured outcome
- REQUIRES_CRITERION / EXCLUDES_CRITERION → demographic/health rule
- TARGETS_BIOMARKER → gene/protein/molecular marker

VERDICTS per relationship:
- valid: literally supported AND ontology-correct
- ungrounded: entity or claim not in source
- ontology_violation: target wrong type. Common failures:
  * vaccine as HAS_INDICATION target (target must be a disease)
  * disease as TARGETS_BIOMARKER target (target must be a molecular marker)
  * drug class (e.g. "Factor Xa Inhibitor", "SGLT2 Inhibitor") as TARGETS_BIOMARKER
  * placebo/comparator/standard-of-care as TESTS_INTERVENTION (belongs under USES_CONTROL)
  * eligibility criteria or lab thresholds as TARGETS_BIOMARKER (belongs under REQUIRES/EXCLUDES_CRITERION)
- partial: entity in source but the specific claim is not stated

When ambiguous, mark ungrounded.

You will be given a numbered list of relationships. Emit exactly one finding for each numbered index, in order from 0 to N-1. Do not skip, merge, or reorder. If a relationship looks trivial or duplicate, still emit a finding for it. Populate "note" only when verdict is not "valid".
"""


def build_per_record_schema(n_rels: int) -> dict:
    """
    Dynamic schema per record. minItems and maxItems both equal n_rels,
    which forces the judge (via XGrammar) to emit exactly one finding
    per relationship — no silent drops, no padding. idx.maximum bounds
    the index range so idx misalignment becomes structurally impossible.
    """
    return {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "minItems": n_rels,
                "maxItems": n_rels,
                "items": {
                    "type": "object",
                    "properties": {
                        "idx": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": max(0, n_rels - 1),
                        },
                        "verdict": {
                            "type": "string",
                            "enum": ["valid", "ungrounded", "ontology_violation", "partial"],
                        },
                        "note": {"type": "string"},
                    },
                    "required": ["idx", "verdict"],
                },
            }
        },
        "required": ["findings"],
    }


def derive_score_and_hallucination(findings: list) -> tuple:
    """
    Deterministic aggregation. LLM-as-judge is unreliable at aggregating
    across a list of its own per-item verdicts (it contradicts itself
    routinely). We compute score and hallucination_detected in Python
    from the verdicts, so both fields are guaranteed self-consistent.

    Score ladder:
      5 — all valid
      3 — one or more partials, no outright failures
      2 — exactly one ungrounded or ontology_violation (no other issues)
      1 — multiple failures (ungrounded + ontology_violation >= 2)
    """
    verdicts = [f.get("verdict", "ungrounded") for f in findings]
    n_ungrounded = sum(1 for v in verdicts if v == "ungrounded")
    n_ontology = sum(1 for v in verdicts if v == "ontology_violation")
    n_partial = sum(1 for v in verdicts if v == "partial")
    n_failures = n_ungrounded + n_ontology

    if n_failures == 0 and n_partial == 0:
        score = 5
    elif n_failures == 0 and n_partial >= 1:
        score = 3
    elif n_failures == 1 and n_partial == 0:
        score = 2
    elif n_failures >= 2:
        score = 1
    else:
        # mixed partial + single failure — treat conservatively
        score = 2

    hallucination_detected = any(v != "valid" for v in verdicts)
    return score, hallucination_detected


async def evaluate_record(record_data, sem, pbar, success_file, error_file, file_lock):
    async with sem:
        record_id = record_data.get("id", "UNKNOWN")
        try:
            source_text = record_data.get("prompt")
            extracted_payload = record_data.get("data", {})
            relationships = extracted_payload.get("relationships", [])
            n_rels = len(relationships)

            # Edge case: no relationships to judge. Skip LLM call entirely.
            if n_rels == 0:
                record_data["judge_evaluation"] = {
                    "findings": [],
                    "score": 5,
                    "hallucination_detected": False,
                }
                async with file_lock:
                    success_file.write(json.dumps(record_data, ensure_ascii=False) + "\n")
                    success_file.flush()
                    os.fsync(success_file.fileno())
                pbar.set_postfix(ID=record_id, Score=5, Hall=False, N=0)
                return

            # Numbered relationship list — the [N] prefix in the prompt matches
            # the idx enforced by the schema. Judge cannot lose alignment.
            numbered_rels = "\n".join(
                f"[{i}] {json.dumps(rel, ensure_ascii=False)}"
                for i, rel in enumerate(relationships)
            )

            USER_PROMPT = (
                f"SOURCE TRIAL TEXT:\n{source_text}\n\n"
                f"RELATIONSHIPS TO EVALUATE ({n_rels} total, indices 0 through {n_rels - 1}):\n"
                f"{numbered_rels}\n\n"
                f"Emit exactly one finding for each of the {n_rels} indices above, "
                f"in order from 0 to {n_rels - 1}. Do not skip, merge, or reorder."
            )

            per_record_schema = build_per_record_schema(n_rels)

            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT},
                ],
                temperature=0.0,     # deterministic judging
                max_tokens=3072,     # headroom for records with 20+ relationships
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "judge_evaluation",
                        "schema": per_record_schema,
                    },
                },
            )

            raw_content = response.choices[0].message.content
            evaluation_json = json.loads(raw_content)
            findings = evaluation_json.get("findings", [])

            # Deterministic derivation — never trust the LLM for aggregation.
            score, hallucination_detected = derive_score_and_hallucination(findings)
            evaluation_json["score"] = score
            evaluation_json["hallucination_detected"] = hallucination_detected

            record_data["judge_evaluation"] = evaluation_json

            # Durable write — survives hard VM kill.
            async with file_lock:
                success_file.write(json.dumps(record_data, ensure_ascii=False) + "\n")
                success_file.flush()
                os.fsync(success_file.fileno())

            pbar.set_postfix(ID=record_id, Score=score, Hall=hallucination_detected, N=n_rels)

        except Exception as e:
            error_payload = {
                "id": record_id,
                "error": str(e),
                "record_data": record_data,
            }
            async with file_lock:
                error_file.write(json.dumps(error_payload, ensure_ascii=False) + "\n")
                error_file.flush()
                os.fsync(error_file.fileno())
        finally:
            pbar.update(1)


async def main():
    if not os.path.exists(INPUT_FILE):
        logger.error(f"Input file not found: {INPUT_FILE}")
        sys.exit(1)

    logger.info(f"Loading data from {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        all_records = [json.loads(line) for line in f]

    # RESUME LOGIC — use ONLY the success file, so any prior errors get
    # retried (transient failures from VM hibernation, etc.)
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line).get("id"))
                except json.JSONDecodeError:
                    # partial trailing line from a hard kill — skip, retry
                    continue

    records_to_process = [r for r in all_records if r.get("id") not in processed_ids]

    if SANITY_MODE:
        records_to_process = records_to_process[:SANITY_LIMIT]
        logger.info(
            f"SANITY_MODE=True — capping run at {SANITY_LIMIT} records, writing to {OUTPUT_FILE}"
        )

    total_records = len(records_to_process)
    skipped = len(all_records) - total_records

    if skipped > 0:
        logger.info(f"Resuming... Skipped {skipped} already evaluated records.")

    if total_records == 0:
        logger.info("All records have been evaluated. Exiting.")
        return

    logger.info(f"Starting async evaluation for {total_records} records using {MODEL_NAME}")

    start_time = time.time()
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    file_lock = asyncio.Lock()

    with open(OUTPUT_FILE, "a", encoding="utf-8") as success_file, \
         open(ERROR_OUTPUT_FILE, "a", encoding="utf-8") as error_file:

        with tqdm(total=total_records, desc="Evaluating records") as pbar:
            tasks = [
                evaluate_record(record, sem, pbar, success_file, error_file, file_lock)
                for record in records_to_process
            ]
            await asyncio.gather(*tasks)

    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    logger.info("Evaluation Phase Complete.")
    logger.info(f"Total Execution Time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")


if __name__ == "__main__":
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())