import json
import re
import time
from pathlib import Path

INPUT_FILE = "/workspace/data/protocol_ai/aact_distilled.jsonl"
OUTPUT_FILE = "/workspace/data/protocol_ai/aact_distilled_sanitized.jsonl"

# The deterministic syntactic filter for biological/clinical targets
BIOMARKER_PATTERN = re.compile(
    r"[<>≤≥%]|positive|negative|mutation|amplified|expression|abnormal|clearance|levels", 
    re.IGNORECASE
)


def sanitize_dataset():
    start_time = time.time()

    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    # Ensure output path exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Tracking metrics
    processed_count = 0
    kept_count = 0
    biomarker_swaps = 0

    # Explicit drop counters
    status_errors = 0
    empty_arrays = 0
    missing_ids = 0

    print(f"\nDATA PRE-PROCESSING IN PROGRESS FOR : {input_path.name}...\n")

    with open(input_path, 'r', encoding='utf-8') as infile, \
    open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            processed_count += 1
            record = json.loads(line.strip())

            # Status purge: drop rows where status is not 'success'
            if record.get("status").lower() != "success":
                status_errors += 1
                continue

            data = record.get("data", {})
            relationships = data.get("relationships", [])

            # Empty array validation: drop if no clinical value extracted
            if not relationships:
                empty_arrays += 1
                continue

            nct_id = record.get("id")
            if not nct_id:
                missing_ids += 1
                continue

            clean_relationships = []
            for rel in relationships:
                # Source node normalization: overwrite long title with NCT_ID
                rel["source"] = nct_id

                # Biomarker regex swap: upgrade standard criteria to biological targets
                if rel.get("type").upper() == "REQUIRES_CRITERION":
                    target_str = rel.get("target", "")
                    
                    if BIOMARKER_PATTERN.search(target_str):
                        rel["type"] = "TARGETS_BIOMARKER"
                        biomarker_swaps += 1

                clean_relationships.append(rel)

            # Overwrite payload with clean, optimized array
            record["data"]["relationships"] = clean_relationships

            # Write the sanitized JSON string to disk
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept_count += 1

    elapsed_time = time.time() - start_time

    print("\n" + "="*40)
    print("SANITIZATION COMPLETE")
    print(f"Total Raw Records Processed: {processed_count}")
    print(f"[-] Dropped (Status Error):  {status_errors}")
    print(f"[-] Dropped (Empty Array):   {empty_arrays}")
    print(f"[-] Dropped (Missing ID):    {missing_ids}")
    print(f"[+] Clean Records Kept:      {kept_count}")
    print(f"\nBiomarker Edges Swapped:  {biomarker_swaps}")
    print(f"Execution Time:           {elapsed_time:.2f} seconds")
    print("="*40)

if __name__ == "__main__":
    sanitize_dataset()
