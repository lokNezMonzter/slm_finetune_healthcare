import time
import json
from pathlib import Path


INPUT_FILE = "/workspace/data/protocol_ai/sanity_evaluated.jsonl"
OUTPUT_FILE = "/workspace/data/protocol_ai/examples.jsonl"


def filter_records():
    """
    Stratified sampler for the sanity run. Pulls a small set of records
    across four categories so the judge output can be inspected before
    the full 34k run:

      Cat A — score=5, rels<=5    (short + clean, sanity baseline)
      Cat B — score in [1, 2]     (real failures — ungrounded/ontology_violation)
      Cat C — score=5, rels>5     (long + clean, tests coverage on complex records)
      Cat D — score=3             (partials — tests the mid-range verdict)

    NOTE ON SCORE LADDER: main.py derives score deterministically as
    5 / 3 / 2 / 1 (no 4). If you're running this against output from an
    older main.py that emitted 4s, adjust the Cat D condition.
    """
    start_time = time.time()
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    kept_count = 0
    status_errors = 0
    missing_ids = 0
    missing_eval = 0

    # Running tallies — these are the actual gate values, not local
    # counters. This is what the earlier buggy version got wrong.
    cat_a_count = 0  # short + clean
    cat_b_count = 0  # failures
    cat_c_count = 0  # long + clean
    cat_d_count = 0  # partials (mid-range)

    # Per-category caps
    CAP_A, CAP_B, CAP_C, CAP_D = 8, 8, 5, 4

    print(f"\nFILTERING : {input_path.name}...\n")

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line in infile:
            processed_count += 1

            try:
                record = json.loads(line.strip())
            except json.JSONDecodeError:
                # partial trailing line from a hard kill — skip
                continue

            status = record.get("status")
            if not status or status.lower() != "success":
                status_errors += 1
                continue

            nct_id = record.get("id")
            if not nct_id:
                missing_ids += 1
                continue

            judge_eval = record.get("judge_evaluation")
            if not judge_eval:
                missing_eval += 1
                continue

            relationships = record.get("data", {}).get("relationships", [])
            n_rels = len(relationships)
            score = judge_eval.get("score", 0)

            # Elif chain: each record fills at most one bucket, first match wins.
            # Order chosen so more specific / rarer categories are checked first
            # (long+clean before short+clean, failures before partials).
            matched_category = None

            if cat_c_count < CAP_C and score == 5 and n_rels > 5:
                matched_category = "cat_c"
            elif cat_a_count < CAP_A and score == 5 and n_rels <= 5:
                matched_category = "cat_a"
            elif cat_b_count < CAP_B and score in [1, 2]:
                matched_category = "cat_b"
            elif cat_d_count < CAP_D and score == 3:
                matched_category = "cat_d"

            if matched_category is None:
                continue

            if matched_category == "cat_a":
                cat_a_count += 1
            elif matched_category == "cat_b":
                cat_b_count += 1
            elif matched_category == "cat_c":
                cat_c_count += 1
            elif matched_category == "cat_d":
                cat_d_count += 1

            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept_count += 1

            # Early exit — every bucket full, no reason to keep scanning.
            if (cat_a_count >= CAP_A and cat_b_count >= CAP_B
                    and cat_c_count >= CAP_C and cat_d_count >= CAP_D):
                break

    elapsed_time = time.time() - start_time

    print("\n" + "=" * 55)
    print("FILTERING COMPLETE")
    print(f"Total Raw Records Scanned:        {processed_count}")
    print(f"[-] Dropped (Status Error):       {status_errors}")
    print(f"[-] Dropped (Missing ID):         {missing_ids}")
    print(f"[-] Dropped (Missing Judge Eval): {missing_eval}")
    print(f"[+] Clean Records Kept:           {kept_count}")
    print(f"[+] Cat A (s=5, rels<=5):         {cat_a_count} / {CAP_A}")
    print(f"[+] Cat B (s in [1,2]):           {cat_b_count} / {CAP_B}")
    print(f"[+] Cat C (s=5, rels>5):          {cat_c_count} / {CAP_C}")
    print(f"[+] Cat D (s=3):                  {cat_d_count} / {CAP_D}")
    print(f"Execution Time:                   {elapsed_time:.2f} seconds")
    print("=" * 55)


if __name__ == "__main__":
    filter_records()