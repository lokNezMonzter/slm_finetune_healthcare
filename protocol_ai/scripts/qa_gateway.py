import json
import re
import time
from collections import Counter
from pathlib import Path

# Paths - Adjust if your workspace directory differs
INPUT_FILE = "/workspace/data/protocol_ai/aact_distilled_sanitized.jsonl"
OUTPUT_FILE = "/workspace/data/protocol_ai/aact_silver_sft.jsonl"

# The deterministic syntactic filter for Endpoints misclassified as Interventions
ENDPOINT_PATTERN = re.compile(
    r"score|index|change|rate|survival|incidence|time to|proportion|number of", 
    re.IGNORECASE
)

def run_qa_gateway():
    print(f"Running Deterministic QA Gateway...")
    start_time = time.time()
    
    # Audit Tracking Metrics
    metrics = {
        "total_processed": 0,
        "dropped_target_collapse": 0,
        "dropped_failed_integrity": 0,
        "duplicate_edges_pruned": 0,
        "enum_swaps": 0,
        "total_kept": 0
    }
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
         
        for line in infile:
            metrics["total_processed"] += 1
            record = json.loads(line.strip())
            
            data = record.get("data", {})
            relationships = data.get("relationships", [])
            reasoning = data.get("reasoning", "")
            
            # -------------------------------------------------------------
            # Phase 1: Target Collapse Check (The Repetition Hallucination)
            # -------------------------------------------------------------
            if len(relationships) > 5:
                # Extract all target strings
                targets = [rel.get("target", "") for rel in relationships]
                if targets:
                    _, count = Counter(targets).most_common(1)[0]
                    # If a single string makes up >75% of all targets, the model collapsed
                    if (count / len(targets)) > 0.75:
                        metrics["dropped_target_collapse"] += 1
                        continue  # SILENT DROP: Skips writing this row entirely
            
            # -------------------------------------------------------------
            # Phase 2 & 3: Graph Deduplication & Predicate Confusion
            # -------------------------------------------------------------
            seen_signatures = set()
            clean_relationships = []
            
            for rel in relationships:
                r_source = rel.get("source", "")
                r_type = rel.get("type", "")
                r_target = rel.get("target", "")
                
                # Phase 2: Predicate Confusion (Lexical Override)
                # If labeled an intervention, but sounds like a clinical metric -> swap to endpoint
                if r_type.upper() == "TESTS_INTERVENTION" and ENDPOINT_PATTERN.search(r_target):
                    r_type = "MEASURES_ENDPOINT"
                    rel["type"] = r_type
                    metrics["enum_swaps"] += 1
                    
                # Phase 3: Cryptographic / Set-Based Deduplication
                # Create a frozen signature of the edge
                signature = f"{r_source}|{r_type}|{r_target}"
                
                if signature in seen_signatures:
                    metrics["duplicate_edges_pruned"] += 1
                    continue  # Skip this edge, it's a duplicate
                
                # If unique, add to tracking set and append to clean list
                seen_signatures.add(signature)
                clean_relationships.append(rel)
                
            # Overwrite the row's payload with the deduplicated array
            record["data"]["relationships"] = clean_relationships
            
            # -------------------------------------------------------------
            # Phase 4: Final Integrity Check
            # -------------------------------------------------------------
            # Must have at least 1 valid edge and reasoning cannot be empty/truncated
            if len(clean_relationships) == 0 or len(reasoning) < 32:
                metrics["dropped_failed_integrity"] += 1
                continue  # SILENT DROP
                
            # -------------------------------------------------------------
            # WRITE TO DISK (Streaming)
            # -------------------------------------------------------------
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            metrics["total_kept"] += 1

    elapsed = time.time() - start_time
    
    # Terminal Output & Audit Log
    print("\n" + "="*50)
    print("QA GATEWAY COMPLETE")
    print("="*50)
    print(f"Total Rows Evaluated:      {metrics['total_processed']}")
    print(f"[-] Dropped (Target Loop): {metrics['dropped_target_collapse']}")
    print(f"[-] Dropped (Integrity):   {metrics['dropped_failed_integrity']}")
    print(f"[-] Duplicate Edges Cut:   {metrics['duplicate_edges_pruned']}")
    print(f"[+] ENUMs Fixed (Regex):   {metrics['enum_swaps']}")
    print(f"\nFinal Gold Rows:      {metrics['total_kept']}")
    print(f"Execution Time:          {elapsed:.2f} seconds")
    print("="*50)

if __name__ == "__main__":
    run_qa_gateway()