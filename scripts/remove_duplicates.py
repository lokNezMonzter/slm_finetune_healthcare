import json
import hashlib

from custom_logger import setup_logger

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = "/workspace/data/distillation_results_cleaned.jsonl"
OUTPUT_FILE = "/workspace/data/distillation_results_cleaned_deduped.jsonl"

logger = setup_logger("RemoveDuplicates")

def get_hash(text):
    """Generates a strict SHA-256 hash of the text to guarantee absolute uniqueness"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def purge_duplicates():
    seen_hashes = set()
    duplicate_count = 0
    unique_count = 0

    logger.info(f"Scanning {INPUT_FILE} for duplicates...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            record = json.loads(line)
            
            # Extract the raw text. If missing, use empty string.
            raw_text = record.get("raw_medical_text", "")
            
            # Create a unique fingerprint for this specific clinical note
            text_fingerprint = get_hash(raw_text)

            if text_fingerprint in seen_hashes:
                # We have seen this exact clinical note before. Drop it.
                duplicate_count += 1
                continue
            
            # First time seeing this note. Save it and remember the fingerprint.
            seen_hashes.add(text_fingerprint)
            f_out.write(line)
            unique_count += 1

    logger.info("\n--- RESULT ---\n")
    logger.info(f"CRITICAL PURGE COMPLETE")
    logger.info(f"Duplicates Destroyed: {duplicate_count}")
    logger.info(f"Unique Records Saved: {unique_count}")
    logger.info(f"New File: {OUTPUT_FILE}")

if __name__ == "__main__":
    purge_duplicates()