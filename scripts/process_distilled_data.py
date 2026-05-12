import sys
import json
from tqdm import tqdm 
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from custom_logger import setup_logger


# Initialize logger
logger = setup_logger("DataProcessing")


def load_dataset_from_hf(dataset_name="zhengyun21/PMC-Patients", file_name="PMC-Patients-V2.json"):
    file_path = hf_hub_download(
        repo_id=dataset_name, 
        filename=file_name, 
        repo_type="dataset"
    )

    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    return raw_data

def load_raw_text_mapping():
    """
    Loads the original PMC-Patients dataset and creates a lookup dictionary.
    This guarantees that raw text is matched to correct ID
    """
    dataset_name = "zhengyun21/PMC-Patients"
    file_name = "PMC-Patients-V2.json"

    logger.info(f"loading {dataset_name} dataset")
    dataset = load_dataset_from_hf(dataset_name, file_name)

    text_mapping = {}
    for record in dataset:
        uid = str(record["patient_uid"])
        text_mapping[uid] = record["patient"]

    logger.info(f"mapped {len(text_mapping)} records")
    return text_mapping


def is_valid(data):
    """
    Validates JSON schema, types and array contents.
    Returns True if valid, False otherwise
    """
    if not isinstance(data, dict):
        return False
    
    # Check exact required keys
    required_keys = {"summary", "clinical_reasoning", "relationships", "keywords"}
    if set(data.keys()) != required_keys:
        return False
    
    # Validate summary
    if not isinstance(data["summary"], str) or not data["summary"].strip():
        return False
    
    # Validate clinical_reasoning
    if not isinstance(data["clinical_reasoning"], str) or not data["clinical_reasoning"].strip():
        return False
    
    # Validate keywords array
    keywords = data["keywords"]
    if not isinstance(keywords, list) or len(keywords) == 0:
        return False
    
    # Ensure every item in keywords array is str
    if not all (isinstance(k, str) for k in keywords):
        return False
    
    # Validate relationships array
    relations = data["relationships"]
    if not isinstance(relations, list) or len(relations) == 0:
        return False

    # Validate every dict inside relationships array
    required_rel_keys = {
        "subject",
        "predicate",
        "object",
        "polarity",
        "certainty",
        "evidence"
    }
    for rel in relations:
        if not isinstance(rel, dict):
            return False
        if set(rel.keys()) != required_rel_keys:
            return False
        if not all(isinstance(rel[k], str) and rel[k].strip() for k in required_rel_keys):
            return False
        
    return True

def process_records(input_file, output_file):
    # Determine if we are inside interactive shell or background process
    is_atty = sys.stderr.isatty()

    # If not running inside terminal, update every 30s
    # If running inside terminal, update every 0.1s
    min_int = 0.1 if is_atty else 30

    text_mapping = load_raw_text_mapping()

    clean_count = 0
    error_count = 0
    missing_text_count = 0

    # Calculate the total lines for progress bar
    total_lines = sum(1 for _ in open(input_file, "r", encoding="utf-8"))

    logger.info(f"processing records from {input_file}...")

    with open(input_file, "r", encoding="utf-8") as infile, \
    open(output_file, "w", encoding="utf-8") as outfile:
        
        # Progress bar config for tqdm
        pbar = tqdm(
            infile,
            total=total_lines,
            desc="processing records",
            mininterval=min_int,
            disable=None
        )
        
        for _, line in enumerate(pbar, 1):
            try:
                record = json.loads(line.strip())

                # Drop records with error status
                if record.get("status") != "success":
                    error_count += 1
                    continue

                record_id = str(record.get("id", ""))
                data = record.get("data", {})

                # Schema validation
                if not is_valid(data):
                    error_count += 1
                    continue

                # Safe text merging
                if record_id in text_mapping:
                    record["raw_medical_text"] = text_mapping[record_id]
                    outfile.write(json.dumps(record) + "\n")
                    clean_count += 1
                else:
                    missing_text_count += 1

            except json.JSONDecodeError:
                error_count += 1
    
    logger.info("---- PIPELINE COMPLETE ----")
    logger.info(f"✅ clean record count: {clean_count}")
    logger.info(f"‼️ dropped (errors/bad schema) count: {error_count}")
    logger.info(f"‼️ dropped (missing text) count: {missing_text_count}")
    logger.info(f"⛔️ TOTAL DROPPED RECORDS: {error_count + missing_text_count}")


if __name__ == "__main__":
    process_records("distillation_results_v2.jsonl", "/workspace/data/distillation_results_cleaned.jsonl")
