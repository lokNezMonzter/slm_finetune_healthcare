import json
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from datasets import load_dataset, load_from_disk

IN_PATH = "data/distilled/pmc_patients"
OUT_PATH = "/mnt/huggingface/data/distilled/pmc_patients"
IN_FILE = "pmc_patients_distilled.jsonl"
CLEANED_FILE = "pmc_patients_distilled_cleaned.jsonl"
PREPARED_FILE = "pmc_patients_distilled_prepared.jsonl"
DATASET_NAME = "pmc-patients-distilled-medgemma-22B"


# Load env variables -  need for pushing to hub
load_dotenv()

dataset = load_dataset(
    "json",
    data_files=f"{IN_PATH}/{IN_FILE}",
    split="train"
)
print(f"\n✅ FILE {IN_FILE} LOADED WITH {len(dataset)} RECORDS...")

filtered_dataset = dataset.filter(
    lambda x: x["status"] is not None and x["status"].strip().lower() == "success",
)

# SAVE TO DISK
filtered_dataset.to_json(f"{IN_PATH}/{CLEANED_FILE}")

print(f"\n✅ CLEANED AND PROCESSED {len(filtered_dataset)} RECORDS FROM {IN_FILE}")
print(f"✅ SAVED CLEANED JSON TO {IN_PATH}/{CLEANED_FILE}")


# TODO: Un-comment this for batched execution
# Filter processes records one at a time by default
# For 1M+ records use batching 
# filtered_dataset = dataset.filter(
#     lambda batch: [s.lower() == "success" if s else False for s in batch["status"]],
#     batched=True
# )


def merge_raw_text(distilled_path: str, output_path: str):
    # 1. Load original PMC-Patients dataset
    orig_path = hf_hub_download(
        repo_id="zhengyun21/PMC-Patients",
        filename="PMC-Patients-V2.json",
        repo_type="dataset"
    )
    with open(orig_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    # 2. Build lookup: patient_uid → raw clinical text
    uid_to_text = {
        r.get("patient_uid", str(i)): r["patient"]
        for i, r in enumerate(raw_data)
    }
    
    # 3. Read distilled output, inject text, write merged file
    matched = 0
    missing = 0
    
    with open(distilled_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        
        for line in fin:
            rec = json.loads(line)
            rec_id = rec.get("id")
            
            if rec_id in uid_to_text:
                rec["text"] = uid_to_text[rec_id]
                matched += 1
            else:
                missing += 1
            
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    print(f"\n✅ Matched: {matched} | Missing: {missing}")
    print(f"Output: {output_path}")


# Run after distillation is fully complete
try:
    merge_raw_text(
        f"{IN_PATH}/{CLEANED_FILE}",
        f"{IN_PATH}/{PREPARED_FILE}"
    )
except Exception as e:
     print(f"⚠️ ERROR WHILE PREPARING {IN_PATH}/{CLEANED_FILE}: {e}")
else:

    try:
        # Load the cleaned jsonl file as HF dataset
        dataset = load_dataset(
            "json",
            data_files=f"{IN_PATH}/{PREPARED_FILE}",
            split="train"
        )
        print(f"✅ LOADED JSON FILE {PREPARED_FILE} FROM {IN_PATH}")

        # Shuffle and split the dataset before saving
        split_dataset = dataset.train_test_split(
            test_size=0.1, 
            seed=42,
            shuffle=True
        )

    except Exception as e:
        print(f"⚠️ ERROR WHILE LOADING FROM DISK: {e}")

    else:
        try:
            # Save to disk
            split_dataset.save_to_disk(f"{OUT_PATH}/{DATASET_NAME}")
            print(f"✅ SAVED {DATASET_NAME} TO {OUT_PATH}/{DATASET_NAME}")
        except Exception as e:
            print(f"⚠️ ERROR WHILE SAVING TO DISK: {e}")
        
        try:
            # Push to hub
            split_dataset.push_to_hub(
                f"loknezmonzter/{DATASET_NAME}",
                revision="distilled-medgemma-15k",
                commit_message="Created initial distilled dataset from PMC-Patients with 15k rows using Medgemma-22B"
            )
            print(f"✅ SAVED {DATASET_NAME} TO HF REPO: loknezmonzter/{DATASET_NAME}")
        except Exception as e:
            print(f"⚠️ ERROR WHILE PUSHING TO HUB: {e}")
