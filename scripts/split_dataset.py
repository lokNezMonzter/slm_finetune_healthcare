import json
import os

from custom_logger import setup_logger

logger = setup_logger("SplitDataset")

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = "/workspace/data/distillation_results_cleaned_deduped.jsonl"
PART1_FILE = "/workspace/data/distillation_part1.jsonl"
PART2_FILE = "/workspace/data/distillation_part2.jsonl"
PART3_FILE = "/workspace/data/distillation_part3.jsonl"
PART4_FILE = "/workspace/data/distillation_part4.jsonl"
PART5_FILE = "/workspace/data/distillation_part5.jsonl"
PART6_FILE = "/workspace/data/distillation_part6.jsonl"
PART7_FILE = "/workspace/data/distillation_part7.jsonl"
PART8_FILE = "/workspace/data/distillation_part8.jsonl"
PART9_FILE = "/workspace/data/distillation_part9.jsonl"
PART10_FILE = "/workspace/data/distillation_part10.jsonl"


def split_data():
    logger.info(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total = len(lines)
    midpoint = total // 10
    
    logger.info(f"Total Records: {total}. Splitting at {midpoint}...")
    
    with open(PART1_FILE, 'w', encoding='utf-8') as f1:
        f1.writelines(lines[:midpoint])
        
    with open(PART2_FILE, 'w', encoding='utf-8') as f2:
        f2.writelines(lines[midpoint:])

    with open(PART3_FILE, 'w', encoding='utf-8') as f3:
        f3.writelines(lines[midpoint:])

    with open(PART4_FILE, 'w', encoding='utf-8') as f4:
        f4.writelines(lines[midpoint:])

    with open(PART5_FILE, 'w', encoding='utf-8') as f5:
        f5.writelines(lines[midpoint:])

    with open(PART6_FILE, 'w', encoding='utf-8') as f6:
        f6.writelines(lines[midpoint:])

    with open(PART7_FILE, 'w', encoding='utf-8') as f7:
        f7.writelines(lines[midpoint:])

    with open(PART8_FILE, 'w', encoding='utf-8') as f8:
        f8.writelines(lines[midpoint:])

    with open(PART9_FILE, 'w', encoding='utf-8') as f9:
        f9.writelines(lines[midpoint:])

    with open(PART10_FILE, 'w', encoding='utf-8') as f10:
        f10.writelines(lines[midpoint:])
        
    logger.info(f"success! created all part files!")

if __name__ == "__main__":
    split_data()