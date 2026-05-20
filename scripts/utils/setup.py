# setup.py
import subprocess
import sys

def run_setup(new, model):
    """Run all setup checks before training"""

    if not new:
        print("Running cleanup operations on broken dirs...")
        try:
            result = subprocess.run(
                ["bash", "-c", f"rm -rf /mnt/huggingface/models/{model}"],
                capture_output=True,
                text=True
            )
            # result = subprocess.run(
            #     ["bash", "-c", f"rm -rf /workspace/logs/*.log"],
            #     capture_output=True,
            #     text=True
            # )
            result = subprocess.run(
                ["bash", "-c", f"rm -rf /workspace/.cache/"],
                capture_output=True,
                text=True
            )
            print("\n✅ Cleanup complete\n")
        except Exception as e:
            print(f"⚠️ Cleanup failed with error: {e}")
    
    print("Checking GPU availability...")
    try:
        result = subprocess.run(
            ["bash", "-c", "nvidia-smi | head -20"],
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except Exception as e:
        print(f"⚠️ GPU check failed with error: {e}")
    
    print("\n✅ Setup complete, ready to train\n")

if __name__ == "__main__":
    run_setup()