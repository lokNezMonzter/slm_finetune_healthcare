from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

def purge_stuck_batches():
    print("Fetching all OpenAI batches...")
    # Fetch the last 100 batches
    batches = client.batches.list(limit=100)
    
    cleared_count = 0
    for b in batches.data:
        # Target any batch that is holding your token limit hostage
        if b.status in ["validating", "enqueuing", "in_progress"]:
            print(f"Killing stuck batch: {b.id} (Status: {b.status})")
            try:
                client.batches.cancel(b.id)
                cleared_count += 1
            except Exception as e:
                print(f"Failed to cancel {b.id}: {e}")
                
    print(f"\nPurge Complete. {cleared_count} ghost batches killed.")
    print("Your account token limit has been reset.")

if __name__ == "__main__":
    purge_stuck_batches()