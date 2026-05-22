import json
import json5
import re
from typing import Dict, Optional


def parse_json(raw_text: str) -> Optional[Dict]:
    """Parse LLM JSON output with repair fallback for common syntax errors.
    """
    text = raw_text.strip()
    
    # Strip markdown fences if the model ignored instructions
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()
    
    # Try strict JSON parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Extract first {...} block (sometimes models add chitchat)
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    if match:
        text = match.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    
    # Repair single-quote JSON using json5 (properly handles ' vs ")
    try:
        return json5.loads(text)
    except Exception:
        pass
    
    # Unrecoverable
    return None