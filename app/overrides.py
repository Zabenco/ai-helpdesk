import json
import os

OVERRIDES_FILE = "overrides.json"


def load_overrides() -> dict:
    if not os.path.exists(OVERRIDES_FILE):
        return {}
    with open(OVERRIDES_FILE, "r") as f:
        return json.load(f)
    
def save_overrides(data: dict):
    with open(OVERRIDES_FILE, "w") as f:
        json.dump(data, f, indent=2)

def get_override_for_question(question: str) -> str | None:
    overrides =load_overrides()
    for keyword, override_text in overrides.items():
        if keyword.lower() in question.lower():
            return override_text
    return None