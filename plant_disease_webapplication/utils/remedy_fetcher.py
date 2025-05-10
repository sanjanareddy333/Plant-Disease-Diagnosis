# utils/remedy_fetcher.py

import json
import os

# Load remedies once at the start
with open("remedies.json") as f:
    all_remedies = json.load(f)

def fetch_remedies_offline(disease_name):
    # Use full key like "Tomato___Late_blight"
    remedy = all_remedies.get(disease_name)
    if remedy:
        return {
            "cause": remedy.get("cause", "Cause information not found."),
            "natural": remedy.get("natural", "Natural remedy not available."),
            "pesticide": remedy.get("pesticide", "Pesticide remedy not available.")
        }
    else:
        return {
            "cause": "No info available.",
            "natural": "Natural remedy not available.",
            "pesticide": "Pesticide remedy not available."
        }
