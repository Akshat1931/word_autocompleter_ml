from fastapi import FastAPI
from model import AutocompleteModel
import json

# Load the dataset
with open("words.json", "r") as f:
    data = json.load(f)
words = list(data.keys())

# Create model instance
model = AutocompleteModel(words)

# Start FastAPI app
app = FastAPI()

@app.get("/suggest")
def get_suggestions(prefix: str):
    return {
        "prefix": prefix,
        "suggestions": model.suggest(prefix)
    }
