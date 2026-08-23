"""
list_groq_models.py

Quick utility: prints every model currently available on your Groq account.
Run this whenever a model name in rag_query.py stops working.

Run it with:  uv run python src/rag/list_groq_models.py
"""

from pathlib import Path
import os
from dotenv import load_dotenv
from groq import Groq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

models = client.models.list()
print("Available Groq models:\n")
for model in models.data:
    print(f"  - {model.id}")
