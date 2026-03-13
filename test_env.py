import os
from dotenv import load_dotenv
import langchain
import faiss
import gradio as gr
import pdfplumber
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load environment
load_dotenv()

# Retrieve API key
api_key = os.getenv("GROQ_API_KEY")

# Validation
if api_key:
    masked_key = f"{api_key[:4]}...{api_key[-4:]}"
    print(f"Success: API Key loaded -> {masked_key}")
else:
    print("Error: GROQ_API_KEY not found in .env file.")

print("Success: All core libraries imported without errors.")