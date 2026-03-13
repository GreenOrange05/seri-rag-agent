import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import pdfplumber
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Setup directories
INPUT_DIR = "data/"
OUTPUT_DIR = "data/cleaned_markdown/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model initialization
llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")

# Prompt template
template = """
You are a macroeconomic data extraction engine. Extract indicators into a standardized Markdown table: | Indicator | Year | Value |

RULES:
1. USE ECONOMIC COMMON SENSE FOR ALIGNMENT: Pair floating labels logically.
2. DO NOT INVENT VALUES: Extract exact mathematical values provided.
3. FIX EXTRACTION TYPOS: Correct obvious OCR errors.
4. Output ONLY the Markdown table.

Raw Text:
{raw_text}
"""
prompt = PromptTemplate.from_template(template)
chain = prompt | llm

# Batch processing loop
for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, f"{filename.replace('.pdf', '')}.md")
        
        print(f"Processing: {filename}...")
        
        with pdfplumber.open(pdf_path) as pdf:
            try:
                # Target a specific page index (Page 4) to avoid rate limits
                raw_text = pdf.pages[3].extract_text()
                
                if raw_text:
                    response = chain.invoke({"raw_text": raw_text})
                    
                    # Save the cleaned output
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(response.content)
                    print(f"-> Saved: {output_path}")
                else:
                    print(f"-> Skipped: No text found on target page.")
            except Exception as e:
                print(f"-> Error on {filename}: {e}")

print("\nBatch extraction complete.")