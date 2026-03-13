import os
import pdfplumber

INPUT_DIR = "data/"
# We will save the full raw text in a separate folder for Day 3
OUTPUT_DIR = "data/full_raw_text/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, f"{filename.replace('.pdf', '')}.txt")
        
        print(f"Extracting all pages from: {filename}...")
        
        full_text = []
        
        with pdfplumber.open(pdf_path) as pdf:
            # Loop through EVERY page in the document
            for i, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text()
                    if text:
                        # Add a page marker so the RAG knows where it is
                        full_text.append(f"\n\n--- PAGE {i + 1} ---\n\n")
                        full_text.append(text)
                except Exception as e:
                    print(f"  -> Error on page {i + 1}: {e}")
        
        # Save the massive text block to a file
        if full_text:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("".join(full_text))
            print(f"-> Saved full document: {output_path}")

print("\nFull corpus extraction complete.")