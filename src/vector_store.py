import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import gc
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

INPUT_DIR = "data/full_raw_text/"
# Use absolute pathing to prevent VS Code terminal directory confusion
DB_PATH = os.path.abspath("data/faiss_index")

# 1. Load the entire corpus of economic text
print("Loading all raw text files...")
all_text = ""
for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".txt"):
        with open(os.path.join(INPUT_DIR, filename), "r", encoding="utf-8") as f:
            all_text += f.read() + "\n\n"

# 2. Execute Semantic Chunking
print("Chunking entire corpus...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.create_documents([all_text])
print(f"-> Created {len(docs)} total chunks for the database.\n")

# 3. Initialize Embedding Engine
print("Waking up embedding model (CPU mode)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# 4. Build and Persist the FAISS Index
print("\nBuilding FAISS Vector Store. This may take a few minutes depending on your CPU...")
vectorstore = FAISS.from_documents(docs, embeddings)

print(f"Saving database locally to: {DB_PATH}")
vectorstore.save_local(DB_PATH)

# 5. Flush RAM
del vectorstore
gc.collect()
print("\n-> Index saved and RAM cleared. The RAG brain is permanently stored.")