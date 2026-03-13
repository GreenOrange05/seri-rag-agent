import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DB_PATH = os.path.abspath("data/faiss_index")

# 1. Re-Initialize Embeddings (FAISS needs this to translate your query into a vector)
print("Initializing embedding model for cold start...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# 2. Load the Index (The Major Gotcha Bypass)
print("Loading FAISS database from disk...")
vectorstore = FAISS.load_local(
    DB_PATH, 
    embeddings, 
    allow_dangerous_deserialization=True # CRITICAL: Bypasses the pickle security block
)
print("-> Database loaded successfully.\n")

# 3. Execute the L2 Similarity Search
query = "What is the projected core inflation rate for the upcoming quarter?"
print(f"Executing search for: '{query}'\n")

# Retrieve the top 4 closest geometric matches
docs = vectorstore.similarity_search(query, k=4)

print("--- TOP 4 RETRIEVED CHUNKS ---")
for i, doc in enumerate(docs):
    print(f"\n[Match {i+1}]")
    print(doc.page_content)
    print("-" * 40)