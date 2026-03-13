import warnings
# Mute the Pydantic warning BEFORE LangChain loads
warnings.filterwarnings("ignore", category=UserWarning)

import gc
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load your raw text from Day 2
sample_file = "data/full_raw_text/FullReport_AES2025.txt"
with open(sample_file, "r", encoding="utf-8") as f:
    raw_text = f.read()

# 2. Configure the Splitter (1000 chars with a 150 char sliding window overlap)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", " ", ""]
)

print("Chunking text...")
docs = text_splitter.create_documents([raw_text])
print(f"-> Created {len(docs)} individual chunks.\n")

# 3. Load the Local Hugging Face Model
print("Loading Hugging Face embedding model (this may take a minute to download weights)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'} # CRITICAL: Forces CPU usage
)

# 4. The Vector Test
print("\nTesting vectorization on Chunk #1...")
test_chunk = docs[0].page_content

vector = embeddings.embed_query(test_chunk)

# 5. Validation & Memory Cleanup
try:
    assert len(vector) == 384, f"FAIL: Expected 384 dimensions, got {len(vector)}"
    print("SUCCESS: Vector array contains exactly 384 dimensions.")
    print(f"Preview: [{vector[0]:.4f}, {vector[1]:.4f}, {vector[2]:.4f} ...]")
except AssertionError as e:
    print(e)
finally:
    del vector
    gc.collect()
    print("-> RAM cleared. Day 3 test complete.")