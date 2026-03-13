import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# 1. Load the FAISS Index (The Brain)
print("Loading local FAISS database...")
DB_PATH = os.path.abspath("data/faiss_index")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)
vectorstore = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print("-> Memory loaded.\n")

# 2. Initialize the Groq LLM (The Mouth)
print("Initializing Groq Llama-3.1-8B-Instant...")
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.0, # Zero-hallucination mode
    max_tokens=500
)

# 3. Design the Guardrail Prompt
system_prompt = (
    "You are an expert Singaporean economic policy analyst. Use the following retrieved context to answer the user's question. "
    "If the answer is not explicitly contained within the text, you must output exactly: 'Insufficient data in policy documents to answer this query.' "
    "\n\nContext: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# 4. Build the PURE LCEL Retrieval Chain (LangChain v1.0+ Standard)
print("Wiring the modern LCEL pipeline...\n")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# This is the modern syntax. It pipes the retriever into the prompt, into the LLM, and out to text.
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Exponential Backoff Wrapper
def invoke_with_backoff(query_text, max_retries=4):
    """Executes the RAG chain with exponential backoff for rate limits."""
    wait_time = 1
    for attempt in range(max_retries):
        try:
            return rag_chain.invoke(query_text)
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "rate limit" in error_msg:
                print(f"-> Rate limit hit. Retrying in {wait_time} seconds (Attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                wait_time *= 2 # Double the wait time
            else:
                raise e
    raise Exception("Max retries exceeded. Groq API is overwhelmed.")

# 6. The Adversarial Test (Hallucination Check)
adversarial_query = "What is the targeted inflation rate for the Bank of Japan in 2025?"
print(f"Executing Adversarial Test: '{adversarial_query}'\n")

response = invoke_with_backoff(adversarial_query)

print("--- AGENT RESPONSE ---")
print(response)
print("-" * 22)