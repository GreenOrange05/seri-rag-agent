import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import time
import gradio as gr
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. GLOBAL STATE: LOAD HEAVY ASSETS ONCE
# ==========================================
print("Booting up SERI Agent Backend...")
print("Loading FAISS Database...")

# Use absolute pathing so Gradio doesn't get confused
DB_PATH = os.path.abspath("data/faiss_index")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)
vectorstore = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

print("Initializing Groq Llama-3.1-8B-Instant...")
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.0,
    max_tokens=500
)

system_prompt = (
    "You are an expert Singaporean economic policy analyst. Use the following retrieved context to answer the user's question. "
    "If the answer is not explicitly contained within the text, you must output exactly: 'Insufficient data in policy documents to answer this query.' "
    "\n\nContext: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# The Modern LCEL Chain
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print("-> Backend fully initialized and ready.\n")


# ==========================================
# 2. THE GRADIO INFERENCE WRAPPER
# ==========================================
def predict(message, history):
    """Executes the RAG chain with exponential backoff and streams to Gradio."""
    wait_time = 1
    max_retries = 4
    
    for attempt in range(max_retries):
        try:
            # Because of StrOutputParser, 'response' is already a clean string
            response = rag_chain.invoke(message)
            return response
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "rate limit" in error_msg:
                print(f"-> Rate limit hit. Retrying in {wait_time} seconds (Attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                wait_time *= 2
            else:
                return f"System Error: {str(e)}"
                
    return "Error: Max retries exceeded. Groq API is overwhelmed."


# ==========================================
# 3. CONSTRUCT THE WEB INTERFACE
# ==========================================
# Gradio ChatInterface handles all the UI, history, and styling automatically
demo = gr.ChatInterface(
    fn=predict,
    title="Singapore Macroeconomic Policy Agent",
    description="Zero-hallucination analysis of MAS and MTI economic reports."
)

if __name__ == "__main__":
    print("Launching Gradio Server...")
    # This locks up the terminal and opens the local web port
    demo.launch()