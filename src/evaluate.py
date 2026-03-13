import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import time
import json
import pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ==========================================
# 1. INITIALIZE THE RAG PIPELINE (The Student)
# ==========================================
print("Booting up RAG Pipeline...")
DB_PATH = os.path.abspath("data/faiss_index")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
vectorstore = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) # Keeping it at 4 for the baseline test

llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.0, max_tokens=500)

system_prompt = (
    "You are an expert Singaporean economic policy analyst. Use the following retrieved context to answer the user's question. "
    "If the answer is not explicitly contained within the text, you must output exactly: 'Insufficient data in policy documents to answer this query.' "
    "\n\nContext: {context}"
)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = {"context": retriever | format_docs, "input": RunnablePassthrough()} | prompt | llm | StrOutputParser()

# ==========================================
# 2. INITIALIZE THE EVALUATOR (The Judge)
# ==========================================
print("Booting up LLM-as-a-Judge...")
# Forcing JSON output mode
judge_llm = ChatGroq(
    model_name="llama-3.1-8b-instant", 
    temperature=0.0
).bind(response_format={"type": "json_object"})

judge_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an impartial grading AI. You will be provided with a User Question, an Expected Answer, and a Generated Answer. Your job is to determine if the Generated Answer is factually consistent with the Expected Answer. Output strictly valid JSON containing a 'score' of 1 for Pass or 0 for Fail, and a short 'reason'."),
    ("human", "Question: {question}\nExpected: {expected}\nGenerated: {generated}")
])

judge_chain = judge_prompt | judge_llm | StrOutputParser()

def parse_json_response(response_str):
    """Strips markdown artifacts and parses JSON safely."""
    try:
        clean_str = response_str.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_str)
    except Exception as e:
        return {"score": 0, "reason": f"JSON Parsing Error: {str(e)}"}

# ==========================================
# 3. THE EVALUATION LOOP
# ==========================================
print("\nLoading Ground Truth Dataset...")
df = pd.read_csv("data/ground_truth.csv")
total_questions = len(df)
passed = 0

print("--- STARTING EVALUATION ---\n")

for index, row in df.iterrows():
    question = row['question']
    expected = row['expected_answer']
    
    print(f"[{index + 1}/{total_questions}] Evaluating: {question}")
    
    # Step A: Generate Answer (The Student)
    try:
        generated = rag_chain.invoke(question)
    except Exception as e:
        generated = f"Error: {str(e)}"
        
    # Step B: Grade Answer (The Judge)
    try:
        judge_raw = judge_chain.invoke({
            "question": question,
            "expected": expected,
            "generated": generated
        })
        evaluation = parse_json_response(judge_raw)
    except Exception as e:
        evaluation = {"score": 0, "reason": "Judge API Error"}
        
    # Step C: Log Results
    score = evaluation.get("score", 0)
    passed += score
    
    print(f"  Expected:  {expected}")
    print(f"  Generated: {generated[:100]}...") # Print first 100 chars to save terminal space
    print(f"  Score:     {'PASS' if score == 1 else 'FAIL'}")
    print(f"  Reason:    {evaluation.get('reason', 'N/A')}\n")
    
    # CRITICAL: Sleep to avoid Groq 429 Rate Limits
    time.sleep(2)

# ==========================================
# 4. FINAL SCORECARD
# ==========================================
print("--- RAG EVALUATION COMPLETE ---")
print(f"Total Questions: {total_questions}")
print(f"Faithfulness Score: {(passed / total_questions) * 100:.1f}%")
print("-------------------------------")