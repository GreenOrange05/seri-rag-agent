# SERI: Macroeconomic Policy RAG Agent 🇸🇬

A fully local, zero-hallucination Retrieval-Augmented Generation (RAG) pipeline designed to analyze decades of macroeconomic policy reports from the Monetary Authority of Singapore (MAS) and the Ministry of Trade and Industry (MTI). 

Built as the qualitative foundation for the **Singapore Economic Resilience Index (SERI)**, this agent bridges the gap between raw, unstructured economic PDFs and strict, deterministic data extraction.



##  System Architecture
This pipeline avoids basic API wrappers in favor of an enterprise-grade, fully local memory framework:
1. **Data Ingestion:** Automated extraction and semantic chunking of highly formatted MAS/MTI PDFs using `pdfplumber` and LangChain's `RecursiveCharacterTextSplitter`.
2. **Local Vectorization:** Text chunks are embedded into a 384-dimensional mathematical space using Hugging Face (`all-MiniLM-L6-v2`) and persistently indexed on local storage via **FAISS**.
3. **LCEL Retrieval Chain:** A modern LangChain Expression Language (LCEL) pipeline retrieves the $L_2$ nearest-neighbor chunks and pipes them into a highly constrained context window.
4. **Zero-Hallucination Generation:** Powered by `Llama-3.1-8B-Instant` via the Groq API, operating at `temperature=0.0`. A strict guardrail prompt forces the model to reject queries ("Insufficient data...") rather than hallucinate external knowledge.
5. **Interactive UI:** A lightweight, stateless web interface built with **Gradio**.

##  Quantitative Evaluation (LLM-as-a-Judge)
To scientifically prove the pipeline's reliability, I engineered an automated evaluation framework. Using a manually curated Ground Truth dataset of MAS/MTI facts, a secondary LLM pipeline mathematically scored the agent's outputs, resulting in:
* **High Faithfulness:** The agent consistently refuses to hallucinate when faced with out-of-domain adversarial queries.
* **Proven Edge Cases:** The framework successfully identified the limitations of semantic vector search for isolated numerical lookups, highlighting the future necessity of a hybrid BM25/FAISS approach or a dedicated Pandas CSV agent for structured quantitative data.

## Tech Stack
* **Framework:** LangChain (v1.0+ native LCEL)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** Hugging Face (`sentence-transformers`)
* **LLM:** Meta Llama 3.1 (via Groq API)
* **Frontend:** Gradio
* **Data Engineering:** Pandas, pdfplumber

##  Quick Start
```bash
# Clone the repository
git clone [https://github.com/yourusername/seri-rag-agent.git](https://github.com/yourusername/seri-rag-agent.git)
cd seri-rag-agent

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Add your API key
# Create a .env file in the root directory and add: GROQ_API_KEY="your_key_here"

# Launch the Web Interface
python src/app.py