# full_orchestration.py
"""
Hybrid SQL + RAG Orchestration (Updated for new LangChain APIs)
This version:
- Loads a SINGLE prebuilt FAISS index (accident table only)
- Uses SentenceTransformer embeddings (all-MiniLM-L6-v2)
- Uses retriever.invoke() instead of get_relevant_documents()
- Supports SQL, RAG, or BOTH routing
"""

import os
from typing import Literal
from dotenv import load_dotenv
load_dotenv("../config/.env")

# ----------------------------------------------
# 1) IMPORT MODULES
# ----------------------------------------------
from sql_query_chain import ask_fars_database
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ----------------------------------------------
# 2) LOAD ACCIDENT FAISS INDEX
# ----------------------------------------------

FAISS_PATH = "../../accident_master_faiss"

print(f"Loading FAISS vectorstore from: {FAISS_PATH}")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.load_local(
    FAISS_PATH,
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

accident_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ----------------------------------------------
# 2a) LOAD FARS CODEBOOK FAISS INDEX
# ----------------------------------------------
CODEBOOK_FAISS_PATH = "../../fars_codebook_faiss"

print(f"Loading Codebook FAISS vectorstore from: {CODEBOOK_FAISS_PATH}")

codebook_vectorstore = FAISS.load_local(
    CODEBOOK_FAISS_PATH,
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

codebook_retriever = codebook_vectorstore.as_retriever(search_kwargs={"k": 4})

# ----------------------------------------------
# 3) SIMPLE RAG QA (supporting both retrievers)
# ----------------------------------------------
class SimpleRAGQA:
    def __init__(self, accident_retriever, codebook_retriever, llm):
        self.accident_retriever = accident_retriever
        self.codebook_retriever = codebook_retriever
        self.llm = llm

    def answer(self, query: str):
        # Query both FAISS indexes
        accident_docs = self.accident_retriever.invoke(query)
        codebook_docs = self.codebook_retriever.invoke(query)

        # Combine results
        source_docs = accident_docs + codebook_docs

        context = "\n\n".join(doc.page_content for doc in source_docs)

        prompt = f"""
You are a helpful assistant answering questions based on the provided context.

Use ONLY the information in the context. If the answer isn't in the context,
say you don't know.

Context:
{context}

Question:
{query}

Answer in clear, concise English:
"""
        response = self.llm.invoke(prompt)
        return getattr(response, "content", str(response)).strip()

rag_llm = ChatOllama(model="llama3", temperature=0.0)
rag_qa = SimpleRAGQA(accident_retriever, codebook_retriever, rag_llm)

def run_rag(question: str) -> str:
    return rag_qa.answer(question)

# ----------------------------------------------
# 4) ROUTER LLM
# ----------------------------------------------
router_llm = ChatOllama(model="llama3", temperature=0)

ROUTER_PROMPT = """
You are a routing classifier for a hybrid SQL + RAG system for the FARS dataset.

Your job:
- Route questions about the FARS tables, their columns, meaning, definitions, or descriptive explanations → "rag"
- Route questions asking for counts, sums, averages, filters, comparisons on the FARS data → "sql"

IMPORTANT EXCEPTIONS:
- If the question is general knowledge (e.g., geography, history, definitions NOT related to FARS),
  ALWAYS return "rag", NEVER "sql".
- If the question is not about the dataset at all, return "rag".

Return ONLY one of these EXACT labels:
"sql", "rag"

QUESTION:
{question}

LABEL ONLY:
"""

def route(question: str) -> Literal["sql", "rag"]:
    prompt = ROUTER_PROMPT.format(question=question)
    response = router_llm.invoke(prompt)
    label = getattr(response, "content", "").strip().lower()
    return label if label in {"sql", "rag"} else "rag"

# ----------------------------------------------
# 5) ORCHESTRATION LAYER
# ----------------------------------------------
def answer_question(question: str):
    choice = route(question)

    if choice == "sql":
        return ask_fars_database(question)
    else:  # rag
        return run_rag(question)

# ----------------------------------------------
# 6) CLI
# ----------------------------------------------
if __name__ == "__main__":
    print("Hybrid SQL + RAG Assistant Ready (Ollama + FAISS Accident Index + Databricks)")
    print("Type 'quit' to exit.\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit"):
            break
        print("\nAssistant:", answer_question(q), "\n")