# Felix

"""
faiss_rag_retriever.py

RAG pipeline using:
- Ollama embeddings + FAISS as vector store
- Ollama (local, free) as the LLM
- A simple custom Retrieval-QA function (no langchain.chains dependency)

Now supports loading data directly from a Databricks SQL table
instead of a CSV.

Make sure you have installed (in your environment):

%pip install -U langchain-core langchain-text-splitters langchain-community langchain-ollama faiss-cpu pandas

Also ensure Ollama is running and you have pulled the models, e.g.:

ollama pull llama3
ollama pull nomic-embed-text
"""

import os
from typing import List, Optional, Tuple

import pandas as pd

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# 🔁 Ollama integrations
from langchain_ollama import ChatOllama, OllamaEmbeddings


# ------------------------------------------------------------------------
# 1A. Load dataset from CSV (still available if you ever need it)
# ------------------------------------------------------------------------

def load_dataset_as_documents(
    csv_path: str,
    text_cols: Optional[List[str]] = None,
    id_col: Optional[str] = None,
) -> List[Document]:
    df = pd.read_csv(csv_path)

    if text_cols is None:
        text_cols = list(df.columns)

    docs: List[Document] = []
    for _, row in df.iterrows():
        parts = [f"{col}: {row[col]}" for col in text_cols]
        text = "\n".join(parts)

        metadata = {}
        if id_col and id_col in df.columns:
            metadata["id"] = row[id_col]

        docs.append(Document(page_content=text, metadata=metadata))
    return docs


# ------------------------------------------------------------------------
# 1B. Load dataset directly from a Databricks SQL table
# ------------------------------------------------------------------------

def load_table_as_documents(
    table_name: str,
    text_cols: Optional[List[str]] = None,
    id_col: Optional[str] = None,
) -> List[Document]:
    """
    Load a Databricks table with Spark and convert each row into a Document.

    Parameters
    ----------
    table_name : str
        Fully qualified table name, e.g. "workspace.fars_database.accident_master".
    text_cols : list[str], optional
        Columns to include in the text. If None, use all columns.
    id_col : str, optional
        Column to use as a unique identifier in metadata.

    Returns
    -------
    docs : list[Document]
    """
    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    sdf = spark.table(table_name)

    if text_cols is not None:
        sdf = sdf.select(*text_cols)

    # Convert to pandas to reuse the same row → Document logic
    df = sdf.toPandas()

    if text_cols is None:
        text_cols = list(df.columns)

    docs: List[Document] = []
    for _, row in df.iterrows():
        parts = [f"{col}: {row[col]}" for col in text_cols]
        text = "\n".join(parts)

        metadata = {}
        if id_col and id_col in df.columns:
            metadata["id"] = row[id_col]

        docs.append(Document(page_content=text, metadata=metadata))
    return docs


# ------------------------------------------------------------------------
# 2–4. Chunk, embed, and store in FAISS
# ------------------------------------------------------------------------

def build_faiss_vectorstore(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model: str = "nomic-embed-text",
) -> FAISS:
    """
    Build a FAISS vector store using Ollama embeddings.

    Parameters
    ----------
    docs : list[Document]
        Documents to index.
    chunk_size : int
        Max characters per chunk.
    chunk_overlap : int
        Overlap between chunks.
    embedding_model : str
        Name of the Ollama embedding model (e.g. "nomic-embed-text").

    Returns
    -------
    vectorstore : FAISS
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    split_docs = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model=embedding_model)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore


# ------------------------------------------------------------------------
# 5–6. Simple custom Retrieval-QA (no langchain.chains)
# ------------------------------------------------------------------------

class SimpleRAGQA:
    def __init__(self, retriever, llm: ChatOllama):
        self.retriever = retriever
        self.llm = llm

    def answer(self, query: str) -> Tuple[str, List[Document]]:
        """
        Retrieve relevant docs and ask the LLM to answer using them.
        """
        source_docs: List[Document] = self.retriever.get_relevant_documents(query)
        context = "\n\n".join(doc.page_content for doc in source_docs)

        prompt = f"""
You are a helpful assistant answering questions based on the provided context.

Use ONLY the information in the context to answer the question. If the answer
is not in the context, say you don't know.

Context:
{context}

Question:
{query}

Answer in clear, concise English:
"""

        response = self.llm.invoke(prompt)

        # ChatOllama returns an AIMessage with `.content`
        if hasattr(response, "content"):
            answer_text = response.content
        else:
            answer_text = str(response)

        return answer_text.strip(), source_docs


def build_simple_rag_qa(
    vectorstore: FAISS,
    llm_model: str = "llama3",
    temperature: float = 0.0,
    k: int = 4,
) -> SimpleRAGQA:
    """
    Create a SimpleRAGQA that uses:
      - FAISS as retriever
      - Ollama Chat model (ChatOllama) as the LLM

    Parameters
    ----------
    vectorstore : FAISS
        Vector store created by build_faiss_vectorstore.
    llm_model : str
        Name of the Ollama chat model (e.g. "llama3", "mistral").
    temperature : float
        Sampling temperature.
    k : int
        Number of documents to retrieve per query.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    llm = ChatOllama(
        model=llm_model,
        temperature=temperature,
    )

    return SimpleRAGQA(retriever=retriever, llm=llm)


# ------------------------------------------------------------------------
# 7. Simple interactive interface
# ------------------------------------------------------------------------

def interactive_chat(rag_qa: SimpleRAGQA):
    print("RAG chat with Ollama ready. Type 'exit' or 'quit' to end.\n")

    while True:
        try:
            user_q = input("You: ").strip()
        except EOFError:
            break

        if user_q.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not user_q:
            continue

        answer, _ = rag_qa.answer(user_q)
        print("\nAssistant:", answer, "\n")


# ------------------------------------------------------------------------
# Main entry point: still uses a SQL TABLE instead of CSV
# ------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Example run:

    1. Ensure you have:
       - Ollama running locally
       - An Ollama chat model pulled (e.g. 'llama3')
       - An Ollama embedding model pulled (e.g. 'nomic-embed-text')
       - A Databricks table with your data
         (e.g. workspace.fars_database.accident_master)

    2. Update:
       - TABLE_NAME
       - EMBEDDING_MODEL
       - LLM_MODEL
    """

    # ---- CONFIG: change these for your environment ----
    TABLE_NAME = "workspace.fars_database.accident_master"
    EMBEDDING_MODEL = "nomic-embed-text"
    LLM_MODEL = "llama3"
    # --------------------------------------------------

    # 1–2. Load and convert table rows to Documents
    documents = load_table_as_documents(
        table_name=TABLE_NAME,
        text_cols=None,   # or e.g. ["ST_CASE", "STATE", "FATALS", "WEATHER"]
        id_col=None,      # or an ID like "ST_CASE"
    )

    # 3–4. Build FAISS index
    vectorstore = build_faiss_vectorstore(
        docs=documents,
        chunk_size=1000,
        chunk_overlap=200,
        embedding_model=EMBEDDING_MODEL,
    )

    # 5–6. Build Simple RAG QA with Ollama
    rag_qa = build_simple_rag_qa(
        vectorstore=vectorstore,
        llm_model=LLM_MODEL,
        temperature=0.0,
        k=4,
    )

    # 7. Interact
    interactive_chat(rag_qa)
