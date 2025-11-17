#Felix


"""
faiss_rag_retriever.py

RAG pipeline using:
- Databricks embeddings + FAISS as vector store
- DBRX (via Databricks model serving) as the LLM
- A simple custom Retrieval-QA function (no langchain.chains dependency)

Now supports loading data directly from a Databricks SQL table
instead of a CSV.

Make sure you have installed (in your Databricks cluster or notebook):

%pip install -U langchain-core langchain-text-splitters langchain-community databricks-langchain faiss-cpu pandas
"""

import os
from typing import List, Optional, Tuple

import pandas as pd

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from databricks_langchain import DatabricksEmbeddings, ChatDatabricks


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
    embedding_endpoint: str = "databricks-bge-large-en",
) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    split_docs = splitter.split_documents(docs)

    embeddings = DatabricksEmbeddings(endpoint=embedding_endpoint)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore


# ------------------------------------------------------------------------
# 5–6. Simple custom Retrieval-QA (no langchain.chains)
# ------------------------------------------------------------------------

class SimpleRAGQA:
    def __init__(self, retriever, llm: ChatDatabricks):
        self.retriever = retriever
        self.llm = llm

    def answer(self, query: str) -> Tuple[str, List[Document]]:
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

        if hasattr(response, "content"):
            answer_text = response.content
        else:
            answer_text = str(response)

        return answer_text.strip(), source_docs


def build_simple_rag_qa(
    vectorstore: FAISS,
    dbrx_endpoint: str = "databricks-dbrx-instruct",
    temperature: float = 0.0,
    k: int = 4,
) -> SimpleRAGQA:
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    llm = ChatDatabricks(
        endpoint=dbrx_endpoint,
        temperature=temperature,
    )

    return SimpleRAGQA(retriever=retriever, llm=llm)


# ------------------------------------------------------------------------
# 7. Simple interactive interface
# ------------------------------------------------------------------------

def interactive_chat(rag_qa: SimpleRAGQA):
    print("RAG chat with DBRX ready. Type 'exit' or 'quit' to end.\n")

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
# Main entry point: now uses a SQL TABLE instead of CSV
# ------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Example run:

    1. Ensure you have:
       - A Databricks model serving endpoint for DBRX (e.g. 'databricks-dbrx-instruct')
       - A Databricks embedding endpoint (e.g. 'databricks-bge-large-en')
       - A Databricks table with your data (e.g. workspace.fars_database.accident_master)

    """

    # ---- CONFIG: change these for your environment ----
    TABLE_NAME = "workspace.fars_database.accident_master"
    EMBEDDING_ENDPOINT = "databricks-bge-large-en"
    DBRX_ENDPOINT = "databricks-dbrx-instruct"
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
        embedding_endpoint=EMBEDDING_ENDPOINT,
    )

    # 5–6. Build Simple RAG QA with DBRX
    rag_qa = build_simple_rag_qa(
        vectorstore=vectorstore,
        dbrx_endpoint=DBRX_ENDPOINT,
        temperature=0.0,
        k=4,
    )

    # 7. Interact
    interactive_chat(rag_qa)
