# Felix

"""
faiss_rag_retriever.py

RAG pipeline using:
- Databricks embeddings + FAISS as vector store
- DBRX (via Databricks model serving) as the LLM
- A simple custom Retrieval-QA function (no langchain.chains dependency)

Pipeline steps:
1. Load dataset with pandas (CSV by default; adapt as needed)
2. Convert rows to LangChain `Document`s
3. Chunk text with RecursiveCharacterTextSplitter
4. Embed chunks with DatabricksEmbeddings and store in FAISS
5. Turn FAISS into a retriever
6. Ask DBRX questions using retrieved context
7. Optional interactive CLI-style chat


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
# 1. Load dataset and convert to Documents
# ------------------------------------------------------------------------

def load_dataset_as_documents(
    csv_path: str,
    text_cols: Optional[List[str]] = None,
    id_col: Optional[str] = None,
) -> List[Document]:
    """
    Load a tabular dataset with pandas and convert each row into a LangChain Document.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file. On Databricks this can be a DBFS path like '/dbfs/FileStore/...'.
    text_cols : list of str, optional
        Columns whose values will be concatenated into the main document text.
        If None, all columns in the CSV will be used.
    id_col : str, optional
        Column to use as a unique identifier in metadata.

    Returns
    -------
    docs : list[Document]
    """
    df = pd.read_csv(csv_path)

    if text_cols is None:
        text_cols = list(df.columns)

    docs: List[Document] = []
    for _, row in df.iterrows():
        # Build a single text field from selected columns
        parts = [f"{col}: {row[col]}" for col in text_cols]
        text = "\n".join(parts)

        metadata = {}
        if id_col and id_col in df.columns:
            metadata["id"] = row[id_col]

        doc = Document(page_content=text, metadata=metadata)
        docs.append(doc)

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
    """
    Chunk documents, create embeddings with DatabricksEmbeddings, and store them in a FAISS index.

    Parameters
    ----------
    docs : list[Document]
        The original (possibly long) documents.
    chunk_size : int
        Max characters per chunk.
    chunk_overlap : int
        Overlap between chunks to preserve context.
    embedding_endpoint : str
        Name of the Databricks embedding endpoint (check Serving > Endpoints).

    Returns
    -------
    vectorstore : FAISS
        A FAISS vector store ready to be used as a retriever.
    """
    # 3. Chunk text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    split_docs = splitter.split_documents(docs)

    # 4. Embed & store in FAISS (Databricks embedding model)
    embeddings = DatabricksEmbeddings(endpoint=embedding_endpoint)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore


# ------------------------------------------------------------------------
# 5–6. Simple custom Retrieval-QA (no langchain.chains)
# ------------------------------------------------------------------------

class SimpleRAGQA:
    """
    A minimal Retrieval-QA helper that:
      - uses a retriever (FAISS.as_retriever())
      - calls DBRX with context + question
    """

    def __init__(self, retriever, llm: ChatDatabricks):
        self.retriever = retriever
        self.llm = llm

    def answer(self, query: str) -> Tuple[str, List[Document]]:
        """
        Retrieve relevant docs and ask DBRX to answer using them.

        Returns
        -------
        answer : str
        source_docs : list[Document]
        """
        # Retrieve top-k similar chunks
        source_docs: List[Document] = self.retriever.get_relevant_documents(query)

        # Build context string
        context = "\n\n".join(doc.page_content for doc in source_docs)

        # Simple prompt template
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

        # Call DBRX via ChatDatabricks
        response = self.llm.invoke(prompt)

        # response is usually a string (depending on the version of databricks_langchain)
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
    """
    Create a SimpleRAGQA that uses:
      - FAISS as retriever
      - DBRX (via ChatDatabricks) as the chat model

    Parameters
    ----------
    vectorstore : FAISS
        Vector store created by build_faiss_vectorstore.
    dbrx_endpoint : str
        Name of the Databricks DBRX endpoint (Serving > Endpoints).
    temperature : float
        Sampling temperature (0 = deterministic).
    k : int
        Number of documents to retrieve per query.

    Returns
    -------
    rag_qa : SimpleRAGQA
    """
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
# 7. Simple interactive interface (Notebook / terminal)
# ------------------------------------------------------------------------

def interactive_chat(rag_qa: SimpleRAGQA):
    """
    Simple REPL for chatting with the RAG system.

    Type 'exit' or 'quit' to end the session.
    """
    print("RAG chat with DBRX ready. Ask questions about your data.")
    print("Type 'exit' or 'quit' to end.\n")

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
# Main entry point (example usage)
# ------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Example run:

    1. Ensure you have:
       - A Databricks model serving endpoint for DBRX (e.g. 'databricks-dbrx-instruct')
       - A Databricks embedding endpoint (e.g. 'databricks-bge-large-en')
       - A CSV file in DBFS or local environment.

    2. Update:
       - CSV_PATH
       - EMBEDDING_ENDPOINT
       - DBRX_ENDPOINT

    3. Run:
       python faiss_rag_retriever.py
       or run this in a Databricks notebook with %run.
    """

    # ---- CONFIG: change these for your environment ----
    CSV_PATH = "data/example_dataset.csv"  # e.g. "/dbfs/FileStore/mydata/fars_joined.csv"
    EMBEDDING_ENDPOINT = "databricks-bge-large-en"
    DBRX_ENDPOINT = "databricks-dbrx-instruct"
    # --------------------------------------------------

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"CSV file not found at {CSV_PATH}. "
            "Update CSV_PATH in __main__ to point to your dataset."
        )

    # 1–2. Load and convert to Documents
    documents = load_dataset_as_documents(
        csv_path=CSV_PATH,
        text_cols=None,   # or specify e.g. ["column1", "column2"]
        id_col=None,      # or specify an ID column if you have one
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
