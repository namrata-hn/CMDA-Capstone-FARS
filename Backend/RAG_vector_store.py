import sqlalchemy
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import warnings

# --- CONFIGURATION ---
DB_URI = "mysql+pymysql://root:NewStrongPass!123@localhost:3306/fars"
VECTORSTORE_DIR = "./capstone_chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SQL_QUERY_LIMIT = 5000 # Use 5000 for demo, set to None for production

# Ignore common warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
print("--- RAG Vector Store Builder ---")

def fetch_data_from_sql(engine):
    """
    Connects to the MySQL database, runs the JOIN query, 
    and serializes the results into LangChain Documents.
    """
    print(f"Connecting to database at {DB_URI}...")
    documents_to_index = []
    
    with engine.connect() as connection:
        # This query JOINS your three tables to get rich data
        query_text = """
            SELECT 
                a.ST_CASE, a.YEAR, a.STATE, a.MONTH, a.PERSONS, a.VE_FORMS,
                p.AGE, p.SEX, p.PER_TYP,
                v.MAKE, v.MODEL
            FROM 
                accident_master a
            LEFT JOIN 
                person_master p ON a.ST_CASE = p.ST_CASE
            LEFT JOIN 
                vehicle_master v ON a.ST_CASE = v.ST_CASE
        """
        if SQL_QUERY_LIMIT:
            query_text += f" LIMIT {SQL_QUERY_LIMIT}"
        
        query = sqlalchemy.text(query_text)
        
        print("Executing SQL query...")
        result = connection.execute(query)
        
        print("Serializing rows into documents...")
        for row in result:
            # 1. Create the text snippet (page_content)
            content_snippet = (
                f"Accident Case {row.ST_CASE} in {row.YEAR} involved "
                f"{row.PERSONS} persons and {row.VE_FORMS} vehicles. "
                f"Details include: Person (Age: {row.AGE}, Sex: {row.SEX}), "
                f"Vehicle (Make: {row.MAKE}, Model: {row.MODEL})."
            )
            # 2. Create the metadata (for 100% traceability)
            metadata = {
                "source_table": "accident_master",
                "ST_CASE": str(row.ST_CASE),
                "YEAR": int(row.YEAR),
            }
            
            doc = Document(page_content=content_snippet, metadata=metadata)
            documents_to_index.append(doc)

    print(f"Successfully serialized {len(documents_to_index)} documents from MySQL.")
    return documents_to_index

def build_vector_store(documents):
    """
    Takes the list of documents, loads the embedding model,
    and builds/persists the Chroma vector store.
    """
    if not documents:
        print("No documents to index.")
        return

    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    print(f"Building and persisting vector store at {VECTORSTORE_DIR}...")
    # This creates the vector store and saves it to disk
    vectorstore = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR
    )
    
    print("\n--- SUCCESS ---")
    print(f"Vector store created with {vectorstore._collection.count()} documents.")
    print(f"Database saved to: {VECTORSTORE_DIR}")

def main():
    try:
        engine = sqlalchemy.create_engine(DB_URI)
        documents = fetch_data_from_sql(engine)
        build_vector_store(documents)
    except Exception as e:
        print(f"\n--- ERROR ---")
        if "Can't connect to MySQL server" in str(e):
            print("Connection Error: Could not connect to the MySQL database.")
            print("Please ensure MySQL is running and credentials are correct.")
        else:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
