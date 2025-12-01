from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import warnings

# --- CONFIGURATION ---
VECTORSTORE_DIR = "./capstone_chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3"

# Ignore common warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

print("--- FARS RAG Backend ---")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for your web frontend

# --- GLOBAL VARIABLES (Load models once on startup) ---
llm = None
retriever = None
rag_chain = None

def initialize_models():
    """
    Loads the LLM, Embedding Model, and pre-built Vector Store
    into memory. This ensures the backend is ready to answer
    questions immediately.
    """
    global llm, retriever, rag_chain
    
    print("Initializing models...")
    
    try:
        # 1. Load the LLM (the "brain")
        print(f"Loading LLM: {LLM_MODEL}...")
        llm = ChatOllama(model=LLM_MODEL)
        # Quick test to ensure Ollama is running
        llm.invoke("hello")
        print("LLM loaded.")

        # 2. Load the Embedding Model (the "vectorizer")
        print(f"Loading embedding model: {EMBEDDING_MODEL}...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        print("Embedding model loaded.")

        # 3. Load the pre-built Vector Store (the "knowledge base")
        print(f"Loading vector store from: {VECTORSTORE_DIR}...")
        vectorstore = Chroma(
            persist_directory=VECTORSTORE_DIR, 
            embedding_function=embeddings
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        print("Vector store loaded and retriever created.")

        # 4. Build the RAG Chain (the "pipeline")
        print("Building RAG chain...")
        prompt_template = """
        Answer the question based ONLY on the following context:
        
        {context}
        
        Question: {question}
        
        Answer:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        print("RAG chain built successfully.")
        
        print("\n--- Backend is Ready to Serve Requests ---")
        
    except Exception as e:
        print("\n--- FATAL ERROR DURING INITIALIZATION ---")
        if "Connection refused" in str(e):
             print("Ollama Error: Could not connect to Ollama.")
             print("Please ensure the Ollama application is running.")
        elif "No such file or directory" in str(e) and VECTORSTORE_DIR in str(e):
            print(f"Vector Store Error: Cannot find '{VECTORSTORE_DIR}'.")
            # --- THIS LINE IS UPDATED ---
            print("Please run `python RAG_vector_store.py` first.")
        else:
            print(f"An unexpected error occurred: {e}")
        exit(1) # Exit if models can't be loaded


@app.route('/query', methods=['POST'])
def handle_query():
    """
    This is the main API endpoint. It receives a JSON payload
    with a "question" and returns the RAG-powered answer.
    """
    if not request.json or 'question' not in request.json:
        return jsonify({"error": "Missing 'question' in JSON payload"}), 400
    
    question = request.json['question']
    print(f"\nReceived query: {question}")
    
    if not rag_chain:
        return jsonify({"error": "RAG chain is not initialized."}), 500
        
    try:
        # .stream() is also an option for a "live typing" effect
        response = rag_chain.invoke(question)
        print(f"Generated response: {response}")
        
        return jsonify({
            "question": question,
            "answer": response
        })
        
    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({"error": "Failed to process query."}), 500

if __name__ == "__main__":
    # Load all models *before* starting the server
    initialize_models()
    
    # Start the Flask web server
    app.run(host='0.0.0.0', port=5000)
    