# **FARS RAG Backend Server**

This backend powers the "RAG and Roll" chatbot, providing a "backend API" that your web-based frontend can call.  
It uses the logic from LangChain.ipynb and formalizes it into a scalable web server.

## **How It Works**

This project is in two parts:

1. **RAG\_vector\_store.py** (formerly vector\_store.py): A one-time script that connects to your MySQL fars database, serializes all the data into text snippets, and builds a persistent Chroma vector store.  
2. **RAG\_Pipeline.py** (formerly RAG\_Backend.py): The main web server. It loads the pre-built vector store, initializes the RAG chain (using LangChain and a local Ollama LLM), and exposes a /query endpoint to answer questions.

## **How to Run and Test**

### **Step 0: Prerequisites**

1. **MySQL**: Ensure your MySQL server is running with the fars database and tables (accident\_master, person\_master, etc.) already loaded.  
2. **Ollama**: Ensure the Ollama application is running on your Mac and you have the llama3 model.  
   ollama pull llama3

3. **Python Environment**: Make sure you are in your capstone\_env or fars\_env and have all necessary packages installed:  
   pip install langchain langchain\_community langchain\_core chromadb sentence-transformers sqlalchemy pymysql flask flask-cors

### **Step 1: Build the Vector Store (One-Time Only)**

Run this script from your terminal. This will connect to MySQL and create the ./capstone\_chroma\_db folder. You only need to do this once.  
python RAG\_vector\_store.py

### **Step 2: Start the Backend Server**

Once the vector store is built, run this script to start the server. This terminal window will be "busy" running the server.  
python RAG\_Pipeline.py

You should see output indicating that all models are loaded and the server is running on http://0.0.0.0:5000.

### **Step 3: Test the Backend**

Open a **new, separate terminal window** (leave the server from Step 2 running).  
Run the curl command to send a test question to your server. This is how your frontend will talk to it.  
curl \-X POST http://localhost:5000/query \\  
     \-H "Content-Type: application/json" \\  
     \-d '{"question":"Tell me about an accident in Virginia involving a person over 50"}'

**You should get a JSON response back like this:**  
{  
  "answer": "According to the provided context, Accident Case 40229 in 75 involved a person aged 67...",  
  "question": "Tell me about an accident in Virginia involving a person over 50"  
}  
