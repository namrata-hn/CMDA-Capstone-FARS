from langchain_ollama import ChatOllama
from sql_query_chain import ask_fars_database
from rag_pipeline import run_rag

router_llm = ChatOllama(model="llama3", temperature=0)

ROUTER_PROMPT = """
Decide how to route this question.

Return ONLY:
- sql
- rag

SQL if: asking for counts, filters, data values, comparisons, totals.
RAG if: asking for explanations, descriptions, meaning, summaries.

QUESTION:
{question}

LABEL:
"""

def classify(question: str) -> str:
    result = router_llm.invoke(ROUTER_PROMPT.format(question=question)).content.strip().lower()
    return result if result in ["sql", "rag"] else "sql"

def answer_question(question: str):
    route = classify(question)
    if route == "sql":
        return ask_fars_database(question)
    else:
        return run_rag(question)
