# test_sql_query_chain.py
# Test Ollama SQL generation, Databricks execution, and NL answer creation

from sql_query_chain import ask_fars_database
import pandas as pd

questions = [
    "How many total fatalities were there in 2023?",
    "Show me the weather (WEATHER) and number of fatalities (FATALS) for accidents in Virginia (STATE=51) in 2022",
    "How many accidents involved a 17-year-old (AGE) driver (PER_TYP=1)?"
]

for i, q in enumerate(questions, start=1):
    print(f"\n====================")
    print(f"Question {i}: {q}")
    print(f"====================")

    # ask the full SQL → Databricks → NL answer pipeline
    response = ask_fars_database(q)

    # expected structure:
    # {
    #     "query": "...",
    #     "results": <DataFrame>,
    #     "answer": "..."
    # }

    # Safety check
    if not isinstance(response, dict):
        print("❌ Unexpected return type from ask_fars_database()")
        print(response)
        continue

    # --- Show generated SQL ---
    print("\n📌 Generated SQL:")
    print(response.get("query", "[No SQL produced]"))

    # --- Show DataFrame results ---
    df = response.get("results")

    print("\n📊 Query Results:")
    if isinstance(df, pd.DataFrame) and not df.empty:
        print(df)
    else:
        print("[No rows returned]")

    # --- Show sentence-style answer ---
    print("\n📝 Natural Language Answer:")
    print(response.get("answer", "[No NL answer generated]"))