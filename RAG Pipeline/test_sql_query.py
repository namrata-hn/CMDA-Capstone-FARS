# test_sql_query_chain.py
# Test Ollama SQL generation and Databricks execution separately

from sql_query_chain import ask_fars_database
import pandas as pd

# ---------------- Example Usage with Ollama SQL generation ----------------
questions = [
    "How many total fatalities were there in 2023?",
    "Show me the weather (WEATHER) and number of fatalities (FATALS) for accidents in Virginia (STATE=51) in 2022",
    "How many accidents involved a 17-year-old driver (PER_TYP = 1)?"
]

for i, q in enumerate(questions, start=1):
    print(f"\nQ{i}: {q}")
    # Generate SQL via Ollama and run on Databricks
    df = ask_fars_database(q)

    if isinstance(df, pd.DataFrame) and not df.empty:
        # If query returns multiple rows/columns, print as DataFrame
        print(f"A{i}:\n{df}\n")
        # Optionally, for single-value queries like COUNT or SUM, extract scalar
        if df.shape[0] == 1 and df.shape[1] == 1:
            print(f"Scalar value: {df.iloc[0,0]}")
    else:
        print(f"A{i}: No results returned")