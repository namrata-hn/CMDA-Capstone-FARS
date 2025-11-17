# Rafael

# ----------- Imports -----------
from langchain_community.utilities import SQLDatabase
from databricks import sql
import os

from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.base import create_sql_agent


# ----------- Connect LangChain to Databricks Database ------------
os.environ["DATABRICKS_SERVER_HOSTNAME"] = "dbc-05e81ced-b6f1.cloud.databricks.com"
os.environ["DATABRICKS_HTTP_PATH"] = "/sql/1.0/warehouses/dc6d2c2d31f9b31c" # Get this from your Databricks cluster
os.environ["DATABRICKS_TOKEN"] = "dapi98b277fcfdfc7b98946f01bf21310a10"
os.environ["OPENAI_API_KEY"] = "your-new-openai-key" # You'll need an LLM

server_hostname = os.getenv("DATABRICKS_SERVER_HOSTNAME")
http_path = os.getenv("DATABRICKS_HTTP_PATH")
access_token = os.getenv("DATABRICKS_TOKEN")

db = SQLDatabase.from_databricks(
    catalog="workspace", 
    schema="fars_database",   
    api_token=access_token,
    server_hostname=server_hostname,
    http_path=http_path,
    include_tables=["accident_master", "person_master", "vehicle_master"]
)
print(db.table_info)



# -------------------------- LLM Setup ----------------------------
# model="gpt-4-turbo" is powerful and good at SQL
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)



# ---------------------- Create SQL Agent --------------------------
from langchain.agents import create_sql_agent
'''
This agent will:
1) Look at the user's question.
2) Look at the database schema (your db.table_info).
3) Write a SQL query.
4) Run the query on Databricks.
5) Get the result and formulate a natural language answer.
'''
# Create the agent
agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="openai-tools", # Use the latest agent type
    verbose=True # Set to True to see the agent think
)



# --------------------- Fnction to Run Queries ----------------------
def ask_fars_database(question: str):
    """
    Takes a natural language question and uses the SQL agent
    to query the FARS Databricks database.
    """
    print(f"Executing query for: {question}")
    try:
        # The agent.invoke() call is where the magic happens
        result = agent_executor.invoke({"input": question})
        return result["output"]
    except Exception as e:
        print(f"Error executing query: {e}")
        return "Sorry, I couldn't answer that question."

# --- Example Usage ---
q1 = "How many total fatalities were there in 2023?"
# This will make the LLM generate: 
# SELECT SUM(FATALS) FROM master_accident WHERE YEAR = 2023
print(ask_fars_database(q1))

q2 = "Show me the weather (WEATHER) and number of fatalities (FATALS) for accidents in Virginia (STATE=51) in 2022"
# Note: The LLM needs to know the state code. Your schema (metadata.sql)
# should be descriptive for this to work well.
print(ask_fars_database(q2))

q3 = "How many accidents involved a 17-year-old driver?"
# This will require the LLM to JOIN master_accident with master_person
# on the case ID, and filter where AGE = 17 and PER_TYP = 1 (Driver)
print(ask_fars_database(q3))