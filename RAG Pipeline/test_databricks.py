from databricks import sql
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv("../config/.env")

def print_table_info(connection, full_table_name: str):
    """
    Prints column info and sample rows for a table using Databricks SQL connector.
    """
    print(f"\n--- Table Info: {full_table_name} ---")
    try:
        with connection.cursor() as cursor:
            # get sample rows
            cursor.execute(f"SELECT * FROM {full_table_name} LIMIT 5")
            arrow_table = cursor.fetchall_arrow()
            df = arrow_table.to_pandas()
            print("Sample Data:")
            print(df)

            # get column info
            cursor.execute(f"DESCRIBE {full_table_name}")
            arrow_table = cursor.fetchall_arrow()
            columns_df = arrow_table.to_pandas()
            print("\nColumns Info:")
            print(columns_df)

    except Exception as e:
        print(f"Error accessing {full_table_name}: {e}")

if __name__ == "__main__":
    # ---------------- Databricks Connection ----------------
    with sql.connect(
        server_hostname=os.getenv("DATABRICKS_HOST"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_TOKEN")
    ) as connection:

        print("Testing Databricks SQL connector connection...")

        tables = [
            "workspace.fars_database.accident_master",
            "workspace.fars_database.person_master",
            "workspace.fars_database.vehicle_master"
        ]

        for table in tables:
            print_table_info(connection, table)