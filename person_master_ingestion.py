from sqlalchemy import create_engine, text, inspect # type: ignore
import pandas as pd # type: ignore

host = "localhost"
port = 3306
database = "fars"
user = "root" # replace with your username
password = "NewStrongPass!123"

# Use this line if your MySQL user has a password
engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}")
# engine = create_engine(f"mysql+pymysql://{user}@{host}:{port}/{database}")

try:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT VERSION();"))
        version = result.scalar()
        print(f"MySQL version: {version}")
except Exception as e:
    print("Connection failed:", e)


inspector = inspect(engine)
tables = inspector.get_table_names()
print("Tables in database:", tables)

person_master = pd.read_csv("/Users/rafaelviray/Downloads/person_master.csv") # change to your path


person_master.head(0).to_sql(
    name="person_master",
    con=engine,
    if_exists="replace",  
    index=False
)

person_master.to_sql(
    name="person_master",
    con=engine,
    if_exists="append",  # now insert data
    index=False,
    chunksize=1000      
)

with engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM person_master;"))
    count = result.scalar() 
    print(f"Rows now in table: {count}")