from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os
import re 
import pandas as pd
from databricks import sql
import logging
from metadata_loader import load_fars_codebook

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------- Load environment variables --------------
load_dotenv("../../config/.env")

# ---------------- Column Metadata ---------------- 
COLUMN_METADATA = load_fars_codebook("../../fars_codebook.csv")

# --------- Databricks Connection and Execution ---------
def run_databricks_query(query: str) -> pd.DataFrame:
    """
    Executes a SQL query on Databricks using the official connector.
    Returns a pandas DataFrame with nullable dtypes.
    """
    try:
        with sql.connect(
            server_hostname=os.getenv("DATABRICKS_HOST"),
            http_path=os.getenv("DATABRICKS_HTTP_PATH"),
            access_token=os.getenv("DATABRICKS_TOKEN")
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                arrow_table = cursor.fetchall_arrow()
                df = arrow_table.to_pandas()
                logger.info(f"Query executed successfully, returned {len(df)} rows")
                return df
    except Exception as e:
        logger.error(f"Databricks query execution error: {str(e)}")
        raise

# ---------------- Ollama LLM ----------------
llm = None

def get_llm():
    """Lazy initialization of LLM"""
    global llm
    if llm is None:
        try:
            llm = ChatOllama(model="llama3", temperature=0)
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise
    return llm

# ---------------- Table & Schema Info ----------------
TABLE_SCHEMAS = {
    "workspace.fars_database.accident_master": {
        "columns": ["STATE", "ST_CASE", "PEDS", "PERNOTMVIT", "VE_TOTAL", "VE_FORMS", 
                    "PVH_INVL", "PERSONS", "PERMVIT", "COUNTY", "CITY", "MONTH", "DAY", 
                    "DAY_WEEK", "YEAR", "HOUR", "MINUTE", "TWAY_ID", "TWAY_ID2", "CL_TWAY", 
                    "ROUTE", "RUR_URB", "FUNC_SYS","RD_OWNER", "NHS", "SP_JUR", "MILEPT", 
                    "LATITUDE", "LONGITUD", "HARM_EV", "MAN_COLL", "RELJCT1", "REL_JUNC", 
                    "RELJCT2", "TYP_INT", "REL_ROAD", "C_M_ZONE", "WRK_ZONE", "LGT_COND", 
                    "WEATHER", "SCH_BUS", "RAIL", "NOT_HOUR", "NOT_MIN", "ARR_HOUR", 
                    "ARR_MIN", "HOSP_HR", "HOSP_MN", "FATALS"],
        "string_columns": ["RAIL", "TWAY_ID", "TWAY_ID2"]
    },
    "workspace.fars_database.person_master": {
        "columns": ["STATE", "ST_CASE", "PER_NO", "AGE", "SEX", "PER_TYP", "INJ_SEV", 
                   "SEAT_POS", "REST_USE", "REST_MIS", "HELM_USE", "HELM_MIS", "AIR_BAG", 
                   "EJECTION", "EJ_PATH", "EXTRICAT", "DRINKING", "ALC_STATUS", "ATST_TYP", 
                   "TEST_RES", "ALC_RES", "DRUGS", "DSTATUS", "HOSPITAL", "DOA", "DEATH_MO", 
                   "DEATH_DA", "DEATH_YR", "DEATH_TM", "DEATH_HR", "DEATH_MN", "LAG_HRS", 
                   "LAG_MINS", "N_MOT_NO", "STR_VEH", "DEVTYPE", "DEVMOTOR", "LOCATION", 
                   "WORK_INJ", "HISPANIC", "YEAR"],
        "string_columns": []
    },
    "workspace.fars_database.vehicle_master": {
        "columns": ["STATE", "ST_CASE", "VEH_NO", "OCUPANTS", "NUMOCCS", "UNITTYPE", "HIT_RUN", 
                   "REG_STAT", "OWNER", "VIN", "MOD_YEAR", "VPICMAKE", "VPICMODEL", 
                   "VPICBODYCLASS", "MAKE", "MODEL", "BODY_TYP", "ICFINALBODY", "GVWR_FROM", 
                   "GVWR_TO", "TOW_VEH", "TRLR1VIN", "TRLR2VIN", "TRLR3VIN", "TRLR1GVWR", 
                   "TRLR2GVWR", "TRLR3GVWR", "J_KNIFE", "MCARR_ID", "MCARR_I1", "MCARR_I2", 
                   "V_CONFIG", "CARGO_BT", "HAZ_INV", "HAZ_PLAC", "HAZ_ID", "HAZ_CNO", 
                   "HAZ_REL", "BUS_USE", "SPEC_USE", "EMER_USE", "TRAV_SP", "UNDEROVERRIDE", 
                   "ROLLOVER", "ROLINLOC", "IMPACT1", "DEFORMED", "TOWAWAY", "TOWED", "M_HARM", 
                   "FIRE_EXP", "ADS_PRES", "ADS_LEV", "ADS_ENG", "MAK_MOD", "VIN_1", "VIN_2", 
                   "VIN_3", "VIN_4", "VIN_5", "VIN_6", "VIN_7", "VIN_8", "VIN_9", "VIN_10",
                   "VIN_11", "VIN_12", "DEATHS", "DR_DRINK", "DR_PRES", "L_STATE", "DR_ZIP", 
                   "L_TYPE", "L_STATUS", "CDL_STAT", "L_ENDORS", "L_CL_VEH", "L_COMPL",
                   "L_RESTRI", "DR_HGT", "DR_WGT", "PREV_ACC", "PREV_SUS1", "PREV_SUS2",
                   "PREV_SUS3", "PREV_DWI", "PREV_SPD", "PREV_OTH", "FIRST_MO", "FIRST_YR",
                   "LAST_MO", "LAST_YR", "SPEEDREL", "VTRAFWAY", "VNUM_LAN", "VSPD_LIM",
                   "VALIGN", "VPROFILE", "VPAVETYP", "VSURCOND", "VTRAFCON", "VTCONT_F",
                   "P_CRASH1", "P_CRASH2", "P_CRASH3", "PCRASH4", "PCRASH5", "ACC_TYPE", "ACC_CONFIG", "YEAR"],
        "string_columns": ["VIN", "TRLR1VIN", "TRLR2VIN", "TRLR3VIN", "MCARR_ID", "MCARR_I2",
                           "ADS_PRES", "ADS_LEV", "ADS_ENG"] + [f"VIN_{i}" for i in range(1,13)]
    }
}

# ---------------- Schema Prompt Builder ----------------
def build_schema_prompt(tables):
    prompt = (
        "You are an expert SQL generator for a Databricks SQL database.\n"
        "You MUST follow all the rules below:\n\n"
        "Rules:\n"
        "1. ONLY use the tables and columns listed.\n"
        "2. NEVER guess or invent column names.\n"
        "3. NEVER apply SQL functions to columns unless their type supports it.\n"
        "4. If a question requires columns from multiple tables, ALWAYS join them using ST_CASE.\n"
        "5. ST_CASE is the primary key that exists in all three tables and is ALWAYS the join key.\n"
        "6. Prefer SUM() when a question asks for totals of numeric fields.\n"
        "7. Numeric columns must never be quoted.\n"
        "8. Use COALESCE in SUM for numeric columns: `SUM(COALESCE(column, 0))`.\n"
        "9. IMPERATIVE GROUP BY: When using ANY aggregate function (SUM, COUNT, AVG, etc.), you MUST include a GROUP BY clause with ALL non-aggregated columns from the SELECT clause.\n"
        "10. IMPERATIVE JOIN: If columns from different tables are needed, join using `ON t1.ST_CASE = t2.ST_CASE`.\n"
        "11. Output ONLY the SQL query. No comments. No markdown.\n"
        "12. NEVER reference tables that are not listed\n"
        "13. **NEVER use string literals like 'DRIVER' or 'RAIN' in WHERE clauses for these columns.** Use only numeric codes (e.g., `PER_TYP = 1`).\n"
        "Available Tables and Columns:\n"
    )
    
    for t in tables:
        table_info = TABLE_SCHEMAS[t]
        prompt += f"- **{t}** (Join Key: ST_CASE):\n"
        prompt += f"  All Columns: {', '.join(table_info['columns'])}\n"
        numeric_cols = [c for c in table_info['columns'] if c not in table_info.get('string_columns', [])]
        if numeric_cols:
            prompt += f"  Numeric Columns (Use for math/comparison): {', '.join(numeric_cols)}\n"
        if table_info.get("string_columns"):
            prompt += f"  String Columns (May need quotes): {', '.join(table_info['string_columns'])}\n"
    
    prompt += (
        "\nFinal Instruction: Write ONLY the valid Databricks SQL query, starting with SELECT or WITH, and ending with a semicolon, with NO markdown formatting.\n"
    )
    
    return prompt

# ---------------- Clean LLM SQL Output ----------------
def clean_sql_output(sql_text):
    if sql_text is None:
        return ""

    # remove markdown fences
    sql_text = re.sub(r"```(?:sql)?\n?", "", sql_text, flags=re.IGNORECASE)
    sql_text = re.sub(r"\n?```$", "", sql_text, flags=re.IGNORECASE).strip()

    # try to find SELECT or WITH to the first semicolon
    match = re.search(r"((SELECT|WITH)[\s\S]*?;)", sql_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # fallback: ensure trailing semicolon
    sql = sql_text.strip()
    if not sql.endswith(";"):
        sql += ";"
    return sql

# ---------------------- Confirm Table Names ---------------------
FULLY_QUALIFIED_TABLES = {
    "accident_master": "workspace.fars_database.accident_master",
    "person_master": "workspace.fars_database.person_master",
    "vehicle_master": "workspace.fars_database.vehicle_master"
}

def qualify_table_names(sql_query: str) -> str:
    for short_name, fq_name in FULLY_QUALIFIED_TABLES.items():
        # only replace short_name if it is NOT already part of a fully-qualified name
        pattern = rf"(?<!\.)\b{short_name}\b" 
        sql_query = re.sub(pattern, fq_name, sql_query)
    return sql_query

def get_column_metadata_context(df: pd.DataFrame, sql_query: str) -> str:
    """
    Build natural-language context for columns in the query result using
    the metadata loaded from fars_codebook.csv.
    """
    if not COLUMN_METADATA:
        return ""

    columns_in_result = [c.upper() for c in df.columns]
    query_lower = sql_query.lower()

    # Mapping Databricks Table Names -> FARS Codebook File Names
    # The SQL might use "accident_master", but the codebook just uses "accident"
    table_mapping = {
        "accident_master": "accident",
        "vehicle_master": "vehicle",
        "person_master": "person",
        "accidents": "accident", # Catch-alls
        "vehicles": "vehicle",
        "persons": "person",
        "people": "person"
    }

    # Identify which tables are likely in the SQL query
    active_codebook_tables = []
    for sql_key, codebook_key in table_mapping.items():
        if sql_key in query_lower:
            active_codebook_tables.append(codebook_key)

    # If detection fails, default to checking all codebook tables
    if not active_codebook_tables:
        active_codebook_tables = list(COLUMN_METADATA.keys())

    context_lines = []
    context_lines.append("Column Meanings:")

    for col in columns_in_result:
        col_meta = None
        
        # 1. Search in the tables identified in the SQL query
        for table_key in active_codebook_tables:
            if table_key in COLUMN_METADATA:
                if col in COLUMN_METADATA[table_key]:
                    col_meta = COLUMN_METADATA[table_key][col]
                    break
        
        # 2. Fallback: Search ALL tables if not found (e.g. JOINs might obscure table names)
        if not col_meta:
            for table_key in COLUMN_METADATA:
                if col in COLUMN_METADATA[table_key]:
                    col_meta = COLUMN_METADATA[table_key][col]
                    break

        # 3. Build the Context String
        if col_meta:
            desc = col_meta.get("description", f"Meaning of {col}")
            context_lines.append(f"- {col}: {desc}")
            
            codes = col_meta.get("codes", {})
            if codes:
                context_lines.append("  Code Mappings:")
                # Limit to first 30 codes to prevent context overflow if list is huge
                for i, (code, label) in enumerate(codes.items()):
                    if i > 30: 
                        context_lines.append("    ... (more codes truncated)")
                        break
                    context_lines.append(f"    {code} -> {label}")
        else:
            # Handle aggregates or missing metadata
            if any(k in col for k in ["SUM", "COUNT", "AVG", "TOTAL"]):
                context_lines.append(f"- {col}: Calculated/aggregated value")
            else:
                context_lines.append(f"- {col}: (No metadata available)")

    return "\n".join(context_lines)


# ---------------- LLM Explanation --------------------
def llm_explanation(question: str, df: pd.DataFrame, sql_query: str = "") -> str:
    """
    Uses Ollama to convert the Databricks query result into a natural language answer.
    """
    try:
        llm = get_llm()
        
        # Handle empty results
        if df.empty:
            return "The query returned no results. This might mean there's no data matching your criteria."
        
        # convert dataframe to JSON (safer + shorter)
        result_json = df.to_dict(orient="records")

        # Get metadata context for the columns in the result
        metadata_context = get_column_metadata_context(df, sql_query)

        prompt = (
            "You are an expert data interpreter.\n"
            "The user asked the following question:\n"
            f"QUESTION: {question}\n\n"
            "Here is the SQL result from Databricks:\n"
            f"{result_json}\n"
            f"{metadata_context}\n\n"

            "====================\n"
            "CRITICAL INSTRUCTIONS\n"
            "====================\n"

            "At the beginning of EVERY answer, you MUST include the phrase:\n"
            "\"According to the FARS data,\"\n\n"

            "1. FIRST determine whether the query result contains ONE row or MULTIPLE rows.\n\n"

            "2. IF THE RESULT HAS ONE ROW:\n"
            "   - Output ONE short, direct sentence.\n"
            "   - Start with: \"According to the FARS data, ...\"\n"
            "   - Do NOT add explanations, disclaimers, or extra sentences.\n"
            "   - Example:\n"
            "       \"According to the FARS data, there were a total of 40,901 fatalities in 2023.\"\n\n"

           "3. IF THE RESULT HAS MULTIPLE ROWS:\n"
            " - You MUST start the answer with ONLY: \"According to the FARS data, accidents in Virginia (STATE=51) in 2022 had fatalities in various weather conditions.\"\n"
            " - For EACH row/object in the input JSON (e.g., `{'WEATHER': X, 'FATALS': Y}`), output **ONE** full sentence following this template exactly:\n"
            "  \"<count> fatality/fatalities occurred in the weather condition <mapped_label>.\"\n"
            " - **CRITICAL MAPPING:** You MUST use the numerical code **X** from the `WEATHER` column as the *lookup key* in the `metadata_context` Mappings to find the correct English label (<mapped_label>). Pair it ONLY with the fatality count **Y** from the same row.\n"
            " - The list of sentences must immediately follow the introductory sentence, one sentence per line. DO NOT use commas to combine sentences.\n"
            " - If a code (e.g., 98) is present in the result but the mapping is not in the `metadata_context`, use the description provided in the metadata (e.g., 'Not reported').\n"
            " - Each row must be interpreted completely independently. Do NOT try to connect or reorder the rows based on the mappings' order.\n\n"

            "====================\n"
            "STRICT COLUMN INTERPRETATION RULES\n"
            "====================\n"
            "- **ABSOLUTE PRIORITY:** Match the 'FATALS' value with the 'WEATHER' value **from the same dictionary object** in the input JSON list before applying any code mapping.\n" # The most important new rule
            "- Treat each row as completely independent; do not compare or mix it with other rows.\n"
            "- For each row, ensure that the value in one column (e.g., FATALS) is paired with the exact corresponding value in the other columns of the same row (e.g., WEATHER).\n"
            "- Only use metadata_context to map coded values.\n"
            "- If a mapping does not exist, output the numeric code as-is with '(code not reported in metadata)'.\n"
            "- NEVER swap, reorder, or confuse column values across rows.\n"
            "- Ignore row indices or any implicit ordering; only use the actual column values.\n"
            "- Each output sentence should correspond 1:1 to a single row in the result.\n"
            "- Do not infer meaning beyond what metadata_context explicitly provides.\n\n"

            "4. DO NOT:\n"
            "   - Mention SQL, tables, rows, or databases.\n"
            "   - Mention how the answer was generated.\n"
            "   - Use bullet points or label:value formatting.\n"
            "   - Add filler phrases like 'here is the breakdown' or 'based on the results'.\n\n"

            "5. The final output MUST:\n"
            "   - Be clean, concise, and purely natural language.\n"
            "   - Follow the full-sentence-per-row rule EXACTLY.\n"
            "   - Use plural or singular ('fatality' vs 'fatalities') correctly.\n"
        )

        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        return f"Query executed successfully but explanation generation failed: {str(e)}"

# ---------------- Main Query Function ----------------
def ask_fars_database(question: str, max_retries: int = 0):
    """
    Main function to process natural language questions and return SQL results.
    Returns a dict with 'query', 'results', and 'answer' keys.
    """
    try:
        llm = get_llm()
        
        tables = list(TABLE_SCHEMAS.keys())
        schema_prompt = build_schema_prompt(tables)

        # ------------------------ SQL GENERATION ------------------------
        prompt = (
            f"{schema_prompt}"
            f"Question: {question}\n"
            "Write ONLY the valid Databricks SQL query, starting with SELECT or WITH and ending with a semicolon."
        )
        
        logger.info(f"Generating SQL for question: {question}")
        response = llm.invoke(prompt)
        sql_query = clean_sql_output(response.content.strip())
        sql_query = qualify_table_names(sql_query)
        
        logger.info(f"Generated SQL: {sql_query}")

        # Check if SQL was actually generated
        if not sql_query or sql_query == ";":
            error_msg = "Failed to generate valid SQL query from the question."
            logger.error(error_msg)
            return {
                "query": None,
                "results": pd.DataFrame(),
                "answer": error_msg
            }

        # ------------------------ SQL EXECUTION ------------------------
        try:
            df = run_databricks_query(sql_query)
            nl_answer = llm_explanation(question, df, sql_query)

            return {
                "query": sql_query,
                "results": df,
                "answer": nl_answer
            }
        except Exception as e:
            error_msg = f"SQL execution error: {str(e)}"
            logger.error(error_msg)
            return {
                "query": sql_query,
                "results": pd.DataFrame(),
                "answer": error_msg
            }
            
    except Exception as e:
        error_msg = f"Error in ask_fars_database: {str(e)}"
        logger.exception(error_msg)
        return {
            "query": None,
            "results": pd.DataFrame(),
            "answer": error_msg
        }