import os
import zipfile
import pandas as pd
import gc

# === Configuration ===
fars_folder = os.path.join("..", "FARS")
output_file = "master_accident.csv"

# === KEEP COLUMNS (non-discontinued FARS Accident Data Elements) ===
keep_cols = [
    "ST_CASE", "PEDS", "PERNOTMVIT", "VE_TOTAL", "VE_FORMS", "PVH_INVL", "PERSONS",
    "PERMVIT", "COUNTY", "CITY", "MONTH", "DAY", "DAY_WEEK", "YEAR", "HOUR",
    "MINUTE", "TWAY_ID", "TWAY_ID2", "CL_TWAY", "ROUTE", "RUR_URB", "FUNC_SYS",
    "RD_OWNER", "NHS", "SP_JUR", "MILEPT", "LATITUDE", "LONGITUD", "HARM_EV",
    "MAN_COLL", "RELJCT1", "REL_JUNC", "RELJCT2", "TYP_INT", "REL_ROAD",
    "C_M_ZONE", "WRK_ZONE", "LGT_COND", "WEATHER", "SCH_BUS", "RAIL",
    "NOT_HOUR", "NOT_MIN", "ARR_HOUR", "ARR_MIN", "HOSP_HR", "HOSP_MN", "FATALS"
]

file_endings = ["ACCIDENT.CSV", "accident.csv"]
accident_dfs = []

print("=== Starting Accident Merge ===")

# === Loop through all years ===
for year in range(1975, 2024):
    zip_path = os.path.join(fars_folder, f"FARS{year}NationalCSV.zip")
    if not os.path.exists(zip_path):
        print(f"⚠️  Zip file for {year} not found, skipping.")
        continue

    with zipfile.ZipFile(zip_path, 'r') as z:
        # Find accident CSV (case-insensitive)
        file_name = next((name for name in z.namelist() 
                          if any(name.upper().endswith(ending.upper()) for ending in file_endings)), None)
        if file_name is None:
            print(f"⚠️  No accident CSV found in {zip_path}, skipping {year}.")
            continue

        # Load CSV with automatic encoding fallback
        try:
            try:
                df = pd.read_csv(z.open(file_name), encoding="utf-8", low_memory=False, dtype=str)
            except UnicodeDecodeError:
                df = pd.read_csv(z.open(file_name), encoding="latin1", low_memory=False, dtype=str)

            df.columns = df.columns.str.upper().str.strip()

            # Keep only desired columns
            available_cols = [c for c in keep_cols if c in df.columns]
            df = df[available_cols].copy()

            # Add missing columns
            for col in keep_cols:
                if col not in df.columns:
                    df[col] = pd.NA

            # Ensure YEAR column exists
            df["YEAR"] = year

            accident_dfs.append(df)
            print(f"Loaded {year}: {df.shape[0]} rows, {len(available_cols)} columns found.")

            del df
            gc.collect()

        except Exception as e:
            print(f"❌ Error reading {file_name} for {year}: {e}")

# === Merge all years ===
if not accident_dfs:
    raise RuntimeError("❌ No accident data files were successfully loaded.")

master_accident = pd.concat(accident_dfs, ignore_index=True, sort=False)
master_accident = master_accident[keep_cols]  # consistent column order

# === Save output ===
master_accident.to_csv(output_file, index=False)
print(f"\n✅ Master accident dataset created: {output_file}")
print(f"Shape: {master_accident.shape}")
print(f"Columns: {list(master_accident.columns)}")