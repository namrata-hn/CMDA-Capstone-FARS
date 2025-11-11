import os
import zipfile
import pandas as pd
import gc

# === CONFIGURATION ===
script_dir = os.path.dirname(os.path.abspath(__file__))
fars_folder = os.path.join(script_dir, "..", "FARS")
output_file = "master_person.csv"

# === KEEP COLUMNS (non-discontinued FARS Person Data Elements) ===
keep_cols = [
    "ST_CASE", "AGE", "SEX", "PER_TYP", "INJ_SEV", "SEAT_POS", "REST_USE", "REST_MIS",
    "HELM_USE", "HELM_MIS", "AIR_BAG", "EJECTION", "EJ_PATH", "EXTRICAT",
    "DRINKING", "ALC_STATUS", "ATST_TYP", "TEST_RES", "ALC_RES", "DRUGS", "DSTATUS",
    "HOSPITAL", "DOA", "DEATH_MO", "DEATH_DA", "DEATH_YR", "DEATH_TM",
    "DEATH_HR", "DEATH_MN", "LAG_HRS", "LAG_MINS", "N_MOT_NO", "STR_VEH", "DEVTYPE",
    "DEVMOTOR", "LOCATION", "WORK_INJ", "HISPANIC"
]

print("=== Starting Person Merge ===")

# List of possible file endings
file_endings = ["PERSON.CSV", "person.csv"]

person_dfs = []
total_records = 0

for year in range(1975, 2024):
    zip_path = os.path.join(fars_folder, f"FARS{year}NationalCSV.zip")
    if not os.path.exists(zip_path):
        print(f"⚠️  Zip file for {year} not found, skipping.")
        continue

    with zipfile.ZipFile(zip_path, 'r') as z:
        file_name = next((name for name in z.namelist()
                          if any(name.upper().endswith(ending.upper()) for ending in file_endings)), None)
        if file_name is None:
            print(f"⚠️  No person CSV found in {zip_path}, skipping year {year}.")
            continue

        try:
            print(f"→ Loading {year} person data...")
            try:
                df = pd.read_csv(z.open(file_name), encoding="utf-8", low_memory=False, dtype=str)
            except UnicodeDecodeError:
                # fallback if utf-8 fails
                df = pd.read_csv(z.open(file_name), encoding="latin1", low_memory=False, dtype=str)

            # Ensure columns are uppercase and stripped
            df.columns = df.columns.str.upper().str.strip()

            # Add YEAR column first
            df["YEAR"] = str(year)

            # Keep only desired columns plus YEAR
            available_cols = [c for c in keep_cols + ["YEAR"] if c in df.columns]
            df = df[available_cols].copy()

            # Fill missing columns
            for col in keep_cols + ["YEAR"]:
                if col not in df.columns:
                    df[col] = pd.NA

            # Ensure final column order: keep_cols + YEAR
            df = df[keep_cols + ["YEAR"]]

            person_dfs.append(df)
            total_records += len(df)
            print(f"   Added {len(df):,} rows (total so far: {total_records:,})")

            # Clean up memory
            del df
            gc.collect()

        except Exception as e:
            print(f"❌ Error processing {year}: {e}")

# Merge all dataframes
if person_dfs:
    print("\n=== Final Merge ===")
    master_person = pd.concat(person_dfs, ignore_index=True, sort=False)
    print(f"✓ Master Person dataset shape: {master_person.shape}")
    print(f"✓ Total records: {len(master_person):,}")

    # Save output
    master_person.to_csv(output_file, index=False)
    print(f"✅ Master person CSV created: {output_file}")
else:
    print("❌ No person data loaded. Master CSV not created.")