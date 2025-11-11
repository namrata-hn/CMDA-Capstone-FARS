import os
import zipfile
import pandas as pd
import gc
import io

# === CONFIGURATION ===
script_dir = os.path.dirname(os.path.abspath(__file__))
fars_folder = os.path.join(script_dir, "..", "FARS")  # adjust relative path if needed
output_file = "master_vehicle.csv"
file_endings = ["VEHICLE.CSV", "vehicle.csv"]

# === KEEP COLUMNS (non-discontinued FARS Vehicle Data Elements) ===
keep_cols = [
    "ST_CASE", "OCUPANTS", "NUMOCCS", "UNITTYPE", "HIT_RUN", "REG_STAT", "OWNER", "VIN",
    "MOD_YEAR", "VPICMAKE", "VPICMODEL", "VPICBODYCLASS", "MAKE", "MODEL",
    "BODY_TYP", "ICFINALBODY", "GVWR_FROM", "GVWR_TO", "TOW_VEH",
    "TRLR1VIN", "TRLR2VIN", "TRLR3VIN", "TRLR1GVWR", "TRLR2GVWR", "TRLR3GVWR",
    "J_KNIFE", "MCARR_ID", "MCARR_I1", "MCARR_I2", "V_CONFIG", "CARGO_BT",
    "HAZ_INV", "HAZ_PLAC", "HAZ_ID", "HAZ_CNO", "HAZ_REL", "BUS_USE",
    "SPEC_USE", "EMER_USE", "TRAV_SP", "UNDEROVERRIDE", "ROLLOVER",
    "ROLINLOC", "IMPACT1", "DEFORMED", "TOWAWAY", "TOWED", "M_HARM", "FIRE_EXP",
    "ADS_PRES", "ADS_LEV", "ADS_ENG", "MAK_MOD", "VIN_1", "VIN_2", "VIN_3",
    "VIN_4", "VIN_5", "VIN_6", "VIN_7", "VIN_8", "VIN_9", "VIN_10",
    "VIN_11", "VIN_12", "DEATHS", "DR_DRINK", "DR_PRES", "L_STATE",
    "DR_ZIP", "L_TYPE", "L_STATUS", "CDL_STAT", "L_ENDORS", "L_CL_VEH", "L_COMPL",
    "L_RESTRI", "DR_HGT", "DR_WGT", "PREV_ACC", "PREV_SUS1", "PREV_SUS2",
    "PREV_SUS3", "PREV_DWI", "PREV_SPD", "PREV_OTH", "FIRST_MO", "FIRST_YR",
    "LAST_MO", "LAST_YR", "SPEEDREL", "VTRAFWAY", "VNUM_LAN", "VSPD_LIM",
    "VALIGN", "VPROFILE", "VPAVETYP", "VSURCOND", "VTRAFCON", "VTCONT_F",
    "P_CRASH1", "P_CRASH2", "P_CRASH3", "PCRASH4", "PCRASH5", "ACC_TYPE",
    "ACC_CONFIG"
]

print("=== Starting Vehicle Merge ===")

def read_csv_with_fallback(file_obj):
    """Try reading CSV with multiple encodings to avoid decode errors."""
    for enc in ["utf-8-sig", "utf-8", "latin1", "cp1252"]:
        try:
            return pd.read_csv(file_obj, encoding=enc, low_memory=False, dtype=str)
        except Exception:
            file_obj.seek(0)  # reset file pointer for next attempt
    # last resort: decode as utf-8 with replacement
    file_obj.seek(0)
    return pd.read_csv(io.StringIO(file_obj.read().decode("utf-8", errors="replace")),
                       low_memory=False, dtype=str)

# === Process Data in Batches ===
batch_size = 5
vehicle_dfs = []
all_batches = []
total_records = 0

for year in range(1975, 2024):
    zip_path = os.path.join(fars_folder, f"FARS{year}NationalCSV.zip")
    if not os.path.exists(zip_path):
        print(f"⚠️  Missing zip for {year}, skipping.")
        continue

    with zipfile.ZipFile(zip_path, "r") as z:
        file_name = next((name for name in z.namelist()
                          if any(name.upper().endswith(ending.upper()) for ending in file_endings)), None)
        if not file_name:
            print(f"⚠️  No vehicle CSV found for {year}.")
            continue

        try:
            print(f"→ Loading {year} vehicle data...")
            df = read_csv_with_fallback(z.open(file_name))
            df.columns = df.columns.str.upper().str.strip()

            if "YEAR" not in df.columns:
                df["YEAR"] = str(year)

            # Keep only desired columns
            available_cols = [c for c in keep_cols if c in df.columns]
            df = df[available_cols].copy()

            # Fill missing columns
            for col in keep_cols:
                if col not in df.columns:
                    df[col] = pd.NA

            # Ensure final column order
            df = df[keep_cols]
            df["YEAR"] = year

            vehicle_dfs.append(df)
            total_records += len(df)
            print(f"   Added {len(df):,} rows (total so far: {total_records:,})")

            if len(vehicle_dfs) >= batch_size:
                print(f"   Combining batch of {len(vehicle_dfs)} years...")
                batch_df = pd.concat(vehicle_dfs, ignore_index=True)
                all_batches.append(batch_df)
                vehicle_dfs = []
                gc.collect()

        except Exception as e:
            print(f"❌ Error processing {year}: {e}")

# Combine remaining data
if vehicle_dfs:
    print(f"Combining final batch of {len(vehicle_dfs)} years...")
    batch_df = pd.concat(vehicle_dfs, ignore_index=True)
    all_batches.append(batch_df)
    gc.collect()

# Final merge
print("\n=== Final Merge ===")
master_vehicle = pd.concat(all_batches, ignore_index=True, sort=False)
master_vehicle = master_vehicle[keep_cols + ["YEAR"]]

print(f"✓ Master Vehicle dataset shape: {master_vehicle.shape}")
print(f"✓ Total records: {len(master_vehicle):,}")

# Save output
master_vehicle.to_csv(output_file, index=False)
print(f"✅ Master vehicle CSV created: {output_file}")