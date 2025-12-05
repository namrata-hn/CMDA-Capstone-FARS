# test_metadata_loader_main.py
import os
from metadata_loader import load_fars_codebook

def main():
    # Path to your FARS codebook CSV
    codebook_path = os.path.join(os.path.dirname(__file__), "../../fars_codebook.csv")
    
    print(f"Loading metadata from: {codebook_path}")
    metadata = load_fars_codebook(codebook_path)
    
    # Check that top-level keys exist
    expected_tables = ["accident_master", "person_master", "vehicle_master"]
    for table in expected_tables:
        if table in metadata:
            print(f"[PASS] Table '{table}' found in metadata")
        else:
            print(f"[FAIL] Table '{table}' NOT found in metadata")
    
    # Check that a known column exists and has codes
    table = "accident_master"
    column = "WEATHER"
    if column in metadata[table]:
        col_meta = metadata[table][column]
        print(f"[PASS] Column '{column}' found in table '{table}'")
        if "codes" in col_meta and col_meta["codes"]:
            print(f"  Codes for '{column}': {col_meta['codes']}")
        else:
            print(f"  [FAIL] No codes found for '{column}'")
    else:
        print(f"[FAIL] Column '{column}' NOT found in table '{table}'")
    
    # Check a column without codes
    table = "person_master"
    column = "AGE"
    if column in metadata[table]:
        col_meta = metadata[table][column]
        print(f"[PASS] Column '{column}' found in table '{table}'")
        if "codes" in col_meta and col_meta["codes"] == {}:
            print(f"  Correct: '{column}' has no codes (empty dict)")
        else:
            print(f"  [FAIL] Unexpected codes for '{column}': {col_meta.get('codes')}")
    else:
        print(f"[FAIL] Column '{column}' NOT found in table '{table}'")
    
    # Print a small summary
    print("\nSample metadata loaded successfully:")
    for t in expected_tables:
        print(f"- {t}: {list(metadata[t].keys())[:5]} ...")  # show first 5 columns only

if __name__ == "__main__":
    main()
