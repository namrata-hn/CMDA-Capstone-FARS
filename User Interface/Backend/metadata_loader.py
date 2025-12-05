import pandas as pd
from collections import defaultdict

def load_fars_codebook(csv_path: str):
    """
    Load the rfars codebook CSV and convert it into the internal
    COLUMN_METADATA format your system uses.

    Expected CSV columns:
        file          (accident / vehicle / person)
        var           (column name)
        value         (code value)
        label         (meaning of the code)
        description   (optional; sometimes NA)
    """
    df = pd.read_csv(csv_path)

    metadata = defaultdict(dict)

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]

    # Convert to expected structure
    for (file_name, var), group in df.groupby(["file", "var"]):
        file_name = file_name.lower().strip()
        var_name = var.upper().strip()

        codes = {}

        # Build code → label mapping
        for _, row in group.iterrows():
            if pd.notna(row.get("value")) and pd.notna(row.get("label")):
                codes[str(row["value"])] = str(row["label"])

        # Column description (optional)
        description = None
        if "description" in group.columns:
            # pick the first non-null description
            desc = group["description"].dropna()
            if len(desc) > 0:
                description = desc.iloc[0]

        metadata[file_name][var_name] = {
            "description": description or f"Codes for {var_name}",
            "codes": codes
        }

    return dict(metadata)