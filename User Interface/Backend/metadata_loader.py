import pandas as pd
from collections import defaultdict
import logging

# Configure local logger
logger = logging.getLogger(__name__)

def load_fars_codebook(csv_path: str):
    """
    Loads the RFARS codebook CSV and transforms it into a nested dictionary 
    compatible with the AI Metadata Processor.

    Target RFARS Columns:
      - filee:       Table type (accident, vehicle, person)
      - name_ncsa:   The column name (e.g., WEATHER)
      - value:       The code (e.g., 98)
      - value_label: The meaning (e.g., "Not Reported")
      - definition:  General description of the column
    """
    try:
        # Load CSV with low_memory=False to handle mixed types in 'value' columns
        df = pd.read_csv(csv_path, low_memory=False)
        
        # Normalize CSV headers to lowercase to ensure matching works
        df.columns = [c.lower().strip() for c in df.columns]

        # Verify critical columns exist
        required_cols = ['name_ncsa', 'value', 'value_label']
        if not all(col in df.columns for col in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            logger.error(f"Metadata CSV missing required columns: {missing}")
            return {}

        metadata = defaultdict(dict)

        # 1. Clean Data: Ensure filee exists, fill NA with 'common' or inference
        if 'filee' not in df.columns:
            df['filee'] = 'fars' # Fallback if filee is missing
        
        # 2. Group by File (Table) and Variable (Column)
        # We process the dataframe by grouping to build the dictionaries efficiently
        grouped = df.groupby(['filee', 'name_ncsa'])

        for (file_type, var_name), group in grouped:
            
            # Normalize Keys
            # 'accident' -> 'accident', 'WEATHER' -> 'WEATHER'
            table_key = str(file_type).lower().strip()
            column_key = str(var_name).upper().strip()

            # 3. Extract Column Description
            # Use the 'definition' column. Take the first non-null value found.
            description = f"Codes for {column_key}"
            if 'definition' in group.columns:
                desc_vals = group['definition'].dropna()
                if not desc_vals.empty:
                    description = str(desc_vals.iloc[0]).strip()

            # 4. Build the Code Map (Value -> Value Label)
            # Example: {'98': 'Not Reported', '1': 'Clear'}
            code_map = {}
            
            for _, row in group.iterrows():
                val = row.get('value')
                label = row.get('value_label')

                # Only add if both value and label are valid
                if pd.notna(val) and pd.notna(label):
                    # Convert float-like strings (e.g. "1.0") to integers strings ("1")
                    # This is crucial because SQL returns integers (1), not floats (1.0)
                    try:
                        val_str = str(int(float(val)))
                    except (ValueError, TypeError):
                        val_str = str(val).strip()
                    
                    code_map[val_str] = str(label).strip()

            # 5. Assign to Metadata Structure
            metadata[table_key][column_key] = {
                "description": description,
                "codes": code_map
            }

        logger.info(f"Successfully loaded metadata for {len(metadata)} tables.")
        return dict(metadata)

    except Exception as e:
        logger.error(f"Failed to load FARS codebook: {str(e)}")
        return {}