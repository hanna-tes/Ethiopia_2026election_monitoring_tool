import pandas as pd
import requests
from io import StringIO

def load_data_robustly(file_path):
    """STUB: Load data from local file."""
    return pd.read_csv(file_path)

def load_peps_from_github(csv_url):
    """Load PEPs from GitHub CSV (skips title row)."""
    try:
        resp = requests.get(csv_url, timeout=30)
        resp.raise_for_status()
        lines = resp.text.split('\n')
        df = pd.read_csv(StringIO('\n'.join(lines[1:])))  # Skip title row
        return df.to_dict('records')
    except Exception as e:
        print(f"❌ Failed to load PEPs: {e}")
        return []
