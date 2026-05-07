import pandas as pd
import requests
import re
from io import StringIO

def parse_timestamp_robust(timestamp):
    """
    Standardizes various timestamp formats found in Meltwater, 
    TikTok, and OpenMeasure exports.
    """
    if pd.isna(timestamp) or str(timestamp).strip() == "":
        return pd.NaT
    
    # Remove 'GMT' artifacts common in social media exports
    ts_str = re.sub(r'\s+GMT$', '', str(timestamp).strip(), flags=re.IGNORECASE)
    
    # Try standard pandas parsing first
    try:
        parsed = pd.to_datetime(ts_str, errors='coerce', utc=True)
        if pd.notna(parsed): 
            return parsed
    except: 
        pass
    
    # Fallback to specific formats if standard parsing fails
    formats = [
        '%Y-%m-%d %H:%M:%S', 
        '%Y-%m-%d %H:%M', 
        '%d/%m/%Y %H:%M', 
        '%b %d, %Y %H:%M', 
        '%Y-%m-%d'
    ]
    
    for fmt in formats:
        try:
            parsed = pd.to_datetime(ts_str, format=fmt, errors='coerce', utc=True)
            if pd.notna(parsed): 
                return parsed
        except: 
            continue
            
    return pd.NaT

def load_data_robustly(file_path):
    """Load data from local file with basic error handling."""
    try:
        return pd.read_csv(file_path, low_memory=False, on_bad_lines='skip')
    except Exception as e:
        print(f"❌ Error reading CSV {file_path}: {e}")
        return pd.DataFrame()

def load_peps_from_github(csv_url):
    """Load PEPs from GitHub CSV (skips title row)."""
    try:
        resp = requests.get(csv_url, timeout=30)
        resp.raise_for_status()
        lines = resp.text.split('\n')
        # Standard cleaning: remove empty lines and join
        content = '\n'.join([line for line in lines[1:] if line.strip()])
        df = pd.read_csv(StringIO(content))
        return df.to_dict('records')
    except Exception as e:
        print(f"❌ Failed to load PEPs: {e}")
        return []
