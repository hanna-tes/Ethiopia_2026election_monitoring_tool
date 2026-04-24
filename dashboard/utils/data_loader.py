import pandas as pd
import requests
from io import StringIO
import logging

logger = logging.getLogger(__name__)


def load_data_robustly(url, name, default_sep=','):
    """Load CSV from URL - Django compatible"""
    df = pd.DataFrame()
    if not url:
        return df
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content = response.text
        
        attempts = [(',', 'utf-8'), (',', 'utf-8-sig'), ('\t', 'utf-8'), (';', 'utf-8'), (',', 'latin-1')]
        
        for sep, enc in attempts:
            try:
                df = pd.read_csv(StringIO(content), sep=sep, encoding=enc, low_memory=False, on_bad_lines='skip')
                if not df.empty and len(df.columns) > 1:
                    logger.info(f"✅ {name} loaded (Shape: {df.shape})")
                    return df
            except:
                continue
        
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"❌ {name} failed: {str(e)}")
        return pd.DataFrame()


def load_peps_from_github(csv_url):
    """Load PEPs from GitHub CSV"""
    try:
        response = requests.get(csv_url, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        df = df[df['Name (English)'].notna()]
        return df.to_dict('records')
    except Exception as e:
        logger.error(f"❌ Failed to load PEPs: {str(e)}")
        return []
