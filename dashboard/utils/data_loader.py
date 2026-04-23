import pandas as pd
import requests
from io import StringIO
import logging
from django.conf import settings

logger = logging.getLogger(__name__)


def load_data_robustly(url, name, default_sep=','):
    """Load CSV from URL or local path - Django compatible"""
    df = pd.DataFrame()
    if not url:
        logger.warning(f"⚠️ {name}: No URL/path provided")
        return df
    
    try:
        if url.startswith('http'):
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            content = response.text
            
            attempts = [
                (',', 'utf-8'),
                (',', 'utf-8-sig'),
                ('\t', 'utf-8'),
                (';', 'utf-8'),
                (',', 'latin-1'),
            ]
            
            for sep, enc in attempts:
                try:
                    df = pd.read_csv(
                        StringIO(content),
                        sep=sep,
                        encoding=enc,
                        low_memory=False,
                        on_bad_lines='skip'
                    )
                    if not df.empty and len(df.columns) > 1:
                        logger.info(f"✅ {name} loaded (Sep: '{sep}', Enc: '{enc}', Shape: {df.shape})")
                        return df
                except Exception:
                    continue
        else:
            if os.path.exists(url):
                df = pd.read_csv(url, sep=default_sep, low_memory=False, on_bad_lines='skip')
                logger.info(f"✅ {name} loaded from local file (Shape: {df.shape})")
                return df
        
        logger.error(f"❌ {name}: Could not load data")
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"❌ {name}: Failed - {str(e)}")
        return pd.DataFrame()


def load_election_data():
    """Load all election-related data sources from GitHub"""
    sources = {
        'meltwater': settings.MELTWATER_URL,
        'civicsignal': settings.CIVICSIGNALS_URL,
        'tiktok': settings.TIKTOK_URL,
        'openmeasure': settings.OPENMEASURES_URL,
        'original_posts': settings.ORIGINAL_POSTS_URL,
    }
    
    all_data = {}
    for name, url in sources.items():
        all_data[name] = load_data_robustly(url, name.title())
    
    return all_data


def load_peps_from_github(csv_url):
    """Load PEPs from GitHub CSV (your sample dataset)"""
    try:
        response = requests.get(csv_url, timeout=30)
        response.raise_for_status()
        
        # Parse CSV
        df = pd.read_csv(StringIO(response.text))
        
        # Clean and filter
        df = df[df['Name (English)'].notna()]
        df = df[df['Position'].notna()]
        
        return df.to_dict('records')
        
    except Exception as e:
        logger.error(f"❌ Failed to load PEPs: {str(e)}")
        return []
