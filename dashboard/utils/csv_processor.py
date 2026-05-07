import pandas as pd
import numpy as np
import re
import logging

logger = logging.getLogger(__name__)

def get_col(df, cols):
    """Helper to find the first matching column name."""
    for col in cols:
        if col in df.columns:
            return df[col]
    df_cols = [c.lower().strip() for c in df.columns]
    for col in cols:
        norm = col.lower().strip()
        if norm in df_cols:
            return df[df.columns[df_cols.index(norm)]]
    return pd.Series([np.nan] * len(df), index=df.index)

def map_columns_by_type(df, platform):
    """Maps platform-specific CSV headers to a standard format."""
    mapped = pd.DataFrame()
    
    if platform == 'meltwater':
        mapped['account_id'] = get_col(df, ['influencer', 'author'])
        mapped['content_id'] = get_col(df, ['tweet id', 'post id', 'id'])
        mapped['object_id'] = get_col(df, ['hit sentence', 'text', 'content'])
        mapped['URL'] = get_col(df, ['url'])
        mapped['timestamp_share'] = get_col(df, ['date', 'timestamp'])
    elif platform == 'tiktok':
        mapped['account_id'] = get_col(df, ['authorMeta/name', 'author'])
        mapped['content_id'] = get_col(df, ['id', 'video_id'])
        mapped['object_id'] = get_col(df, ['text', 'caption'])
        mapped['URL'] = get_col(df, ['webVideoUrl', 'url'])
        mapped['timestamp_share'] = get_col(df, ['createTimeISO', 'timestamp'])
    # Add other platforms as needed...
    
    mapped['source_dataset'] = platform
    return mapped

def preprocess_dataframe(df):
    """Basic cleaning: remove empty rows and format text."""
    if df.empty: return df
    # Remove rows where the main text is missing
    df = df.dropna(subset=['object_id'])
    df['object_id'] = df['object_id'].astype(str).str.strip()
    return df[df['object_id'] != ""]

def process_uploaded_csv(file, platform):
    """Main entry point for processing a single uploaded file."""
    try:
        df = pd.read_csv(file, low_memory=False, on_bad_lines='skip')
        mapped_df = map_columns_by_type(df, platform)
        clean_df = preprocess_dataframe(mapped_df)
        return clean_df
    except Exception as e:
        logger.error(f"Failed to process CSV: {e}")
        return pd.DataFrame()

def combine_social_media_data(meltwater_df=None, civicsignals_df=None, tiktok_df=None, openmeasures_df=None):
    """Kept for backward compatibility with your views logic."""
    combined = []
    if meltwater_df is not None: combined.append(map_columns_by_type(meltwater_df, 'meltwater'))
    if civicsignals_df is not None: combined.append(map_columns_by_type(civicsignals_df, 'civicsignal'))
    if tiktok_df is not None: combined.append(map_columns_by_type(tiktok_df, 'tiktok'))
    if openmeasures_df is not None: combined.append(map_columns_by_type(openmeasures_df, 'openmeasure'))
    
    if not combined: return pd.DataFrame()
    return pd.concat(combined, ignore_index=True)
