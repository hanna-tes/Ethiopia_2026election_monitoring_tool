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
        # Prioritize your exact column names first, then fallbacks
        mapped['account_id'] = get_col(df, ['Influencer', 'influencer', 'author', 'username', 'account'])
        mapped['content_id'] = get_col(df, ['tweet id', 'post id', 'id', 'ID', 'tweet_id'])
        mapped['object_id'] = get_col(df, ['Hit Sentence', 'hit sentence', 'text', 'content', 'opening text', 'headline', 'post_text', 'message'])
        mapped['URL'] = get_col(df, ['URL', 'url', 'link', 'Link', 'post_url', 'tweet_url', 'web_url', 'permalink', 'external_url'])  # ✅ 'URL' first!
        mapped['timestamp_share'] = get_col(df, ['Date', 'date', 'timestamp', 'alternate date format', 'created_at', 'posted_at'])
        
    elif platform == 'civicsignal':
        mapped['account_id'] = get_col(df, ['media_name', 'author', 'username'])
        mapped['content_id'] = get_col(df, ['stories_id', 'post_id', 'id', 'ID'])
        mapped['object_id'] = get_col(df, ['title', 'text', 'content', 'body'])
        mapped['URL'] = get_col(df, ['url', 'URL', 'link', 'Link', 'post_url', 'web_url', 'permalink', 'external_url'])
        mapped['timestamp_share'] = get_col(df, ['publish_date', 'timestamp', 'date', 'created_at'])
        
    elif platform == 'tiktok':
        mapped['account_id'] = get_col(df, ['authorMeta/name', 'username', 'creator', 'author'])
        mapped['content_id'] = get_col(df, ['id', 'video_id', 'itemId', 'videoId', 'ID'])
        mapped['object_id'] = get_col(df, ['text', 'Transcript', 'caption', 'content', 'description'])
        mapped['URL'] = get_col(df, ['webVideoUrl', 'TikTok Link', 'url', 'URL', 'videoUrl', 'shareUrl', 'web_url', 'link', 'external_url'])
        mapped['timestamp_share'] = get_col(df, ['createTimeISO', 'timestamp', 'date', 'createTime', 'created_at'])
        
    elif platform == 'openmeasure':
        mapped['account_id'] = get_col(df, ['context_name', 'channelusername', 'channeltitle', 'actor_username'])
        mapped['content_id'] = get_col(df, ['id', 'url'])
        mapped['object_id'] = get_col(df, ['text', 'message', 'body'])
        mapped['URL'] = get_col(df, ['url', 'URL', 'link', 'Link', 'web_url', 'permalink', 'external_url'])
        raw_dates = get_col(df, ['created_at', 'date'])
        mapped['timestamp_share'] = raw_dates.astype(str).str.replace(' @ ', ' ', regex=False)
    
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
