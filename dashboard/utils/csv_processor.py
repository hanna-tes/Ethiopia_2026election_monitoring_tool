import pandas as pd
import numpy as np
import re
import logging
import hashlib

logger = logging.getLogger(__name__)

def get_col(df, cols):
    """Helper to find the first matching column name safely."""
    df_cols_lower = {c.lower().strip(): c for c in df.columns}
    for col in cols:
        norm = col.lower().strip()
        if norm in df_cols_lower:
            return df[df_cols_lower[norm]]
    return pd.Series([np.nan] * len(df), index=df.index)

def load_brandwatch_data(filepath):
    """Load and standardize Brandwatch exports (6 rows metadata + UTF-8)"""
    try:
        # Brandwatch exports have 6 rows of metadata before column headers
        df = pd.read_csv(filepath, encoding='utf-8', sep=',', low_memory=False,
                         skiprows=6, on_bad_lines='skip')
    except Exception as e:
        logger.error(f"Brandwatch load failed: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    # 1. Create unified Account column
    df['Account'] = df.get('Weblog Title', pd.Series(dtype='object')).combine_first(
        df.get('Author', pd.Series(dtype='object'))
    ).combine_first(df.get('Full Name', pd.Series(dtype='object')))

    # 2. Map to standard schema (Matches other platforms)
    brandwatch_df = pd.DataFrame()
    brandwatch_df['account_id'] = df['Account'].str.strip().fillna('Unknown')
    brandwatch_df['URL'] = df.get('Url', pd.Series(dtype='object'))  # Uppercase URL
    brandwatch_df['timestamp_share'] = df.get('Date', pd.Series(dtype='object'))
    brandwatch_df['object_id'] = df.get('Full Text', pd.Series(dtype='object')).fillna('')  # object_id
    brandwatch_df['Platform'] = df.get('Page Type', pd.Series(dtype='object')).str.strip().str.title()
    
    # Content ID: Brandwatch doesn't always provide it; fallback to URL
    brandwatch_df['content_id'] = df.get('Content Id', pd.Series(dtype='object')).fillna(brandwatch_df['URL'])

    # 3. Normalize platform names
    platform_map = {'Twitter': 'X', 'Facebook': 'Facebook', 'Instagram': 'Instagram', 
                    'Tiktok': 'TikTok', 'Telegram': 'Telegram', 'Media': 'Media'}
    brandwatch_df['Platform'] = brandwatch_df['Platform'].replace(platform_map)

    # 4. Clean text & handle NaNs
    brandwatch_df['object_id'] = brandwatch_df['object_id'].astype(str).replace('nan', '')
    brandwatch_df['Platform'] = brandwatch_df['Platform'].replace('', 'Unknown')
    brandwatch_df['source_dataset'] = 'Brandwatch'

    return brandwatch_df


def map_columns_by_type(df, platform):
    """Maps platform-specific CSV headers to a standard format."""
    mapped = pd.DataFrame()

    # Brandwatch is handled by load_brandwatch_data, so we skip it here
    
    if platform == 'meltwater':
        mapped['account_id'] = get_col(df, ['Influencer', 'author', 'username', 'account'])
        mapped['content_id'] = get_col(df, ['tweet id', 'post id', 'id', 'ID'])
        mapped['object_id'] = get_col(df, ['Hit Sentence', 'text', 'content', 'opening text'])
        mapped['URL'] = get_col(df, ['URL', 'url', 'link', 'permalink'])
        mapped['timestamp_share'] = get_col(df, ['Date', 'timestamp', 'alternate date format'])

    elif platform == 'civicsignal':
        mapped['account_id'] = get_col(df, ['media_name', 'author', 'username'])
        mapped['content_id'] = get_col(df, ['stories_id', 'post_id', 'id', 'ID'])
        mapped['object_id'] = get_col(df, ['title', 'text', 'content', 'body'])
        mapped['URL'] = get_col(df, ['url', 'URL', 'link', 'permalink'])
        mapped['timestamp_share'] = get_col(df, ['publish_date', 'timestamp', 'date'])

    elif platform == 'tiktok':
        mapped['account_id'] = get_col(df, ['authorMeta/name', 'username', 'creator', 'author'])
        mapped['content_id'] = get_col(df, ['id', 'video_id', 'itemId', 'ID'])
        mapped['object_id'] = get_col(df, ['text', 'Transcript', 'caption', 'content'])
        mapped['URL'] = get_col(df, ['webVideoUrl', 'TikTok Link', 'url', 'URL', 'shareUrl'])
        mapped['timestamp_share'] = get_col(df, ['createTimeISO', 'timestamp', 'createTime'])

    elif platform == 'openmeasure':
        mapped['account_id'] = get_col(df, ['context_name', 'channelusername', 'channeltitle', 'actor_username'])
        mapped['content_id'] = get_col(df, ['id', 'url'])
        mapped['object_id'] = get_col(df, ['text', 'message', 'body'])
        mapped['URL'] = get_col(df, ['url', 'URL', 'link', 'permalink'])
        raw_dates = get_col(df, ['created_at', 'date'])
        mapped['timestamp_share'] = raw_dates.astype(str).str.replace(' @ ', ' ', regex=False)
    else:
        # Fallback or custom
        mapped = df.copy()

    # Standardize fields
    if platform != 'openmeasure':
        mapped['Platform'] = platform.upper()
    else:
        mapped['Platform'] = 'Telegram'
    mapped['source_dataset'] = platform

    # Generate content_id from URL if missing/empty
    if 'content_id' in mapped.columns:
        mask = mapped['content_id'].isna() | (mapped['content_id'] == '')
        if mask.any():
            mapped.loc[mask, 'content_id'] = mapped.loc[mask, 'URL'].apply(
                lambda x: hashlib.md5(str(x).encode()).hexdigest()[:16] 
                if pd.notna(x) and str(x).strip() != '' 
                else hashlib.md5(str(np.random.random())).hexdigest()[:16]
            )

    return mapped

def preprocess_dataframe(df):
    """Basic cleaning: remove empty rows and format text."""
    if df.empty:
        return df
    if 'object_id' in df.columns:
        df = df.dropna(subset=['object_id'])
        df['object_id'] = df['object_id'].astype(str).str.strip()
        df = df[df['object_id'] != "nan"]
    return df

def process_uploaded_csv(file, platform):
    """Main entry point for processing a single uploaded file."""
    try:
        if platform == 'brandwatch':
            # Brandwatch has its own loader that handles skiprows and specific mapping
            return load_brandwatch_data(file)
        else:
            df = pd.read_csv(file, low_memory=False, on_bad_lines='skip')
            mapped_df = map_columns_by_type(df, platform)
            clean_df = preprocess_dataframe(mapped_df)
            return clean_df
    except Exception as e:
        logger.error(f"Failed to process CSV for {platform}: {e}")
        return pd.DataFrame()

def combine_social_media_data(meltwater_df=None, brandwatch_df=None, civicsignals_df=None, tiktok_df=None, openmeasures_df=None):
    """Merges standardized DataFrames from any combination of sources."""
    combined = []
    if meltwater_df is not None: combined.append(meltwater_df)
    if brandwatch_df is not None and not brandwatch_df.empty:
        combined.append(brandwatch_df)  
    if civicsignals_df is not None: combined.append(civicsignals_df)
    if tiktok_df is not None: combined.append(tiktok_df)
    if openmeasures_df is not None: combined.append(openmeasures_df)

    return pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()
