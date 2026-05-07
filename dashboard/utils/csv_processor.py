import pandas as pd
import numpy as np
import re
import logging

logger = logging.getLogger(__name__)

def get_col(df, cols):
    """Helper to find the first matching column name from a list of possibilities."""
    # Try exact match first
    for col in cols:
        if col in df.columns:
            return df[col]
    
    # Try case-insensitive match
    df_cols_lower = {c.lower().strip(): c for c in df.columns}
    for col in cols:
        norm = col.lower().strip()
        if norm in df_cols_lower:
            return df[df_cols_lower[norm]]
            
    return pd.Series([np.nan] * len(df), index=df.index)

def combine_social_media_data(meltwater_df=None, civicsignals_df=None, tiktok_df=None, openmeasures_df=None):
    """
    Normalizes and combines different social media datasets into a standard schema.
    """
    combined = []

    # 1. Process Meltwater
    if meltwater_df is not None and not meltwater_df.empty:
        mw = pd.DataFrame()
        mw['account_id'] = get_col(meltwater_df, ['influencer', 'author', 'user'])
        mw['content_id'] = get_col(meltwater_df, ['tweet id', 'post id', 'id'])
        mw['object_id'] = get_col(meltwater_df, ['hit sentence', 'opening text', 'text', 'content'])
        mw['URL'] = get_col(meltwater_df, ['url'])
        mw['timestamp_share'] = get_col(meltwater_df, ['date', 'timestamp'])
        mw['source_dataset'] = 'Meltwater'
        combined.append(mw)

    # 2. Process CivicSignal
    if civicsignals_df is not None and not civicsignals_df.empty:
        cs = pd.DataFrame()
        cs['account_id'] = get_col(civicsignals_df, ['media_name', 'author', 'username'])
        cs['content_id'] = get_col(civicsignals_df, ['stories_id', 'post_id', 'id'])
        cs['object_id'] = get_col(civicsignals_df, ['title', 'text', 'content'])
        cs['URL'] = get_col(civicsignals_df, ['url', 'link'])
        cs['timestamp_share'] = get_col(civicsignals_df, ['publish_date', 'timestamp'])
        cs['source_dataset'] = 'CivicSignal'
        combined.append(cs)

    # 3. Process TikTok
    if tiktok_df is not None and not tiktok_df.empty:
        tt = pd.DataFrame()
        tt['account_id'] = get_col(tiktok_df, ['authorMeta/name', 'username', 'author'])
        tt['content_id'] = get_col(tiktok_df, ['id', 'video_id', 'itemId'])
        tt['object_id'] = get_col(tiktok_df, ['text', 'caption', 'transcript'])
        tt['URL'] = get_col(tiktok_df, ['webVideoUrl', 'url'])
        tt['timestamp_share'] = get_col(tiktok_df, ['createTimeISO', 'timestamp', 'date'])
        tt['source_dataset'] = 'TikTok'
        combined.append(tt)

    # 4. Process OpenMeasure (Telegram)
    if openmeasures_df is not None and not openmeasures_df.empty:
        om = pd.DataFrame()
        om['account_id'] = get_col(openmeasures_df, ['context_name', 'channelusername', 'author'])
        om['content_id'] = get_col(openmeasures_df, ['id', 'message_id'])
        om['object_id'] = get_col(openmeasures_df, ['text', 'message', 'body'])
        om['URL'] = get_col(openmeasures_df, ['url'])
        raw_dates = get_col(openmeasures_df, ['created_at', 'date'])
        om['timestamp_share'] = raw_dates.astype(str).str.replace(' @ ', ' ', regex=False)
        om['source_dataset'] = 'OpenMeasure'
        combined.append(om)

    if not combined:
        return pd.DataFrame(columns=['account_id', 'content_id', 'object_id', 'URL', 'timestamp_share', 'source_dataset', 'Platform'])

    final_df = pd.concat(combined, ignore_index=True)
    
    # Add platform inferrence
    from ..views import infer_platform_from_url # Internal import to avoid circularity if possible
    final_df['Platform'] = final_df['URL'].apply(lambda x: "TikTok" if "tiktok" in str(x).lower() else ("Telegram" if "t.me" in str(x).lower() else "Social Media"))
    
    return final_df
