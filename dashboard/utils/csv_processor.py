import pandas as pd
import numpy as np
import re
import logging
from io import StringIO
import requests
from .lexicon_engine import scan_text_for_lexicon_terms, calculate_risk_score

logger = logging.getLogger(__name__)


def load_data_robustly(file_path_or_url, name="Data", default_sep=','):
    """
    Load CSV from file path or URL with automatic encoding/separator detection.
    Mirrors your Streamlit app's robust loading logic.
    """
    df = pd.DataFrame()
    
    # --- Handle local files ---
    if not str(file_path_or_url).startswith('http'):
        import os
        if os.path.exists(file_path_or_url):
            try:
                # Try multiple encodings and separators
                attempts = [
                    (',', 'utf-8'), (',', 'utf-8-sig'), (',', 'utf-16'), 
                    ('\t', 'utf-8'), (';', 'utf-8'), (',', 'latin-1'), (',', 'windows-1252')
                ]
                for sep, enc in attempts:
                    try:
                        df = pd.read_csv(
                            file_path_or_url, 
                            sep=sep, 
                            encoding=enc,
                            low_memory=False,
                            on_bad_lines='skip'
                        )
                        if not df.empty and len(df.columns) > 1:
                            logger.info(f"✅ {name} loaded (Sep: '{sep}', Enc: '{enc}', Shape: {df.shape})")
                            return df
                    except (pd.errors.ParserError, UnicodeDecodeError):
                        continue
                logger.error(f"❌ {name}: Could not parse after all encoding/separator attempts")
                return pd.DataFrame()
            except Exception as e:
                logger.error(f"❌ {name} local load failed: {e}")
                return pd.DataFrame()
        logger.error(f"❌ {name}: Local file not found: {file_path_or_url}")
        return pd.DataFrame()
    
    # --- Handle URLs ---
    try:
        response = requests.get(file_path_or_url, timeout=30)
        response.raise_for_status()
        content = response.text
        
        # Try parsing with multiple encodings/separators
        attempts = [
            (',', 'utf-8'), (',', 'utf-8-sig'), (',', 'utf-16'),
            ('\t', 'utf-8'), (';', 'utf-8'), (',', 'latin-1'), (',', 'windows-1252')
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
                    logger.info(f"✅ {name} loaded from URL (Sep: '{sep}', Enc: '{enc}', Shape: {df.shape})")
                    return df
            except (pd.errors.ParserError, UnicodeDecodeError):
                continue
        
        logger.error(f"❌ {name}: Could not parse CSV content after all attempts")
        return pd.DataFrame()
        
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ {name}: Failed to fetch URL - {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"❌ {name}: Unexpected error - {type(e).__name__}: {e}")
        return pd.DataFrame()


def get_col(df, cols):
    """
    Get column from dataframe trying multiple possible names.
    First tries exact match (case-sensitive), then normalized match.
    """
    # First try exact column name match (case-sensitive)
    for col in cols:
        if col in df.columns:
            return df[col]
    
    # Then try normalized match (lowercase, stripped)
    df_cols = [c.lower().strip() for c in df.columns]
    for col in cols:
        norm = col.lower().strip()
        if norm in df_cols:
            return df[df.columns[df_cols.index(norm)]]
    
    return pd.Series([np.nan]*len(df), index=df.index)


def combine_social_media_data(meltwater_df, civicsignals_df, tiktok_df=None, openmeasures_df=None):
    """
    Combine datasets from different sources into unified schema.
    Mirrors your Streamlit app's column mapping logic.
    """
    combined = []
    
    if meltwater_df is not None and not meltwater_df.empty:
        mw = pd.DataFrame()
        mw['account_id'] = get_col(meltwater_df, ['influencer'])
        mw['content_id'] = get_col(meltwater_df, ['tweet id', 'post id', 'id'])
        mw['object_id'] = get_col(meltwater_df, ['hit sentence', 'opening text', 'headline', 'text', 'content'])
        mw['URL'] = get_col(meltwater_df, ['url'])
        mw['timestamp_share'] = get_col(meltwater_df, ['date', 'timestamp', 'alternate date format'])
        mw['source_dataset'] = 'Meltwater'
        combined.append(mw)
    
    if civicsignals_df is not None and not civicsignals_df.empty:
        cs = pd.DataFrame()
        cs['account_id'] = get_col(civicsignals_df, ['media_name', 'author', 'username'])
        cs['content_id'] = get_col(civicsignals_df, ['stories_id', 'post_id', 'id'])
        cs['object_id'] = get_col(civicsignals_df, ['title', 'text', 'content', 'body'])
        cs['URL'] = get_col(civicsignals_df, ['url', 'link'])
        cs['timestamp_share'] = get_col(civicsignals_df, ['publish_date', 'timestamp', 'date'])
        cs['source_dataset'] = 'Civicsignal'
        combined.append(cs)
    
    if tiktok_df is not None and not tiktok_df.empty:
        tt = pd.DataFrame()
        tt['object_id'] = get_col(tiktok_df, ['text', 'Transcript', 'caption', 'content'])
        tt['account_id'] = get_col(tiktok_df, ['authorMeta/name', 'username', 'creator'])
        tt['content_id'] = get_col(tiktok_df, ['id', 'video_id', 'itemId'])
        tt['URL'] = get_col(tiktok_df, ['webVideoUrl', 'TikTok Link', 'url'])
        tt['timestamp_share'] = get_col(tiktok_df, ['createTimeISO', 'timestamp', 'date', 'createTime'])
        tt['source_dataset'] = 'TikTok'
        
        # Preserve engagement metrics
        for col in ['playCount', 'diggCount', 'commentCount', 'shareCount', 'repostCount', 'textLanguage']:
            if col in tiktok_df.columns:
                tt[col] = tiktok_df[col]
        
        # Preserve hashtags
        for i in range(5):
            hashtag_col = f'hashtags/{i}/name'
            if hashtag_col in tiktok_df.columns:
                tt[f'hashtag_{i}'] = tiktok_df[hashtag_col]
        
        combined.append(tt)
    
    if openmeasures_df is not None and not openmeasures_df.empty:
        om = pd.DataFrame()
        om['account_id'] = get_col(openmeasures_df, ['context_name', 'channelusername', 'channeltitle'])
        om['content_id'] = get_col(openmeasures_df, ['id', 'url'])
        om['object_id'] = get_col(openmeasures_df, ['text', 'message', 'body'])
        om['URL'] = get_col(openmeasures_df, ['url'])
        
        # Fix timestamp format: remove '@' for proper parsing
        raw_dates = get_col(openmeasures_df, ['created_at', 'date'])
        om['timestamp_share'] = raw_dates.astype(str).str.replace(' @ ', ' ', regex=False)
        
        om['source_dataset'] = 'OpenMeasure_Telegram'
        combined.append(om)
    
    return pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()


def extract_original_text(text):
    """Clean text by removing RT markers, URLs, mentions, etc."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    cleaned = re.sub(r'^(RT|rt|QT|qt|repost|shared|via|credit)\s*[:@]\s*', '', text, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'@\w+|http\S+|www\S+|https\S+', '', cleaned).strip()
    cleaned = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}\b', '', cleaned)
    return re.sub(r'\s+', ' ', cleaned).strip().lower()


def is_original_post(text):
    """Filter out reposts/retweets."""
    if pd.isna(text) or not isinstance(text, str):
        return False
    lower = text.strip().lower()
    if not lower:
        return False
    patterns = [
        r'^🔁.*reposted', r'\b(reposted|reshared|retweeted)\b',
        r'^(rt|qt|repost)\s*[:@\s]', r'^\s*[🔁↪️➡️]\s*@?\w*'
    ]
    if any(re.search(p, lower, flags=re.IGNORECASE) for p in patterns):
        return False
    if len(re.sub(r'http\S+|\@\w+', '', text).strip()) < 15:
        return False
    return len(lower) >= 20 and not re.search(r'^\s*["\u201c]|\s*@\w+\s*[":]', lower)


def parse_timestamp_robust(timestamp):
    """Parse timestamp with multiple format attempts."""
    if pd.isna(timestamp):
        return pd.NaT
    ts_str = re.sub(r'\s+GMT$', '', str(timestamp).strip(), flags=re.IGNORECASE)
    try:
        parsed = pd.to_datetime(ts_str, errors='coerce', utc=True)
        if pd.notna(parsed):
            return parsed
    except:
        pass
    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%d/%m/%Y %H:%M', '%b %d, %Y %H:%M', '%Y-%m-%d']:
        try:
            parsed = pd.to_datetime(ts_str, format=fmt, errors='coerce', utc=True)
            if pd.notna(parsed):
                return parsed
        except:
            continue
    return pd.NaT


def infer_platform_from_url(url):
    """Infer platform from URL domain."""
    if pd.isna(url) or not isinstance(url, str) or not url.startswith("http"):
        return "Unknown"
    url = url.lower()
    platforms = {
        "tiktok.com": "TikTok", "vt.tiktok.com": "TikTok",
        "facebook.com": "Facebook", "fb.watch": "Facebook",
        "twitter.com": "X", "x.com": "X",
        "youtube.com": "YouTube", "youtu.be": "YouTube",
        "instagram.com": "Instagram",
        "telegram.me": "Telegram", "t.me": "Telegram", "telegram.org": "Telegram"
    }
    for key, val in platforms.items():
        if key in url:
            return val
    if any(d in url for d in ["nytimes.com", "bbc.com", "cnn.com", "reuters.com", "aljazeera.com"]):
        return "News/Media"
    return "Media"


def final_preprocess_and_map_columns(df, coordination_mode="Text Content"):
    """
    Clean and preprocess dataframe, mapping columns to standard schema.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            'account_id','content_id','object_id','URL','timestamp_share',
            'Platform','original_text','Outlet','Channel','cluster','source_dataset','Sentiment'
        ])
    
    dfp = df.copy()
    
    # Filter by sentiment if column exists
    if 'Sentiment' in dfp.columns:
        dfp = dfp[dfp['Sentiment'].isin(['Negative', 'Neutral'])]
    
    # Filter to original posts only
    if 'object_id' in dfp.columns:
        mask = dfp['object_id'].apply(is_original_post) & (~dfp['object_id'].str.contains('🔁', na=False)) & (~dfp['object_id'].str.startswith('RT @', na=False))
        dfp = dfp[mask].copy()
    
    # Clean text columns
    dfp['object_id'] = dfp['object_id'].astype(str).replace('nan','').fillna('')
    dfp = dfp[dfp['object_id'].str.strip() != ""]
    
    # Extract original text
    dfp['original_text'] = dfp['object_id'].apply(extract_original_text) if coordination_mode=="Text Content" else dfp['URL'].astype(str).replace('nan','')
    dfp = dfp[dfp['original_text'].str.strip() != ""].reset_index(drop=True)
    
    # Infer platform from URL
    dfp['Platform'] = dfp['URL'].apply(infer_platform_from_url)
    
    # Map source_dataset to Platform
    if 'source_dataset' in dfp.columns:
        dfp['source_dataset'] = dfp['source_dataset'].fillna('')
        
        # Map TikTok
        tiktok_mask = dfp['source_dataset'].str.contains('TikTok|tiktok|vt.tiktok', case=False, na=False)
        dfp.loc[tiktok_mask, 'Platform'] = 'TikTok'
        
        # Map Telegram/OpenMeasure
        telegram_mask = dfp['source_dataset'].str.contains('Telegram|telegram|t.me|OpenMeasure', case=False, na=False)
        dfp.loc[telegram_mask, 'Platform'] = 'Telegram'
        
        # Map Media/News
        media_mask = dfp['source_dataset'].str.contains('Media|News|Civicsignal', case=False, na=False)
        dfp.loc[media_mask, 'Platform'] = 'Media'
    
    # Fill unknown platforms
    dfp['Platform'] = dfp['Platform'].replace('', 'Unknown').fillna('Unknown')
    
    # Add missing columns with defaults
    dfp['Outlet'] = dfp.get('Outlet', np.nan)
    dfp['Channel'] = dfp.get('Channel', np.nan)
    dfp['cluster'] = dfp.get('cluster', -1)
    if 'Sentiment' not in dfp.columns:
        dfp['Sentiment'] = np.nan
    
    # Return standard columns
    cols = ['account_id','content_id','object_id','URL','timestamp_share','Platform','original_text','Outlet','Channel','cluster','source_dataset','Sentiment']
    return dfp[[c for c in cols if c in dfp.columns]].copy()


def process_uploaded_csv(file_path, data_type='custom', source_name='User Upload'):
    """
    Process uploaded CSV file and save to database.
    Returns: (success: bool, message: str, count: int)
    """
    try:
        # Load CSV with robust encoding detection
        df = load_data_robustly(file_path, name=source_name)
        
        if df.empty:
            return False, "Could not load CSV file. Check encoding and format.", 0
        
        # Combine with other sources (even if empty, this normalizes schema)
        df_combined = combine_social_media_data(
            meltwater_df=df if data_type == 'meltwater' else None,
            civicsignals_df=df if data_type == 'civicsignal' else None,
            tiktok_df=df if data_type == 'tiktok' else None,
            openmeasures_df=df if data_type == 'openmeasure' else None
        )
        
        # Preprocess and map columns
        df_processed = final_preprocess_and_map_columns(df_combined if not df_combined.empty else df)
        
        # Parse timestamps
        if 'timestamp_share' in df_processed.columns:
            df_processed['timestamp_share'] = df_processed['timestamp_share'].apply(parse_timestamp_robust)
        
        # Validate required columns
        required_cols = ['content_id', 'account_id', 'original_text', 'timestamp_share', 'platform']
        # Map 'Platform' to 'platform' for database
        if 'Platform' in df_processed.columns and 'platform' not in df_processed.columns:
            df_processed['platform'] = df_processed['Platform']
        
        missing_cols = [col for col in required_cols if col not in df_processed.columns]
        if missing_cols:
            return False, f"Missing required columns: {', '.join(missing_cols)}. Found: {list(df_processed.columns)}", 0
        
        # Save to database
        from dashboard.models import ProcessedPost, DataSource
        from django.utils import timezone
        from datetime import datetime
        
        # Get or create data source
        data_source, _ = DataSource.objects.get_or_create(
            name=source_name,
            defaults={'description': f'Uploaded via UI - {data_type}', 'record_count': 0}
        )
        
        success_count = 0
        failed_count = 0
        
        for idx, row in df_processed.iterrows():
            try:
                # Parse timestamp
                timestamp = row['timestamp_share']
                if isinstance(timestamp, str):
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d', '%d/%m/%Y %H:%M']:
                        try:
                            timestamp = datetime.strptime(timestamp, fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        timestamp = timezone.now()
                elif not isinstance(timestamp, datetime):
                    timestamp = timezone.now()
                
                # Scan for lexicon matches
                text = row.get('original_text', '')
                lexicon_matches = scan_text_for_lexicon_terms(text)
                risk = calculate_risk_score(lexicon_matches)
                
                # Create or update post
                ProcessedPost.objects.update_or_create(
                    content_id=str(row['content_id']),
                    defaults={
                        'account_id': str(row['account_id']),
                        'object_id': row.get('object_id', ''),
                        'original_text': text,
                        'url': row.get('url', '') or row.get('URL', ''),
                        'timestamp_share': timestamp,
                        'platform': row.get('platform', row.get('Platform', 'Unknown')),
                        'source_dataset': data_source,
                        'is_original_post': True,
                        'sentiment': row.get('Sentiment', ''),
                        'cluster': -1,
                        'is_election_related': True,
                        'election_keywords_matched': [],
                        'hashtags': [],
                        'play_count': row.get('playCount') or row.get('play_count'),
                        'digg_count': row.get('diggCount') or row.get('digg_count'),
                        'comment_count': row.get('commentCount') or row.get('comment_count'),
                        'share_count': row.get('shareCount') or row.get('share_count'),
                        'text_language': row.get('textLanguage') or row.get('text_language', ''),
                        'lexicon_matches': lexicon_matches,
                        'risk_score': risk['score'],
                        'risk_level': risk['level'],
                    }
                )
                success_count += 1
                
            except Exception as e:
                logger.error(f"Failed to process row {idx}: {str(e)}")
                failed_count += 1
                continue
        
        # Update data source record count
        data_source.record_count = ProcessedPost.objects.filter(source_dataset=data_source).count()
        data_source.save()
        
        message = f"Processed {success_count} records ({failed_count} failed). Encoding auto-detected."
        return True, message, success_count
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.error(f"Failed to process {file_path}: {error_msg}")
        return False, error_msg, 0


def map_columns_by_type(df, data_type):
    """
    Map CSV columns to standard schema based on data source type.
    (Kept for backward compatibility - main logic now in combine_social_media_data)
    """
    # This function is now largely handled by combine_social_media_data
    # but kept for any direct calls
    return df
