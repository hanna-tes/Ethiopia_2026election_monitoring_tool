import pandas as pd
import logging
from django.conf import settings
from .data_loader import load_data_robustly
from .election_filter import is_election_related
from .lexicon_engine import scan_text_for_lexicon_terms, calculate_risk_score

logger = logging.getLogger(__name__)


def process_uploaded_csv(file_path, data_type='custom', source_name='User Upload'):
    """
    Process a user-uploaded CSV through your existing pipeline:
    1. Load CSV
    2. Map columns (based on data_type)
    3. Preprocess (clean text, infer platform, etc.)
    4. Filter for election relevance
    5. Scan for lexicon matches
    6. Save to ProcessedPost model
    
    Returns: (success: bool, message: str, records_processed: int)
    """
    try:
        # 1. Load CSV
        df = load_data_robustly(file_path, source_name)
        if df.empty:
            return False, "CSV file is empty or could not be parsed", 0
        
        logger.info(f"✅ Loaded {len(df)} rows from {file_path}")
        
        # 2. Map columns based on data_type
        mapped_df = map_columns_by_type(df, data_type)
        
        # 3. Preprocess (reuse your final_preprocess_and_map_columns logic)
        processed_df = preprocess_dataframe(mapped_df)
        
        # 4. Filter & enrich
        enriched_records = []
        for _, row in processed_df.iterrows():
            # Check election relevance
            is_election = is_election_related(row['original_text'])
            
            # Scan for lexicon matches
            lexicon_matches = scan_text_for_lexicon_terms(row['original_text'])
            risk = calculate_risk_score(lexicon_matches) if lexicon_matches else {'score': 0, 'level': 'low'}
            
            enriched_records.append({
                'content_id': row.get('content_id', f"{row['account_id']}_{row['timestamp_share']}"),
                'account_id': row['account_id'],
                'object_id': row['object_id'],
                'original_text': row['original_text'],
                'url': row.get('url'),
                'timestamp_share': row['timestamp_share'],
                'platform': row['platform'],
                'source_dataset_name': source_name,
                'is_original_post': True,  # Already filtered in preprocess
                'sentiment': row.get('sentiment'),
                'cluster': -1,  # Will be updated by clustering job
                'is_election_related': is_election,
                'election_keywords_matched': [],  # Could extract matched keywords here
                # TikTok fields
                'play_count': row.get('play_count'),
                'digg_count': row.get('digg_count'),
                'comment_count': row.get('comment_count'),
                'share_count': row.get('share_count'),
                'hashtags': row.get('hashtags', []),
                'text_language': row.get('text_language'),
                # Lexicon analysis
                'lexicon_matches': lexicon_matches,
                'risk_score': risk['score'],
                'risk_level': risk['level'],
            })
        
        # 5. Bulk create in database
        from dashboard.models import ProcessedPost, DataSource
        
        # Create or get DataSource
        data_source, _ = DataSource.objects.get_or_create(
            name=source_name,
            defaults={'description': f'User upload: {file_path}'}
        )
        
        # Prepare records for bulk_create
        post_objects = []
        for record in enriched_records:
            source_ds = DataSource.objects.get(name=record.pop('source_dataset_name'))
            record['source_dataset'] = source_ds
            post_objects.append(ProcessedPost(**record))
        
        # Bulk insert (efficient for large files)
        created = ProcessedPost.objects.bulk_create(post_objects, batch_size=1000, ignore_conflicts=True)
        
        logger.info(f"✅ Saved {len(created)} processed posts to database")
        return True, f"Successfully processed {len(created)} records", len(created)
        
    except Exception as e:
        logger.error(f"❌ Error processing CSV: {str(e)}")
        return False, f"Processing failed: {str(e)}", 0


def map_columns_by_type(df, data_type):
    """Map CSV columns to standard schema based on data_type"""
    # Reuse your get_col logic from combine_social_media_data
    def get_col(cols):
        df_cols = [c.lower().strip() for c in df.columns]
        for col in cols:
            norm = col.lower().strip()
            if norm in df_cols:
                return df[df.columns[df_cols.index(norm)]]
        return pd.Series([None]*len(df), index=df.index)
    
    mapped = pd.DataFrame()
    
    if data_type == 'tiktok':
        mapped['object_id'] = get_col(['text', 'Transcript', 'caption', 'content'])
        mapped['account_id'] = get_col(['authorMeta/name', 'username', 'creator'])
        mapped['content_id'] = get_col(['id', 'video_id', 'itemId'])
        mapped['url'] = get_col(['webVideoUrl', 'TikTok Link', 'url'])
        mapped['timestamp_share'] = get_col(['createTimeISO', 'timestamp', 'date', 'createTime'])
        mapped['platform'] = 'TikTok'
        # TikTok-specific
        mapped['play_count'] = get_col(['playCount'])
        mapped['digg_count'] = get_col(['diggCount'])
        mapped['comment_count'] = get_col(['commentCount'])
        mapped['share_count'] = get_col(['shareCount'])
        mapped['text_language'] = get_col(['textLanguage'])
        # Hashtags
        hashtags = []
        for i in range(13):
            col = f'hashtags/{i}/name'
            if col in df.columns:
                hashtags.append(df[col])
        mapped['hashtags'] = hashtags
    
    elif data_type == 'openmeasure':
        mapped['account_id'] = get_col(['context_name', 'channelusername', 'channeltitle'])
        mapped['content_id'] = get_col(['id', 'url'])
        mapped['object_id'] = get_col(['text', 'message', 'body'])
        mapped['url'] = get_col(['url'])
        raw_dates = get_col(['created_at', 'date'])
        mapped['timestamp_share'] = raw_dates.astype(str).str.replace(' @ ', ' ', regex=False)
        mapped['platform'] = 'Telegram'
    
    elif data_type == 'meltwater':
        mapped['account_id'] = get_col(['influencer'])
        mapped['content_id'] = get_col(['tweet id', 'post id', 'id'])
        mapped['object_id'] = get_col(['hit sentence', 'opening text', 'headline', 'text', 'content'])
        mapped['url'] = get_col(['url'])
        mapped['timestamp_share'] = get_col(['date', 'timestamp', 'alternate date format'])
        mapped['platform'] = 'X'
    
    elif data_type == 'civicsignal':
        mapped['account_id'] = get_col(['media_name', 'author', 'username'])
        mapped['content_id'] = get_col(['stories_id', 'post_id', 'id'])
        mapped['object_id'] = get_col(['title', 'text', 'content', 'body'])
        mapped['url'] = get_col(['url', 'link'])
        mapped['timestamp_share'] = get_col(['publish_date', 'timestamp', 'date'])
        mapped['platform'] = 'Media'
    
    else:  # custom - try to auto-detect
        mapped['account_id'] = get_col(['account_id', 'username', 'author', 'influencer', 'account'])
        mapped['content_id'] = get_col(['content_id', 'post_id', 'id', 'tweet_id'])
        mapped['object_id'] = get_col(['object_id', 'text', 'content', 'message', 'body', 'post_text'])
        mapped['url'] = get_col(['url', 'link', 'post_url'])
        mapped['timestamp_share'] = get_col(['timestamp_share', 'timestamp', 'date', 'created_at', 'post_date'])
        mapped['platform'] = get_col(['platform', 'source', 'network'])
    
    return mapped


def preprocess_dataframe(df):
    """Apply your preprocessing logic to a DataFrame"""
    from dashboard.utils.data_loader import extract_original_text, infer_platform_from_url, is_original_post
    import pandas as pd
    import numpy as np
    
    dfp = df.copy()
    
    # Clean object_id
    dfp['object_id'] = dfp['object_id'].astype(str).replace('nan', '').fillna('')
    dfp = dfp[dfp['object_id'].str.strip() != ""]
    
    # Extract original text
    dfp['original_text'] = dfp['object_id'].apply(extract_original_text)
    dfp = dfp[dfp['original_text'].str.strip() != ""].reset_index(drop=True)
    
    # Infer platform from URL if not set
    if 'platform' in dfp.columns:
        mask = dfp['platform'].isna() | (dfp['platform'] == '')
        if mask.any():
            dfp.loc[mask, 'platform'] = dfp.loc[mask, 'url'].apply(infer_platform_from_url)
    
    # Parse timestamps
    if 'timestamp_share' in dfp.columns:
        dfp['timestamp_share'] = pd.to_datetime(dfp['timestamp_share'], errors='coerce')
        dfp = dfp[dfp['timestamp_share'].notna()]
    
    return dfp
