import pandas as pd
import logging
from .lexicon_engine import scan_text_for_lexicon_terms, calculate_risk_score

logger = logging.getLogger(__name__)

def process_uploaded_csv(file_path, data_type='custom', source_name='User Upload'):
    """
    Process uploaded CSV with automatic encoding detection.
    Returns: (success: bool, message: str, count: int)
    """
    try:
        # Try multiple encodings in order of likelihood
        encodings_to_try = ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'windows-1252', 'latin-1', 'iso-8859-1']
        
        df = None
        used_encoding = None
        
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip')
                used_encoding = encoding
                logger.info(f"Successfully read {file_path} with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                # If it's not an encoding error, re-raise
                if 'codec' not in str(e).lower() and 'decode' not in str(e).lower():
                    raise
        
        if df is None:
            return False, f"Could not decode file with any supported encoding. Tried: {', '.join(encodings_to_try)}", 0
        
        # Preprocess the dataframe
        df = preprocess_dataframe(df)
        
        # Map columns based on data type
        df = map_columns_by_type(df, data_type)
        
        # Validate required columns
        required_cols = ['content_id', 'account_id', 'original_text', 'timestamp_share', 'platform']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            return False, f"Missing required columns: {', '.join(missing_cols)}. Found: {list(df.columns)}", 0
        
        # Process each row and save to database
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
        
        for idx, row in df.iterrows():
            try:
                # Parse timestamp
                timestamp = row['timestamp_share']
                if isinstance(timestamp, str):
                    # Try multiple datetime formats
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d', '%d/%m/%Y %H:%M']:
                        try:
                            timestamp = datetime.strptime(timestamp, fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        timestamp = timezone.now()  # Fallback
                elif not isinstance(timestamp, datetime):
                    timestamp = timezone.now()
                
                # Scan for lexicon matches
                text = row.get('original_text', '')
                lexicon_matches = scan_text_for_lexicon_terms(text)
                risk = calculate_risk_score(lexicon_matches)
                
                # Create or update post
                ProcessedPost.objects.update_or_create(
                    content_id=row['content_id'],
                    defaults={
                        'account_id': row['account_id'],
                        'object_id': row.get('object_id', ''),
                        'original_text': text,
                        'url': row.get('url', ''),
                        'timestamp_share': timestamp,
                        'platform': row['platform'],
                        'source_dataset': data_source,
                        'is_original_post': True,
                        'sentiment': row.get('sentiment', ''),
                        'cluster': -1,
                        'is_election_related': True,  # Default to True, can be filtered later
                        'election_keywords_matched': [],
                        'hashtags': row.get('hashtags', []),
                        'play_count': row.get('play_count'),
                        'digg_count': row.get('digg_count'),
                        'comment_count': row.get('comment_count'),
                        'share_count': row.get('share_count'),
                        'text_language': row.get('text_language', ''),
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
        
        message = f"Processed {success_count} records ({failed_count} failed). Encoding: {used_encoding}"
        return True, message, success_count
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.error(f"Failed to process {file_path}: {error_msg}")
        return False, error_msg, 0


def map_columns_by_type(df, data_type):
    """
    Map CSV columns to standard schema based on data source type.
    """
    # Standard column mappings for different platforms
    mappings = {
        'meltwater': {
            'post_id': 'content_id',
            'author_id': 'account_id',
            'text': 'original_text',
            'posted_at': 'timestamp_share',
            'platform_type': 'platform',
            'url': 'url',
        },
        'tiktok': {
            'video_id': 'content_id',
            'author_id': 'account_id',
            'description': 'original_text',
            'create_time': 'timestamp_share',
            'platform': 'platform',
            'video_url': 'url',
            'play_count': 'play_count',
            'digg_count': 'digg_count',
            'comment_count': 'comment_count',
            'share_count': 'share_count',
        },
        'civicsignal': {
            'id': 'content_id',
            'account': 'account_id',
            'text': 'original_text',
            'timestamp': 'timestamp_share',
            'platform': 'platform',
            'link': 'url',
        },
        'openmeasure': {
            'message_id': 'content_id',
            'account_id': 'account_id',
            'text': 'original_text',
            'timestamp': 'timestamp_share',
            'platform': 'platform',
            'url': 'url',
        },
    }
    
    # Get mapping for data type, or use generic mapping
    mapping = mappings.get(data_type.lower(), {})
    
    # Rename columns if mapping exists
    if mapping:
        df = df.rename(columns={v: k for k, v in mapping.items() if v in df.columns})
    
    # Ensure platform column exists
    if 'platform' not in df.columns:
        df['platform'] = data_type
    
    return df


def preprocess_dataframe(df):
    """
    Clean and preprocess dataframe.
    """
    # Remove completely empty rows
    df = df.dropna(how='all')
    
    # Strip whitespace from string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip() if hasattr(df[col], 'str') else df[col]
    
    # Fill NaN values with empty strings for text columns
    text_cols = ['original_text', 'account_id', 'content_id', 'platform']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('')
    
    return df
