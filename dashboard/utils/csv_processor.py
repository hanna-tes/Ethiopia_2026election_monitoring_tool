import pandas as pd
import logging
import re
from dashboard.models import ProcessedPost, DataSource
from .lexicon_engine import scan_text_for_lexicon_terms, calculate_risk_score

logger = logging.getLogger(__name__)

def clean_text(text):
    """Basic text cleaning"""
    if pd.isna(text):
        return ''
    return str(text).strip()

def parse_timestamp_robust(val):
    """Parse timestamps with fallback to UTC"""
    if pd.isna(val):
        return pd.NaT
    s = str(val).strip().replace(' @ ', ' ')  # Fix OpenMeasure format
    try:
        return pd.to_datetime(s, utc=True, errors='coerce')
    except Exception:
        return pd.NaT

def load_csv_for_platform(file_path, data_type):
    """Load CSV with platform-specific encoding & separator"""
    dt = data_type.lower()
    
    if dt in ['meltwater', 'x', 'twitter']:
        # Your script uses: encoding='utf-16', sep='\t'
        return pd.read_csv(file_path, encoding='utf-16', sep='\t', low_memory=False, on_bad_lines='skip')
    elif dt in ['openmeasure', 'telegram']:
        return pd.read_csv(file_path, encoding='utf-8', low_memory=False, on_bad_lines='skip')
    elif dt in ['media', 'civicsignal', 'news']:
        return pd.read_csv(file_path, encoding='utf-8', low_memory=False, on_bad_lines='skip')
    elif dt in ['tiktok']:
        return pd.read_csv(file_path, encoding='utf-8', low_memory=False, on_bad_lines='skip')
    else:
        # Fallback: try multiple encodings
        for enc in ['utf-8', 'utf-8-sig', 'utf-16', 'latin-1', 'windows-1252']:
            try:
                return pd.read_csv(file_path, encoding=enc, low_memory=False, on_bad_lines='skip')
            except Exception:
                continue
        raise ValueError(f"Could not decode {file_path} with any supported encoding")

def standardize_to_common_schema(df, data_type):
    """
    Map platform-specific columns to your proven common schema:
    Source, URL, Timestamp, text, Platform
    """
    dt = data_type.lower()
    df = df.copy()
    
    # Helper: find column case-insensitively or by partial match
    def find_col(possible_names):
        for name in possible_names:
            # Exact match
            if name in df.columns:
                return name
            # Case-insensitive match
            match = next((c for c in df.columns if c.lower().strip() == name.lower()), None)
            if match:
                return match
            # Partial match
            match = next((c for c in df.columns if name.lower() in c.lower()), None)
            if match:
                return match
        return None

    if dt in ['meltwater', 'x', 'twitter']:
        col_map = {
            find_col(['Influencer', 'influencer', 'author', 'account']): 'Source',
            find_col(['URL', 'url', 'link']): 'URL',
            find_col(['Date', 'date', 'timestamp', 'posted_at']): 'Timestamp',
            find_col(['Hit Sentence', 'hit sentence', 'text', 'content', 'opening text']): 'text'
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k is not None})
        df['Platform'] = 'X'
        
    elif dt in ['openmeasure', 'telegram']:
        col_map = {
            find_col(['actor_username', 'channeltitle', 'channelusername', 'context_name', 'author']): 'Source',
            find_col(['url', 'link']): 'URL',
            find_col(['created_at', 'date', 'timestamp']): 'Timestamp',
            find_col(['text', 'message', 'body', 'content']): 'text'
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k is not None})
        # Try to extract platform from metadata if available
        plat_col = find_col(['openmeasures_meta_source_endpoint', 'platform', 'source_type'])
        if plat_col:
            df['Platform'] = df[plat_col]
        else:
            df['Platform'] = 'Telegram'
            
    elif dt in ['media', 'civicsignal', 'news']:
        col_map = {
            find_col(['media_name', 'outlet', 'author', 'source']): 'Source',
            find_col(['url', 'link', 'story_url']): 'URL',
            find_col(['publish_date', 'date', 'timestamp', 'published_at']): 'Timestamp',
            find_col(['title', 'text', 'headline', 'content']): 'text'
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k is not None})
        df['Platform'] = 'Media'
        
    elif dt in ['tiktok']:
        col_map = {
            find_col(['authorMeta/name', 'authorMeta/name', 'username', 'creator', 'author']): 'Source',
            find_col(['webVideoUrl', 'TikTok Link', 'url', 'link']): 'URL',
            find_col(['createTimeISO', 'createTime', 'date', 'timestamp']): 'Timestamp',
            find_col(['text', 'Transcript', 'caption', 'description']): 'text'
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k is not None})
        df['Platform'] = 'TikTok'
        
    else:
        # Generic fallback
        col_map = {
            find_col(['source', 'author', 'account', 'influencer', 'username']): 'Source',
            find_col(['url', 'link']): 'URL',
            find_col(['timestamp', 'date', 'created_at', 'posted_at']): 'Timestamp',
            find_col(['text', 'content', 'message', 'body', 'title']): 'text'
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k is not None})
        df['Platform'] = data_type.title()

    # Ensure all required columns exist
    for col in ['Source', 'URL', 'Timestamp', 'text', 'Platform']:
        if col not in df.columns:
            df[col] = ''
            
    return df[['Source', 'URL', 'Timestamp', 'text', 'Platform']]

def process_uploaded_csv(file_path, data_type='custom', source_name='User Upload'):
    """
    Main upload processor. Matches your working script's logic exactly.
    Returns: (success: bool, message: str, count: int)
    """
    try:
        logger.info(f"📥 Processing {file_path} | Type: {data_type} | Source: {source_name}")
        
        # 1. Load with correct encoding/separator
        df = load_csv_for_platform(file_path, data_type)
        logger.info(f"✅ Loaded {len(df)} rows. Columns: {list(df.columns)}")
        
        # 2. Standardize to common schema
        df = standardize_to_common_schema(df, data_type)
        
        # 3. Clean text & filter empty rows
        df['text'] = df['text'].apply(clean_text)
        df = df[df['text'].str.len() > 3].copy()
        logger.info(f"🧹 After cleaning: {len(df)} valid rows")
        
        # 4. Parse timestamps
        df['Timestamp'] = df['Timestamp'].apply(parse_timestamp_robust)
        df = df[df['Timestamp'].notna()].copy()
        logger.info(f"📅 After timestamp parsing: {len(df)} rows")
        
        if df.empty:
            return False, "No valid rows after cleaning. Check column names & date formats.", 0
            
        # 5. Save to PostgreSQL
        data_source, _ = DataSource.objects.get_or_create(
            name=source_name,
            defaults={'description': f'Uploaded via UI - {data_type}', 'record_count': 0}
        )
        
        success_count = 0
        fail_count = 0
        
        for _, row in df.iterrows():
            try:
                # Generate unique content_id
                content_id = str(row.get('URL', row.get('Source', '')) + str(row['Timestamp']))[:255]
                if not content_id or content_id.lower() in ['nan', 'nat']:
                    content_id = f"{source_name}_{success_count}_{hash(str(row['text']))}"
                
                text = str(row['text'])
                matches = scan_text_for_lexicon_terms(text)
                risk = calculate_risk_score(matches)
                
                ProcessedPost.objects.update_or_create(
                    content_id=content_id,
                    defaults={
                        'account_id': str(row.get('Source', ''))[:255],
                        'original_text': text,
                        'object_id': text[:500],
                        'url': str(row.get('URL', ''))[:500],
                        'timestamp_share': row['Timestamp'],
                        'platform': str(row.get('Platform', 'Unknown'))[:50],
                        'source_dataset': data_source,
                        'lexicon_matches': matches,
                        'risk_score': risk['score'],
                        'risk_level': risk['level'],
                        'is_election_related': True,
                        'is_original_post': True,
                        'sentiment': '',
                        'cluster': -1,
                        'election_keywords_matched': [],
                        'hashtags': [],
                    }
                )
                success_count += 1
            except Exception as e:
                logger.warning(f"⚠️ Row failed: {e}")
                fail_count += 1
                continue
                
        # Update source record count
        data_source.record_count = ProcessedPost.objects.filter(source_dataset=data_source).count()
        data_source.save()
        
        msg = f"✅ {success_count} records saved"
        if fail_count > 0:
            msg += f" ({fail_count} skipped)"
        logger.info(f"🎉 {msg}")
        return True, msg, success_count
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}", exc_info=True)
        return False, f"Error: {str(e)}", 0


# === Backward Compatibility Stubs ===
def preprocess_dataframe(df):
    if df.empty: return df
    df = df.dropna(how='all')
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip() if hasattr(df[col], 'str') else df[col]
    return df

def map_columns_by_type(df, data_type):
    return df
