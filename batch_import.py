import os
import sys
import django
import pandas as pd
from django.utils import timezone

# 1. Setup Django Context Environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'election_monitor.settings')
django.setup()

from dashboard.models import ProcessedPost, DataSource
from dashboard.utils.csv_processor import process_uploaded_csv, map_columns_by_type, preprocess_dataframe
from dashboard.utils.data_loader import parse_timestamp_robust

def safe_str(val):
    if pd.isna(val) or val is None:
        return ""
    return str(val).strip()

folder = 'media/uploads/social_media'
print('🚀 Starting High-Yield Adaptive Batch Import (v8)...')

for filename in sorted(os.listdir(folder)):
    if not filename.endswith('.csv'): 
        continue
        
    filepath = os.path.join(folder, filename)
    print(f'\n📂 Processing: {filename}')

    name = filename.lower()
    dtype = 'custom'

    # Set platform type tags matching initialization layout
    if 'brandwatch' in name or 'hatespeech' in name or 'polarization' in name:
        dtype = 'brandwatch'
    elif 'meltwater' in name or 'x.csv' in name or 'apri2026x' in name: 
        dtype = 'meltwater'
    elif 'civicsignal' in name or 'media' in name: 
        dtype = 'civicsignal'
    elif 'openmeasure' in name: 
        dtype = 'openmeasure'
    elif 'tiktok' in name: 
        dtype = 'tiktok'

    try:
        # --- SMART EXPLICIT FORMAT HANDLING FOR MELTWATER ---
        if dtype == 'meltwater':
            df = None
            # Attempt 1: Try UTF-16 Tab-Separated (Meltwater Excel style)
            try:
                df = pd.read_csv(filepath, encoding='utf-16', sep='\t', low_memory=False, on_bad_lines='skip')
                if len(df.columns) <= 1:
                    # If it loaded but only found 1 giant column, it's not actually tab-separated
                    df = None
                else:
                    print("  📥 Loaded via Meltwater UTF-16 Tab-Separated settings.")
            except Exception:
                df = None

            # Attempt 2: Fallback to standard UTF-8/CSV settings
            if df is None:
                try:
                    df = pd.read_csv(filepath, encoding='utf-8-sig', low_memory=False, on_bad_lines='skip')
                    print("  📥 Fallback loaded via standard UTF-8 settings.")
                except Exception:
                    df = pd.read_csv(filepath, encoding='latin-1', low_memory=False, on_bad_lines='skip')
                    print("  📥 Fallback loaded via Latin-1 settings.")

            mapped_df = map_columns_by_type(df, 'meltwater')
            processed_df = preprocess_dataframe(mapped_df)
        else:
            # Fallback to standard validation wrappers for non-Meltwater trackers
            processed_df = process_uploaded_csv(filepath, dtype)
        
        if processed_df is None or processed_df.empty:
            print("  ⚠️ No data rows parsed by the structural mapping configuration layout. Skipping.")
            continue
            
        print(f"  🔄 CSV Pipeline Output Verified. Total clean rows parsed: {len(processed_df)}")

        # Drop metrics tracker tracking variables
        dropped_empty = 0
        dropped_duplicate = 0
        count = 0
        
        source_obj, _ = DataSource.objects.get_or_create(name=f"Import_{dtype}_{filename}")
        
        for _, row in processed_df.iterrows():
            text = safe_str(row.get('object_id') or row.get('original_text') or '')
            
            if not text or text.lower() in ['nan', 'none', '']: 
                dropped_empty += 1
                continue

            cid = safe_str(row.get('content_id') or '')
            url = safe_str(row.get('url') or row.get('URL') or row.get('link') or row.get('Link') or '')
            
            # Anomaly content checkpoint
            if not cid and not url:
                dropped_empty += 1
                continue

            # Check duplication parameters to preserve single entry unique indices 
            if cid and cid.lower() not in ['nan', 'none', ''] and ProcessedPost.objects.filter(content_id=cid).exists(): 
                dropped_duplicate += 1
                continue
            if url.startswith('http') and ProcessedPost.objects.filter(url=url).exists(): 
                dropped_duplicate += 1
                continue

            platform_name = row.get('platform') or row.get('Platform') or dtype.title()

            ProcessedPost.objects.create(
                account_id=safe_str(row.get('account_id', 'Unknown'))[:100],
                content_id=cid[:100] if cid else None,
                original_text=text,
                url=url[:500] if url.startswith('http') else None,
                platform=str(platform_name).upper() if str(platform_name).lower() == 'x' else str(platform_name).title(),
                timestamp_share=parse_timestamp_robust(row.get('timestamp_share')),
                source_dataset=source_obj,
                is_election_related=True, 
                ingested_at=timezone.now()
            )
            count += 1

        print(f"  📊 Import Statistics for {filename}:")
        print(f"    - Null / Empty text discarded: {dropped_empty}")
        print(f"    - Existing database duplicates skipped: {dropped_duplicate}")
        print(f"  ✅ Saved {count} fully verified posts into your database system.")
        
    except Exception as e:
        print(f"  ❌ Import operation execution failure pipeline layer error: {e}")
        import traceback
        traceback.print_exc()

print('\n🏁 Import complete.')
