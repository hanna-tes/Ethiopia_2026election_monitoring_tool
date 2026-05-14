import os
import sys
import django
import pandas as pd
from django.utils import timezone

# Setup Django
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'election_monitor.settings')
django.setup()

from dashboard.models import ProcessedPost, DataSource
from dashboard.utils.csv_processor import combine_social_media_data, final_preprocess_and_map_columns, preprocess_dataframe, parse_timestamp_robust
from dashboard.utils.election_filter import is_election_related

folder = 'media/uploads/social_media'
print(' Starting direct batch import...')

for filename in sorted(os.listdir(folder)):
    if not filename.endswith('.csv'): 
        continue
        
    filepath = os.path.join(folder, filename)
    print(f'\n📂 Processing: {filename}')

    # 1. Load with encoding fallback (fixes 0xff UTF-16 BOM error)
    df = None
    for enc in ['utf-8-sig', 'utf-16', 'latin-1']:
        try:
            df = pd.read_csv(filepath, encoding=enc, on_bad_lines='skip', low_memory=False)
            print(f"  ✅ Loaded with {enc} encoding. Shape: {df.shape}")
            break
        except Exception:
            continue
            
    if df is None:
        print("  ❌ Failed to load with any encoding. Skipping.")
        continue

    # 2. Determine data type
    name = filename.lower()
    if 'brandwatch' in name or 'hatespeech' in name or 'polarization' in name: dtype = 'brandwatch'
    elif 'meltwater' in name or 'x.csv' in name: dtype = 'meltwater'
    elif 'civicsignal' in name or 'media' in name: dtype = 'civicsignal'
    elif 'openmeasure' in name: dtype = 'openmeasure'
    elif 'tiktok' in name: dtype = 'tiktok'
    else: dtype = 'custom'

    # 3. Run through your existing pipeline
    try:
        if dtype in ['meltwater', 'brandwatch']:
            combined = combine_social_media_data(meltwater_df=df)
        elif dtype == 'civicsignal':
            combined = combine_social_media_data(civicsignals_df=df)
        elif dtype == 'tiktok':
            combined = combine_social_media_data(tiktok_df=df)
        elif dtype == 'openmeasure':
            combined = combine_social_media_data(openmeasures_df=df)
        else:
            combined = preprocess_dataframe(df)

        processed = final_preprocess_and_map_columns(combined)
        print(f"  🔄 Cleaned rows: {len(processed)}")

        if processed.empty:
            print("  ⚠️ No valid rows after preprocessing.")
            continue

        # Fix hashing/type errors: ensure strings before DB save
        processed = processed.astype(str)
        if 'timestamp_share' in processed.columns:
            processed['timestamp_share'] = processed['timestamp_share'].apply(parse_timestamp_robust)

        # 4. Save to DB
        count = 0
        source_obj, _ = DataSource.objects.get_or_create(name=f"Import_{dtype}_{filename}")
        
        for _, row in processed.iterrows():
            text = row.get('original_text', '').strip()
            if not text or text.lower() in ['nan', 'none', '']: 
                continue

            cid = row.get('content_id', '')
            url = row.get('URL', '')
            
            # Skip duplicates
            if cid and cid.lower() not in ['nan', 'none', ''] and ProcessedPost.objects.filter(content_id=cid).exists(): 
                continue
            if url.startswith('http') and ProcessedPost.objects.filter(url=url).exists(): 
                continue

            ProcessedPost.objects.create(
                account_id=row.get('account_id', '')[:100],
                content_id=cid[:100] if cid else None,
                original_text=text,
                url=url[:500] if url.startswith('http') else None,
                platform=row.get('Platform', dtype.title()),
                timestamp_share=row.get('timestamp_share'),
                source_dataset=source_obj,
                is_election_related=is_election_related(text),
                ingested_at=timezone.now()
            )
            count += 1

        print(f"  ✅ Saved {count} posts to DB.")
        
    except Exception as e:
        print(f"  ❌ Pipeline error: {e}")

print('\n Import complete.')
