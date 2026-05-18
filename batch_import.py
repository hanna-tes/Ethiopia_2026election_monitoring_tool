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
from dashboard.utils.csv_processor import process_uploaded_csv  # Official, unified parser
from dashboard.utils.data_loader import parse_timestamp_robust
from dashboard.utils.election_filter import is_election_related

def safe_str(val):
    if pd.isna(val) or val is None:
        return ""
    return str(val).strip()

folder = 'media/uploads/social_media'
print('🚀 Starting Fully Aligned Batch Import (v5)...')

for filename in sorted(os.listdir(folder)):
    if not filename.endswith('.csv'): 
        continue
        
    filepath = os.path.join(folder, filename)
    print(f'\n📂 Processing: {filename}')

    name = filename.lower()
    dtype = 'custom'

    # Set appropriate dtype parameters matching your csv_processor.py rules
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
        # 2. Call your official unified entry point
        # This automatically applies load_brandwatch_data, skip_rows, or encoding rules safely
        processed_df = process_uploaded_csv(filepath, dtype)
        
        if processed_df is None or processed_df.empty:
            print("  ⚠️ No data parsed by the CSV Processor. Skipping.")
            continue
            
        print(f"  🔄 CSV Processor output verified. Clean rows: {len(processed_df)}")

        # 3. Store entries to the data model layer securely
        count = 0
        source_obj, _ = DataSource.objects.get_or_create(name=f"Import_{dtype}_{filename}")
        
        for _, row in processed_df.iterrows():
            # ROBUST FALLBACK: Grab whichever text column label your engine outputted
            text = safe_str(row.get('object_id') or row.get('original_text') or '')
            
            if not text or text.lower() in ['nan', 'none', '']: 
                continue

            # Run Election-Related Verification
            if not is_election_related(text):
                continue 

            cid = safe_str(row.get('content_id') or '')
            url = safe_str(row.get('url') or row.get('URL') or row.get('link') or row.get('Link') or '')
            
            # Anomaly safety checkpoint
            if not cid and not url:
                continue

            # Check for existing records to prevent unique-constraint crashes
            if cid and cid.lower() not in ['nan', 'none', ''] and ProcessedPost.objects.filter(content_id=cid).exists(): 
                continue
            if url.startswith('http') and ProcessedPost.objects.filter(url=url).exists(): 
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

        print(f"  ✅ Saved {count} verified posts to your database.")
        
    except Exception as e:
        print(f"  ❌ Import operation pipeline execution failure: {e}")
        import traceback
        traceback.print_exc()

print('\n🏁 Import complete.')
