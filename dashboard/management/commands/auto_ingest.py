# ingestion/management/commands/auto_ingest.py
import os
import uuid
import logging
import requests
import pandas as pd
from datetime import datetime
from django.core.management.base import BaseCommand
from django.utils import timezone
from django.conf import settings

from dashboard.models import ProcessedPost, DataSource
# ⚠️ Adjust this import path to match your actual project structure
from dashboard.csv_processor import load_brandwatch_data, map_columns_by_type, final_preprocess_and_map_columns

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Automatically ingest latest election data from configured remote sources'

    # 🔧 CONFIGURE YOUR DATA SOURCES HERE
    # Use environment variables in production for security
    DATA_SOURCES = {
        'brandwatch': {
            'type': 'brandwatch',
            'url': os.getenv('BRANDWATCH_CSV_URL', ''),
            'skip_rows': 6  # Brandwatch exports usually have 6 metadata rows
        },
        'openmeasure': {
            'type': 'openmeasure',
            'url': os.getenv('OPENMEASURE_CSV_URL', ''),
        },
        'media': {
            'type': 'civicsignal',
            'url': os.getenv('MEDIA_CSV_URL', ''),
        },
        'tiktok': {
            'type': 'tiktok',
            'url': os.getenv('TIKTOK_CSV_URL', ''),
        }
    }

    def add_arguments(self, parser):
        parser.add_argument('--source', type=str, help='Ingest only a specific source (e.g., brandwatch)')
        parser.add_argument('--test', action='store_true', help='Run in test mode (fetch & process but do NOT save to DB)')

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('🚀 Starting automated data ingestion...'))
        batch_id = f"batch_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d')}"
        self.stdout.write(f"📦 Batch ID: {batch_id}")

        if options['test']:
            self.stdout.write(self.style.WARNING('⚠️ TEST MODE: No data will be saved to the database.'))

        sources_to_run = [options['source']] if options['source'] else list(self.DATA_SOURCES.keys())

        for source_key in sources_to_run:
            if source_key not in self.DATA_SOURCES:
                self.stdout.write(self.style.ERROR(f'❌ Unknown source: {source_key}'))
                continue

            config = self.DATA_SOURCES[source_key]
            url = config.get('url')
            if not url:
                self.stdout.write(self.style.WARNING(f'️ Skipping {source_key}: No URL configured.'))
                continue

            try:
                self.stdout.write(f'📥 Fetching {source_key} from {url}...')
                response = requests.get(url, timeout=120, headers={'User-Agent': 'EthiopiaElectionMonitor/1.0'})
                response.raise_for_status()

                # Load CSV safely
                skiprows = config.get('skip_rows', 0)
                df = pd.read_csv(
                    pd.io.common.BytesIO(response.content), 
                    skiprows=skiprows, 
                    low_memory=False, 
                    on_bad_lines='skip'
                )
                self.stdout.write(f'✅ Downloaded {len(df)} raw rows for {source_key}')

                # Process DataFrame through your pipeline
                processed_df = self._process_dataframe(df, config['type'])
                if processed_df.empty:
                    self.stdout.write(self.style.WARNING(f'⚠️ {source_key} yielded 0 valid records after preprocessing.'))
                    continue

                self.stdout.write(f'✨ Preprocessed {len(processed_df)} records for {source_key}')

                if not options['test']:
                    self._save_to_database(processed_df, source_key, batch_id)
                    self.stdout.write(self.style.SUCCESS(f'💾 Saved {source_key} to database.'))

            except requests.RequestException as e:
                self.stdout.write(self.style.ERROR(f'🌐 Network error for {source_key}: {e}'))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'💥 Failed to process {source_key}: {e}'))
                logger.exception(f'Auto-ingest failed for {source_key}')

        self.stdout.write(self.style.SUCCESS('✅ Automated ingestion complete.'))

    def _process_dataframe(self, df, data_type):
        """Apply your existing pipeline logic safely"""
        try:
            # Step 1: Map platform-specific columns to standard schema
            mapped_df = map_columns_by_type(df, data_type)

            # Step 2: Run your advanced preprocessing & filtering
            processed_df = final_preprocess_and_map_columns(mapped_df)

            # Step 3: Final cleanup: drop rows missing critical fields
            required_cols = ['content_id', 'account_id', 'original_text']
            for col in required_cols:
                if col not in processed_df.columns:
                    processed_df[col] = pd.NA

            processed_df = processed_df.dropna(subset=['content_id', 'account_id', 'original_text'])
            processed_df['original_text'] = processed_df['original_text'].astype(str).str.strip()
            processed_df = processed_df[processed_df['original_text'] != 'nan']

            return processed_df.reset_index(drop=True)
        except Exception as e:
            logger.error(f"Processing failed for {data_type}: {e}")
            return pd.DataFrame()

    def _save_to_database(self, df, source_key, batch_id):
        """Efficiently batch-save records to ProcessedPost model"""
        self.stdout.write(f'💾 Saving {len(df)} records to DB...')

        # Get or create DataSource reference for tracking
        source_obj, _ = DataSource.objects.get_or_create(name=source_key)

        records = []
        for _, row in df.iterrows():
            try:
                # Parse timestamp safely (handles strings, NaN, etc.)
                ts = row.get('timestamp_share')
                if isinstance(ts, str):
                    ts = pd.to_datetime(ts, errors='coerce')
                elif pd.isna(ts):
                    ts = None

                records.append(ProcessedPost(
                    account_id=str(row.get('account_id', ''))[:100],
                    content_id=str(row.get('content_id', ''))[:100],
                    object_id=str(row.get('object_id', '')),
                    original_text=str(row.get('original_text', '')),
                    url=str(row.get('URL', ''))[:500] if pd.notna(row.get('URL')) else None,
                    timestamp_share=ts,
                    platform=row.get('Platform', 'Unknown'),
                    source_dataset=source_obj,
                    batch_id=batch_id,
                    ingested_at=timezone.now(),
                    sentiment=row.get('Sentiment', 'Neutral'),
                    cluster=-1,
                    is_original_post=True,
                    is_election_related=False  # Default; can be updated via separate task
                ))
            except Exception as e:
                logger.warning(f"Skipping row due to error: {e}")
                continue

        if records:
            # Bulk insert for performance (batch_size=1000 prevents memory spikes)
            ProcessedPost.objects.bulk_create(records, batch_size=1000)
            self.stdout.write(f'✅ Successfully saved {len(records)} records.')
        else:
            self.stdout.write(self.style.WARNING('⚠️ No valid records to save.'))
