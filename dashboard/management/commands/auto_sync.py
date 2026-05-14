# ingestion/management/commands/auto_sync.py
import os
import uuid
import logging
import requests
import pandas as pd
from datetime import datetime
from django.core.management.base import BaseCommand
from django.utils import timezone
from django.conf import settings

from dashboard.models import SyncSource, ProcessedPost, DataSource
from dashboard.csv_processor import load_brandwatch_data, map_columns_by_type, final_preprocess_and_map_columns
# Import your other processors as needed
# from .utils import sync_peps_from_github, parse_monthly_report_pdf

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Auto-sync data from registered remote sources'

    def add_arguments(self, parser):
        parser.add_argument('--source', type=str, help='Sync only a specific source by name')
        parser.add_argument('--force', action='store_true', help='Ignore frequency limits & sync now')

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('🚀 Starting automated backend sync...'))
        batch_id = f"batch_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d')}"
        
        sources = SyncSource.objects.filter(is_active=True)
        if options['source']:
            sources = sources.filter(name__icontains=options['source'])
            
        for source in sources:
            if not self._should_sync(source) and not options['force']:
                self.stdout.write(f"⏭️ Skipping {source.name} (synced recently)")
                continue
                
            try:
                self.stdout.write(f"📥 Fetching {source.name} from {source.url}...")
                resp = requests.get(source.url, timeout=120, headers={'User-Agent': 'EthiopiaElectionMonitor/1.0'})
                resp.raise_for_status()
                
                if source.file_type == 'csv_posts':
                    self._sync_csv_posts(resp.content, source, batch_id)
                elif source.file_type == 'csv_peps':
                    self._sync_csv_peps(resp.content, source)
                elif source.file_type == 'pdf_report':
                    self._sync_pdf_report(resp.content, source)
                # Add other file types as needed
                    
                source.last_synced = timezone.now()
                source.save()
                self.stdout.write(self.style.SUCCESS(f'✅ Synced: {source.name}'))
                
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'❌ Failed {source.name}: {e}'))
                logger.exception(f"Sync failed for {source.name}")
                
        self.stdout.write(self.style.SUCCESS('✅ Backend sync complete.'))

    def _should_sync(self, source):
        if not source.last_synced:
            return True
        now = timezone.now()
        if source.sync_frequency == 'daily':
            return (now - source.last_synced).days >= 1
        elif source.sync_frequency == 'weekly':
            return (now - source.last_synced).days >= 7
        elif source.sync_frequency == 'monthly':
            return (now - source.last_synced).days >= 30
        return False

    def _sync_csv_posts(self, content, source, batch_id):
        """Route CSV election posts to ProcessedPost"""
        df = pd.read_csv(pd.io.common.BytesIO(content), skiprows=6 if 'brandwatch' in source.name.lower() else 0, low_memory=False, on_bad_lines='skip')
        platform = 'brandwatch' if 'brandwatch' in source.name.lower() else 'meltwater'
        
        mapped = map_columns_by_type(df, platform)
        processed = final_preprocess_and_map_columns(mapped)
        
        if processed.empty:
            return
            
        # Bulk save to ProcessedPost (reuse your existing save logic)
        records = []
        for _, row in processed.iterrows():
            records.append(ProcessedPost(
                content_id=str(row.get('content_id', ''))[:100],
                account_id=str(row.get('account_id', ''))[:100],
                original_text=str(row.get('original_text', '')),
                url=str(row.get('URL', ''))[:500] if pd.notna(row.get('URL')) else None,
                timestamp_share=pd.to_datetime(row.get('timestamp_share'), errors='coerce'),
                platform=row.get('Platform', 'Unknown'),
                batch_id=batch_id,
                ingested_at=timezone.now(),
            ))
        if records:
            ProcessedPost.objects.bulk_create(records, batch_size=1000)
            
    def _sync_csv_peps(self, content, source):
        """Route CSV to PEP model (reuse your existing PEP sync logic)"""
        df = pd.read_csv(pd.io.common.BytesIO(content))
        for _, row in df.iterrows():
            PEP.objects.update_or_create(
                name=row.get('Name (English)', row.get('full_name_en', 'Unknown')),
                defaults={
                    'title': row.get('Position', ''),
                    'x_link': row.get('X (Twitter) Link') if row.get('X (Twitter) Link') not in ['No verified personal account found', 'None'] else None,
                    'facebook_link': row.get('Facebook Link') if row.get('Facebook Link') not in ['No verified personal account found', 'None'] else None,
                    'last_updated': timezone.now()
                }
            )
            
    def _sync_pdf_report(self, content, source):
        """Route PDF to MonthlyReport model (reuse your LLM parser)"""
        from dashboard.models import MonthlyReport
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(content)
            tmp.flush()
            report = MonthlyReport.objects.create(file=os.path.basename(tmp.name), report_month=source.name)
            report.file.save(f'reports/{source.name}_{timezone.now().strftime("%Y%m")}.pdf', tmp)
            report.parse_with_llm()  # Your existing LLM extraction method
            os.unlink(tmp.name)
