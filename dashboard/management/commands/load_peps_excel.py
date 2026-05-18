import os
import pandas as pd
from django.core.management.base import BaseCommand
from dashboard.models import PEP
from django.utils import timezone

class Command(BaseCommand):
    help = 'Load PEPs from Excel files into the database'

    def add_arguments(self, parser):
        parser.add_argument('folder_path', type=str, help='Path to folder containing PEP Excel files')

    def handle(self, *args, **options):
        folder = options['folder_path']
        files = [f for f in os.listdir(folder) if f.endswith('.xlsx') or f.endswith('.xls')]
        
        if not files:
            self.stdout.write(self.style.WARNING('⚠️ No Excel files found in the specified folder.'))
            return

        total_loaded = 0
        total_skipped = 0

        for file in files:
            self.stdout.write(f'\n📂 Processing: {file}')
            df = pd.read_excel(os.path.join(folder, file))
            
            # Normalize column names to lowercase & strip spaces
            df.columns = [str(c).lower().strip() for c in df.columns]
            
            for _, row in df.iterrows():
                name = str(row.get('name (english)', row.get('full_name_en', ''))).strip()
                if not name or name.lower() in ['nan', 'none', '']:
                    total_skipped += 1
                    continue

                obj, created = PEP.objects.update_or_create(
                    name=name,
                    defaults={
                        'title': row.get('position', ''),
                        'x_link': row.get('x (twitter) link') if pd.notna(row.get('x (twitter) link')) else None,
                        'facebook_link': row.get('facebook link') if pd.notna(row.get('facebook link')) else None,
                        'confidence_level': str(row.get('confidence', 'medium')).lower(),
                        'last_updated': timezone.now()
                    }
                )
                if created:
                    total_loaded += 1
                else:
                    total_skipped += 1

        self.stdout.write(self.style.SUCCESS(f'\n✅ Import Complete!'))
        self.stdout.write(f'🆕 Added: {total_loaded} | ⏭️ Updated/Skipped: {total_skipped}')
