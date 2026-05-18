import os
import pandas as pd
from django.core.management.base import BaseCommand
from dashboard.models import ElectionOfficeholder
from django.db import transaction

class Command(BaseCommand):
    help = 'Load all sheets from election Excel files into the database'

    def add_arguments(self, parser):
        parser.add_argument('folder_path', type=str, help='Path to folder containing Excel files')

    def handle(self, *args, **kwargs):
        folder = kwargs['folder_path']
        for filename in os.listdir(folder):
            if not filename.endswith(('.xlsx', '.xls')):
                continue
                
            filepath = os.path.join(folder, filename)
            self.stdout.write(f"\n📂 Processing: {filename}")
            
            try:
                xls = pd.ExcelFile(filepath)
                total_rows = 0
                
                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    # Normalize headers to match model fields
                    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
                    
                    objs = []
                    for _, row in df.iterrows():
                        objs.append(ElectionOfficeholder(
                            membership_id=str(row.get('id', '')).strip(),
                            role_id=str(row.get('role_id', '')).strip(),
                            person_id=str(row.get('person_id', '')).strip(),
                            party_id=str(row.get('party_id', '')).strip(),
                            membership_type=str(row.get('membership_type', 'officeholder')).strip(),
                            start_date=row.get('start_date'),
                            end_date=row.get('end_date'),
                            is_partisan=bool(row.get('is_partisan', True)),
                            has_end_date=bool(row.get('has_end_date', True)),
                            contest_id=str(row.get('contest_id', '')).strip(),
                            source_file=filename,
                            source_sheet=sheet_name
                        ))
                    
                    # Bulk insert, skip duplicates on membership_id
                    with transaction.atomic():
                        created = ElectionOfficeholder.objects.bulk_create(objs, ignore_conflicts=True)
                        total_rows += len(created)
                        
                self.stdout.write(self.style.SUCCESS(f"✅ Successfully loaded {total_rows} new rows from {filename}"))
                
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"❌ Failed to process {filename}: {e}"))
