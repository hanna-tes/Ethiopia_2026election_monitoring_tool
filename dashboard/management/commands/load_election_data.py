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
                total_loaded = 0
                
                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    # Normalize headers to lowercase + underscores
                    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
                    
                    # Skip sheets that don't contain the required ID column
                    if 'id' not in df.columns:
                        self.stdout.write(f"  ⏭️ Skipped sheet '{sheet_name}' (no 'id' column)")
                        continue

                    objs = []
                    for _, row in df.iterrows():
                        # Skip metadata/empty rows
                        mid = str(row.get('id', '')).strip()
                        if not mid or mid.lower() in ['nan', 'none', '']:
                            continue

                        # Safe date parsing (handles pandas NaT -> None)
                        sd = row.get('start_date')
                        ed = row.get('end_date')
                        start = pd.to_datetime(sd).date() if pd.notna(sd) else None
                        end = pd.to_datetime(ed).date() if pd.notna(ed) else None

                        objs.append(ElectionOfficeholder(
                            membership_id=mid,
                            role_id=str(row.get('role_id', '')).strip(),
                            person_id=str(row.get('person_id', '')).strip(),
                            party_id=str(row.get('party_id', '')).strip(),
                            membership_type=str(row.get('membership_type', 'officeholder')).strip(),
                            start_date=start,
                            end_date=end,
                            is_partisan=bool(row.get('is_partisan', True)),
                            has_end_date=bool(row.get('has_end_date', True)),
                            contest_id=str(row.get('contest_id', '')).strip(),
                            source_file=filename,
                            source_sheet=sheet_name
                        ))

                    if objs:
                        with transaction.atomic():
                            created = ElectionOfficeholder.objects.bulk_create(objs, ignore_conflicts=True)
                            total_loaded += len(created)
                            
                self.stdout.write(self.style.SUCCESS(f"✅ Loaded {total_loaded} records from {filename}"))
                
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"❌ Failed to process {filename}: {e}"))
                import traceback
                traceback.print_exc()
