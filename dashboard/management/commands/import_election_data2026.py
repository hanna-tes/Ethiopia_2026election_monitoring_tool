import os
import pandas as pd
from django.core.management.base import BaseCommand
from django.conf import settings
from dashboard.models import ElectionOfficeholder

class Command(BaseCommand):
    help = 'Import 2026 Election Candidate data (HoPR, RC, Regional, Executive) into Key Officials'

    # Define the exact files you moved to /media/peps/
    FILES_TO_IMPORT = [
        'HoPR_Candidates.xlsx',
        'Regional_Candidates.xlsx',
        'RC_Members.xlsx',
        'Executive_Members.xlsx'
    ]

    def handle(self, *args, **kwargs):
        peps_dir = os.path.join(settings.MEDIA_ROOT, 'peps')
        
        if not os.path.exists(peps_dir):
            self.stdout.write(self.style.ERROR(f"❌ Directory not found: {peps_dir}"))
            return

        for filename in self.FILES_TO_IMPORT:
            filepath = os.path.join(peps_dir, filename)
            
            if not os.path.exists(filepath):
                self.stdout.write(self.style.WARNING(f"⚠️  File not found, skipping: {filename}"))
                continue

            self.stdout.write(f"\n📂 Processing: {filename}...")
            
            try:
                # Clear old records for this specific file to prevent duplicates
                deleted_count, _ = ElectionOfficeholder.objects.filter(source_file=filename).delete()
                if deleted_count > 0:
                    self.stdout.write(f"   🗑️  Removed {deleted_count} old records for {filename}")

                # Read all sheets in the Excel file
                excel_file = pd.ExcelFile(filepath)
                total_records = 0

                for sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    if df.empty:
                        continue
                    
                    # Save exact column order so the UI can align the table headers
                    col_order = [str(c).strip() for c in df.columns]
                    records_to_create = []
                    
                    for idx, row in df.iterrows():
                        # Skip completely empty rows
                        if all(pd.isna(val) for val in row.values):
                            continue
                        
                        # Clean row data for JSON storage
                        raw_data = {}
                        for col in col_order:
                            val = row.get(col)
                            if pd.isna(val):
                                raw_data[col] = None
                            elif isinstance(val, (int, float, bool)):
                                raw_data[col] = val
                            else:
                                s = str(val).strip()
                                raw_data[col] = None if s.lower() in ['nan', 'none', ''] else s
                                
                        records_to_create.append(ElectionOfficeholder(
                            source_file=filename,
                            source_sheet=sheet_name,
                            row_index=idx,
                            column_order=col_order,
                            raw_data=raw_data
                        ))
                        total_records += 1

                # Bulk create for speed
                if records_to_create:
                    ElectionOfficeholder.objects.bulk_create(records_to_create, batch_size=500)
                    self.stdout.write(self.style.SUCCESS(f"   ✅ Imported {total_records} records from {len(excel_file.sheet_names)} sheets"))

            except Exception as e:
                self.stdout.write(self.style.ERROR(f"   ❌ Error processing {filename}: {e}"))

        self.stdout.write(self.style.SUCCESS("\n🎉 All 2026 election data import complete! Check the 'Key Officials' tab."))
