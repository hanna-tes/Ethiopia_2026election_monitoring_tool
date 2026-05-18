import os
import pandas as pd
import numpy as np
from django.core.management.base import BaseCommand
from dashboard.models import ElectionOfficeholder

class Command(BaseCommand):
    help = 'Load election candidate Excel files into ElectionOfficeholder model'

    def add_arguments(self, parser):
        parser.add_argument('folder', type=str, help='Path to folder containing Excel files')

    def clean_value(self, val):
        """Convert pandas/Excel values to JSON-safe Python types"""
        if pd.isna(val) or val is None:
            return None
        if isinstance(val, float):
            if np.isnan(val) or np.isinf(val):
                return None
            return val
        if isinstance(val, (np.integer, np.floating)):
            return int(val) if isinstance(val, np.integer) else float(val)
        
        str_val = str(val).strip()
        if str_val.lower() in ['nan', 'none', 'null', '']:
            return None
        return str_val

    def clean_date(self, val):
        """Safely parse dates for Django DateField"""
        if pd.isna(val) or val is None:
            return None
        if isinstance(val, pd.Timestamp):
            return val.date()
        if isinstance(val, str):
            val = val.strip()
            if val.lower() in ['nan', 'none', 'null', '']:
                return None
            try:
                return pd.to_datetime(val).date()
            except Exception:
                return None
        return None

    def handle(self, *args, **options):
        folder = options['folder']
        if not os.path.isdir(folder):
            self.stderr.write(f"❌ Folder not found: {folder}")
            return

        self.stdout.write(f"📂 Processing folder: {folder}")

        for filename in sorted(os.listdir(folder)):
            if not filename.endswith('.xlsx'):
                continue

            filepath = os.path.join(folder, filename)
            self.stdout.write(f"\n📂 Processing: {filename}")

            try:
                xl = pd.ExcelFile(filepath)
                for sheet_name in xl.sheet_names:
                    # Read sheet, keep original types for better cleaning
                    df = pd.read_excel(xl, sheet_name=sheet_name)
                    
                    created = 0
                    for _, row in df.iterrows():
                        # Skip completely empty rows
                        if all(pd.isna(v) or str(v).strip().lower() in ['nan', 'none', 'null', ''] for v in row.values):
                            continue

                        # Build clean dictionary for all fields
                        clean_row = {}
                        for k, v in row.items():
                            clean_row[k] = self.clean_value(v)

                        # Extract key identifiers
                        person_id_val = clean_row.get('id') or clean_row.get('name_id') or ''
                        
                        obj, created_obj = ElectionOfficeholder.objects.update_or_create(
                            source_file=filename,
                            source_sheet=sheet_name,
                            person_id=person_id_val,
                            defaults={
                                # === PERSON FIELDS ===
                                'person_full_name': clean_row.get('full_name'),
                                'person_first_name': clean_row.get('first_name'),
                                'person_last_name': clean_row.get('last_name'),
                                'person_name_amharic': clean_row.get('name_amharic'),
                                'person_region': clean_row.get('region'),
                                'person_gender': clean_row.get('gender'),
                                'person_fb_url': clean_row.get('fb_url'),
                                'person_twitter_url': clean_row.get('twitter_url'),
                                'person_telegram': clean_row.get('telegram'),

                                # === PARTY FIELDS ===
                                'party_id': clean_row.get('party_id') or clean_row.get('partyID'),
                                'party_name_english': clean_row.get('Party Name') or clean_row.get('name_english'),
                                'party_abbrv': clean_row.get('Abbrv') or clean_row.get('abbrv'),
                                'party_name_amharic': clean_row.get('party_name_amharic'),

                                # === ROLE / CHAMBER / AREA FIELDS ===
                                'role_id': clean_row.get('role_id'),
                                'role_title': clean_row.get('title'),
                                'role_title_amharic': clean_row.get('title_amharic'),
                                'chamber_name_english': clean_row.get('name_english'),
                                'area_name': clean_row.get('name'),

                                # === MEMBERSHIP / CONTEST FIELDS ===
                                'membership_id': clean_row.get('id'),
                                'membership_type': clean_row.get('membership_type'),
                                'contest_id': clean_row.get('contest_id'),
                                'start_date': self.clean_date(clean_row.get('start_date')),
                                'end_date': self.clean_date(clean_row.get('end_date')),
                                'is_partisan': str(clean_row.get('is_partisan', '')).lower() in ['true', 'yes', '1'],
                                'has_end_date': str(clean_row.get('has_end_date', '')).lower() in ['true', 'yes', '1'],

                                # === FLEXIBLE JSON STORAGE ===
                                'raw_data': clean_row,
                            }
                        )
                        if created_obj:
                            created += 1

                    self.stdout.write(f"  ✅ {sheet_name}: {created} new/updated records")

            except Exception as e:
                self.stderr.write(f"❌ Error processing {filename}: {e}")
                import traceback
                traceback.print_exc()

        self.stdout.write("\n🏁 Import complete.")
