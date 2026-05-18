import os
import pandas as pd
from django.core.management.base import BaseCommand
from dashboard.models import ElectionOfficeholder

class Command(BaseCommand):
    help = 'Load election candidate Excel files into ElectionOfficeholder'

    def add_arguments(self, parser):
        parser.add_argument('folder', type=str, help='Path to folder containing Excel files')

    def handle(self, *args, **options):
        folder = options['folder']
        if not os.path.isdir(folder):
            self.stderr.write(f"❌ Folder not found: {folder}")
            return

        for filename in os.listdir(folder):
            if not filename.endswith('.xlsx'):
                continue
                
            filepath = os.path.join(folder, filename)
            self.stdout.write(f"\n📂 Processing: {filename}")
            
            xl = pd.ExcelFile(filepath)
            for sheet_name in xl.sheet_names:
                df = pd.read_excel(xl, sheet_name=sheet_name)
                df = df.astype(str).replace('nan', '')  # Clean NaNs
                
                created = 0
                for _, row in df.iterrows():
                    # Skip completely empty rows
                    if all(v == '' for v in row.values):
                        continue
                        
                    obj, created_obj = ElectionOfficeholder.objects.update_or_create(
                        source_file=filename,
                        source_sheet=sheet_name,
                        person_id=str(row.get('id', '') or row.get('name_id', '')),
                        defaults={
                            # Person
                            'person_full_name': row.get('full_name', ''),
                            'person_first_name': row.get('first_name', ''),
                            'person_last_name': row.get('last_name', ''),
                            'person_name_amharic': row.get('name_amharic', ''),
                            'person_region': row.get('region', ''),
                            'person_gender': row.get('gender', ''),
                            'person_fb_url': row.get('fb_url', ''),
                            'person_twitter_url': row.get('twitter_url', ''),
                            'person_telegram': row.get('telegram', ''),
                            # Party
                            'party_id': row.get('party_id', row.get('partyID', '')),
                            'party_name_english': row.get('Party Name', row.get('name_english', '')),
                            'party_abbrv': row.get('Abbrv', row.get('abbrv', '')),
                            'party_name_amharic': row.get('party_name_amharic', row.get('name_amharic', '')),
                            # Role/Chamber/Area
                            'role_id': row.get('role_id', ''),
                            'role_title': row.get('title', ''),
                            'role_title_amharic': row.get('title_amharic', ''),
                            'chamber_name_english': row.get('name_english', ''),
                            'area_name': row.get('name', ''),
                            # Membership/Contest
                            'membership_id': row.get('id', ''),
                            'membership_type': row.get('membership_type', ''),
                            'contest_id': row.get('contest_id', ''),
                            'start_date': row.get('start_date', None),
                            'end_date': row.get('end_date', None),
                            'is_partisan': str(row.get('is_partisan', '')).lower() in ['true', 'yes', '1'],
                            'has_end_date': str(row.get('has_end_date', '')).lower() in ['true', 'yes', '1'],
                            # Store everything else as JSON
                            'raw_data': row.to_dict(),
                        }
                    )
                    if created_obj:
                        created += 1
                        
                self.stdout.write(f"  ✅ {sheet_name}: {created} new/updated records")
                
        self.stdout.write("\n🏁 Import complete.")
