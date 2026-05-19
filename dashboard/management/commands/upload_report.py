import os
import logging
from django.core.management.base import BaseCommand
from dashboard.models import MonitoringReport

logger = logging.getLogger(__name__)

def extract_text(file_path):
    """Extract text from PDF, DOCX, or TXT files"""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.txt':
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
            
    elif ext == '.pdf':
        try:
            import pdfplumber
            text = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text: text.append(page_text)
            return '\n'.join(text)
        except ImportError:
            raise ImportError("Install pdfplumber: pip install pdfplumber")
            
    elif ext in ['.docx', '.doc']:
        try:
            from docx import Document
            doc = Document(file_path)
            return '\n'.join([p.text for p in doc.paragraphs])
        except ImportError:
            raise ImportError("Install python-docx: pip install python-docx")
            
    raise ValueError(f"Unsupported file type: {ext}")

class Command(BaseCommand):
    help = 'Upload election monitoring reports (text extraction only)'

    def add_arguments(self, parser):
        parser.add_argument('file_path', type=str)
        parser.add_argument('--title', type=str)
        parser.add_argument('--analyst', type=str, default='Internal Analyst')
        parser.add_argument('--type', type=str, default='Investigative', choices=['Investigative', 'Monthly', 'Special'])

    def handle(self, *args, **options):
        file_path = options['file_path']
        if not os.path.exists(file_path):
            self.stderr.write(f"❌ File not found: {file_path}")
            return

        title = options['title'] or os.path.splitext(os.path.basename(file_path))[0].replace('_', ' ').title()
        
        self.stdout.write(f"📄 Processing: {title}")
        self.stdout.write("🔍 Extracting text...")
        
        try:
            raw_text = extract_text(file_path)
            if not raw_text or len(raw_text.strip()) < 50:
                self.stderr.write("⚠️ Could not extract meaningful text.")
                return
            
            # Save directly to DB - NO LLM CALL
            report = MonitoringReport.objects.create(
                title=title,
                source_analyst=options['analyst'],
                file_path=file_path,
                report_type=options['type'],
                extracted_text=raw_text[:10000],  # Store first 10k chars for preview
                # Leave insight fields empty for manual entry via admin/UI
                summary='',
                key_findings=[],
                mentioned_entities=[],
                risk_level='medium',  # Default, can be updated later
                weaponised_narratives=[],
                actor_spotlight=[],
                ttp_infrastructure=[],
                is_processed=True
            )
            
            self.stdout.write(f"\n✅ SUCCESS! Report saved to database.")
            self.stdout.write(f"📝 Extracted {len(raw_text):,} characters")
            self.stdout.write(f"✏️  Edit insights via Django Admin: http://localhost:8505/admin/dashboard/monitoringreport/{report.id}/change/")
            self.stdout.write(f"🔗 View in UI: http://localhost:8505/narratives/")
            
        except Exception as e:
            self.stderr.write(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
