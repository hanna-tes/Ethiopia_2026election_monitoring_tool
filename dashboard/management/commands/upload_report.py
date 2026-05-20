import os
import re
import json
import logging
from django.core.management.base import BaseCommand
from dashboard.models import MonitoringReport
from dashboard.utils.llm_service import safe_llm_call

logger = logging.getLogger(__name__)

def extract_text(file_path):
    """Extract text, preserving line breaks for section detection"""
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
                    if page_text: 
                        text.append(page_text)
            # Join with newlines to keep structure
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

def parse_sections_precisely(full_text):
    """
    Scans the document for specific headers and extracts ONLY the actual content items
    (bullet points, numbered lists, substantive claims) - skipping intro/descriptive text.
    """
    # Define headers we are looking for (Case Insensitive)
    headers_map = {
        'executive_summary': r'(?i)executive\s+summary',
        'key_findings': r'(?i)key\s+findings|findings|main\s+conclusions',
        'weaponised_narratives': r'(?i)weaponised\s+narratives|weaponized\s+narratives|harmful\s+narratives|polarising\s+narratives',
        'actor_spotlight': r'(?i)actor\s+spotlight|key\s+actors|mentioned\s+actors|people\s+and\s+organisations',
        'ttp_infrastructure': r'(?i)tactics,\s*techniques,\s*and\s*procedures|ttp|infrastructure|information\s+manipulation'
    }
    
    extracted = {
        'executive_summary': [],
        'key_findings': [],
        'weaponised_narratives': [],
        'actor_spotlight': [],
        'ttp_infrastructure': [],
    }
    
    lines = full_text.split('\n')
    current_section = None
    skip_intro = True  # Flag to skip introductory text after header
    
    # Helper to check if line is a real content item (not intro/description)
    def is_content_item(line):
        clean = line.strip()
        if len(clean) < 20: return False  # Too short
        # Skip lines that are clearly descriptive/intro
        intro_phrases = [
            'this section examines', 'the section lists', 'cite this research', 
            'get more information', 'share your leads', 'this report is part',
            'among these', 'collectively they', 'operations or other forms'
        ]
        if any(phrase in clean.lower() for phrase in intro_phrases):
            return False
        # Skip citation lines
        if clean.lower().startswith('cite this') or 'www.disinfo' in clean.lower():
            return False
        # Keep bullet points, numbered items, or substantive claims
        if re.match(r'^[-•*]\s+', clean) or re.match(r'^\d+[\.\)]\s+', clean):
            return True
        # Keep lines that look like actual findings/narratives (contain specific entities/actions)
        if any(w in clean.lower() for w in ['hate speech', 'targeting', 'community', 'ethnic', 'polarisation', 'disinformation', 'manipulation', 'amplifying', 'coordinated']):
            return True
        return False
    
    # Helper to clean bullet points
    def clean_text(text):
        return re.sub(r'^[-*•\d.\s]+', '', text).strip()

    # 1. Extract URLs from the whole document
    all_urls = re.findall(r'https?://\S+', full_text)
    all_urls = list(set([re.sub(r'[.,;)]$', '', u) for u in all_urls if len(u) > 10]))[:10]

    # 2. Scan lines for sections
    for i, line in enumerate(lines):
        clean_line = line.strip()
        if not clean_line: 
            continue
        
        # Check if line is a header
        matched_header = None
        for key, pattern in headers_map.items():
            if re.search(pattern, clean_line):
                matched_header = key
                break
        
        if matched_header:
            current_section = matched_header
            skip_intro = True  # Reset skip flag for new section
            continue
            
        # If we are inside a section, decide whether to capture
        if current_section:
            # Stop if we hit a new major section header
            if any(re.search(pat, clean_line) for pat in headers_map.values()):
                current_section = None
                continue
            
            # Skip intro text for first few lines after header
            if skip_intro:
                if not is_content_item(clean_line):
                    continue
                else:
                    skip_intro = False  # Stop skipping once we find real content
            
            # Capture valid content items
            if is_content_item(clean_line):
                cleaned = clean_text(clean_line)
                if len(cleaned) > 30:  # Only keep substantive items
                    extracted[current_section].append(cleaned)
                    
    # Process lists into readable strings
    processed = {
        'summary': " ".join([l for l in extracted['executive_summary'][:3] if len(l) > 50]),
        'findings': [l for l in extracted['key_findings'][:5] if len(l) > 30],
        'narratives': [l for l in extracted['weaponised_narratives'][:5] if len(l) > 30],
        'actors': [l for l in extracted['actor_spotlight'][:5] if len(l) > 30],
        'ttps': [l for l in extracted['ttp_infrastructure'][:5] if len(l) > 30],
        'urls': all_urls
    }
    
    return processed
def get_risk_level(text, parsed_data):
    """Context-aware risk assessment"""
    prompt = f"""Analyze this report summary and risk level:
{parsed_data['summary'][:1000]}

Return ONLY the risk level string: "low", "medium", "high", or "critical".
JSON:"""
    
    try:
        # ✅ FIXED: Removed temperature/max_tokens
        response = safe_llm_call(prompt)
        match = re.search(r'low|medium|high|critical', response.lower())
        if match:
            return match.group()
    except Exception as e:
        logger.error(f"LLM Risk failed: {e}")

    # Fallback
    t = text.lower()
    if any(w in t for w in ['violence', 'kill', 'incitement', 'civil war', 'genocide']): return 'critical'
    if any(w in t for w in ['hate speech', 'disinformation', 'manipulation', 'polarization']): return 'high'
    if any(w in t for w in ['bias', 'protest', 'distrust']): return 'medium'
    return 'low'

class Command(BaseCommand):
    help = 'Upload & parse election monitoring reports'

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
            if not raw_text or len(raw_text.strip()) < 100:
                self.stderr.write("⚠️ Could not extract meaningful text.")
                return
            
            self.stdout.write(f"✅ Extracted {len(raw_text):,} characters")
            self.stdout.write("📖 Scanning for specific sections (Executive, Narratives, TTPs)...")
            
            # Parse sections precisely
            data = parse_sections_precisely(raw_text)
            
            # Get Risk Level via LLM (Lightweight call)
            self.stdout.write("🤖 Assessing risk level...")
            risk = get_risk_level(raw_text, data) or 'medium'
            
            # Save to DB
            report = MonitoringReport.objects.create(
                title=title,
                source_analyst=options['analyst'],
                file_path=file_path,
                report_type=options['type'],
                extracted_text=raw_text[:10000], # Store preview
                
                # ✅ Use parsed sections
                summary=data['summary'] or "See full document.",
                key_findings=data['findings'],
                weaponised_narratives=data['narratives'],
                actor_spotlight=data['actors'],
                ttp_infrastructure=data['ttps'],
                sample_urls=data['urls'], # ✅ New field
                risk_level=risk,
                is_processed=True
            )
            
            self.stdout.write(f"\n✅ SUCCESS! Report saved.")
            self.stdout.write(f"📊 Risk: {report.risk_level.upper()}")
            self.stdout.write(f"🔗 Found {len(report.sample_urls)} sample links")
            self.stdout.write(f"🔗 View: http://localhost:8505/investigative-reports/")
            
        except Exception as e:
            self.stderr.write(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
