import os
import re
import json
import logging
from django.core.management.base import BaseCommand
from dashboard.models import MonitoringReport
from dashboard.utils.llm_service import safe_llm_call

logger = logging.getLogger(__name__)

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.txt':
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    elif ext == '.pdf':
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                return '\n'.join(page.extract_text() or '' for page in pdf.pages)
        except ImportError:
            raise ImportError("Install pdfplumber: pip install pdfplumber")
    elif ext in ['.docx', '.doc']:
        try:
            from docx import Document
            doc = Document(file_path)
            return '\n'.join(p.text for p in doc.paragraphs)
        except ImportError:
            raise ImportError("Install python-docx: pip install python-docx")
    raise ValueError(f"Unsupported file type: {ext}")

def parse_document_sections(text):
    """Directly extract sections by header names instead of relying on LLM"""
    sections = {
        'summary': r'(?i)(executive\s+summary|executive\s+summary\s*:)',
        'key_findings': r'(?i)(key\s+findings|findings|main\s+conclusions)',
        'weaponised_narratives': r'(?i)(weaponised\s+narratives|weaponized\s+narratives|harmful\s+narratives|polarising\s+narratives)',
        'actor_spotlight': r'(?i)(actor\s+spotlight|key\s+actors|mentioned\s+actors|organizations\s+and\s+actors)',
        'ttp_infrastructure': r'(?i)(notable\s+tactics|tactics,\s*techniques,\s*and\s*procedures|TTP|infrastructure|information\s+manipulation)',
        'mentioned_entities': r'(?i)(mentioned\s+entities|entities|countries|organizations|parties)'
    }
    
    extracted = {k: [] for k in sections}
    extracted['summary'] = ""
    
    lines = text.split('\n')
    current_section = None
    buffer = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped: continue
        
        # Check if this line is a section header
        matched = None
        for key, pattern in sections.items():
            if re.search(pattern, stripped):
                matched = key
                break
                
        if matched:
            # Save previous section
            if current_section and buffer:
                content = '\n'.join(buffer).strip()
                if current_section in ['key_findings', 'weaponised_narratives', 'actor_spotlight', 'ttp_infrastructure', 'mentioned_entities']:
                    # Split by bullets, numbers, or newlines
                    items = [item.strip('- •*1234567890.').strip() for item in re.split(r'\n|\n\s*[-•*]', content) if item.strip() and len(item.strip()) > 10]
                    extracted[current_section] = items[:8]
                else:
                    extracted[current_section] = content[:800]
            current_section = matched
            buffer = []
        else:
            buffer.append(stripped)
            
    # Save last section
    if current_section and buffer:
        content = '\n'.join(buffer).strip()
        if current_section in ['key_findings', 'weaponised_narratives', 'actor_spotlight', 'ttp_infrastructure', 'mentioned_entities']:
            extracted[current_section] = [item.strip('- •*1234567890.').strip() for item in re.split(r'\n|\n\s*[-•*]', content) if item.strip() and len(item.strip()) > 10][:8]
        else:
            extracted[current_section] = content[:800]
            
    return extracted

def get_llm_summary_and_risk(text):
    """Use LLM only for concise summary & contextual risk level"""
    prompt = f"""Analyze this election monitoring report. Return ONLY valid JSON:
{{
  "summary": "2-3 sentence executive summary",
  "risk_level": "low|medium|high|critical"
}}

RISK GUIDELINES:
- critical: Violence incitement, electoral sabotage, coordinated hate speech
- high: Widespread disinformation, ethnic/political polarization, severe institutional distrust
- medium: Bias, unverified claims, moderate polarization, standard political criticism
- low: Factual reporting, neutral analysis, procedural updates

REPORT:
{text[:6000]}

JSON:"""
    
    try:
        # ✅ FIXED: Removed temperature/max_tokens that caused the error
        response = safe_llm_call(prompt)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        logger.warning(f"LLM summary failed: {e}")
        
    # Fallback risk assessment based on keywords
    t = text.lower()
    if any(w in t for w in ['violence', 'kill', 'incitement', 'hate speech', 'civil war', 'rigged election']):
        risk = 'high'
    elif any(w in t for w in ['polarization', 'disinformation', 'manipulation', 'distrust', 'protest']):
        risk = 'medium'
    else:
        risk = 'low'
        
    return {"summary": text[:400] + "...", "risk_level": risk}

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
            
            # 1. Directly parse existing sections
            self.stdout.write("📖 Parsing document sections...")
            parsed = parse_document_sections(raw_text)
            
            # 2. Use LLM only for summary & risk
            self.stdout.write("🤖 Generating concise summary & risk level...")
            llm_data = get_llm_summary_and_risk(raw_text)
            
            # 3. Save to DB
            report = MonitoringReport.objects.create(
                title=title,
                source_analyst=options['analyst'],
                file_path=file_path,
                report_type=options['type'],
                extracted_text=raw_text[:10000],
                summary=llm_data.get('summary', parsed.get('summary', '')),
                key_findings=parsed['key_findings'] or ["See full document for findings"],
                weaponised_narratives=parsed['weaponised_narratives'],
                actor_spotlight=parsed['actor_spotlight'],
                ttp_infrastructure=parsed['ttp_infrastructure'],
                mentioned_entities=parsed['mentioned_entities'],
                risk_level=llm_data.get('risk_level', 'low'),
                is_processed=True
            )
            
            self.stdout.write(f"\n✅ SUCCESS! Report saved.")
            self.stdout.write(f"📊 Risk: {report.risk_level.upper()}")
            self.stdout.write(f"🔑 Findings: {len(report.key_findings)}")
            self.stdout.write(f"🎯 Narratives: {len(report.weaponised_narratives)}")
            self.stdout.write(f"👤 Actors: {len(report.actor_spotlight)}")
            self.stdout.write(f"⚙️ TTPs: {len(report.ttp_infrastructure)}")
            self.stdout.write(f"🔗 View: http://localhost:8505/investigative-reports/")
            
        except Exception as e:
            self.stderr.write(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
