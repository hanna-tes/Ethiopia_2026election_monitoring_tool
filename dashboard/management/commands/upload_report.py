import os
import re
import json
import logging
from django.core.management.base import BaseCommand
from dashboard.models import MonitoringReport
from dashboard.utils.llm_service import safe_llm_call

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

def llm_extract_insights(text):
    """
    Use LLM to extract structured insights.
    FIX: Risk level is now determined by content severity, not default.
    """
    # Smart truncation to handle token limits while keeping context
    context = text[:2000] + "\n...\n" + text[-2000:] if len(text) > 4000 else text
    
    prompt = f"""You are a senior election monitoring analyst. Analyze the report and extract insights into STRICT JSON. Return ONLY valid JSON. No markdown, no explanations.

REQUIRED JSON STRUCTURE:
{{
  "summary": "2-3 sentence executive summary",
  "key_findings": ["finding 1", "finding 2"],
  "mentioned_entities": ["entity 1", "entity 2"],
  "risk_level": "low|medium|high|critical", 
  "weaponised_narratives": ["Narrative/theme with brief example/context"],
  "actor_spotlight": ["Account/Org/Person + brief role/amplification method"],
  "ttp_infrastructure": ["Tactic/Infrastructure + how it enables manipulation"]
}}

RISK LEVEL GUIDELINES:
- "critical": Incitement to violence, imminent harm, widespread hate speech, state-sponsored disinfo
- "high": Coordinated manipulation, ethnic polarization, targeted harassment, false election info
- "medium": Bias, misleading framing, unverified claims, moderate polarization
- "low": Factual reporting, neutral analysis, constructive criticism

REPORT TEXT:
{context}

JSON:"""
    
    try:
        response = safe_llm_call(prompt, temperature=0.1, max_tokens=600)
        # Clean response to ensure valid JSON (remove markdown code blocks)
        cleaned = re.sub(r'^```json\s*|\s*```$', '', response.strip(), flags=re.MULTILINE)
        return json.loads(cleaned)
    except Exception as e:
        logger.error(f"LLM extraction failed: {e}")
        # Fallback: Return empty lists but force risk to 'medium' ONLY as a safe fallback for DB integrity
        return {
            "summary": "AI extraction failed. Manual review recommended.",
            "key_findings": ["Review original document for details"],
            "mentioned_entities": [],
            "risk_level": "medium", 
            "weaponised_narratives": [],
            "actor_spotlight": [],
            "ttp_infrastructure": []
        }

class Command(BaseCommand):
    help = 'Upload & AI-analyze election monitoring reports'

    def add_arguments(self, parser):
        parser.add_argument('file_path', type=str, help='Path to PDF/DOCX/TXT file')
        parser.add_argument('--title', type=str, help='Report title')
        parser.add_argument('--analyst', type=str, default='Internal Analyst', help='Analyst/Source name')
        parser.add_argument('--type', type=str, default='Investigative', choices=['Investigative', 'Monthly', 'Special'], help='Report type')

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
                self.stderr.write("⚠️ Could not extract meaningful text from file.")
                return
                
            self.stdout.write("🤖 Running AI insight extraction...")
            insights = llm_extract_insights(raw_text)
            
            # Save to DB
            report = MonitoringReport.objects.create(
                title=title,
                source_analyst=options['analyst'],
                file_path=file_path,
                report_type=options['type'],
                summary=insights.get('summary', ''),
                key_findings=insights.get('key_findings', []),
                mentioned_entities=insights.get('mentioned_entities', []),
                risk_level=insights.get('risk_level', 'medium'), # Uses LLM value, fallback only on error
                weaponised_narratives=insights.get('weaponised_narratives', []),
                actor_spotlight=insights.get('actor_spotlight', []),
                ttp_infrastructure=insights.get('ttp_infrastructure', []),
                is_processed=True
            )
            
            self.stdout.write(f"\n✅ SUCCESS! Report saved to database.")
            self.stdout.write(f"📊 Risk Level: {report.risk_level.upper()} (AI-Assessed)")
            self.stdout.write(f"🔑 Key Findings: {len(report.key_findings)}")
            self.stdout.write(f"👥 Entities Mentioned: {len(report.mentioned_entities)}")
            self.stdout.write(f"🔗 View in UI: http://localhost:8505/narratives/")
            
        except Exception as e:
            self.stderr.write(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
