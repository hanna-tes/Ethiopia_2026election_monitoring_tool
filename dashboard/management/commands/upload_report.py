import os
import re
import json
import logging
from django.core.management.base import BaseCommand
from dashboard.models import MonitoringReport
from dashboard.utils.llm_service import safe_llm_call

logger = logging.getLogger(__name__)

def extract_text(file_path):
    """Extract text from PDF, DOCX, or TXT"""
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

def extract_insights_with_llm(text):
    """Use LLM to extract structured insights & assess risk contextually"""
    context = text[:8000] if len(text) > 8000 else text
    
    prompt = f"""You are an election monitoring analyst. Analyze the document and return ONLY valid JSON.

**CONTEXTUAL RISK ASSESSMENT RULES:**
- "critical": Incitement to violence, coordinated hate speech, threats to electoral integrity, imminent harm
- "high": Widespread disinformation, ethnic/political polarization, coordinated manipulation, severe institutional distrust
- "medium": Bias, unverified claims, moderate polarization, standard political criticism
- "low": Factual reporting, neutral analysis, constructive dialogue, procedural updates

**Extract these sections:**
1. "summary": 2-3 sentence executive summary
2. "key_findings": List of 3-5 main findings (directly from document)
3. "weaponised_narratives": List of harmful/polarizing narratives or disinformation themes
4. "actor_spotlight": List of key actors/organizations/groups mentioned
5. "ttp_infrastructure": List of tactics/platforms/infrastructure for information operations
6. "mentioned_entities": List of specific entities (countries, parties, institutions, ethnic groups, dates)
7. "risk_level": "low", "medium", "high", or "critical" (MUST be determined by document content severity)

**DOCUMENT TEXT:**
{context}

Return ONLY valid JSON matching this exact structure:
{{
  "summary": "...",
  "key_findings": ["..."],
  "weaponised_narratives": ["..."],
  "actor_spotlight": ["..."],
  "ttp_infrastructure": ["..."],
  "mentioned_entities": ["..."],
  "risk_level": "..."
}}
JSON:"""
    
    try:
        response = safe_llm_call(prompt, temperature=0.1, max_tokens=800)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            # Validate risk level
            valid_risks = ['low', 'medium', 'high', 'critical']
            if data.get('risk_level', '').lower() not in valid_risks:
                data['risk_level'] = 'low'
            return data
        raise ValueError("No valid JSON found")
    except Exception as e:
        logger.error(f"LLM extraction failed: {e}")
        
        # 🧠 CONTEXT-AWARE FALLBACK: Scan document for risk indicators
        text_lower = text.lower()
        high_risk_terms = ['violence', 'kill', 'attack', 'hate speech', 'incitement', 'rigged', 'stolen election', 'civil war', 'genocide', 'ethnic cleansing', 'armed conflict']
        med_risk_terms = ['polarization', 'disinformation', 'fake news', 'manipulation', 'distrust', 'protest', 'conflict', 'crisis', 'instability']
        
        if any(term in text_lower for term in high_risk_terms):
            fallback_risk = 'high'
        elif any(term in text_lower for term in med_risk_terms):
            fallback_risk = 'medium'
        else:
            fallback_risk = 'low'
            
        return {
            "summary": text[:500] + "..." if len(text) > 500 else text,
            "key_findings": ["AI parsing failed. See full extracted text."],
            "weaponised_narratives": [],
            "actor_spotlight": [],
            "ttp_infrastructure": [],
            "mentioned_entities": [],
            "risk_level": fallback_risk
        }

class Command(BaseCommand):
    help = 'Upload & AI-parse election monitoring reports'

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
        self.stdout.write("🔍 Extracting text from document...")
        
        try:
            raw_text = extract_text(file_path)
            if not raw_text or len(raw_text.strip()) < 100:
                self.stderr.write("⚠️  Could not extract meaningful text.")
                return
            
            self.stdout.write(f"✅ Extracted {len(raw_text):,} characters")
            self.stdout.write("🤖 AI parsing insights & assessing risk contextually...")
            
            insights = extract_insights_with_llm(raw_text)
            
            report = MonitoringReport.objects.create(
                title=title,
                source_analyst=options['analyst'],
                file_path=file_path,
                report_type=options['type'],
                extracted_text=raw_text[:10000],
                summary=insights.get('summary', ''),
                key_findings=insights.get('key_findings', []),
                weaponised_narratives=insights.get('weaponised_narratives', []),
                actor_spotlight=insights.get('actor_spotlight', []),
                ttp_infrastructure=insights.get('ttp_infrastructure', []),
                mentioned_entities=insights.get('mentioned_entities', []),
                risk_level=insights.get('risk_level', 'low'),  # ✅ No more hardcoded "medium"
                is_processed=True
            )
            
            self.stdout.write(f"\n✅ SUCCESS! Report saved.")
            self.stdout.write(f"📊 Risk Level: {report.risk_level.upper()} (Context-Assessed)")
            self.stdout.write(f"🔑 Key Findings: {len(report.key_findings)}")
            self.stdout.write(f"🎯 Narratives: {len(report.weaponised_narratives)}")
            self.stdout.write(f"👤 Actors: {len(report.actor_spotlight)}")
            self.stdout.write(f"⚙️  TTPs: {len(report.ttp_infrastructure)}")
            self.stdout.write(f"🔗 View: http://localhost:8505/investigative-reports/")
            
        except Exception as e:
            self.stderr.write(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
