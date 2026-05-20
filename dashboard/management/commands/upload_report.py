import os
import re
import json
import logging
from django.core.management.base import BaseCommand
from dashboard.models import MonitoringReport
from dashboard.utils.llm_service import safe_llm_call

logger = logging.getLogger(__name__)

def extract_text(file_path):
    """Extract text, preserving structure for section detection"""
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

def extract_section_content(full_text, header_pattern):
    """Extract ALL content under a specific header until the next major header"""
    lines = full_text.split('\n')
    content = []
    in_section = False
    
    all_headers = [
        r'(?i)executive\s+summary',
        r'(?i)key\s+findings|findings|main\s+conclusions',
        r'(?i)weaponised\s+narratives|weaponized\s+narratives|harmful\s+narratives',
        r'(?i)actor\s+spotlight|key\s+actors|mentioned\s+actors',
        r'(?i)tactics,\s*techniques,\s*and\s*procedures|ttp|infrastructure',
        r'(?i)\d+\.\s+[A-Z]',
    ]
    
    for line in lines:
        clean = line.strip()
        if not clean: continue
        
        if re.search(header_pattern, clean) and not in_section:
            in_section = True
            continue
            
        if in_section:
            if any(re.search(h, clean) for h in all_headers if h != header_pattern):
                break
            content.append(clean)
    
    return '\n'.join(content).strip()

def llm_summarize_section(section_name, full_content):
    """Use LLM to create a concise, insightful summary (3-5 bullets max)"""
    context = full_content[:4000] if len(full_content) > 4000 else full_content
    
    prompt = f"""You are an election monitoring analyst. Summarize this {section_name} section concisely.

**Instructions:**
1. Extract EXACTLY 3-5 KEY bullet points that capture the MOST important insights
2. Preserve SPECIFIC examples, entities, dates, and claims mentioned
3. Keep the tone factual and analytical
4. Each bullet should be 1-2 sentences MAX
5. Focus on actionable intelligence for election monitoring

**Section Content:**
{context}

**Return ONLY valid JSON in this exact format:**
{{
  "summary_bullets": [
    "Key insight 1 with specific example",
    "Key insight 2 with entity/date",
    "Key insight 3 with claim/evidence"
  ]
}}
JSON:"""
    
    try:
        response = safe_llm_call(prompt)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            bullets = data.get('summary_bullets', [])
            return bullets[:5]  # Always limit to 5
    except Exception as e:
        logger.error(f"LLM summarization failed for {section_name}: {e}")
    
    # Smart fallback: extract only substantive bullet points, limit to 5
    lines = [l.strip() for l in full_content.split('\n') 
             if l.strip().startswith(('•', '-', '*', '▸', '→', '▪')) and len(l.strip()) > 30]
    
    # Filter out intro/descriptive lines
    skip_phrases = ['this section', 'cite this', 'get more', 'share your', 'the report', 
                   'among these', 'collectively', 'operations or other']
    filtered = [l for l in lines if not any(skip in l.lower() for skip in skip_phrases)]
    
    # Clean and return max 5 bullets
    cleaned = [re.sub(r'^[•\-\*\▸→▪\s]+', '', l).strip() for l in filtered[:5]]
    return cleaned if cleaned else [full_content[:200] + "..."]

def parse_and_summarize_sections(full_text):
    """Extract full sections and use LLM to create concise summaries"""
    
    sections_config = {
        'executive_summary': r'(?i)executive\s+summary',
        'weaponised_narratives': r'(?i)weaponised\s+narratives|weaponized\s+narratives|harmful\s+narratives',
        'actor_spotlight': r'(?i)actor\s+spotlight|key\s+actors|mentioned\s+actors',
        'ttp_infrastructure': r'(?i)tactics,\s*techniques,\s*and\s*procedures|ttp|infrastructure',
        'key_findings': r'(?i)key\s+findings|findings|main\s+conclusions',
    }
    
    results = {}
    
    # Extract URLs from entire document
    all_urls = re.findall(r'https?://\S+', full_text)
    results['sample_urls'] = list(set([re.sub(r'[.,;)]$', '', u) for u in all_urls if len(u) > 10]))[:10]
    
    # Process each section
    for section_key, pattern in sections_config.items():
        content = extract_section_content(full_text, pattern)
        
        if content and len(content) > 100:
            logger.info(f"🤖 Summarizing {section_key} ({len(content)} chars)...")
            bullets = llm_summarize_section(section_key.replace('_', ' ').title(), content)
            
            results[section_key] = {
                'full_text': content,  # ✅ NO CHARACTER LIMIT
                'summary_bullets': bullets[:5],  # Max 5 concise bullets
                'key_entities': []  # Can be extended if needed
            }
        else:
            results[section_key] = {
                'full_text': content if content else '',
                'summary_bullets': [],
                'key_entities': []
            }
    
    return results

def assess_risk_level(full_text, parsed_sections):
    """Context-aware risk assessment based on document content"""
    prompt = f"""Analyze this election monitoring report and assign a risk level.

**RISK GUIDELINES:**
- critical: Incitement to violence, coordinated hate speech, threats to electoral integrity
- high: Widespread disinformation, ethnic/political polarization, coordinated manipulation
- medium: Bias, unverified claims, moderate polarization, standard political criticism
- low: Factual reporting, neutral analysis, procedural updates

**Key sections:**
Executive Summary: {parsed_sections.get('executive_summary', {}).get('full_text', '')[:500]}
Weaponised Narratives: {parsed_sections.get('weaponised_narratives', {}).get('full_text', '')[:500]}

Return ONLY the risk level string: "low", "medium", "high", or "critical".
JSON:"""
    
    try:
        response = safe_llm_call(prompt)
        match = re.search(r'low|medium|high|critical', response.lower())
        if match:
            return match.group()
    except Exception as e:
        logger.error(f"LLM risk assessment failed: {e}")
    
    # Fallback keyword scan
    t = full_text.lower()
    if any(w in t for w in ['violence', 'kill', 'incitement', 'civil war', 'genocide', 'hate speech']):
        return 'critical'
    if any(w in t for w in ['polarization', 'disinformation', 'manipulation', 'distrust', 'coordinated']):
        return 'high'
    if any(w in t for w in ['bias', 'protest', 'unverified', 'criticism']):
        return 'medium'
    return 'low'

class Command(BaseCommand):
    help = 'Upload & AI-summarize election monitoring reports'

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
                self.stderr.write("⚠️ Could not extract meaningful text.")
                return
            
            self.stdout.write(f"✅ Extracted {len(raw_text):,} characters")
            
            # Parse sections and generate LLM summaries
            self.stdout.write("📖 Extracting sections & generating AI summaries...")
            parsed = parse_and_summarize_sections(raw_text)
            
            # Assess risk level
            self.stdout.write("🤖 Assessing contextual risk level...")
            risk_level = assess_risk_level(raw_text, parsed)
            
            # Helper to join bullets with ||| separator for easy template parsing
            def bullets_to_text(bullets):
                return '|||'.join(bullets) if bullets else ''
            
            # Save to database - FULL CONTENT + CONCISE SUMMARIES
            report = MonitoringReport.objects.create(
                title=title,
                source_analyst=options['analyst'],
                file_path=file_path,
                report_type=options['type'],
                
                # ✅ Store FULL extracted text (NO LIMIT)
                extracted_text=raw_text,
                
                # Executive summary (concise)
                summary=" ".join(parsed.get('executive_summary', {}).get('summary_bullets', [])[:2]),
                
                # Key findings (JSONField - concise bullets)
                key_findings=parsed.get('key_findings', {}).get('summary_bullets', []),
                
                # ✅ Store concise summaries as |||-separated text for template parsing
                weaponised_narratives=bullets_to_text(parsed.get('weaponised_narratives', {}).get('summary_bullets', [])),
                actor_spotlight=bullets_to_text(parsed.get('actor_spotlight', {}).get('summary_bullets', [])),
                ttp_infrastructure=bullets_to_text(parsed.get('ttp_infrastructure', {}).get('summary_bullets', [])),
                
                # ✅ Store FULL section content (NO LIMITS)
                weaponised_narratives_full=parsed.get('weaponised_narratives', {}).get('full_text', ''),
                actor_spotlight_full=parsed.get('actor_spotlight', {}).get('full_text', ''),
                ttp_infrastructure_full=parsed.get('ttp_infrastructure', {}).get('full_text', ''),
                key_findings_full=parsed.get('key_findings', {}).get('full_text', ''),
                
                # Entities
                mentioned_entities=parsed.get('weaponised_narratives', {}).get('key_entities', []) + 
                                  parsed.get('actor_spotlight', {}).get('key_entities', []),
                
                sample_urls=parsed.get('sample_urls', []),
                risk_level=risk_level,
                is_processed=True
            )
            
            self.stdout.write(f"\n✅ SUCCESS! Report saved with AI-generated insights.")
            self.stdout.write(f"📊 Risk Level: {report.risk_level.upper()}")
            self.stdout.write(f"🔑 Key Findings: {len(report.key_findings)} bullets")
            self.stdout.write(f"🎯 Narratives: {len(report.weaponised_narratives.split('|||')) if report.weaponised_narratives else 0} bullets")
            self.stdout.write(f"👤 Actors: {len(report.actor_spotlight.split('|||')) if report.actor_spotlight else 0} bullets")
            self.stdout.write(f"⚙️ TTPs: {len(report.ttp_infrastructure.split('|||')) if report.ttp_infrastructure else 0} bullets")
            self.stdout.write(f"🔗 Sample Links: {len(report.sample_urls)}")
            self.stdout.write(f"🔗 View: http://localhost:8505/investigative-reports/")
            
        except Exception as e:
            self.stderr.write(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
