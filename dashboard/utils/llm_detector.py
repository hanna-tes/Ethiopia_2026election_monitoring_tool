import logging
import json
import re
from .llm_service import safe_llm_call 

logger = logging.getLogger(__name__)

def detect_hate_speech_llm(text):
    """
    Use LLM to detect hate speech, ethnic targeting, and derogatory content.
    Returns dict with detection results.
    """
    if not text or len(text.strip()) < 10:
        return {
            'is_hate_speech': False,
            'confidence': 0.0,
            'categories': [],
            'explanation': 'Text too short for analysis'
        }
    
    prompt = f"""You are an expert content moderator analyzing text for hate speech and harmful content.

Analyze the following text for:
1. **Hate Speech**: Content that attacks, demeans, or dehumanizes groups based on ethnicity, religion, nationality, etc.
2. **Derogatory Language**: Insults, slurs, or dehumanizing terms
3. **Ethnic/Religious Targeting**: Content that targets specific ethnic or religious groups negatively
4. **Incitement**: Content that encourages harm or discrimination

**Text to analyze:**
"{text}"

**Respond in JSON format ONLY:**
{{
    "is_hate_speech": true/false,
    "confidence": 0.0-1.0,
    "categories": ["ethnic_hate", "religious_hate", "derogatory", "incitement"],
    "targeted_groups": ["list targeted groups"],
    "severity": "low/medium/high/critical",
    "explanation": "Brief explanation"
}}

Be strict but fair. Consider context and cultural nuances. Output ONLY valid JSON."""

    try:
        response = safe_llm_call(prompt, temperature=0.1)
        
        # Extract JSON from response (handles LLMs that add extra text)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                'is_hate_speech': result.get('is_hate_speech', False),
                'confidence': result.get('confidence', 0.0),
                'categories': result.get('categories', []),
                'targeted_groups': result.get('targeted_groups', []),
                'severity': result.get('severity', 'low'),
                'explanation': result.get('explanation', ''),
                'llm_detected': True
            }
        else:
            return {
                'is_hate_speech': False,
                'confidence': 0.0,
                'categories': [],
                'explanation': 'Failed to parse LLM response',
                'llm_detected': False
            }
            
    except Exception as e:
        logger.error(f"LLM hate speech detection failed: {e}")
        return {
            'is_hate_speech': False,
            'confidence': 0.0,
            'categories': [],
            'explanation': f'LLM error: {str(e)}',
            'llm_detected': False
        }
