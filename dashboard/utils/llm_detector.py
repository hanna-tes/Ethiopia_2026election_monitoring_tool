import logging
import json
import re
from .llm_service import safe_llm_call 

logger = logging.getLogger(__name__)

def detect_hate_speech_llm(text):
    """
    Use LLM to detect hate speech. Falls back gracefully if LLM unavailable.
    """
    if not text or len(text.strip()) < 10:
        return {
            'is_hate_speech': False,
            'confidence': 0.0,
            'categories': [],
            'explanation': 'Text too short for analysis',
            'llm_detected': False
        }
    
    # Check if API key exists
    import os
    if not os.getenv('GROQ_API_KEY'):
        # LLM not available - return gracefully without error
        return {
            'is_hate_speech': False,
            'confidence': 0.0,
            'categories': [],
            'explanation': 'LLM service not configured (lexicon detection only)',
            'llm_detected': False
        }
    
    prompt = f"""You are an expert content moderator analyzing text for hate speech.

Analyze: "{text}"

Respond in JSON:
{{
    "is_hate_speech": true/false,
    "confidence": 0.0-1.0,
    "categories": ["ethnic_hate", "religious_hate", "derogatory"],
    "targeted_groups": ["groups"],
    "severity": "low/medium/high/critical",
    "explanation": "Brief explanation"
}}"""

    try:
        response = safe_llm_call(prompt)
        
        if not response:  # Handle None response
            return {
                'is_hate_speech': False,
                'confidence': 0.0,
                'categories': [],
                'explanation': 'LLM returned empty response',
                'llm_detected': False
            }
        
        import json
        import re
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
        return {
            'is_hate_speech': False,
            'confidence': 0.0,
            'categories': [],
            'explanation': 'Failed to parse LLM response',
            'llm_detected': False
        }
    except Exception as e:
        # Don't log to user, just return gracefully
        return {
            'is_hate_speech': False,
            'confidence': 0.0,
            'categories': [],
            'explanation': 'LLM unavailable',
            'llm_detected': False
        }
