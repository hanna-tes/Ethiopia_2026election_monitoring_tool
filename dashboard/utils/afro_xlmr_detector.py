import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from django.conf import settings 

logger = logging.getLogger(__name__)

# Use Django's BASE_DIR for bulletproof path resolution ---
# settings.BASE_DIR points to /home/ubuntu/Ethiopia_2026election_monitoring_tool/
MODEL_PATH = os.path.join(settings.BASE_DIR, 'dashboard', 'models_cache', 'afro_xlmr')

logger.info(f"🎯 AFRO-XLMR Model Path: {MODEL_PATH}")

# Global cache
_AFRO_XLMR_MODEL = None
_AFRO_XLMR_TOKENIZER = None

def get_afro_xlmr_detector():
    """Loads and returns the AFRO-XLMR model and tokenizer."""
    global _AFRO_XLMR_MODEL, _AFRO_XLMR_TOKENIZER
    
    if _AFRO_XLMR_MODEL is not None:
        return _AFRO_XLMR_MODEL, _AFRO_XLMR_TOKENIZER

    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Model path does not exist: {MODEL_PATH}")
        return None, None

    try:
        logger.info(f"⏳ Loading AFRO-XLMR from: {MODEL_PATH}")
        _AFRO_XLMR_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        _AFRO_XLMR_MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _AFRO_XLMR_MODEL.to(device)
        _AFRO_XLMR_MODEL.eval()
        
        logger.info("✅ AFRO-XLMR model loaded successfully!")
        return _AFRO_XLMR_MODEL, _AFRO_XLMR_TOKENIZER
        
    except Exception as e:
        logger.error(f"❌ Failed to load AFRO-XLMR model: {e}")
        return None, None

class AfroXlmrDetector:
    def __init__(self):
        self.model, self.tokenizer = get_afro_xlmr_detector()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def detect(self, text: str) -> dict:
        if not self.model or not self.tokenizer:
            return {
                'is_hate_speech': False,
                'confidence': 0.0,
                'category': 'error',
                'severity': 'low',
                'language_detected': 'unknown',
                'error': 'Model not loaded'
            }

        try:
            inputs = self.tokenizer(
                text, return_tensors='pt', truncation=True, padding=True, max_length=128
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().cpu().numpy()

            # Map to your 20 classes (from your notebook)
            ID2LABEL = {
                0: 'Ancestry', 1: 'Ethnicity', 2: 'Gender disinformation', 3: 'Homophobic',
                4: 'Misognistic', 5: 'Religion', 6: 'Xenophobia', 7: 'Deragatory',
                8: 'Dehumanization', 9: 'Ethnic slur', 10: 'Slur', 11: 'Stereotype',
                12: 'Call for action', 13: 'Inciteful', 14: 'Violence', 15: 'Class',
                16: 'Extremism', 17: 'Inflammatory', 18: 'Stractural', 19: 'Neutral'
            }
            
            SEVERITY_MAP = {
                'Violence': 'critical', 'Inciteful': 'critical', 'Call for action': 'critical', 
                'Dehumanization': 'critical', 'Extremism': 'high', 'Ethnic slur': 'high', 
                'Slur': 'high', 'Misognistic': 'high', 'Deragatory': 'medium', 
                'Inflammatory': 'high', 'Gender disinformation': 'high', 'Stereotype': 'high', 
                'Homophobic': 'high', 'Ethnicity': 'high', 'Xenophobia': 'high', 'Religion': 'high',
                'Ancestry': 'medium', 'Class': 'low', 'Stractural': 'low', 'Neutral': 'low'
            }

            top_idx = int(probs.argmax())
            confidence = float(probs[top_idx])
            category = ID2LABEL.get(top_idx, 'Neutral')
            
            # Detect language
            import re
            language = 'amharic' if re.search(r'[\u1200-\u137F]', text) else 'english'

            return {
                'is_hate_speech': category != 'Neutral',
                'confidence': round(confidence, 4),
                'category': category,
                'severity': SEVERITY_MAP.get(category, 'low'),
                'language_detected': language,
                'model': 'AFRO-XLMR'
            }
        except Exception as e:
            logger.error(f"AFRO-XLMR detection error: {e}")
            return {
                'is_hate_speech': False,
                'confidence': 0.0,
                'category': 'error',
                'severity': 'low',
                'language_detected': 'unknown',
                'error': str(e)
            }

# Singleton instance
_detector_instance = None
def get_detector():
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = AfroXlmrDetector()
    return _detector_instance
