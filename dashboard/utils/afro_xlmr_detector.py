"""
AFRO-XLMR Hate Speech Detector
Specialized for Ethiopian languages (Amharic, Oromo, Tigrinya, etc.)
"""
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

logger = logging.getLogger(__name__)

class AfroXLMRDetector:
    """AFRO-XLMR based hate speech detector for Ethiopian languages"""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self):
        """Load the AFRO-XLMR model and tokenizer"""
        try:
            if self.model_path is None:
                # Default to HuggingFace model
                self.model_path = "mesolitica/afro-xlmr-base-hate-speech"  # Replace with actual model
            
            logger.info(f"Loading AFRO-XLMR model from: {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True if self.model_path else False
            )
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                local_files_only=True if self.model_path else False
            )
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✅ AFRO-XLMR model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load AFRO-XLMR model: {e}")
            return False
    
    def detect(self, text: str) -> dict:
        """
        Detect hate speech in text using AFRO-XLMR
        
        Returns:
            dict: {
                'is_hate_speech': bool,
                'confidence': float,
                'category': str,
                'severity': str,
                'language_detected': str,
                'all_predictions': dict
            }
        """
        if not text or len(text.strip()) < 5:
            return {
                'is_hate_speech': False,
                'confidence': 0.0,
                'category': 'neutral',
                'severity': 'low',
                'language_detected': 'unknown',
                'error': 'Text too short'
            }
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                confidence, predicted_class = torch.max(probabilities, dim=-1)
            
            # Map predictions to categories (adjust based on your model's labels)
            categories = {
                0: 'neutral',
                1: 'hate_speech',
                2: 'offensive',
                3: 'threat'
            }
            
            category = categories.get(predicted_class.item(), 'neutral')
            conf_score = confidence.item()
            
            # Determine severity
            if category == 'hate_speech':
                if conf_score > 0.9:
                    severity = 'critical'
                elif conf_score > 0.75:
                    severity = 'high'
                elif conf_score > 0.6:
                    severity = 'medium'
                else:
                    severity = 'low'
            elif category == 'threat':
                severity = 'high' if conf_score > 0.7 else 'medium'
            elif category == 'offensive':
                severity = 'medium' if conf_score > 0.6 else 'low'
            else:
                severity = 'low'
            
            # Detect language (simple heuristic)
            language = self._detect_language(text)
            
            return {
                'is_hate_speech': category in ['hate_speech', 'threat'],
                'confidence': round(conf_score, 4),
                'category': category,
                'severity': severity,
                'language_detected': language,
                'all_predictions': {
                    'category': category,
                    'confidence': conf_score
                }
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
    
    def detect_batch(self, texts: list) -> list:
        """Detect hate speech in multiple texts at once"""
        if not texts:
            return []
        
        results = []
        batch_size = 16  # Process in batches
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                inputs = self.tokenizer(
                    batch,
                    return_tensors='pt',
                    truncation=True,
                    padding=True,
                    max_length=512
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=-1)
                    confidences, predicted_classes = torch.max(probabilities, dim=-1)
                
                for j, (text, pred_class, conf) in enumerate(zip(batch, predicted_classes, confidences)):
                    category = {0: 'neutral', 1: 'hate_speech', 2: 'offensive', 3: 'threat'}.get(
                        pred_class.item(), 'neutral'
                    )
                    
                    results.append({
                        'text': text[:100],
                        'is_hate_speech': category in ['hate_speech', 'threat'],
                        'confidence': round(conf.item(), 4),
                        'category': category,
                        'severity': self._get_severity(category, conf.item()),
                        'language_detected': self._detect_language(text)
                    })
                    
            except Exception as e:
                logger.error(f"Batch detection error: {e}")
                # Add error results for this batch
                for text in batch:
                    results.append({
                        'text': text[:100],
                        'is_hate_speech': False,
                        'confidence': 0.0,
                        'category': 'error',
                        'severity': 'low',
                        'language_detected': 'unknown',
                        'error': str(e)
                    })
        
        return results
    
    def _detect_language(self, text: str) -> str:
        """Detect language of the text (simple heuristic)"""
        # Amharic characters
        if re.search(r'[\u1200-\u137F]', text):
            return 'amharic'
        # Oromo/Latin with specific patterns
        elif re.search(r'(oo|aa|uu|ii|ee)', text.lower()):
            return 'oromo'
        # Tigrinya
        elif re.search(r'[\u1200-\u137F]', text) and any(w in text for w in ['ትግርኛ', 'ሰላም']):
            return 'tigrinya'
        # English
        elif re.search(r'[a-zA-Z]{4,}', text):
            return 'english'
        else:
            return 'unknown'
    
    def _get_severity(self, category: str, confidence: float) -> str:
        """Determine severity based on category and confidence"""
        if category == 'hate_speech':
            if confidence > 0.9:
                return 'critical'
            elif confidence > 0.75:
                return 'high'
            elif confidence > 0.6:
                return 'medium'
        elif category == 'threat':
            return 'high' if confidence > 0.7 else 'medium'
        elif category == 'offensive':
            return 'medium' if confidence > 0.6 else 'low'
        return 'low'


# Global instance for caching
_afro_xlmr_detector = None

def get_afro_xlmr_detector(model_path: str = None) -> AfroXLMRDetector:
    """Get or create AFRO-XLMR detector instance"""
    global _afro_xlmr_detector
    
    if _afro_xlmr_detector is None:
        _afro_xlmr_detector = AfroXLMRDetector(model_path=model_path)
        _afro_xlmr_detector.load_model()
    
    return _afro_xlmr_detector
