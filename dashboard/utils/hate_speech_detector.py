import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logger = logging.getLogger(__name__)

# These are the EXACT categories from your trained model
HATE_SPEECH_CATEGORIES = [
    'ethnicity', 'xenophobia', 'ancestry', 'violence', 'extremism',
    'gender disinformation', 'stereotype', 'class', 'derogatory',
    'slur', 'misogynistic', 'religion', 'ethnic slur',
    'inflammatory', 'inciteful', 'call for action', 'homophobic',
    'structural', 'dehumanization', 'neutral'
]

# Map categories to severity levels
CATEGORY_SEVERITY = {
    'violence': 'critical',
    'inciteful': 'critical',
    'call for action': 'critical',
    'dehumanization': 'critical',
    'extremism': 'high',
    'ethnic slur': 'high',
    'slur': 'high',
    'misogynistic': 'high',
    'derogatory': 'medium',
    'inflammatory': 'medium',
    'gender disinformation': 'medium',
    'stereotype': 'medium',
    'homophobic': 'medium',
    'ethnicity': 'high',
    'xenophobia': 'high',
    'religion': 'high',
    'ancestry': 'medium',
    'class': 'low',
    'structural': 'low',
    'neutral': 'low'
}

class GemmaHateSpeechDetector:
    def __init__(self, model_path: str, base_model: str = "google/gemma-2-2b-it"):
        self.model_path = model_path
        self.base_model_name = base_model
        self.model = None
        self.tokenizer = None
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
    def load_model(self):
        if self.model is not None:
            return
            
        try:
            logger.info(f"Loading tokenizer from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            logger.info(f"Loading base model: {self.base_model_name}...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map="auto" if self.device != "cpu" else None
            )
            
            logger.info(f"Applying LoRA adapter from {self.model_path}...")
            self.model = PeftModel.from_pretrained(
                base_model,
                self.model_path,
                device_map="auto" if self.device != "cpu" else None
            )
            self.model.eval()
            logger.info("✅ Gemma LoRA Hate Speech model loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Gemma LoRA model: {e}")
            raise

    def detect(self, text: str) -> dict:
        if self.model is None:
            self.load_model()
            
        try:
            # Use the EXACT prompt format from your training
            prompt = f"""Classify the following text into one of these categories:
{', '.join(HATE_SPEECH_CATEGORIES)}

Text: "{text}"

Category:"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=30,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            prediction = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip().lower()
            
            # Find which category was predicted
            detected_category = self._parse_category(prediction)
            severity = CATEGORY_SEVERITY.get(detected_category, 'medium')
            confidence = self._estimate_confidence(prediction, detected_category)
            
            return {
                'category': detected_category,
                'severity': severity,
                'confidence': confidence,
                'raw_prediction': prediction,
                'is_hate_speech': detected_category != 'neutral',
                'model_type': 'gemma_lora_multiclass_19categories'
            }
                
        except Exception as e:
            logger.error(f"Gemma LoRA detection failed: {e}")
            return {
                'category': 'error',
                'severity': 'low',
                'confidence': 0.0,
                'raw_prediction': str(e),
                'is_hate_speech': False
            }
    
    def _parse_category(self, prediction: str) -> str:
        """Find which category from our list matches the prediction."""
        prediction_lower = prediction.lower().strip()
        
        # Check for exact matches first
        for category in HATE_SPEECH_CATEGORIES:
            if category in prediction_lower or prediction_lower in category:
                return category
        
        # Fallback: check for key terms
        if any(word in prediction_lower for word in ['violence', 'violent']):
            return 'violence'
        elif any(word in prediction_lower for word in ['incite', 'incitement']):
            return 'inciteful'
        elif any(word in prediction_lower for word in ['call', 'action']):
            return 'call for action'
        elif any(word in prediction_lower for word in ['dehumaniz', 'dehumanization']):
            return 'dehumanization'
        elif any(word in prediction_lower for word in ['extrem', 'extremism']):
            return 'extremism'
        elif any(word in prediction_lower for word in ['slur', 'ethnic slur']):
            return 'slur' if 'ethnic' not in prediction_lower else 'ethnic slur'
        elif any(word in prediction_lower for word in ['misogyn', 'misogynistic']):
            return 'misogynistic'
        elif any(word in prediction_lower for word in ['derogat', 'derogatory']):
            return 'derogatory'
        elif any(word in prediction_lower for word in ['inflamm', 'inflammatory']):
            return 'inflammatory'
        elif any(word in prediction_lower for word in ['gender', 'disinformation']):
            return 'gender disinformation'
        elif any(word in prediction_lower for word in ['stereotype']):
            return 'stereotype'
        elif any(word in prediction_lower for word in ['homophobic', 'homophobia']):
            return 'homophobic'
        elif any(word in prediction_lower for word in ['ethnicity', 'ethnic']):
            return 'ethnicity'
        elif any(word in prediction_lower for word in ['xenophobia', 'xenophobic']):
            return 'xenophobia'
        elif any(word in prediction_lower for word in ['religion', 'religious']):
            return 'religion'
        elif any(word in prediction_lower for word in ['ancestry']):
            return 'ancestry'
        elif any(word in prediction_lower for word in ['class']):
            return 'class'
        elif any(word in prediction_lower for word in ['structural']):
            return 'structural'
        else:
            return 'neutral'
    
    def _estimate_confidence(self, prediction: str, detected_category: str) -> float:
        """Estimate confidence based on prediction clarity."""
        if detected_category == 'neutral':
            return 0.90
        
        # Higher confidence if category name appears clearly in prediction
        if detected_category in prediction.lower():
            return 0.85
        elif any(word in prediction.lower() for word in detected_category.split()):
            return 0.75
        else:
            return 0.65

# Global instance for caching
_detector_instance = None

def get_hate_speech_detector(model_path: str = None) -> GemmaHateSpeechDetector:
    global _detector_instance
    
    if _detector_instance is None:
        if model_path is None:
            from django.conf import settings
            model_path = getattr(settings, 'GEMMA_LOKA_MODEL_PATH', 
                               './dashboard/model_cache/gemma_hate_lexicon_lora')
        
        _detector_instance = GemmaHateSpeechDetector(model_path)
        _detector_instance.load_model()
    
    return _detector_instance
