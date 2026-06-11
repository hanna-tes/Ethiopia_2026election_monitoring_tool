import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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
    'violence': 'critical', 'inciteful': 'critical', 'call for action': 'critical', 'dehumanization': 'critical',
    'extremism': 'high', 'ethnic slur': 'high', 'slur': 'high', 'misogynistic': 'high',
    'derogatory': 'medium', 'inflammatory': 'medium', 'gender disinformation': 'medium',
    'stereotype': 'medium', 'homophobic': 'high', 'ethnicity': 'high', 'xenophobia': 'high', 'religion': 'high',
    'ancestry': 'low', 'class': 'low', 'structural': 'low', 'neutral': 'low'
}

class GemmaHateSpeechDetector:
    def __init__(self, model_path: str, base_model: str = "./dashboard/model_cache/base_model_4bit"):
        self.model_path = model_path
        self.base_model_name = base_model
        self.model = None
        self.tokenizer = None
        # Use MPS for Mac, CUDA for Linux/EC2, CPU as fallback
        self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self):
        if self.model is not None:
            return
            
        try:
            logger.info(f"Loading tokenizer from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            
            logger.info(f"Loading base model: {self.base_model_name}...")
            
            # The Unsloth model is ALREADY 4-bit quantized.
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16,
                device_map=None,  # Changed from "auto"
                trust_remote_code=True
            )
            
            # Explicitly move the model to Mac GPU (MPS) or EC2 GPU (CUDA)
            base_model = base_model.to(self.device)
            
            logger.info(f"Applying LoRA adapter from {self.model_path}...")
            self.model = PeftModel.from_pretrained(
                base_model,
                self.model_path
                
            )
            
            # Ensure the adapter is also on the correct device
            self.model.to(self.device)
            self.model.eval()
            logger.info("✅ Gemma LoRA Hate Speech model loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Gemma LoRA model: {e}")
            raise
            
    def detect(self, text: str) -> dict:
        if self.model is None:
            self.load_model()
            
        try:
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
            
            detected_category = self._parse_category(prediction)
            severity = CATEGORY_SEVERITY.get(detected_category, 'medium')
            confidence = self._estimate_confidence(prediction, detected_category)
            
            return {
                'category': detected_category,
                'severity': severity,
                'confidence': confidence,
                'raw_prediction': prediction,
                'is_hate_speech': detected_category != 'neutral',
                'model_type': 'gemma_4_lora_multiclass_19categories'
            }
                
        except Exception as e:
            logger.error(f"Gemma LoRA detection failed: {e}")
            return {
                'category': 'error', 'severity': 'low', 'confidence': 0.0,
                'raw_prediction': str(e), 'is_hate_speech': False
            }
    
    def _parse_category(self, prediction: str) -> str:
        prediction_lower = prediction.lower().strip()
        for category in HATE_SPEECH_CATEGORIES:
            if category in prediction_lower or prediction_lower in category:
                return category
        # Fallback keyword matching
        if any(w in prediction_lower for w in ['violence', 'violent']): return 'violence'
        elif any(w in prediction_lower for w in ['incite', 'incitement']): return 'inciteful'
        elif any(w in prediction_lower for w in ['call', 'action']): return 'call for action'
        elif any(w in prediction_lower for w in ['dehumaniz']): return 'dehumanization'
        elif any(w in prediction_lower for w in ['extrem']): return 'extremism'
        elif any(w in prediction_lower for w in ['slur']): return 'slur'
        elif any(w in prediction_lower for w in ['misogyn']): return 'misogynistic'
        elif any(w in prediction_lower for w in ['derogat']): return 'derogatory'
        elif any(w in prediction_lower for w in ['inflamm']): return 'inflammatory'
        elif any(w in prediction_lower for w in ['gender']): return 'gender disinformation'
        elif any(w in prediction_lower for w in ['stereotype']): return 'stereotype'
        elif any(w in prediction_lower for w in ['homophobic']): return 'homophobic'
        elif any(w in prediction_lower for w in ['ethnicity', 'ethnic']): return 'ethnicity'
        elif any(w in prediction_lower for w in ['xenophobia']): return 'xenophobia'
        elif any(w in prediction_lower for w in ['religion']): return 'religion'
        elif any(w in prediction_lower for w in ['ancestry']): return 'ancestry'
        elif any(w in prediction_lower for w in ['class']): return 'class'
        elif any(w in prediction_lower for w in ['structural']): return 'structural'
        else: return 'neutral'
    
    def _estimate_confidence(self, prediction: str, detected_category: str) -> float:
        if detected_category == 'neutral': return 0.90
        if detected_category in prediction.lower(): return 0.85
        elif any(word in prediction.lower() for word in detected_category.split()): return 0.75
        else: return 0.65

_detector_instance = None

def get_hate_speech_detector(model_path: str = None) -> GemmaHateSpeechDetector:
    global _detector_instance
    if _detector_instance is None:
        if model_path is None:
            from django.conf import settings
            model_path = getattr(settings, 'GEMMA_LOKA_MODEL_PATH', './dashboard/model_cache/gemma_hate_lexicon_lora')
        _detector_instance = GemmaHateSpeechDetector(model_path)
        _detector_instance.load_model()
    return _detector_instance
