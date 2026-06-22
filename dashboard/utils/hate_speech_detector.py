import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from django.conf import settings

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

class FallbackDetector:
    """
    Fallback when Gemma model fails to load or is intentionally disabled.
    Returns a clear 'unavailable' status so the UI knows NOT to display it as a real detection.
    """
    def detect(self, text: str) -> dict:
        return {
            'category': 'neutral',
            'severity': 'low',
            'confidence': 0.0,
            'raw_prediction': 'Local Gemma model is not loaded or disabled.',
            'is_hate_speech': False,
            'model_status': 'unavailable'  
        }

# 🔥 DISABLED: GemmaHateSpeechDetector class commented out to prevent RAM crashes
'''
class GemmaHateSpeechDetector:
    def __init__(self, model_path: str, base_model: str = "unsloth/gemma-4-e4b-it-unsloth-bnb-4bit"):
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
            
            # The Unsloth model is ALREADY 4-bit quantized
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16,
                device_map=None,
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
                    pad_token_id=self.tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True
                )
            
            # Calculate confidence from logits
            if hasattr(outputs, 'scores') and outputs.scores:
                logits = outputs.scores[0][0]
                probs = torch.softmax(logits, dim=-1)
                confidence = probs.max().item()
            else:
                confidence = 0.5
            
            prediction = self.tokenizer.decode(outputs.sequences[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip().lower()
            
            # Detect if the model is stuck in a repetition loop
            if prediction.count('category') > 2 or (len(prediction) > 10 and len(set(prediction.split())) < 3):
                logger.warning(f"Gemma model stuck in repetition loop. Falling back to LLM/Lexicon.")
                return {
                    'category': 'error', 
                    'severity': 'low', 
                    'confidence': 0.0,
                    'raw_prediction': prediction[:100], 
                    'is_hate_speech': False,
                    'model_status': 'error'
                }
            
            detected_category = self._parse_category(prediction)
            severity = CATEGORY_SEVERITY.get(detected_category, 'medium')
            
            return {
                'category': detected_category,
                'severity': severity,
                'confidence': confidence,
                'raw_prediction': prediction,
                'is_hate_speech': detected_category != 'neutral',
                'model_type': 'gemma_4_lora_multiclass_19categories',
                'model_status': 'active'  # 🔥 Crucial flag for the UI
            }
                
        except Exception as e:
            logger.error(f"Gemma LoRA detection failed: {e}")
            return {
                'category': 'error', 'severity': 'low', 'confidence': 0.0,
                'raw_prediction': str(e), 'is_hate_speech': False,
                'model_status': 'error'
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
'''

_detector_instance = None

def get_hate_speech_detector():
    """Get or create the hate speech detector - DISABLED due to RAM limitations"""
    global _detector_instance
    
    # Always use FallbackDetector to prevent RAM crashes
    # TODO: Re-enable GemmaHateSpeechDetector after upgrading EC2 to 16GB RAM
    logger.warning("Gemma LoRA model is DISABLED - using FallbackDetector (insufficient RAM)")
    
    if _detector_instance is None:
        _detector_instance = FallbackDetector()
        logger.info("✅ FallbackDetector initialized (Gemma model disabled)")
    
    return _detector_instance

# gamma model  (commented out):
'''
def get_hate_speech_detector():
    """Get or create the hate speech detector using Gemma LoRA"""
    global _detector_instance
    
    if _detector_instance is None:
        try:
            lora_adapter_path = getattr(settings, 'GEMMA_LORA_ADAPTER_PATH', './model_cache/gemma-lora-hate-speech')
            base_model_name = getattr(settings, 'GEMMA_BASE_MODEL_NAME', 'unsloth/gemma-4-e4b-it-unsloth-bnb-4bit')
            
            logger.info(f"Initializing hate speech detector with adapter: {lora_adapter_path}")
            
            # Create the detector instance with the path
            _detector_instance = GemmaHateSpeechDetector(
                model_path=lora_adapter_path,
                base_model=base_model_name
            )
            
            # Load the model
            _detector_instance.load_model()
            logger.info("✅ Hate speech detector initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize hate speech detector: {e}")
            logger.info("Using FallbackDetector instead (Model Unavailable)")
            _detector_instance = FallbackDetector()
    
    return _detector_instance
'''
