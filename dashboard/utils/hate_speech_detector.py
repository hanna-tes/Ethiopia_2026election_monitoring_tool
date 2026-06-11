"""
Hate Speech Detection using Fine-tuned Gemma + LoRA Adapter
"""
import logging
import torch
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logger = logging.getLogger(__name__)

class GemmaHateSpeechDetector:
    """
    Multiclass hate speech detector using Gemma model with LoRA adapter.
    """
    
    def __init__(self, model_path: str, base_model: str = "google/gemma-2b"):
        """
        Initialize the hate speech detector.
        
        Args:
            model_path: Path to the LoRA adapter files
            base_model: Base Gemma model to load (default: gemma-2b)
        """
        self.model_path = model_path
        self.base_model_name = base_model
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
    def load_model(self):
        """Load the base model and apply LoRA adapter."""
        if self.model is not None:
            logger.info("Model already loaded")
            return
            
        try:
            logger.info(f"Loading base model: {self.base_model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Load LoRA adapter
            logger.info(f"Loading LoRA adapter from: {self.model_path}")
            self.model = PeftModel.from_pretrained(
                base_model,
                self.model_path,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            self.model.eval()
            logger.info("✅ Hate speech detector loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def detect(self, text: str) -> Dict:
        """
        Detect hate speech in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with detection results
        """
        if self.model is None:
            self.load_model()
        
        try:
            # Prepare input
            prompt = self._create_prompt(text)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate prediction
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode prediction
            prediction = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            prediction = prediction.strip()
            
            # Parse result
            result = self._parse_prediction(prediction, text)
            
            return result
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return {
                'is_hate_speech': False,
                'category': 'error',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _create_prompt(self, text: str) -> str:
        """Create prompt for hate speech detection."""
        return f"""Classify the following text into one of these categories:
- hate_speech
- offensive
- neutral

Text: "{text}"

Category:"""
    
    def _parse_prediction(self, prediction: str, original_text: str) -> Dict:
        """Parse model prediction into structured result."""
        prediction_lower = prediction.lower().strip()
        
        # Determine category
        if 'hate' in prediction_lower or 'hate_speech' in prediction_lower:
            category = 'hate_speech'
            is_hate_speech = True
            confidence = 0.85  # Base confidence
        elif 'offensive' in prediction_lower:
            category = 'offensive'
            is_hate_speech = False
            confidence = 0.75
        else:
            category = 'neutral'
            is_hate_speech = False
            confidence = 0.90
        
        return {
            'is_hate_speech': is_hate_speech,
            'category': category,
            'confidence': confidence,
            'raw_prediction': prediction,
            'model_type': 'gemma_lora_multiclass',
            'text_analyzed': original_text
        }
    
    def batch_detect(self, texts: List[str]) -> List[Dict]:
        """
        Detect hate speech in multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of detection results
        """
        results = []
        for text in texts:
            result = self.detect(text)
            results.append(result)
        return results


# Global instance for caching
_detector_instance = None

def get_hate_speech_detector(model_path: str = None) -> GemmaHateSpeechDetector:
    """
    Get or create the hate speech detector instance.
    
    Args:
        model_path: Path to the model (optional, uses default if not provided)
        
    Returns:
        GemmaHateSpeechDetector instance
    """
    global _detector_instance
    
    if _detector_instance is None:
        if model_path is None:
            from django.conf import settings
            model_path = getattr(settings, 'HATE_SPEECH_MODEL_PATH', 
                               './dashboard/model_cache/gemma_hate_lexicon_lora')
        
        _detector_instance = GemmaHateSpeechDetector(model_path)
        _detector_instance.load_model()
    
    return _detector_instance
