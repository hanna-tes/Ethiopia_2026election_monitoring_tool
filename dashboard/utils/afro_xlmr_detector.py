import os
import logging
import torch
import torch.nn as nn
import joblib
import re
import numpy as np
from transformers import AutoTokenizer, AutoModel
from django.conf import settings 

logger = logging.getLogger(__name__)

# Use Django's BASE_DIR for bulletproof path resolution
MODEL_DIR = os.path.join(settings.BASE_DIR, 'dashboard', 'models_cache', 'afro_xlmr')
CHECKPOINT_PATH = os.path.join(MODEL_DIR, 'hateguard_finetuned_v7.pt')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')

logger.info(f"🎯 AFRO-XLMR Checkpoint Path: {CHECKPOINT_PATH}")

# --- Model Architecture  ---
class AfroXLMRClassifier(nn.Module):
    def __init__(self, num_classes=20, dropout=0.4):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("Davlan/afro-xlmr-base")
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.last_hidden_state[:, 0, :])

# --- Detector Class ---
class AfroXlmrDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.classes = []
        self._load_model()

    def _load_model(self):
        try:
            if not os.path.exists(CHECKPOINT_PATH):
                logger.error(f"❌ Checkpoint does not exist: {CHECKPOINT_PATH}")
                return

            logger.info(f"⏳ Loading AFRO-XLMR from checkpoint: {CHECKPOINT_PATH}")
            
            # 1. Load label encoder to get exact class names from training
            if os.path.exists(LABEL_ENCODER_PATH):
                self.le = joblib.load(LABEL_ENCODER_PATH)
                self.classes = list(self.le.classes_)
                num_classes = len(self.classes)
                logger.info(f"✅ Loaded {num_classes} classes from label_encoder.pkl")
            else:
                # Fallback to the 20 classes if pkl is missing
                self.classes = [
                    'Ancestry', 'Ethnicity', 'Gender disinformation', 'Homophobic',
                    'Misognistic', 'Religion', 'Xenophobia', 'Deragatory',
                    'Dehumanization', 'Ethnic slur', 'Slur', 'Stereotype',
                    'Call for action', 'Inciteful', 'Violence', 'Class',
                    'Extremism', 'Inflammatory', 'Stractural', 'Neutral'
                ]
                num_classes = len(self.classes)

            # 2. Initialize model architecture
            self.model = AfroXLMRClassifier(num_classes=num_classes).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained("Davlan/afro-xlmr-base")

            # 3. Load fine-tuned weights from the .pt file
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.eval()
            
            logger.info(f"✅ AFRO-XLMR model loaded successfully! (Epoch {checkpoint.get('epoch', '?')}, Val Acc: {checkpoint.get('val_acc', 0):.4f})")
            
        except Exception as e:
            logger.error(f"❌ Failed to load AFRO-XLMR model: {e}")
            import traceback
            logger.error(traceback.format_exc())

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

            # Handle edge case where squeeze makes it 0-dimensional
            if isinstance(probs, np.ndarray) and probs.ndim == 0:
                probs = np.array([probs])
            
            top_idx = int(np.argmax(probs))
            confidence = float(probs[top_idx])
            
            # Map to category name
            if top_idx < len(self.classes):
                category = self.classes[top_idx]
            else:
                category = 'Neutral'
            
            SEVERITY_MAP = {
                'Violence': 'critical', 'Inciteful': 'critical', 'Call for action': 'critical', 
                'Dehumanization': 'critical', 'Extremism': 'high', 'Ethnic slur': 'high', 
                'Slur': 'high', 'Misognistic': 'high', 'Deragatory': 'medium', 
                'Inflammatory': 'high', 'Gender disinformation': 'high', 'Stereotype': 'high', 
                'Homophobic': 'high', 'Ethnicity': 'high', 'Xenophobia': 'high', 'Religion': 'high',
                'Ancestry': 'medium', 'Class': 'low', 'Stractural': 'low', 'Neutral': 'low'
            }

            # Detect language
            language = 'amharic' if re.search(r'[\u1200-\u137F]', text) else 'english'

            return {
                'is_hate_speech': category != 'Neutral',
                'confidence': round(confidence, 4),
                'category': category,
                'severity': SEVERITY_MAP.get(category, 'low'),
                'language_detected': language,
                'model': 'AFRO-XLMR-Finetuned'
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
