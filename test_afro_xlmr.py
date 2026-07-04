#!/usr/bin/env python
"""
Test script for AFRO-XLMR model
Run this locally before deploying to production
"""
import os
import sys
import django

# Setup Django
sys.path.append('/path/to/your/project')  # Update this path
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'your_project.settings')
django.setup()

from dashboard.utils.afro_xlmr_detector import get_afro_xlmr_detector

def test_afro_xlmr():
    """Test AFRO-XLMR detector with sample texts"""
    
    # Test cases in different Ethiopian languages
    test_texts = [
        # Amharic - Hate speech example (Fixed "ክ" to "ክፉ")
        "ይህ ሰው በጣም ክፉ ነው እሱን ማጥፋት አለብን።",
        
        # Amharic - Neutral (Fixed "ዛ" to "ዛሬ")
        "ዛሬ ጥሩ ቀን ነው። ለስራ እሄዳለሁ።",
        
        # Oromo - Offensive
        "Namni kun baay'ee hamaa dha. Isa fixuu qabna.",
        
        # English - Threat
        "I will destroy this person and their family",
        
        # Mixed - Hate speech
        "These people are animals and should be eliminated"
    ]
    
    print("="*70)
    print("Testing AFRO-XLMR Hate Speech Detector")
    print("="*70)
    
    detector = get_afro_xlmr_detector(model_path="/path/to/your/model")  # Update path
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n[Test {i}]")
        print(f"Text: {text[:80]}{'...' if len(text) > 80 else ''}")
        print("-" * 70)
        
        result = detector.detect(text)
        
        print(f"Is Hate Speech: {result.get('is_hate_speech')}")
        print(f"Category: {result.get('category')}")
        print(f"Severity: {result.get('severity')}")
        print(f"Confidence: {result.get('confidence', 0)*100:.2f}%")
        print(f"Language: {result.get('language_detected')}")
        
        if result.get('error'):
            print(f"Error: {result.get('error')}")
        
        print()
    
    print("="*70)
    print("Testing Complete!")
    print("="*70)

if __name__ == "__main__":
    test_afro_xlmr()
