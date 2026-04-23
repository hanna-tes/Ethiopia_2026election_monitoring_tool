import logging
from groq import Groq
from django.conf import settings

logger = logging.getLogger(__name__)


def get_groq_client():
    if not settings.GROQ_API_KEY:
        logger.warning("⚠️ GROQ_API_KEY not configured")
        return None
    try:
        return Groq(api_key=settings.GROQ_API_KEY)
    except Exception as e:
        logger.error(f"❌ Failed to initialize Groq client: {str(e)}")
        return None


def safe_llm_call(prompt, max_tokens=2048, model=None):
    client = get_groq_client()
    if not client:
        return None
    
    try:
        response = client.chat.completions.create(
            model=model or settings.GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"❌ LLM call failed: {str(e)}")
        return None
