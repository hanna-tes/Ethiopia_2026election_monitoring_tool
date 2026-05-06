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

def summarize_cluster_ethiopia(posts, cluster_id=None):
    """
    Generate LLM summary for an Ethiopia election narrative cluster.
    STUB: Returns a placeholder summary for now.
    
    Args:
        posts: QuerySet or list of ProcessedPost objects
        cluster_id: Optional cluster identifier
    
    Returns:
        str: Summary text
    """
    if not posts:
        return "No posts available for this narrative cluster."
    
    # Extract sample texts for context
    sample_texts = [p.original_text[:200] for p in posts[:5] if p.original_text]
    
    # STUB: Return a placeholder summary
    # TODO: Implement actual Groq/LLM call here
    return f"""🇪🇹 Ethiopia Election Narrative Summary (Cluster {cluster_id or 'N/A'})

📊 Analysis of {len(posts)} election-related posts:

🔍 Key Themes Detected:
• Election integrity and process discussions
• Political party positioning and messaging
• Regional dynamics and ethnic considerations

⚠️ Risk Assessment:
• Sentiment: Mixed (requires deeper analysis)
• Coordination signals: Not detected in sample
• Foreign interference indicators: None identified

💡 Recommended Actions:
• Monitor for escalation in rhetoric
• Track narrative spread across platforms
• Cross-reference with verified news sources

---
*This is a placeholder summary. Full LLM analysis requires Groq API configuration.*
"""
