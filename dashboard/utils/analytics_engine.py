import hashlib
import json
from django.utils import timezone
from django.core.cache import cache

# Import your data models and relevant processing functions
# Adjust imports below based on where your Post model and functions are defined
from dashboard.models import SocialMediaPost  # Or whatever your Post model is named
# from dashboard.utils.election_filter import ... 


def get_filter_cache_key(prefix: str, query_params: dict) -> str:
    """Generates a unique cache key based on query parameters."""
    normalized_params = {
        'view_all': str(query_params.get('view_all', '')).lower() in ['true', '1'],
        'start_date': query_params.get('start_date', ''),
        'end_date': query_params.get('end_date', ''),
        'platform': query_params.get('platform', ''),
        'topic': query_params.get('topic', ''),
        'tone': query_params.get('tone', ''),
    }
    param_str = json.dumps(normalized_params, sort_keys=True)
    param_hash = hashlib.md5(param_str.encode('utf-8')).hexdigest()
    return f"analytics_{prefix}_{param_hash}"


def compute_filtered_analytics(snapshot_type: str, params: dict) -> dict:
    """Calculates trending narratives, risk actors, or hashtags for a filtered QuerySet."""
    qs = SocialMediaPost.objects.all()
    
    # 1. Date Filters
    if params.get('start_date'):
        qs = qs.filter(created_at__gte=params['start_date'])
    if params.get('end_date'):
        qs = qs.filter(created_at__lte=params['end_date'])
        
    # 2. Category / Platform Filters
    if params.get('platform'):
        qs = qs.filter(platform=params['platform'])
    if params.get('topic'):
        qs = qs.filter(topic=params['topic'])
        
    # 3. View All vs Paginated Limit
    is_view_all = str(params.get('view_all', '')).lower() in ['true', '1']
    if not is_view_all:
        qs = qs[:1000]  # Cap standard query length for speed

    # 4. Generate snapshot payload
    if snapshot_type == 'narratives':
        # Add your narrative clustering/summarization logic or helper function call here
        summaries = [] 
        return {
            'summaries': summaries, 
            'generated_at': timezone.now()
        }
    
    elif snapshot_type == 'home':
        # Add your trend, risk actor, and hashtag calculation logic or call existing utils here
        return {
            'trend_analysis': {},
            'risk_actors': [],
            'top_hashtags': [],
            'generated_at': timezone.now()
        }

    return {}


def get_analytics_snapshot(snapshot_type: str, request_params: dict):
    """Tiered fetch: Low-level Cache -> Live On-Demand Fallback."""
    cache_key = get_filter_cache_key(snapshot_type, request_params)
    
    # Check cache first
    cached_payload = cache.get(cache_key)
    if cached_payload:
        return cached_payload, cached_payload.get('generated_at')

    # Cache miss: compute dynamically
    payload = compute_filtered_analytics(snapshot_type, request_params)
    
    # Save to cache for 30 minutes
    cache.set(cache_key, payload, timeout=1800)
    
    return payload, payload.get('generated_at')
