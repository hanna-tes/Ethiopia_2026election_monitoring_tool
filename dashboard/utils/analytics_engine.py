import json
import hashlib
from collections import Counter
from django.utils import timezone
from django.core.cache import cache
from dashboard.models import ProcessedPost


def get_filter_cache_key(prefix: str, query_params: dict) -> str:
    """
    Generates a unique, deterministic cache key based on query parameters.
    Converts values to strings to prevent JSON serialization errors with dates/objects.
    """
    normalized_params = {
        'view_all': str(query_params.get('view_all', '')).lower() in ['true', '1'],
        'start_date': str(query_params.get('start_date', '')),
        'end_date': str(query_params.get('end_date', '')),
        'platform': str(query_params.get('platform', '')),
        'topic': str(query_params.get('topic', '')),
        'tone': str(query_params.get('tone', '')),
    }
    param_str = json.dumps(normalized_params, sort_keys=True)
    param_hash = hashlib.md5(param_str.encode('utf-8')).hexdigest()
    return f"analytics_{prefix}_{param_hash}"


def compute_filtered_analytics(snapshot_type: str, params: dict) -> dict:
    """
    Calculates trending narratives, risk actors, or hashtags for a filtered QuerySet.
    """
   
    from dashboard.views import get_ethiopia_summaries, scan_text_for_lexicon_terms

    qs = ProcessedPost.objects.all()

    # 1. Apply Date Filters (using timestamp_share)
    if params.get('start_date'):
        qs = qs.filter(timestamp_share__gte=params['start_date'])
    if params.get('end_date'):
        qs = qs.filter(timestamp_share__lte=params['end_date'])

    # 2. Apply Platform / Category Filters
    if params.get('platform'):
        qs = qs.filter(platform__iexact=params['platform'])

    # 3. Handle View All vs Sliced Evaluation
    is_view_all = str(params.get('view_all', '')).lower() in ['true', '1']
    eval_qs = qs if is_view_all else qs[:1000]

    # 4. Generate Snapshot Payload based on type
    if snapshot_type == 'narratives':
        summaries = get_ethiopia_summaries(eval_qs)
        return {
            'summaries': summaries,
            'total_analyzed': eval_qs.count() if is_view_all else len(eval_qs),
            'generated_at': timezone.now()
        }

    elif snapshot_type == 'home':
        # Platform distribution
        platform_counts = dict(Counter(
            eval_qs.values_list('platform', flat=True)
        ))

        # Risk actor and hashtag scanning
        risk_actors = []
        hashtag_counter = Counter()

        for post in eval_qs:
            text = post.original_text or ''
            
            # Aggregate hashtags
            hashtags = [tag.strip().lower() for tag in text.split() if tag.startswith('#')]
            hashtag_counter.update(hashtags)

            # Check risk level via lexicon scanner
            matches = scan_text_for_lexicon_terms(text)
            if matches and post.account_id and post.account_id != 'Unknown':
                high_risk = any(m.get('severity') in ['high', 'critical'] for m in matches)
                if high_risk:
                    risk_actors.append({
                        'account_id': post.account_id,
                        'platform': post.platform,
                        'matched_terms': [m['term'] for m in matches]
                    })

        # Deduplicate risk actors by account_id
        unique_risk_actors = {actor['account_id']: actor for actor in risk_actors}

        return {
            'trend_analysis': {
                'platform_distribution': platform_counts,
                'total_posts': eval_qs.count() if is_view_all else len(eval_qs)
            },
            'risk_actors': list(unique_risk_actors.values())[:10],
            'top_hashtags': hashtag_counter.most_common(10),
            'generated_at': timezone.now()
        }

    return {}


def get_analytics_snapshot(snapshot_type: str, request_params: dict):
    """
    Tiered fetch: Low-level Cache -> Live On-Demand Fallback.
    """
    cache_key = get_filter_cache_key(snapshot_type, request_params)

    # Check cache first
    cached_payload = cache.get(cache_key)
    if cached_payload:
        return cached_payload, cached_payload.get('generated_at')

    # Cache miss: compute dynamically
    payload = compute_filtered_analytics(snapshot_type, request_params)

    # Save to Redis / Cache store for 2hr
    cache.set(cache_key, payload, timeout=7200)

    return payload, payload.get('generated_at')
