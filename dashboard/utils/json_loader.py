import json
import os
import logging
from django.conf import settings
from django.core.cache import cache

logger = logging.getLogger(__name__)

def load_jsonl_data(filename, cache_key=None, max_lines=None):
    """
    Safely load a .jsonl file line-by-line and cache the result.
    """
    # 1. Check Cache first (prevents reading 80k lines on every page load)
    if cache_key:
        cached_data = cache.get(cache_key)
        if cached_data is not None:
            return cached_data

    try:
        file_path = os.path.join(settings.BASE_DIR, 'dashboard', 'data', filename)
        
        if not os.path.exists(file_path):
            logger.warning(f"⚠️ JSONL file not found: {file_path}")
            return []
        
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue # Skip bad lines
                    
        # 2. Save to cache for 24 hours (86400 seconds)
        if cache_key:
            cache.set(cache_key, data, 86400)
            
        return data
            
    except Exception as e:
        logger.error(f" Error loading {filename}: {e}")
        return []

def get_disarm_ttp_reference():
    """
    Extracts the most common DISARM techniques from the 80k training file.
    """
    # Load the training data (cached for 24 hours)
    train_data = load_jsonl_data('disarm_phase3_train.jsonl', cache_key='disarm_train_data')
    
    if not train_data:
        return []

    # Extract unique techniques and count their frequency
    techniques_map = {}
    
    for record in train_data:
        # The Radar Suite Phase 3 schema usually stores techniques in a list
        # We look for common field names like 'techniques', 'allowed_techniques', or 'ttps'
        technique_list = record.get('allowed_techniques') or record.get('techniques') or record.get('ttps') or []
        
        if isinstance(technique_list, list):
            for tech_id in technique_list:
                # Ensure it looks like a DISARM ID (e.g., "T0049")
                if isinstance(tech_id, str) and tech_id.startswith('T'):
                    if tech_id not in techniques_map:
                        techniques_map[tech_id] = {'id': tech_id, 'count': 0}
                    techniques_map[tech_id]['count'] += 1
                    
    # Convert to list and sort by frequency (most common first)
    unique_ttps = list(techniques_map.values())
    unique_ttps.sort(key=lambda x: x['count'], reverse=True)
    
    # Return top 30 most common TTPs for the UI
    return unique_ttps[:30]
