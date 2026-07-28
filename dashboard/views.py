"""
Django views for Ethiopia Election Monitor
"""
import json
import logging
import os
import re
from django.db.models import Q
import requests
import threading 
import random
import hashlib
from django.core.cache import cache
import csv
from django.utils import timezone
from io import StringIO
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from .utils.hate_speech_detector import get_hate_speech_detector
from .utils.json_loader import get_disarm_ttp_reference
from django.shortcuts import render, redirect
from django.contrib import messages
from django.views.generic import TemplateView, View
from django.http import JsonResponse, HttpResponse
from django.db.models import Count, Q, F, Case, When, Value, CharField, Max, Avg, Min
from django.contrib.postgres.aggregates import ArrayAgg
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.paginator import Paginator
from django.db.models.functions import TruncDay
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from django.utils import timezone
from .models import ProcessedPost, NarrativeCluster, PEP, LexiconTerm, DataUpload
from .utils.llm_service import safe_llm_call, summarize_cluster_ethiopia
from .utils.data_loader import load_data_robustly, load_peps_from_github
from .utils.csv_processor import process_uploaded_csv, map_columns_by_type, preprocess_dataframe
from .utils.lexicon_engine import scan_text_for_lexicon_terms, calculate_risk_score, generate_lexicon_analytics
from .utils.election_filter import is_election_related
from .utils.wordcloud import generate_trigger_wordcloud, wordcloud_to_base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from .utils.data_loader import parse_timestamp_robust
from .utils.csv_processor import combine_social_media_data
from .models import ProcessedPost, DataSource
from django.views.decorators.cache import never_cache
from .utils.llm_detector import detect_hate_speech_llm
from .models import ElectionOfficeholder
from django.shortcuts import render, get_object_or_404
from .models import MonitoringReport
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from django.contrib.auth.decorators import login_required
from .utils.afro_xlmr_detector import get_detector
from .detectors import is_election_related
from dashboard.utils.analytics_engine import get_analytics_snapshot


logger = logging.getLogger(__name__)

# Global model cache
_GEMMA_MODEL = None
_GEMMA_TOKENIZER = None

#Cache to merge CONFIG and Database lexicon terms
_COMBINED_LEXICON_CACHE = None
_CACHE_TIMESTAMP = 0

def load_gemma_lora_model():
    """Gemma LoRA model - DISABLED due to RAM limitations """
    # TODO: Re-enable after upgrading EC2 instance to t3.xlarge (16GB RAM)
    logger.warning("Gemma LoRA model is DISABLED - insufficient RAM (current: 8GB, required: 16GB+)")
    return None, None

def get_combined_lexicon():
    """
    Merges CONFIG lexicon with Database lexicon terms.
    Cached for 60 seconds to prevent database hammering during scans.
    """
    global _COMBINED_LEXICON_CACHE, _CACHE_TIMESTAMP
    import time
    if _COMBINED_LEXICON_CACHE is None or (time.time() - _CACHE_TIMESTAMP) > 60:
        try:
            # Start with a copy of CONFIG lexicon
            combined = {k: v.copy() for k, v in CONFIG.get("lexicon", {}).items()}
            
            # Fetch all active DB terms
            db_terms = LexiconTerm.objects.filter(is_election_related=True).values(
                'term', 'category', 'severity', 'target_entity', 'language'
            )
            for term_obj in db_terms:
                cat = term_obj['category']
                if cat not in combined:
                    combined[cat] = {}
                # DB terms override or add to CONFIG
                combined[cat][term_obj['term']] = {
                    'severity': term_obj['severity'],
                    'target_entity': term_obj['target_entity'],
                    'language': term_obj['language']
                }
            _COMBINED_LEXICON_CACHE = combined
            _CACHE_TIMESTAMP = time.time()
        except Exception as e:
            logger.warning(f"Failed to load DB lexicon terms, falling back to CONFIG: {e}")
            _COMBINED_LEXICON_CACHE = CONFIG.get("lexicon", {})
            _CACHE_TIMESTAMP = time.time()
    return _COMBINED_LEXICON_CACHE
    
WEAPONIZED_KEYWORDS = [
    # === ENGLISH: VIOLENCE & CONFLICT ===
    'genocide', 'kill', 'attack', 'war', 'slur', 'hate', 
    'ethnic cleansing', 'massacre', 'slaughter', 'destroy',
    'eliminate', 'terrorist', 'extremist', 'traitor', 'criminal',
    'dictator', 'oppressor', 'failed state', 'invasion', 'exterminate',
    'wipe out', 'blood on hands', 'step down', 'death to', 'puppet of',

    # === ENGLISH: LOCALIZED POLITICAL SLURS & CONTEXT ===
    'woyane', 'junta', 'banda', 'neftegna', 'chilot', 'galla', 
    'fano', 'ola', 'tpdf', 'pp', 'prosperity party', 'cabal',

    # === AMHARIC (አማርኛ): VIOLENCE, INCITEMENT & EXTREMISM ===
    'ግድያ', 'ጦርነት', 'ፈጅ', 'ጥቃት', 'ማጥፋት', 'አጥፋቸው', 'ደም', 
    'እርምጃ', 'ክህደት', 'ሽብርተኛ', 'አሸባሪ', 'ፅንፈኛ', 'አክራሪ', 'ወራሪ',
    'ህገ-ወጥ', 'ማጥቃት', 'ፋኖ', 'ኦነግ', 'ህወሃት', 'ትህነግ',

    # === AMHARIC (አማርኛ): LOCALIZED WEAPONIZED SLURS ===
    'ባንዳ', 'ወያኔ', 'ጁንታ', 'ነፍጠኛ', 'ጋላ', 'ብልጽግና', 'ሌባ', 
    'ከሃዲ', 'ውሸታም', 'አምባገነን', 'የሰፈር', 'ክልል', 'ጎሳ', 'ዘረኛ', 
    'ተላላኪ', 'መሳሪያ', 'ሀገር አጥፊ', 'አገር አፈራሽ', 'ይውረድ', 'ውድቀት ለ',

    # === AFAAN OROMO: VIOLENCE & CONFLICT ===
    'ajjeesa', 'fixiinsa', 'warraana', 'lola', 'du\'a', 'balleesuu', 
    'diina', 'shororkeessaa', 'gara-laafina', 'gantummaa', 'waraana',
    'miidhaa', 'reeffa', 'dhiiga', 'wrrana',

    # === AFAAN OROMO: LOCALIZED WEAPONIZED SLURS ===
    'gantuu', 'habashaa', 'nefxanyaa', 'neftenga', 'bandaa', 'peepii',
    'ergamtoota', 'xuraawaa', 'saamtuu', 'sobaa', 'kijibduu', 'opheree',
    'PP', 'nafxanyaa'
]


def should_process_post(text: str) -> bool:
    """
    Allows all posts through EXCEPT those that have no mention of Ethiopian 
    affairs/geography or are clearly spam/walls of irrelevant hashtags.
    """
    if not text:
        return False
        
    text_lower = text.lower()
    
    # 1. Broadest possible anchors to confirm the post is about Ethiopia/local affairs
    ethiopian_anchors = [
        "ethiopia", "etiopia", "itopia", "habesha", "addis", "orom", "amhar", 
        "tigray", "somali", "afar", "sidama", "fano", "shene", "ager", "hizb", 
        "hezb", "biher", "gosa", "abiy", "prosperity", "pp", "tplf"
    ]
    
    has_local_context = any(anchor in text_lower for anchor in ethiopian_anchors)
    
    # 2. Identify hashtag spam not talking about Ethiopian affairs
    # Find all hashtags in the text
    hashtags = re.findall(r"#\w+", text_lower)
    
    if hashtags:
        # Define high-frequency non-Ethiopian spam hashtags
        spam_hashtags = {
            "#crypto", "#bitcoin", "#nft", "#forex", "#marketing", "#digitalmarketing",
            "#makeup", "#fashion", "#travel", "#football", "#cricket", "#ecommerce", 
            "#jobs", "#hiring", "#win", "#giveaway", "#fitness", "#motivation"
        }
        
        # Count how many spam hashtags are present
        spam_count = sum(1 for tag in hashtags if tag in spam_hashtags)
        
        # If MORE THAN HALF of the hashtags are completely unrelated spam, drop it
        if len(hashtags) >= 3 and (spam_count / len(hashtags)) > 0.5:
            return False

    # 3. Final Decision: Keep it if it has local context, or if it's just regular text
    # (We only reject if it explicitly matches the spam rules or completely lacks local context)
    return has_local_context
    
def detect_hate_speech_afro_xlmr(text: str) -> dict:
    """
    Detect hate speech using AFRO-XLMR model
    Separate from Gemma model to avoid confusion
    """
    try:
        detector = get_detector()
        return detector.detect(text)
    except Exception as e:
        logger.error(f"AFRO-XLMR detection failed: {e}")
        return {
            'is_hate_speech': False,
            'confidence': 0.0,
            'category': 'error',
            'severity': 'low',
            'error': str(e)
        }
def translate_amharic_terms_llm(terms_list):
    """
    Use LLM to translate Amharic terms to English with context
    """
    if not terms_list:
        return []
    
    # Filter only Amharic terms
    amharic_terms = [t for t in terms_list if t.get('metadata', {}).get('language', '').lower() == 'amharic']
    
    if not amharic_terms:
        return []
    
    try:
        # Prepare prompt for LLM
        terms_text = "\n".join([f"- {t['term']} (Severity: {t['metadata'].get('severity', 'unknown')}, Category: {t['metadata'].get('category', 'unknown')})" for t in amharic_terms])
        
        prompt = f"""Translate these Amharic hate speech terms to English. Provide contextual meaning and explain the severity:

{terms_text}

Format each translation as:
Term: [Amharic term]
English: [Translation]
Context: [Explanation of usage and severity]
---
"""
        
        # Call LLM (using your existing detect_hate_speech_llm function or similar)
        from .utils.llm_integration import call_llm_api  # Adjust import based on your setup
        
        response = call_llm_api(prompt, temperature=0.3)
        
        # Parse the response and attach translations to terms
        translations = parse_llm_translations(response)
        
        # Merge translations back into terms
        for term in amharic_terms:
            term_term = term['term']
            if term_term in translations:
                term['translation'] = translations[term_term]
        
        return amharic_terms
        
    except Exception as e:
        logger.error(f"LLM translation failed: {e}")
        # Return original terms without translation
        return amharic_terms

# ==========================================
# 1. DEFINE UTILITY FUNCTIONS (Put at top)
# ==========================================

def preprocess_and_clean_post(post_text, threshold=5):
    """
    Checks for hashtag stuffing. If the number of hashtags exceeds the threshold,
    it strips them from the text so the classifier isn't misled.
    
    Returns:
        cleaned_text (str): The text to be processed by the classifier.
        confidence_flag (str): "normal" or "low-confidence"
    """
    hashtag_pattern = r'#\S+'
    hashtags = re.findall(hashtag_pattern, post_text)
    
    if len(hashtags) > threshold:
        # Strip all hashtags and clean up trailing/double whitespaces
        cleaned_text = re.sub(hashtag_pattern, '', post_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        confidence_flag = "low-confidence"
    else:
        cleaned_text = post_text
        confidence_flag = "normal"
        
    return cleaned_text, confidence_flag


# ==========================================
# 2. THE PROCESSING PIPELINE 
# ==========================================

def process_incoming_post(raw_post, classifier, publish_function):
    # Step A: Clean the post
    cleaned_text, confidence = preprocess_and_clean_post(raw_post, threshold=5)

    # Step B: Use the passed classifier
    category_prediction = classifier.predict(cleaned_text) 

    # Step C: Drop if low-confidence
    if confidence == "low-confidence":
        print(f"Post dropped! Caught hashtag stuffing. (Predicted: {category_prediction})")
        return False  

    # Step D: Use the passed publish function
    publish_function(category_prediction, raw_post)
    return True

def parse_llm_translations(response_text):
    """Parse LLM response to extract translations"""
    translations = {}
    current_term = None
    current_translation = {}
    
    for line in response_text.strip().split('\n'):
        line = line.strip()
        if line.startswith('Term:'):
            if current_term and current_translation:
                translations[current_term] = current_translation
            current_term = line.replace('Term:', '').strip()
            current_translation = {}
        elif line.startswith('English:'):
            current_translation['english'] = line.replace('English:', '').strip()
        elif line.startswith('Context:'):
            current_translation['context'] = line.replace('Context:', '').strip()
    
    # Don't forget the last one
    if current_term and current_translation:
        translations[current_term] = current_translation
    
    return translations
    

def export_merged_gephi_csv(request):
    """Generates and downloads a Gephi-compatible CSV edge-list representing the coordination network"""
    import re
    from collections import defaultdict
    
    # Extract query filters
    min_connections = int(request.GET.get('min_connections', 2))
    cluster_filter = request.GET.get('cluster') or request.GET.get('group_id') or request.GET.get('coordination_group')
    
    query_kwargs = {'is_election_related': True}
    if cluster_filter and cluster_filter.strip():
        min_connections = 1  # Relax edge constraints for dedicated groups
        if hasattr(ProcessedPost, 'coordination_group_id'):
            query_kwargs['coordination_group_id'] = cluster_filter
        elif hasattr(ProcessedPost, 'coordination_group'):
            query_kwargs['coordination_group'] = cluster_filter
        else:
            try: query_kwargs['cluster'] = int(cluster_filter)
            except ValueError: pass
    else:
        if hasattr(ProcessedPost, 'cluster'):
            query_kwargs['cluster__gte'] = 0

    # Stream values from database
    posts_data = ProcessedPost.objects.filter(**query_kwargs).values('account_id', 'original_text')
    
    G = nx.Graph()
    rt_pattern = re.compile(r'RT\s+@([a-zA-Z0-9_]+)', re.IGNORECASE)
    text_to_accounts = defaultdict(set)
    
    for post in posts_data:
        acc = post['account_id']
        text = post['original_text'] or ''
        if not acc or acc == 'unknown':
            continue
            
        rt_match = rt_pattern.search(text)
        if rt_match:
            target_user = rt_match.group(1)
            if acc != target_user:
                if G.has_edge(acc, target_user): G[acc][target_user]['weight'] += 1
                else: G.add_edge(acc, target_user, weight=1)
                    
        text_to_accounts[text].add(acc)

    for acc_set in text_to_accounts.values():
        if len(acc_set) >= 2:
            acc_list = list(acc_set)
            for i in range(len(acc_list)):
                for j in range(i + 1, len(acc_list)):
                    if acc_list[i] != acc_list[j]:
                        if G.has_edge(acc_list[i], acc_list[j]): G[acc_list[i]][acc_list[j]]['weight'] += 1
                        else: G.add_edge(acc_list[i], acc_list[j], weight=1)

    # Filter graph nodes by calculated min degree constraint
    nodes_to_keep = [n for n, d in G.degree() if d >= min_connections]
    G = G.subgraph(nodes_to_keep)

    # Set up HTTP streaming response headers for CSV distribution
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="coordination_network_edges_{timezone.now().strftime("%Y%m%d")}.csv"'
    
    writer = csv.writer(response)
    # Gephi natively matches 'Source', 'Target', and 'Weight' columns
    writer.writerow(['Source', 'Target', 'Type', 'Weight'])
    
    for u, v, data in G.edges(data=True):
        writer.writerow([u, v, 'Undirected', data.get('weight', 1)])
        
    return response

def export_network_csv(request):
    """Export coordination network data as Gephi-ready CSV files"""
    import csv
    from django.http import HttpResponse
    from django.db.models import Count
    
    min_connections = int(request.GET.get('min_connections', 2))
    posts = ProcessedPost.objects.filter(is_election_related=True)
    coordination_groups = get_coordination_groups(posts, min_accounts=min_connections, max_groups=15)
    
    # Create HTTP response with CSV content type
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="coordination_network_edges.csv"'
    
    writer = csv.writer(response)
    # Write header
    writer.writerow(['Source', 'Target', 'Weight', 'Type', 'Id'])
    
    edge_id = 1
    for group in coordination_groups:
        # Get all accounts in this coordination group
        accounts = group.get('accounts', [])
        
        # Create edges between all pairs of accounts in the group
        for i in range(len(accounts)):
            for j in range(i+1, len(accounts)):
                source = accounts[i]
                target = accounts[j]
                weight = group.get('post_count', 1)
                
                writer.writerow([
                    source,
                    target,
                    weight,
                    'Undirected',
                    f'edge_{edge_id}'
                ])
                edge_id += 1
    
    return response


def export_network_nodes_csv(request):
    """Export nodes CSV for Gephi"""
    import csv
    from django.http import HttpResponse
    from django.db.models import Count
    
    min_connections = int(request.GET.get('min_connections', 2))
    posts = ProcessedPost.objects.filter(is_election_related=True)
    coordination_groups = get_coordination_groups(posts, min_accounts=min_connections, max_groups=15)
    
    # Collect all unique accounts
    all_accounts = {}
    for group in coordination_groups:
        for account in group.get('accounts', []):
            if account not in all_accounts:
                all_accounts[account] = {
                    'post_count': 0,
                    'group_count': 0,
                    'platforms': set()
                }
            all_accounts[account]['post_count'] += group.get('post_count', 0)
            all_accounts[account]['group_count'] += 1
            # Add platforms if available
            if 'platforms' in group:
                all_accounts[account]['platforms'].update(group['platforms'])
    
    # Create HTTP response
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="coordination_network_nodes.csv"'
    
    writer = csv.writer(response)
    # Write header - matching Gephi format
    writer.writerow(['Id', 'Label', 'Post Count', 'Group Count', 'Platforms', 'Type'])
    
    for account, data in all_accounts.items():
        platforms = ', '.join(data['platforms']) if data['platforms'] else 'Unknown'
        writer.writerow([
            account,
            account,  # Label same as Id
            data['post_count'],
            data['group_count'],
            platforms,
            'Account'
        ])
    
    return response

# === GEPHI CSV EXPORT FUNCTIONS ===
def _get_coordination_edges():
    """Helper to extract coordination edges based on identical text."""
    # Find texts posted by multiple accounts
    coordinated_texts = ProcessedPost.objects.filter(is_election_related=True) \
        .values('original_text') \
        .annotate(account_count=Count('account_id', distinct=True)) \
        .filter(account_count__gte=2) \
        .filter(original_text__isnull=False) \
        .exclude(original_text='')
    
    edges = []
    nodes_dict = {} # account -> {'originated': 0, 'amplified': 0}
    
    for text_group in coordinated_texts:
        text = text_group['original_text']
        # Skip very short texts to avoid noise
        if len(text.strip()) < 20:
            continue
            
        # Get all posts for this text, ordered by timestamp
        posts = list(ProcessedPost.objects.filter(original_text=text)
                     .order_by('timestamp_share')
                     .values('account_id', 'timestamp_share', 'url', 'platform'))
        
        if not posts:
            continue
            
        # The first poster is the Source (Originator)
        source_account = posts[0]['account_id']
        first_date = posts[0]['timestamp_share']
        url = posts[0]['url']
        platform = posts[0]['platform']
        
        # Initialize nodes
        if source_account not in nodes_dict:
            nodes_dict[source_account] = {'originated': 0, 'amplified': 0}
        nodes_dict[source_account]['originated'] += 1
        
        # The rest are Targets (Amplifiers)
        for post in posts[1:]:
            target_account = post['account_id']
            if target_account == source_account:
                continue # Skip self-loops
                
            if target_account not in nodes_dict:
                nodes_dict[target_account] = {'originated': 0, 'amplified': 0}
            nodes_dict[target_account]['amplified'] += 1
            
            edges.append({
                'source': source_account,
                'target': target_account,
                'tweet': text,
                'date': first_date.isoformat() if hasattr(first_date, 'isoformat') else (first_date.strftime('%Y-%m-%d %H:%M:%S') if first_date else ''),
                'platform': platform,
                'url': url or ''
            })
            
    return edges, nodes_dict

def export_gephi_nodes_csv(request):
    """Export Nodes CSV for Gephi with source/amplifier classification"""
    min_connections = int(request.GET.get('min_connections', 2))
    posts = ProcessedPost.objects.filter(is_election_related=True)
    coordination_groups = get_coordination_groups(posts, min_accounts=min_connections, max_groups=15)
    
    # Collect all unique accounts with their roles
    all_accounts = {}
    for group in coordination_groups:
        # Get posts for this coordination group
        text = group.get('text_sample', '')
        if not text:
            continue
            
        account_posts = posts.filter(original_text=text).order_by('timestamp_share')
        
        # First poster is Source, rest are Amplifiers
        for idx, post in enumerate(account_posts):
            username = clean_username(post.account_id)
            if username and len(username) > 2:
                if username not in all_accounts:
                    all_accounts[username] = {
                        'post_count': 0,
                        'group_count': 0,
                        'platforms': set(),
                        'is_source': False,
                        'is_amplifier': False,
                        'sub_narrative': group.get('sub_narrative', 'General Coordination')
                    }
                
                all_accounts[username]['post_count'] += 1
                all_accounts[username]['group_count'] += 1
                
                if post.platform:
                    all_accounts[username]['platforms'].add(post.platform)
                
                # First poster in time order is Source
                if idx == 0:
                    all_accounts[username]['is_source'] = True
                else:
                    all_accounts[username]['is_amplifier'] = True
    
    # Create HTTP response
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="gephi_nodes.csv"'
    writer = csv.writer(response)
    
    # Write header
    writer.writerow(['Id', 'Label', 'Post Count', 'Group Count', 'Platforms', 
                    'Node Type', 'Sub Narrative', 'Is Source', 'Is Amplifier'])
    
    for account, data in all_accounts.items():
        platforms = ', '.join(data['platforms']) if data['platforms'] else 'Unknown'
        node_type = 'Source' if data['is_source'] and not data['is_amplifier'] else \
                   'Amplifier' if data['is_amplifier'] else 'Both'
        
        writer.writerow([
            account,
            account,
            data['post_count'],
            data['group_count'],
            platforms,
            node_type,
            data['sub_narrative'],
            data['is_source'],
            data['is_amplifier']
        ])
    
    return response


def export_gephi_edges_csv(request):
    """Export Edges CSV for Gephi with tweet content filtered explicitly for Ethiopian context"""
    min_connections = int(request.GET.get('min_connections', 2))
    

    posts = ProcessedPost.objects.filter(
        is_election_related=True
    ).filter(
        Q(original_text__icontains='Ethiopia') |
        Q(original_text__icontains='ኢትዮጵያ')
    )
    
    coordination_groups = get_coordination_groups(posts, min_accounts=min_connections, max_groups=15)
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="gephi_edges.csv"'
    writer = csv.writer(response)
    
    # Updated header: Source, Target, Timestamp, Text, URL
    writer.writerow(['Source', 'Target', 'Timestamp', 'Text', 'URL'])
    
    for group in coordination_groups:
        accounts = group.get('accounts', [])
        text_sample = group.get('text_sample', '')
        
        # Get timestamp and URL from sample posts
        timestamp = ''
        url = ''
        if group.get('sample_posts_with_urls'):
            first_post = group['sample_posts_with_urls'][0]
            timestamp = first_post.get('timestamp', '')
            url = first_post.get('url', '')
            
        # Clean tweet text for seamless CSV importing
        tweet_clean = text_sample.replace('\n', ' ').replace('\r', '').replace('"', '""').strip()
        
        # Create edges between all pairs
        for i in range(len(accounts)):
            for j in range(i+1, len(accounts)):
                writer.writerow([
                    accounts[i],
                    accounts[j],
                    timestamp,
                    tweet_clean,
                    url
                ])
    return response

def export_gephi_edges_with_roles_csv(request):
    """Export Edges CSV showing Source -> Amplifier relationships"""
    min_connections = int(request.GET.get('min_connections', 2))
    posts = ProcessedPost.objects.filter(is_election_related=True)
    coordination_groups = get_coordination_groups(posts, min_accounts=min_connections, max_groups=15)
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="gephi_source_amplifier_edges.csv"'
    writer = csv.writer(response)
    
    writer.writerow(['Source Account', 'Amplifier Account', 'Weight', 'Type', 
                    'Tweet', 'Sub Narrative', 'First Post Time'])
    
    for group in coordination_groups:
        text = group.get('text_sample', '')
        if not text:
            continue
        
        # Get posts ordered by timestamp to identify source vs amplifier
        account_posts = posts.filter(original_text=text).order_by('timestamp_share')
        
        source_account = None
        first_time = None
        
        for idx, post in enumerate(account_posts):
            username = clean_username(post.account_id)
            if not username or len(username) < 2:
                continue
            
            if idx == 0:
                # This is the source
                source_account = username
                first_time = post.timestamp_share.isoformat() if post.timestamp_share else ''
            else:
                # This is an amplifier
                tweet_clean = text[:200].replace('\n', ' ').replace('\r', '').replace('"', '""')
                
                writer.writerow([
                    source_account,
                    username,
                    1,
                    'Directed',
                    f'"{tweet_clean}"',
                    group.get('sub_narrative', 'General Coordination'),
                    first_time
                ])
    
    return response
    
def export_gephi_edges_tweets_csv(request):
    """Export Edges + Tweets CSV for Gephi"""
    edges, _ = _get_coordination_edges()
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="gephi_edges_tweets.csv"'
    writer = csv.writer(response)
    # Updated header: removed Platform, keeping Source, Target, Tweet, Date, URL
    writer.writerow(['Source', 'Target', 'Tweet', 'Date', 'URL'])
    for edge in edges:
        # Truncate long tweets to prevent CSV issues
        tweet_text = edge['tweet'][:200].replace('\n', ' ').replace('\r', '')
        writer.writerow([
            edge['source'], 
            edge['target'], 
            tweet_text,
            edge['date'],
            edge['url']
        ])
    return response
    
def export_network_edges_with_tweets(request):
    """Export edges with tweet content for Gephi (similar to notebook)"""
    import csv
    from django.http import HttpResponse
    
    min_connections = int(request.GET.get('min_connections', 2))
    posts = ProcessedPost.objects.filter(is_election_related=True)
    coordination_groups = get_coordination_groups(posts, min_accounts=min_connections, max_groups=15)
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="coordination_edges_with_tweets.csv"'
    
    writer = csv.writer(response)
    # Write header matching the notebook format
    writer.writerow(['Source', 'Target', 'Tweet', 'Post Count', 'Platforms', 'Timestamp'])
    
    for group in coordination_groups:
        accounts = group.get('accounts', [])
        text_sample = group.get('text_sample', '')
        post_count = group.get('post_count', 0)
        platforms = ', '.join(group.get('platforms', [])) if 'platforms' in group else 'Unknown'
        
        # Get timestamp from sample posts if available
        timestamp = ''
        if group.get('sample_posts_with_urls'):
            first_post = group['sample_posts_with_urls'][0]
            timestamp = first_post.get('timestamp', '')
        
        # Create edges between all pairs
        for i in range(len(accounts)):
            for j in range(i+1, len(accounts)):
                writer.writerow([
                    accounts[i],
                    accounts[j],
                    text_sample[:200],  # Truncate long tweets
                    post_count,
                    platforms,
                    timestamp
                ])
    
    return response
    
def export_complete_network_csv(request):
    """Export COMPLETE network data - both nodes and edges with full metadata"""
    import csv
    from django.http import HttpResponse
    from django.utils import timezone
    
    min_connections = int(request.GET.get('min_connections', 2))
    
    # Get all election-related posts
    posts = ProcessedPost.objects.filter(is_election_related=True)
    
    # Get coordination groups
    coordination_groups = get_coordination_groups(
        posts, 
        min_accounts=min_connections, 
        max_groups=50  # Increased to capture more groups
    )
    
    # Build comprehensive network data
    all_nodes = {}
    all_edges = []
    
    # Process ALL coordination groups
    for group_idx, group in enumerate(coordination_groups):
        accounts = group.get('accounts', [])
        text_sample = group.get('text_sample', '')
        post_count = group.get('post_count', 0)
        platforms = group.get('platforms', [])
        coordination_type = group.get('coordination_type', 'Unknown')
        sub_narrative = group.get('sub_narrative', 'General Coordination')
        bot_count = group.get('bot_count', 0)
        bot_percentage = group.get('bot_percentage', 0)
        
        # Get timestamp from sample posts
        timestamp = ''
        url = ''
        if group.get('sample_posts_with_urls'):
            first_post = group['sample_posts_with_urls'][0]
            timestamp = first_post.get('timestamp', '')
            url = first_post.get('url', '')
        
        # Add nodes with metadata
        for account in accounts:
            if account not in all_nodes:
                all_nodes[account] = {
                    'account_id': account,
                    'group_count': 0,
                    'total_posts': 0,
                    'platforms': set(),
                    'coordination_types': set(),
                    'sub_narratives': set(),
                    'is_bot': False,
                    'bot_percentage': 0,
                    'first_seen': timestamp,
                    'sample_url': url
                }
            
            all_nodes[account]['group_count'] += 1
            all_nodes[account]['total_posts'] += post_count
            all_nodes[account]['platforms'].update(platforms)
            all_nodes[account]['coordination_types'].add(coordination_type)
            all_nodes[account]['sub_narratives'].add(sub_narrative)
            
            # Mark as bot if bot percentage is high
            if bot_percentage >= 50:
                all_nodes[account]['is_bot'] = True
                all_nodes[account]['bot_percentage'] = max(
                    all_nodes[account]['bot_percentage'], 
                    bot_percentage
                )
        
        # Create edges between all pairs in this group
        for i in range(len(accounts)):
            for j in range(i + 1, len(accounts)):
                all_edges.append({
                    'source': accounts[i],
                    'target': accounts[j],
                    'weight': post_count,
                    'group_id': group_idx + 1,
                    'text_sample': text_sample[:200],
                    'timestamp': timestamp,
                    'url': url,
                    'coordination_type': coordination_type,
                    'sub_narrative': sub_narrative
                })
    
    # Create CSV response
    response = HttpResponse(content_type='text/csv')
    timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
    response['Content-Disposition'] = f'attachment; filename="complete_network_data_{timestamp}.csv"'
    
    # Write CSV with UTF-8 BOM for Excel compatibility
    response.write('\ufeff')
    writer = csv.writer(response)
    
    # Write NODES section
    writer.writerow(['=== NODES (Accounts) ==='])
    writer.writerow([
        'Account ID', 
        'Group Count', 
        'Total Posts', 
        'Platforms', 
        'Coordination Types', 
        'Sub Narratives', 
        'Is Bot', 
        'Bot Percentage', 
        'First Seen', 
        'Sample URL'
    ])
    
    for account, data in sorted(all_nodes.items()):
        writer.writerow([
            account,
            data['group_count'],
            data['total_posts'],
            ', '.join(data['platforms']),
            ', '.join(data['coordination_types']),
            ', '.join(data['sub_narratives']),
            'Yes' if data['is_bot'] else 'No',
            f"{data['bot_percentage']:.1f}%",
            data['first_seen'],
            data['sample_url']
        ])
    
    # Add separator
    writer.writerow([])
    writer.writerow(['=== EDGES (Connections) ==='])
    writer.writerow([
        'Source', 
        'Target', 
        'Weight', 
        'Group ID', 
        'Text Sample', 
        'Timestamp', 
        'URL', 
        'Coordination Type', 
        'Sub Narrative'
    ])
    
    # Write EDGES section
    for edge in all_edges:
        # Clean text for CSV
        text_clean = edge['text_sample'].replace('\n', ' ').replace('\r', '').replace('"', '""')
        writer.writerow([
            edge['source'],
            edge['target'],
            edge['weight'],
            edge['group_id'],
            text_clean,
            edge['timestamp'],
            edge['url'],
            edge['coordination_type'],
            edge['sub_narrative']
        ])
    
    # Add summary
    writer.writerow([])
    writer.writerow(['=== NETWORK SUMMARY ==='])
    writer.writerow(['Total Nodes (Accounts):', len(all_nodes)])
    writer.writerow(['Total Edges (Connections):', len(all_edges)])
    writer.writerow(['Total Coordination Groups:', len(coordination_groups)])
    writer.writerow(['Export Date:', timezone.now().strftime('%Y-%m-%d %H:%M:%S UTC')])
    
    messages.success(request, f"Exported complete network: {len(all_nodes)} nodes, {len(all_edges)} edges")
    return response
   
def extract_sub_narrative(text_sample):
    """
    Extract the primary sub-narrative from a text sample using the CONFIG lexicon.
    1. Uses 1,000+ terms across Amharic, Oromo, and English
    2. Leverages the existing scan_text_for_lexicon_terms function
    3. Counts category matches to determine the dominant narrative
    """
    if not text_sample or len(text_sample.strip()) < 10:
        return "General News"
    
    # Scan the text against the full lexicon
    matches = scan_text_for_lexicon_terms(text_sample)
    
    if not matches:
        return "General News"
    
    # Count matches by category
    category_counts = Counter([m['category'] for m in matches])
    
    # Get the top category
    top_category = category_counts.most_common(1)[0][0]
    
    # Map lexicon categories to readable sub-narrative names
    narrative_map = {
        'ethnic_identity': 'Ethnic Tensions',
        'political_groups': 'Political Conflict',
        'violence_incitement': 'Violence & Incitement',
        'dehumanizing': 'Dehumanizing Language',
        'election_governance': 'Election & Governance',
        'foreign_interference': 'Foreign Interference',
        'religious_cultural': 'Religious/Cultural',
        'gender_misogynistic': 'Gender-Based Attacks',
        'discriminatory_homophobic': 'Discrimination',
        'socio_economic_caste': 'Socio-Economic'
    }
    
    return narrative_map.get(top_category, 'General News')
    
def format_ttp_input(coordination_groups: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Format coordination groups into the input structure expected by the Gemma model
    Following the Phase 3 schema from the notebook
    """
    # Aggregate all posts from coordination groups
    all_posts = []
    platforms = set()
    account_ids = set()
    
    for group in coordination_groups:
        platforms.update(group.get('platforms', []))
        account_ids.update(group.get('accounts', []))
        
        # Extract sample posts if available
        sample_posts = group.get('sample_posts_with_urls', [])
        for post in sample_posts:
            post_data = {
                "account": post.get('username', ''),
                "platform": post.get('platform', ''),
                "post": post.get('text_preview', '')[:200], 
                "primary_url": post.get('url', ''),
                "publication_date": post.get('timestamp', ''),
                "hashtags": [],  # Extract if available
                "domain": "",  # Extract from URL if needed
            }
            all_posts.append(post_data)
    
    # Build the input structure matching Phase 3 schema
    input_data = {
        "network_id": "ethiopia_election_monitor",
        "case_title": "Coordination Detection",
        "source_organization": "Ethiopia Election Monitor",
        "geography": ["Ethiopia"],
        "account_count": len(account_ids),
        "platforms": list(platforms),
        "signal_totals": {
            "coLink": 0,
            "coText": len(coordination_groups),
            "nearPosting": 0,
            "domainBurst": 0,
            "lexicalFlood": 0,
            "crossPost": 0,
            "personaCue": 0,
            "plagiarism": 0,
            "replyTarget": 0,
            "repostChain": 0
        },
        "manipulation_share": 0.0,
        "shared_manipulation_categories": [],
        "top_domains": [],
        "top_urls": [],
        "top_hashtags": [],
        "evidence_posts": all_posts[:3],  
        "allowed_techniques": [
            "T0049", "T0049.002", "T0049.003", "T0049.005",
            "T0016", "T0060",
            "T0119", "T0119.001", "T0119.002",
            "T0097.102", "T0097.202",
            "T0143.002", "T0143.003",
            "T0149.003",
            "T0084.002"
        ]
    }
    
    return input_data
    
def detect_hate_speech_llm_enhanced(text: str) -> dict:
    """
    Enhanced LLM detection that returns category label and accurate context.
    Not limited to the 20 AFRO-XLMR categories.
    """
    if not text or len(text.strip()) < 20:
        return {
            'is_hate_speech': False,
            'confidence': 0.0,
            'category': 'neutral',
            'explanation': 'Text too short for analysis',
            'severity': 'low'
        }
    
    # Flexible categories that LLM can choose from (not limited to 20)
    category_options = """
    - Ethnic Slur
    - Religious Hate
    - Gender-Based Violence/Misogyny
    - Political Incitement
    - Violence/Threats
    - Dehumanization
    - Xenophobia
    - Homophobic/LGBTQ+ Hate
    - Disability Hate
    - Class/Caste Discrimination
    - Conspiracy Theory
    - Misinformation/Disinformation
    - Harassment/Bullying
    - Hate Speech (General)
    - Neutral/Not Hate Speech
    - Other (specify in explanation)
    """
    
    # SAFE PROMPT CONSTRUCTION: Uses string concatenation to avoid editor syntax highlighting bugs
    prompt = (
        "You are an elite expert content moderator specialized in Ethiopian political discourse, Amharic sociolinguistics, ethnic conflict dynamics, and dangerous speech.\n\n"
        "SYSTEM ROLE & GOAL:\n"
        "Your task is to analyze social media text from Ethiopia and identify whether it contains hate speech, ethnic targeting, dehumanization, or dangerous atrocity claims.\n\n"
        f"TEXT TO ANALYZE:\n\"{text}\"\n\n"
        "ALLOWED CATEGORIES (Select the single best match):\n"
        f"{category_options}\n\n"
        "CRITICAL ETHIOPIAN DISCOURSE & MODERATION RULES:\n"
        "1. ATROCITY ALLEGATIONS & SEXUAL VIOLENCE:\n"
        "   - Posts alleging massacres, gang rape (e.g., \"መደፈር\", \"ለሶስት እንደፈሯት\"), or war crimes by ethnically labeled armed forces (e.g., \"የኦሮሙማ ወታሮች\", \"የአምሃ ኃይሎች\") must NEVER be classified as neutral news.\n"
        "   - They are HIGH-RISK INCITEMENT / DANGEROUS SPEECH (Severity: HIGH or CRITICAL) because they are weaponized to drive immediate offline ethnic retaliation.\n\n"
        "2. COLLECTIVE ETHNIC GUILT & ELITE CAPTURE:\n"
        "   - Framing an entire ethnic group as \"looters\", \"thieves\", or \"monopolizing state assets\" (e.g., claiming \"all looters come from one family/ethnicity\" or \"Oromo/Oromuma took over all banks\") is NOT fair economic critique. It is Collective Guilt & Ethnic Generalization (Severity: HIGH).\n\n"
        "3. ETHNIC SLURS & POLITICAL DOGWHISTLES:\n"
        "   - Slurs & Subservient Terms: Words like \"ጋላ\" (Galla) or pairing ethnic groups with \"አሽከሮች\" (slaves/lackeys) are DEHUMANIZING SLURS (Severity: HIGH/CRITICAL).\n"
        "   - Manipulated Names: Words like \"ብልግና\" (vulgarity) used to insult \"ብልጽግና\" (Prosperity Party) are political insults.\n"
        "   - Opportunist Labels: Terms like \"ባለጊዜዎቹ\" used to dismiss an entire ethnic group as temporary illegitimate rulers carry targeted ethnic hostility.\n\n"
        "4. FAIR CRITIQUE vs. HATE SPEECH:\n"
        "   - FAIR CRITIQUE: Criticizing policy, economic inflation, or named state officials on performance grounds (e.g., \"The bank president mismanaged funds\").\n"
        "   - HATE SPEECH: Attributing institutional failure or corruption to an entire ethnic identity or cultural concept (e.g., \"Oromuma is looting the country\").\n\n"
        "RESPONSE FORMAT:\n"
        "You must return your output strictly in JSON format. Do not add markdown outside the JSON block.\n\n"
        "{\n"
        "    \"category\": \"Chosen category from list\",\n"
        "    \"explanation\": \"Detailed step-by-step reasoning grounded in the specialized rules above\",\n"
        "    \"severity\": \"low|medium|high|critical\",\n"
        "    \"confidence\": 0.0-1.0,\n"
        "    \"is_hate_speech\": true|false,\n"
        "    \"target_group\": \"Name of targeted group or 'none'\",\n"
        "    \"harmful_elements\": [\"list\", \"of\", \"specific\", \"harmful\", \"phrases/elements\"]\n"
        "}"
    )
    
    try:
        response = safe_llm_call(prompt, max_tokens=600)
        if not response:
            return {
                'is_hate_speech': False,
                'confidence': 0.0,
                'category': 'error',
                'explanation': 'LLM failed to respond',
                'severity': 'low'
            }
        
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            return {
                'is_hate_speech': result.get('is_hate_speech', False),
                'confidence': float(result.get('confidence', 0.0)),
                'category': result.get('category', 'uncategorized'),
                'explanation': result.get('explanation', ''),
                'severity': result.get('severity', 'low'),
                'target_group': result.get('target_group', ''),
                'harmful_elements': result.get('harmful_elements', [])
            }
        else:
            return {
                'is_hate_speech': 'true' in response.lower() and 'is_hate_speech": true' in response.lower(),
                'confidence': 0.5,
                'category': 'uncategorized',
                'explanation': response,
                'severity': 'medium'
            }
    except Exception as e:
        logger.error(f"Enhanced LLM detection failed: {e}")
        return {
            'is_hate_speech': False,
            'confidence': 0.0,
            'category': 'error',
            'explanation': f'Analysis failed: {str(e)}',
            'severity': 'low'
        }
        
def clean_username(raw_name):
    if not raw_name or pd.isna(raw_name):
        return "Unknown"
    
    # Convert to string and preserve the full name
    name = str(raw_name).strip()
    
    # Use flags=re.IGNORECASE instead of inline (?i)
    # This prevents the Python 3.11+ "global flags not at start" crash
    name = re.sub(r'\s+(name|source|nan|none)$', '', name, flags=re.IGNORECASE).strip()
    
    # Remove leading/trailing special characters
    name = re.sub(r'^[@\s]+|[\s@]+$', '', name)
    
    # If name is empty after cleaning, return Unknown
    if not name or name.lower() in ['nan', 'none', '-', '', 'unknown']:
        return "Unknown"
    
    return name
   
def normalize_platform(platform_name):
    """Normalize platform names to consistent format"""
    if not platform_name:
        return "Unknown"
    p = str(platform_name).lower().strip()
    if p in ['x', 'twitter', 't.co', 'x.com', 'twitter source', 'twitter.com']:
        return 'X'
    elif p in ['facebook', 'fb.watch', 'facebook.com', 'fb']:
        return 'Facebook'
    elif p in ['telegram', 't.me', 'tg']:
        return 'Telegram'
    elif p in ['tiktok', 'tik tok', 'tik-tok']:
        return 'TikTok'
    elif p in ['media', 'news', 'news/media', 'civicsignal']:
        return 'Media'
    elif p in ['youtube', 'youtu.be', 'yt']:
        return 'YouTube'
    elif p in ['instagram', 'insta', 'ig']:
        return 'Instagram'
    return platform_name.title()  
   
def get_queryset(self):
    qs = ProcessedPost.objects.all()
    start = self.request.GET.get('start_date')
    end = self.request.GET.get('end_date')
    if start and end:
        qs = qs.filter(timestamp_share__range=[start, end])
    else:
        # Default: last 30 days
        from django.utils import timezone
        qs = qs.filter(timestamp_share__gte=timezone.now() - timezone.timedelta(days=30))
    return qs.order_by('-timestamp_share')

# Constants
MAX_UPLOAD_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB


def dynamically_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scans an arbitrary DataFrame's columns and maps custom/generic headers 
    to our standardized schema names:
    - account_id
    - original_text
    - url
    - timestamp_share
    - platform
    """
    # Create mapping dictionary
    mapping = {}

    # Common aliases grouped by our target database/pipeline fields
    aliases = {
        'account_id': ['account_id', 'account', 'username', 'user', 'author', 'handle', 'sender', 'screen_name'],
        'original_text': ['original_text', 'text', 'content', 'body', 'tweet', 'post', 'message', 'caption', 'description', 'comment'],
        'url': ['url', 'link', 'uri', 'href', 'permalink', 'post_url'],
        'timestamp_share': ['timestamp_share', 'timestamp', 'date', 'created_at', 'published', 'time', 'datetime', 'published_date'],
        'platform': ['platform', 'source', 'channel', 'network', 'social_network']
    }

    # Match actual columns with aliases case-insensitively
    for target_field, alias_list in aliases.items():
        for col_name in df.columns:
            clean_col = str(col_name).lower().strip()
            if clean_col in alias_list:
                mapping[col_name] = target_field
                break  # Pick the first matching alias and move on to next target

    if mapping:
        df = df.rename(columns=mapping)

    # --- Robust Fallbacks ---
    
    # 1. Fallback for account_id
    if 'account_id' not in df.columns:
        df['account_id'] = 'Unknown'

    # 2. Fallback for original_text (Crucial: Try to find a high-length text column if no headers matched)
    if 'original_text' not in df.columns:
        string_cols = df.select_dtypes(include=['object', 'string']).columns
        if len(string_cols) > 0:
            # Fallback to the text-type column with the highest average string length
            fallback_col = max(string_cols, key=lambda c: df[c].astype(str).str.len().mean())
            df = df.rename(columns={fallback_col: 'original_text'})
        else:
            df['original_text'] = ''

    # 3. Fallback for other expected keys
    if 'url' not in df.columns:
        df['url'] = ''
    if 'timestamp_share' not in df.columns:
        df['timestamp_share'] = timezone.now()
    if 'platform' not in df.columns:
        df['platform'] = 'Generic CSV'

    return df


@never_cache  
def dashboard_view(request):
    """Main Dashboard View with Sidebar Upload and Stats Reporting"""
    
    # 1. Handle File Uploads via POST
    if request.method == 'POST' and request.FILES.getlist('files'):
        platform_type = request.POST.get('platform', 'generic')
        uploaded_files = request.FILES.getlist('files')
        
        stats = {
            'files_count': 0,
            'total_rows': 0,
            'saved': 0,
            'duplicates': 0
        }
        
        for f in uploaded_files:
            # --- FILE SIZE VALIDATION ---
            if f.size > MAX_UPLOAD_SIZE_BYTES:
                messages.error(
                    request, 
                    f"The file '{f.name}' exceeds our 500MB limit. "
                    "Please optimize, compress, or split your CSV before uploading."
                )
                return redirect(request.POST.get('next', 'home'))

            try:
                # Brandwatch needs specific handling (skiprows=6 for metadata)
                if platform_type == 'brandwatch':
                    df = pd.read_csv(f, sep=',', low_memory=False, skiprows=6, on_bad_lines='skip')
                else:
                    df = load_data_robustly(f) 
                
                stats['total_rows'] += len(df)
                
                # Check if it is a known/hardcoded pipeline
                known_platforms = ['meltwater', 'tiktok', 'openmeasure', 'brandwatch', 'civicsignals']
                
                if platform_type in known_platforms:
                    # Leverage your existing fixed mappings
                    if platform_type == 'meltwater':
                        processed_df = combine_social_media_data(meltwater_df=df)
                    elif platform_type == 'tiktok':
                        processed_df = combine_social_media_data(tiktok_df=df)
                    elif platform_type == 'openmeasure':
                        processed_df = combine_social_media_data(openmeasures_df=df)
                    elif platform_type == 'brandwatch':
                        processed_df = combine_social_media_data(brandwatch_df=df)
                    else:
                        processed_df = combine_social_media_data(civicsignals_df=df)
                else:
                    # Dynamic Fallback pipeline: Handles ANY arbitrary or manually selected CSV dataset
                    processed_df = dynamically_map_columns(df)

                # ONE CLEAN LOOP
                for _, row in processed_df.iterrows():
                    cid = row.get('content_id')
                    
                    # Generate a unique content ID fallback if empty to prevent collision
                    if not cid:
                        text_hash = hashlib.md5(str(row.get('original_text', '')).encode('utf-8')).hexdigest()
                        cid = f"gen_{text_hash[:16]}"
                    
                    # Check for duplicates before saving
                    if ProcessedPost.objects.filter(content_id=cid).exists():
                        stats['duplicates'] += 1
                        continue
                        
                    source_name = row.get('source_dataset', platform_type)
                    source_obj, _ = DataSource.objects.get_or_create(name=source_name)
            
                    ProcessedPost.objects.create(
                        account_id=str(row.get('account_id', 'Unknown'))[:100],
                        content_id=cid,
                        original_text=str(row.get('original_text', '')),
                        url=row.get('url') or row.get('URL') or row.get('link') or row.get('Link') or '',
                        platform=row.get('platform', platform_type.title()),
                        timestamp_share=parse_timestamp_robust(row.get('timestamp_share')),
                        source_dataset=source_obj,
                        is_election_related=is_election_related(str(row.get('original_text', '')))
                    )
                    stats['saved'] += 1
                
                stats['files_count'] += 1
            except Exception as e:
                logger.error(f"Upload error processing '{f.name}': {e}")
                messages.error(request, f"Error processing '{f.name}'. Ensure it's a valid CSV format.")

        detail_msg = (
            f"<strong>Data Upload Details:</strong><br>"
            f"• Source: {platform_type.title()}<br>"
            f"• Files processed: {stats['files_count']}<br>"
            f"• Rows analyzed: {stats['total_rows']}<br>"
            f"• <strong>New unique posts: {stats['saved']}</strong><br>"
            f"• Duplicates ignored: {stats['duplicates']}"
        )
        messages.success(request, detail_msg)
        return redirect(request.POST.get('next', 'home'))

    # 2. Page Load Logic (GET)
    # Check for hard refresh query param
    if request.GET.get('refresh'):
        logger.info("Performing hard refresh of election metrics")

    all_posts = ProcessedPost.objects.all().order_by('-timestamp_share')
    
    summaries = get_ethiopia_summaries(all_posts)
    coordination = get_coordination_groups(all_posts)
    
    context = {
        'tabs': [
            {'name': 'Overview', 'url_name': 'home', 'icon': '📊'},
            {'name': 'Narratives', 'url_name': 'narratives', 'icon': '🗣️'},
            {'name': 'Coordination', 'url_name': 'networks', 'icon': '🕸️'},
        ],
        'active_tab': 'home',
        'summaries': summaries,
        'coordination': coordination,
        'total_posts': all_posts.count(),
    }
    
    return render(request, 'dashboard.html', context)
    

# Compile patterns once, reuse them forever
_COMPILED_PATTERNS = {}

def get_cached_pattern(term, language):  # 🔥 FIXED: Removed underscore
    """Get or compile a regex pattern, caching it to prevent memory spikes."""
    cache_key = f"{term}_{language}"
    
    if cache_key not in _COMPILED_PATTERNS:
        try:
            if language == "amharic" or re.match(r'^[\u1200-\u137F]+$', term):
                pattern = r'(?<![\u1200-\u137F])' + re.escape(term) + r'(?![\u1200-\u137F])'
            else:
                pattern = r'\b' + re.escape(term) + r'\b'
            
            _COMPILED_PATTERNS[cache_key] = re.compile(pattern, re.IGNORECASE)
        except re.error:
            # If a pattern is invalid, cache None so we don't try to compile it again
            _COMPILED_PATTERNS[cache_key] = None
            
    return _COMPILED_PATTERNS[cache_key]

def scan_text_for_lexicon_terms(text, category_filter=None):
    """
    FAST PATH: Only regex matching - NO LLM calls during page load.
    Combines CONFIG lexicon with Database lexicon terms.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    text_lower = text.lower()
    matches = []
    
    # Use the combined lexicon (CONFIG + Database)
    lexicon = get_combined_lexicon()
    categories_to_check = category_filter if category_filter else lexicon.keys()
    
    # Neutral context indicators
    neutral_indicators = [
        'regional state', 'development', 'news', 'media', 'platform',
        'solar', 'water access', 'farmers', 'installed', 'modernization',
        'studied', 'experience', 'applied', 'wrote', 'seen',
        'diaspora', 'followers', 'condemns', 'urges', 'respect',
        'sovereignty', 'ministry', 'foreign affairs'
    ]
    is_neutral_context = any(indicator in text_lower for indicator in neutral_indicators)
    
    for category in categories_to_check:
        if category not in lexicon:
            continue
        for term, metadata in lexicon[category].items():
            if len(term.strip()) < 2 and not re.match(r'^[\u1200-\u137F]+$', term):
                continue
            if is_neutral_context and metadata.get('severity') == 'low':
                continue
            if term.lower() in ['amhara', 'oromo', 'tigray', 'somali', 'afar'] and metadata.get('severity') == 'low':
                has_hate_terms = any(
                    other_term in text_lower
                    for other_cat, other_terms in lexicon.items()
                    for other_term, other_meta in other_terms.items()
                    if other_meta.get('severity') in ['high', 'critical']
                )
                if not has_hate_terms:
                    continue
            
            # This call now matches the function name above
            pattern = get_cached_pattern(term, metadata.get("language", "english"))
            if pattern and pattern.search(text_lower):
                matches.append({
                    'term': term,
                    'category': category,
                    'severity': metadata.get('severity', 'medium'),
                    'target_entity': metadata.get('target_entity', ''),
                    'language': metadata.get('language', 'english'),
                    'source': 'Lexicon'
                })
    return matches

def auto_save_important_llm_terms(llm_terms):
    """
    Automatically save high-confidence LLM-discovered terms to database.
    """
    from dashboard.models import LexiconTerm
    
    saved_count = 0
    for term_data in llm_terms:
        # Only save high-severity terms with good confidence
        if term_data.get('severity') in ['high', 'critical'] and term_data.get('confidence', 0) >= 0.6:
            
            # 1. Normalize language to match system values
            detected_language = term_data.get('language', 'Unknown')
            if detected_language:
                detected_language = detected_language.strip().title()
                # Map common variations to standard values
                language_map = {
                    'Amharic': 'amharic',
                    'Oromo': 'oromo', 
                    'Tigrinya': 'tigrinya',
                    'English': 'english',
                    'Somali': 'somali'
                }
                # Use mapped value or fallback to lowercase version
                normalized_language = language_map.get(detected_language, detected_language.lower())
            else:
                normalized_language = 'unknown'
            
            # 2. Save to database (get_or_create prevents duplicates)
            obj, created = LexiconTerm.objects.get_or_create(
                term=term_data['term'],
                defaults={
                    'category': term_data['category'],
                    'severity': term_data['severity'],
                    'target_entity': term_data.get('target_entity', ''),
                    'language': normalized_language,  # Uses detected language, not hardcoded
                    'is_election_related': True,
                    'is_active': True
                }
            )
            
            # 3. Log result only if it was actually new
            if created:
                saved_count += 1
                logger.info(f"✅ Auto-saved NEW term: '{term_data['term']}' (Category: {term_data['category']}, Lang: {normalized_language})")
            else:
                logger.info(f"ℹ️ Term already exists: '{term_data['term']}'")
                
    return saved_count
   
def calculate_risk_score(matches):
    """Calculate risk score based on matched terms"""
    if not matches:
        return {'score': 0, 'level': 'low', 'breakdown': {}, 'term_count': 0}
    
    scoring = CONFIG.get("risk_scoring", {})
    severity_weights = scoring.get("severity_weights", {'low': 1, 'medium': 2, 'high': 3, 'critical': 4})
    category_weights = scoring.get("category_weights", {})
    thresholds = scoring.get("risk_thresholds", {'low': 3, 'medium': 6, 'high': 10, 'critical': 15})
    
    total_score = 0
    breakdown = defaultdict(int)
    
    for match in matches:
        sev = match.get('severity', 'medium')
        cat = match.get('category', 'general')
        weight = severity_weights.get(sev, 2) * category_weights.get(cat, 1.0)
        total_score += weight
        breakdown[cat] += weight
    
    if total_score >= thresholds.get('critical', 15): level = 'critical'
    elif total_score >= thresholds.get('high', 10): level = 'high'
    elif total_score >= thresholds.get('medium', 6): level = 'medium'
    else: level = 'low'
    
    return {'score': round(total_score, 2), 'level': level, 'breakdown': dict(breakdown), 'term_count': len(matches)}


def assign_virality_tier(n):
    if n >= 500: return "Tier 4: Viral Emergency"
    elif n >= 100: return "Tier 3: High Spread"
    elif n >= 20: return "Tier 2: Moderate"
    else: return "Tier 1: Limited"


def summarize_cluster_ethiopia(texts, urls, cluster_data, min_ts, max_ts):
    """
    Generate structured IMI-style intelligence report for Ethiopia election narratives.
    Uses ONLY explicit claims from provided posts - no invention or assumption.
    """
    # Join first 50 texts for context (avoids token limits)
    joined = "\n".join([f"[{i+1}] {t}" for i, t in enumerate(texts[:50]) if t and len(t.strip()) > 10])
    
    # Add real URLs for reference
    url_context = "\nRelevant post links:\n" + "\n".join(urls[:5]) if urls else ""
    
    # Ethiopia-specific prompt 
    prompt = f"""
Generate a structured IMI intelligence report on online narratives related to the Ethiopia election.
Focus on pre and post-election tensions and emerging narratives, including:
- Allegations of political suppression or intimidation
- Electoral Commission (NEBE) corruption, bias, or procedural issues
- Economic distress, state fund misuse, or resource allocation claims
- Hate speech, ethnic targeting, tribalism, or xenophobia
- Gender-based attacks or disinformation against women candidates
- Foreign interference narratives ("Western puppet", anti-EU, China/Russia influence, etc.)
- Marginalization of minorities or regional groups (Amhara, Oromo, Tigray, Somali, etc.)
- Claims of election fraud, rigging, tally center manipulation, or result tampering
- Calls for protests, civic resistance, or boycotts
- Viral slogans, hashtags, or coordinated messaging campaigns

**Strict Instructions:**
- Only report claims **explicitly present** in the provided posts below.
- Identify **originators**: accounts that first posted the core claim (from cluster_data timestamps).
- Note **amplification**: how widely it spread (Total posts count).
- Do NOT invent, cut out, assume, extrapolate, or fact-check claims.
- Summarize clearly and concisely using simple language.

**Output Format (Use plain text, NO markdown headers or bold):**
Narrative Title: [Short, descriptive title]
Core Claim(s):
- [Claim 1, directly quoted or closely paraphrased from posts]
- [Claim 2]
- [Claim 3]
Originator(s): [Account IDs or "Unknown" if not determinable]
Amplification: [Total posts] posts across [X] unique accounts
First Detected: {min_ts}
Last Updated: {max_ts}
Language/Tone Observed: [e.g., accusatory, urgent, informational, mixed]
Sample Quotes (exact phrases from posts, max 3):
1. '[exact quote 1]'
2. '[exact quote 2]'
3. '[exact quote 3]'

Documents for analysis:
{joined}{url_context}
"""
    
    try:
        response = safe_llm_call(prompt, max_tokens=2048)
        raw_summary = response.strip() if response else ""
    except Exception as e:
        logger.warning(f"LLM summary failed: {e}")
        # Fallback: extract key phrases directly from texts
        return _fallback_summary(texts, urls, min_ts, max_ts)
    
    # Clean LLM output: remove markdown, code blocks, instruction echoes
    cleaned = re.sub(r'\*\*.*?(Instructions|strict|Output Format).*?\*\*', '', raw_summary, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'```.*?```', '', cleaned, flags=re.DOTALL)  # Remove code blocks
    cleaned = re.sub(r'###|##|#|\*\*|\*', '', cleaned)  # Remove markdown headers/bold
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Normalize whitespace
    return cleaned.strip()


def _fallback_summary(texts, urls, min_ts, max_ts):
    """Fallback when LLM fails: extract key phrases directly from posts"""
    if not texts:
        return "No content available for summarization."
    
    # Extract top keywords (simple frequency)
    from collections import Counter
    import re
    words = []
    for t in texts[:20]:
        words.extend(re.findall(r'\b[a-zA-Z]{4,}\b', t.lower()))
    top_words = [w for w, _ in Counter(words).most_common(10) if w not in {'the', 'and', 'for', 'that', 'with', 'this', 'from', 'have', 'were', 'will'}]
    
    # Get first 3 substantive sentences
    sentences = []
    for t in texts:
        sents = [s.strip() for s in t.split('.') if len(s.strip()) > 30]
        sentences.extend(sents[:2])
        if len(sentences) >= 3:
            break
    
    return f"""Narrative Title: Cluster discussing {', '.join(top_words[:3]) if top_words else 'election-related topics'}

Core Claim(s):
- Posts reference electoral processes and political developments in Ethiopia
- Content includes mentions of specific incidents, actors, or regional dynamics

Originator(s): Unknown (automated extraction)
Amplification: {len(texts)} posts
First Detected: {min_ts}
Last Updated: {max_ts}
Language/Tone Observed: Mixed tones with informational and analytical content

Sample Quotes:
1. '{sentences[0][:150]}...'
2. '{sentences[1][:150]}...'
3. '{sentences[2][:150]}...'"""


def get_ethiopia_summaries(posts_queryset, max_clusters=15):
    """
    Generates IMI-style structured summaries with Ethiopia-specific topic/tone mapping.
    Uses clustering + LLM with robust fallback when model fails.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import DBSCAN
    from collections import defaultdict, Counter
    import re
    
    all_summaries = []
    
    # Get posts with required fields
    post_data = list(posts_queryset.values(
        'id', 'original_text', 'url', 'account_id', 'platform', 'timestamp_share'
    ).filter(original_text__isnull=False).exclude(original_text='').order_by('-timestamp_share')[:2000])
    
    if len(post_data) < 20:
        return all_summaries
    
    # Filter to substantive texts
    texts = [p['original_text'] for p in post_data if p['original_text'] and len(p['original_text'].strip()) > 50]
    urls = [p['url'] for p in post_data if p.get('url') and str(p['url']).startswith('http')]
    
    if len(texts) < 20:
        return all_summaries
    
    try:
        # 1. Cluster posts by semantic similarity (DBSCAN for flexible cluster shapes)
        vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2), min_df=2)
        X = vectorizer.fit_transform(texts)
        
        # DBSCAN: eps=0.3, min_samples=2 finds tight coordination groups
        clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine').fit(X)
        labels = clustering.labels_
        
        # Group posts by cluster label (exclude noise: label=-1)
        cluster_posts = defaultdict(list)
        for idx, label in enumerate(labels):
            if label != -1:
                cluster_posts[label].append(post_data[idx])
        
        # Process top 15 clusters by size
        top_clusters = sorted(cluster_posts.items(), key=lambda x: len(x[1]), reverse=True)[:max_clusters]
        
        for cluster_id, cluster_data in top_clusters:
            if len(cluster_data) < 3:  # Skip tiny clusters
                continue
            
            # Extract data for summarization
            cluster_texts = [p['original_text'] for p in cluster_data if p['original_text']]
            cluster_urls = [p['url'] for p in cluster_data if p.get('url') and str(p['url']).startswith('http')]
            
            # Timestamps for originator tracking
            timestamps = [p['timestamp_share'] for p in cluster_data if p.get('timestamp_share')]
            if not timestamps:
                continue
                
            min_ts = min(timestamps)
            max_ts = max(timestamps)
            min_ts_str = min_ts.strftime('%Y-%m-%d') if min_ts else 'N/A'
            max_ts_str = max_ts.strftime('%Y-%m-%d') if max_ts else 'N/A'
            
            # Originators: first 5 accounts by timestamp
            sorted_by_time = sorted(cluster_data, key=lambda x: x['timestamp_share'] or datetime.max)
            originators = list(dict.fromkeys([p['account_id'] for p in sorted_by_time[:10] if p.get('account_id')]))[:5]
            originators = [str(o)[:40] for o in originators] if originators else ["Unknown"]
            
            # Amplification metrics
            total_reach = len(cluster_data)
            unique_accounts = len(set(p['account_id'] for p in cluster_data if p.get('account_id')))
            platforms = [p['platform'] for p in cluster_data if p.get('platform')]
            platform_counts = Counter(platforms)
            top_platforms = ", ".join([f"{p} ({c})" for p, c in platform_counts.most_common(3)])
            
            # === ETHIOPIA-SPECIFIC TOPIC & TONE MAPPING (from your old version) ===
            all_text_lower = ' '.join(cluster_texts[:10]).lower()
            
            topic_map = {
                'election process': ['election', 'vote', 'ballot', 'nebe', 'results', 'polling', 'tally', 'counting'],
                'ethnic dynamics': ['amhara', 'oromo', 'tigray', 'somali', 'afar', 'sidama', 'ethnic', 'tribal', 'regional'],
                'political actors': ['abiy', 'prosperity party', 'opposition', 'fano', 'government', 'parliament', 'minister'],
                'security concerns': ['conflict', 'violence', 'attack', 'militia', 'drone', 'security', 'war', 'tension'],
                'information environment': ['media', 'social media', 'disinformation', 'fake news', 'platform', 'censorship'],
                'foreign interference': ['foreign', 'international', 'au', 'un', 'eu', 'china', 'russia', 'usa', 'egypt'],
                'economic issues': ['economy', 'inflation', 'unemployment', 'poverty', 'development', 'funds', 'corruption'],
            }
            detected_topics = [t for t, kws in topic_map.items() if any(kw in all_text_lower for kw in kws)]
            
            tone_map = {
                'accusatory and urgent': ['accuse', 'blame', 'demand', 'condemn', 'outrage', 'crisis', 'emergency', 'immediate'],
                'informational and analytical': ['report', 'analysis', 'data', 'found', 'according', 'study', 'research', 'evidence'],
                'emotional and concerned': ['worried', 'concerned', 'fear', 'alarming', 'devastating', 'tragic', 'heartbreaking'],
                'critical and skeptical': ['criticize', 'fail', 'question', 'doubt', 'skeptical', 'misleading', 'manipulated'],
                'hopeful and constructive': ['hope', 'solution', 'progress', 'unity', 'peace', 'dialogue', 'reform'],
            }
            detected_tone = 'Mixed tones with informational and analytical content'
            for tone, kws in tone_map.items():
                if any(kw in all_text_lower for kw in kws):
                    detected_tone = tone.title()
                    break
            
            # === GENERATE STRUCTURED SUMMARY (IMI-style) ===
            joined = "\n".join([f"[{i+1}] {t}" for i, t in enumerate(cluster_texts[:50]) if t and len(t.strip()) > 10])
            url_context = "\nRelevant post links:\n" + "\n".join(cluster_urls[:5]) if cluster_urls else ""
            
            prompt = f"""
Generate a structured IMI intelligence report on online narratives related to the Ethiopia election.
Focus on pre and post-election tensions and emerging narratives, including:
- Allegations of political suppression or intimidation
- Electoral Commission (NEBE) corruption, bias, or procedural issues
- Economic distress, state fund misuse, or resource allocation claims
- Hate speech, ethnic targeting, tribalism, or xenophobia
- Gender-based attacks or disinformation against women candidates
- Foreign interference narratives ("Western puppet", anti-EU, China/Russia influence, etc.)
- Marginalization of minorities or regional groups (Amhara, Oromo, Tigray, Somali, etc.)
- Claims of election fraud, rigging, tally center manipulation, or result tampering
- Calls for protests, civic resistance, or boycotts
- Viral slogans, hashtags, or coordinated messaging campaigns

**Strict Instructions:**
- Only report claims **explicitly present** in the provided posts below.
- Identify **originators**: accounts that first posted the core claim (from cluster_data timestamps).
- Note **amplification**: how widely it spread (Total posts count).
- Do NOT invent, cut out, assume, extrapolate, or fact-check claims.
- Summarize clearly and concisely using simple language.

**Output Format (Use plain text, NO markdown headers or bold):**
Narrative Title: [Short, descriptive title]
Core Claim(s):
- [Claim 1, directly quoted or closely paraphrased from posts]
- [Claim 2]
- [Claim 3]
Originator(s): [Account IDs or "Unknown" if not determinable]
Amplification: [Total posts] posts across [X] unique accounts
First Detected: {min_ts_str}
Last Updated: {max_ts_str}
Language/Tone Observed: {detected_tone}
Primary Topics: {', '.join(detected_topics[:3]) if detected_topics else 'Ethiopian electoral discourse'}
Sample Quotes (exact phrases from posts, max 3):
1. '[exact quote 1]'
2. '[exact quote 2]'
3. '[exact quote 3]'

Documents for analysis:
{joined}{url_context}
"""
            
            try:
                response = safe_llm_call(prompt, max_tokens=2048)
                raw_summary = response.strip() if response else ""
            except Exception as e:
                logger.warning(f"LLM summary failed for cluster {cluster_id}: {e}")
                raw_summary = ""
            
            # Clean LLM output
            cleaned_summary = re.sub(r'\*\*.*?(Instructions|strict|Output Format).*?\*\*', '', raw_summary, flags=re.IGNORECASE | re.DOTALL)
            cleaned_summary = re.sub(r'```.*?```', '', cleaned_summary, flags=re.DOTALL)
            cleaned_summary = re.sub(r'###|##|#|\*\*|\*', '', cleaned_summary)
            cleaned_summary = re.sub(r'\n{3,}', '\n\n', cleaned_summary).strip()
            
            # === FALLBACK: If LLM fails or returns empty, use rule-based extraction ===
            if not cleaned_summary or len(cleaned_summary) < 50:
                # Extract key claims using keywords (from your old fallback logic)
                specific_claims = []
                seen_claims = set()
                claim_keywords = ['alleged', 'claims', 'reported', 'found', 'accused', 'stated',
                                  'manipulated', 'rigged', 'violence', 'attack', 'killed', 'hate',
                                  'disinformation', 'fake', 'biased', 'unfair', 'targeting', 'condemn']
                
                for text in cluster_texts[:30]:
                    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 40]
                    for sentence in sentences:
                        if sentence.lower().startswith(('http', 'read more', 'via @', 'source:', 'follow', 'rt ')):
                            continue
                        if any(kw in sentence.lower() for kw in claim_keywords):
                            claim_key = sentence[:80].lower()
                            if claim_key not in seen_claims:
                                seen_claims.add(claim_key)
                                specific_claims.append(sentence[:250] + ('...' if len(sentence) > 250 else ''))
                                if len(specific_claims) >= 3:
                                    break
                    if len(specific_claims) >= 3:
                        break
                
                claims_text = '\n'.join([f"- {c}" for c in specific_claims[:3]]) if specific_claims else \
                    "- Posts discuss electoral processes and political developments in Ethiopia\n" \
                    "- Content includes references to specific incidents and political actors\n" \
                    "- Themes include governance, security, and information dynamics"
                
                # Get sample quotes for fallback
                sample_quotes = []
                for text in cluster_texts[:3]:
                    clean = text[:150].replace("'", "'").strip()
                    if clean and len(clean) > 20:
                        sample_quotes.append(f"'{clean}...'")
                
                cleaned_summary = f"""Narrative Title: Cluster discussing {', '.join(detected_topics[:2]) if detected_topics else 'election-related topics'}

Core Claim(s):
{claims_text}

Originator(s): {', '.join(originators) if originators != ["Unknown"] else "Unknown"}
Amplification: {total_reach} posts across {unique_accounts} unique accounts
First Detected: {min_ts_str}
Last Updated: {max_ts_str}
Language/Tone Observed: {detected_tone}
Primary Topics: {', '.join(detected_topics[:3]) if detected_topics else 'Ethiopian electoral discourse'}
Sample Quotes:
{chr(10).join(sample_quotes[:3]) if sample_quotes else "1. '[No substantial quotes available]'" }"""
            
            # Filter out noise/irrelevant clusters
            summary_lower = cleaned_summary.lower()
            noise_indicators = ["no relevant claims", "no explicit claims", "insufficient data", "not applicable", "unable to determine"]
            if any(ind in summary_lower for ind in noise_indicators):
                continue
            
            # Assign virality tier
            virality = assign_virality_tier(total_reach)
            
            title_match = re.search(r'Narrative Title:\s*(.*?)(?:\n|Core Claim)', cleaned_summary, re.IGNORECASE | re.DOTALL)
            if title_match:
                narrative_title = title_match.group(1).strip()
            else:
                # Fallback: take the first line if "Narrative Title:" is missing
                first_line = cleaned_summary.split('\n')[0].strip()
                narrative_title = first_line if first_line and len(first_line) < 100 else f"Cluster #{int(cluster_id) + 1}"
            
            # Final cleanup: remove any trailing punctuation or labels that slipped through
            narrative_title = re.sub(r'\s*Core Claim.*$', '', narrative_title, flags=re.IGNORECASE).strip()

            all_summaries.append({
                'cluster_id': int(cluster_id) + 1,
                'Narrative_Title': narrative_title,  
                'Context': cleaned_summary,
                'Originators': ", ".join(originators),
                'Amplifiers_Count': unique_accounts,
                'Total_Reach': total_reach,
                'Emerging_Virality': virality,
                'Top_Platforms': top_platforms,
                'Min_TS': min_ts_str,
                'Max_TS': max_ts_str,
                'Posts_Data': cluster_data,
                'Platform_Diversity': len(platform_counts),
                'Detected_Topics': detected_topics,
                'Detected_Tone': detected_tone,
            })
        
        # Sort by reach (most amplified first)
        all_summaries.sort(key=lambda x: x['Total_Reach'], reverse=True)
        
    except Exception as e:
        logger.error(f"Narrative clustering failed: {e}")
        import traceback
        traceback.print_exc()
    
    return all_summaries[:max_clusters]
    
def extract_narrative_description(summary_text, sample_posts):
    """Generate a specific, meaningful 1-sentence narrative description from actual cluster posts"""
    
    if not sample_posts:
        return "Analyzing narrative content..."
    
    # Combine all posts in this cluster for analysis
    all_text = ' '.join([p for p in sample_posts if p and isinstance(p, str)]).lower()
    
    # Define topic keywords for Ethiopia election context
    topic_keywords = {
        'election fraud': ['rigged', 'fraud', 'stolen', 'manipulated', 'fake results', 'cheating', 'ballot', 'marked cards', 'vote cards', 'nebe'],
        'voter intimidation': ['intimidation', 'threat', 'forced', 'coerced', 'violence', 'fear', 'suppress', 'arrest'],
        'ethnic tension': ['amhara', 'oromo', 'tigray', 'somali', 'afar', 'sidama', 'ethnic', 'tribal', 'discrimination'],
        'political violence': ['kill', 'attack', 'war', 'conflict', 'militia', 'armed', 'bloodshed', 'massacre'],
        'international observation': ['observer', 'international', 'AU', 'UN', 'monitor', 'transparency', 'legitimate', 'credible'],
        'government criticism': ['government', 'authorities', 'regime', 'corrupt', 'failed', 'oppression', 'tyranny'],
        'opposition support': ['opposition', 'protest', 'resistance', 'freedom', 'democracy', 'rights', 'liberation'],
        'media manipulation': ['propaganda', 'fake news', 'disinformation', 'censorship', 'biased media', 'state media'],
        'humanitarian crisis': ['displaced', 'refugee', 'hunger', 'famine', 'aid', 'crisis', 'suffering'],
        'youth engagement': ['youth', 'young', 'students', 'next generation', 'future', 'university']
    }
    
    # Count topic matches in THIS cluster's posts
    topic_scores = {}
    for topic, keywords in topic_keywords.items():
        score = sum(1 for kw in keywords if kw in all_text)
        if score > 0:
            topic_scores[topic] = score
    
    # If we found topics, generate a specific description for THIS cluster
    if topic_scores:
        # Get top 3 topics by score for this cluster
        top_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        topic_list = [t[0] for t in top_topics]
        
        # Format the description
        if len(topic_list) == 1:
            return f"Posts discussing {topic_list[0]}."
        elif len(topic_list) == 2:
            return f"Posts discussing {topic_list[0]} and {topic_list[1]}."
        else:
            return f"Posts discussing {topic_list[0]}, {topic_list[1]}, and {topic_list[2]}."
    
    # Fallback: Extract key phrase from the most representative post in THIS cluster
    best_post = None
    for post in sample_posts:
        if post and isinstance(post, str) and len(post.strip()) > 50:
            best_post = post
            break
    
    if best_post:
        # Clean and extract first meaningful sentence
        clean = re.sub(r'http\S+|@\w+|#\w+', '', best_post).strip()
        sentences = [s.strip() for s in clean.split('.') if len(s.strip()) > 30]
        if sentences:
            return sentences[0][:200] + ('...' if len(sentences[0]) > 200 else '')
        return clean[:200] + ('...' if len(clean) > 200 else '')
    
    return "Analyzing narrative content from posts..."
    

def detect_ttps_with_gemma(coordination_groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    DISABLED: Gemma TTP model removed. Using rule-based + LLM only.
    """
    logger.info("Gemma TTP detection skipped (model removed). Using rule-based + LLM only.")
    return []
    
def _convert_techniques_to_ttp_format(techniques: List[Dict]) -> List[Dict]:
    """Convert Gemma-format techniques to view-compatible format"""
    ttps = []
    for technique in techniques:
        ttp_data = {
            'name': technique.get('technique_id', ''),
            'description': technique.get('description', ''),
            'severity': _get_ttp_severity(technique.get('technique_id', '')),
            'evidence': f"Detected via Gemma model. {technique.get('evidence', '')}",
            'confidence': technique.get('confidence', 0.8),
            'model_source': 'gemma_finetuned'
        }
        ttps.append(ttp_data)
    return ttps


    
def _get_ttp_severity(technique_id: str) -> str:
    """Map technique IDs to severity levels"""
    high_severity = ['T0049', 'T0049.002', 'T0049.003', 'T0049.005']  # Coordinated behavior
    medium_severity = ['T0119', 'T0119.001', 'T0119.002', 'T0097.102', 'T0097.202']  # Platform manipulation
    
    if technique_id in high_severity:
        return 'High'
    elif technique_id in medium_severity:
        return 'Medium'
    else:
        return 'Low'
       
def is_likely_normal_news(text):
    """Check if text appears to be normal news rather than coordinated manipulation"""
    if not text or len(text.strip()) < 20:
        return False
    
    text_lower = text.lower()
    
    # Strong indicators of legitimate news/media
    news_indicators = [
        'ambassador', 'briefing', 'press', 'spokesperson', 'ministry',
        'official', 'statement', 'announced', 'reported', 'according to',
        'development', 'progress', 'achievement', 'inauguration', 'ceremony',
        'award', 'recognition', 'honor', 'appointed', 'champion',
        'expo', 'exhibition', 'airline', 'flight', 'service',
        'diplomatic', 'bilateral', 'cooperation', 'partnership', 'agreement',
        'current affairs', 'news', 'media', 'journalist', 'reporter'
    ]
    
    # Count news indicators
    news_score = sum(1 for indicator in news_indicators if indicator in text_lower)
    
    # Check for URLs to news sites (legitimate sources)
    news_domains = ['bbc.com', 'reuters.com', 'aljazeera.com', 'nytimes.com', 'washingtonpost.com']
    has_news_url = any(domain in text_lower for domain in news_domains)
    
    # If 3+ news indicators OR has news URL, likely normal news
    return news_score >= 3 or has_news_url


def detect_llm_ttps(coordination_groups, posts):
    """Use LLM to detect additional TTPs - HIGHLY CONSERVATIVE VERSION"""
    if not coordination_groups or len(coordination_groups) < 3:
        return []
    
    # Check if groups appear to be normal news
    normal_news_count = sum(1 for g in coordination_groups if is_likely_normal_news(g.get('text_sample', '')))
    
    # If majority are normal news, don't detect TTPs
    if normal_news_count > len(coordination_groups) * 0.5:
        logger.info(f"Skipping TTP detection: {normal_news_count}/{len(coordination_groups)} groups appear to be normal news")
        return []
    
    # Prepare a concise summary for the LLM
    data_summary = {
        "total_groups": len(coordination_groups),
        "total_accounts": sum(g.get('account_count', 0) for g in coordination_groups),
        "sample_texts": [g.get('text_sample', '')[:200] for g in coordination_groups[:5]],
        "platforms": list(set(p for g in coordination_groups for p in g.get('platforms', []))),
        "has_urls": any(len(g.get('unique_urls', [])) > 0 for g in coordination_groups),
        "has_hashtags": any('#' in g.get('text_sample', '') for g in coordination_groups),
        "avg_posts_per_group": sum(g.get('post_count', 0) for g in coordination_groups) / len(coordination_groups)
    }
    
    prompt = f"""You are an expert DISARM framework analyst detecting coordinated inauthentic behavior.

Data Summary:
{json.dumps(data_summary, indent=2)}

CRITICAL INSTRUCTIONS:
1. ONLY detect TTPs if there is UNAMBIGUOUS evidence of coordinated manipulation.
2. DO NOT flag:
   - Normal news sharing or legitimate media coverage
   - Official government announcements or diplomatic statements
   - Legitimate political discourse or campaign messaging
   - Content about development, achievements, or positive events
   - Posts with URLs to legitimate news sources
3. ONLY flag if you see:
   - Identical text posted by 5+ accounts within hours
   - Bot-like accounts (generic names like "user123", "news_bot")
   - Coordinated hashtag campaigns with no organic engagement
   - Cross-platform amplification of the same manipulative content
   - Temporal coordination (posts within minutes of each other)

KNOWN DISARM TECHNIQUES (only use these IDs):
- T0049: Coordinated Inauthentic Behavior
- T0049.002: Controlled Profiles
- T0049.003: Bot Networks
- T0060: Multi-Platform Manipulation
- T0119: Rapid Response
- T0143: Hashtag Hijacking

OUTPUT FORMAT:
Return a JSON array of detected TTPs. Each object must have:
- "technique_id": One of the IDs listed above
- "name": Human-readable name
- "description": Brief explanation of how it applies
- "severity": "High", "Medium", or "Low"
- "confidence": 0.0 to 1.0
- "evidence": Specific evidence from the data

If no clear manipulation is detected, return an empty array: []

EXAMPLE VALID RESPONSE:
[
  {{
    "technique_id": "T0049.003",
    "name": "Bot Network Amplification",
    "description": "Multiple accounts with generic names posting identical content",
    "severity": "High",
    "confidence": 0.85,
    "evidence": "5 accounts with names like 'user123' posted same text within 2 hours"
  }}
]

Return ONLY the JSON array, no other text."""
    
    try:
        response = safe_llm_call(prompt, max_tokens=1024)
        if not response:
            return []
        
        # Extract JSON from response
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            ttps = json.loads(json_match.group(0))
            
            # Validate and filter TTPs
            valid_ttps = []
            valid_technique_ids = ['T0049', 'T0049.002', 'T0049.003', 'T0060', 'T0119', 'T0143']
            
            for ttp in ttps:
                if not isinstance(ttp, dict):
                    continue
                
                technique_id = ttp.get('technique_id', '')
                
                # Only accept known technique IDs
                if technique_id not in valid_technique_ids:
                    logger.warning(f"Skipping unknown technique ID: {technique_id}")
                    continue
                
                # Require minimum confidence
                confidence = ttp.get('confidence', 0)
                if confidence < 0.6:
                    logger.info(f"Skipping low-confidence TTP: {technique_id} (confidence: {confidence})")
                    continue
                
                # Require evidence
                evidence = ttp.get('evidence', '')
                if not evidence or len(evidence) < 10:
                    logger.warning(f"Skipping TTP without evidence: {technique_id}")
                    continue
                
                valid_ttps.append({
                    'name': ttp.get('name', technique_id),
                    'description': ttp.get('description', ''),
                    'severity': ttp.get('severity', 'Medium'),
                    'evidence': evidence,
                    'confidence': confidence,
                    'source': 'LLM_DISARM',
                    'technique_id': technique_id
                })
            
            if valid_ttps:
                logger.info(f"LLM detected {len(valid_ttps)} valid TTPs")
            else:
                logger.info("LLM found no valid TTPs (all filtered out)")
            
            return valid_ttps
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM TTP response: {e}")
    except Exception as e:
        logger.warning(f"LLM TTP detection failed: {e}")
    
    return []
   
def _make_post_dict(post: dict, reason: str) -> dict:
    """
    Normalise a raw post dict (as returned by get_coordination_groups
    sample_posts_with_urls) into the shape the template expects, and
    attach a ttp_reason string that explains WHY this post is evidence.
    """
    return {
        "username":     post.get("username", "Unknown"),
        "text_preview": post.get("text_preview", ""),
        "platform":     post.get("platform", ""),
        "timestamp":    post.get("timestamp", ""),
        "url":          post.get("url"),
        "ttp_reason":   reason,
    }
 
 
def _parse_ts(ts_str: str):
    """Parse a 'YYYY-MM-DD HH:MM' string; returns datetime.max on failure."""
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M")
    except Exception:
        return datetime.max
 
 
def _posts_within_window(posts: list, window_minutes: int = 60) -> list:
    """
    Return pairs of posts whose timestamps are within *window_minutes* of
    each other.  Each pair is returned as a flat list of 2 post dicts.
    """
    pairs = []
    timed = []
    for p in posts:
        ts = p.get("timestamp", "")
        if ts and ts != "N/A":
            timed.append((p, _parse_ts(ts)))
 
    timed.sort(key=lambda x: x[1])
    for i in range(len(timed) - 1):
        diff = (timed[i + 1][1] - timed[i][1]).total_seconds() / 60
        if 0 <= diff <= window_minutes:
            pairs.append([timed[i][0], timed[i + 1][0]])
    return pairs
 
 
def _post_has_weaponized(text):
    """Check full text against the module-level WEAPONIZED_KEYWORDS list."""
    text_lower = text.lower()
    # Word-boundary aware check to avoid "war" matching "award"
    for kw in WEAPONIZED_KEYWORDS:
        # Use word boundary for short English words, substring for Amharic/Oromo
        if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
            return True
    return False
 
 
def _make_evidence_post(post, reason):
    return {
        'username':     post.get('username', 'Unknown'),
        'text_preview': post.get('text_preview', post.get('full_text', '')[:300]),
        'platform':     post.get('platform', ''),
        'timestamp':    post.get('timestamp', ''),
        'url':          post.get('url'),
        'ttp_reason':   reason,
    }


def _normalize_platform(raw):
    """
    Collapse platform-name variants (e.g. 'Twitter', 'twitter.com', 'X')
    into one canonical label. Without this, a group whose posts are all on
    the same network but tagged inconsistently can falsely look
    'cross-platform' because two different strings for the same platform
    both land in a set() and count as 2 distinct platforms.
    """
    if not raw:
        return ''
    p = str(raw).strip().lower()
    if p in ('x', 'x.com', 'twitter', 'twitter.com', 'twitter/x'):
        return 'X'
    if p in ('facebook', 'fb', 'facebook.com'):
        return 'Facebook'
    if p in ('instagram', 'ig', 'instagram.com'):
        return 'Instagram'
    if p in ('tiktok', 'tiktok.com'):
        return 'TikTok'
    return str(raw).strip().title()


def _prioritize_unused_groups(candidate_groups, used_ids):
    """
    Stable-sort candidate groups so ones whose evidence hasn't already been
    shown under a different TTP come first. This is what stops every TTP
    card from repeatedly surfacing the same dominant group/post just
    because it happens to satisfy every filter.
    """
    return sorted(candidate_groups, key=lambda g: g.get('id') in used_ids)


def analyze_ttps(coordination_groups, posts):
    """
    Analyze Tactics, Techniques, and Procedures.
    Returns 12 TTPs with specific 'example_posts' that prove the TTP exists.
    """
    ttps = []
    if not coordination_groups:
        return ttps

    # Helper to format posts for the template
    def make_evidence(post, reason):
        return {
            'username': post.get('username', 'Unknown'),
            'text_preview': post.get('text_preview', post.get('full_text', '')[:300]),
            'platform': post.get('platform', ''),
            'timestamp': post.get('timestamp', ''),
            'url': post.get('url'),
            'ttp_reason': reason,
        }

    # TTP 1: Coordinated Inauthentic Behavior (CIB)
    cib_groups = [g for g in coordination_groups if g.get('account_count', 0) >= 5]
    if cib_groups:
        evidence = []
        for g in cib_groups[:2]:
            for p in g.get('sample_posts_with_urls', [])[:2]:
                evidence.append(make_evidence(p, f"One of {g.get('account_count')} accounts sharing near-identical content."))
        ttps.append({
            'name': 'Coordinated Inauthentic Behavior (CIB)',
            'description': f'{len(cib_groups)} group(s) with 5+ distinct accounts posting near-identical content.',
            'severity': 'High',
            'evidence': f'{sum(g.get("post_count", 0) for g in cib_groups)} posts across {sum(g.get("account_count", 0) for g in cib_groups)} accounts.',
            'example_posts': evidence[:4]
        })

    # TTP 2: Cross-Platform Amplification
    cross_groups = [g for g in coordination_groups if len(g.get('platforms', [])) > 1]
    if cross_groups:
        all_plats = set()
        for g in cross_groups: all_plats.update(g.get('platforms', []))
        evidence = []
        for g in cross_groups:
            for p in g.get('sample_posts_with_urls', []):
                if p.get('platform') and len(evidence) < 4:
                    evidence.append(make_evidence(p, f"Posted on {p.get('platform')} as part of cross-platform campaign."))
        ttps.append({
            'name': 'Cross-Platform Amplification',
            'description': f'{len(cross_groups)} group(s) spreading identical content across {len(all_plats)} platforms.',
            'severity': 'Medium',
            'evidence': f"Platforms: {', '.join(sorted(all_plats))}",
            'example_posts': evidence
        })

    # TTP 3: Rapid Response / Burst Posting
    burst_groups = [g for g in coordination_groups if g.get('post_count', 0) > 10]
    if burst_groups:
        biggest = max(burst_groups, key=lambda x: x.get('post_count', 0))
        evidence = [make_evidence(p, f"Part of {biggest.get('post_count')}-post burst.") for p in biggest.get('sample_posts_with_urls', [])[:3]]
        ttps.append({
            'name': 'Rapid Response / Burst Posting',
            'description': f'{len(burst_groups)} group(s) with high-volume posting (max: {biggest.get("post_count")} posts).',
            'severity': 'Medium',
            'evidence': f"Content repeated across {sum(g.get('account_count',0) for g in burst_groups)} accounts.",
            'example_posts': evidence
        })

    # TTP 4: Hashtag Manipulation
    hashtag_groups = [g for g in coordination_groups if g.get('hashtags')]
    if hashtag_groups:
        all_tags = set()
        for g in hashtag_groups: all_tags.update(g.get('hashtags', []))
        evidence = []
        for g in hashtag_groups:
            for p in g.get('sample_posts_with_urls', []):
                if len(evidence) < 4: evidence.append(make_evidence(p, f"Contains coordinated hashtags."))
        ttps.append({
            'name': 'Hashtag Manipulation',
            'description': f'Coordinated use of {len(all_tags)} hashtag(s) across {len(hashtag_groups)} groups.',
            'severity': 'Low',
            'evidence': f"Tags: {', '.join(list(all_tags)[:5])}",
            'example_posts': evidence
        })

    # TTP 5: URL Amplification
    url_groups = [g for g in coordination_groups if g.get('unique_urls')]
    if url_groups:
        evidence = []
        for g in url_groups:
            for p in g.get('sample_posts_with_urls', []):
                if p.get('url') and len(evidence) < 4:
                    evidence.append(make_evidence(p, f"Amplifying external URL."))
        ttps.append({
            'name': 'URL Amplification',
            'description': f'{len(url_groups)} group(s) amplifying external URLs.',
            'severity': 'Low',
            'evidence': 'Multiple accounts sharing same external links.',
            'example_posts': evidence
        })

    # TTP 6: Narrative Weaponization
    weaponized_keywords = ['genocide', 'kill', 'attack', 'war', 'slur', 'hate', 'ethnic cleansing', 'massacre']
    weap_groups = [g for g in coordination_groups if any(kw in g.get('text_sample', '').lower() for kw in weaponized_keywords)]
    if weap_groups:
        evidence = []
        for g in weap_groups:
            for p in g.get('sample_posts_with_urls', []):
                text = (p.get('text_preview') or '').lower()
                if any(kw in text for kw in weaponized_keywords) and len(evidence) < 4:
                    evidence.append(make_evidence(p, f"Contains weaponized keyword."))
        ttps.append({
            'name': 'Narrative Weaponization',
            'description': f'{len(weap_groups)} group(s) using high-risk weaponized keywords.',
            'severity': 'Critical',
            'evidence': 'Coordinated amplification of inflammatory language.',
            'example_posts': evidence
        })

    # TTP 7: Temporal Coordination
    sync_groups = []
    for g in coordination_groups:
        timestamps = []
        for p in g.get('sample_posts_with_urls', []):
            if p.get('timestamp') and p['timestamp'] != 'N/A':
                try: timestamps.append(datetime.strptime(p['timestamp'], '%Y-%m-%d %H:%M'))
                except: pass
        if len(timestamps) >= 2:
            timestamps.sort()
            for i in range(len(timestamps) - 1):
                if (timestamps[i+1] - timestamps[i]).total_seconds() / 60 <= 60:
                    sync_groups.append(g)
                    break
    if sync_groups:
        evidence = [make_evidence(p, f"Posted within 60 mins of other accounts.") for p in sync_groups[0].get('sample_posts_with_urls', [])[:3]]
        ttps.append({
            'name': 'Temporal Coordination (Synchronized Posting)',
            'description': f'{len(sync_groups)} group(s) posted identical content within 1 hour.',
            'severity': 'High',
            'evidence': 'Real-time coordination or scheduling tools suspected.',
            'example_posts': evidence
        })

    # TTP 8: Multi-Platform Narrative Seeding
    seed_groups = [g for g in coordination_groups if len(g.get('platforms', [])) >= 2 and g.get('account_count', 0) >= 3]
    if seed_groups:
        evidence = [make_evidence(p, f"Seeding narrative across platforms.") for p in seed_groups[0].get('sample_posts_with_urls', [])[:3]]
        ttps.append({
            'name': 'Multi-Platform Narrative Seeding',
            'description': f'{len(seed_groups)} group(s) seeding narratives across 2+ platforms with 3+ accounts.',
            'severity': 'High',
            'evidence': 'Cross-platform push to maximise reach.',
            'example_posts': evidence
        })

    # TTP 9: Bot-like Account Behavior
    bot_groups = [g for g in coordination_groups if g.get('bot_count', 0) > 0 or any(x in acc.lower() for acc in g.get('accounts', []) for x in ['bot', 'auto', 'news_bot'])]
    if bot_groups:
        evidence = [make_evidence(p, f"Posted by bot-like account.") for p in bot_groups[0].get('sample_posts_with_urls', [])[:3]]
        ttps.append({
            'name': 'Bot-like Account Behavior',
            'description': f'{len(bot_groups)} group(s) contain bot-like accounts or automated patterns.',
            'severity': 'Medium',
            'evidence': 'Potential use of automated accounts.',
            'example_posts': evidence
        })

    # TTP 10: Account Clustering
    cluster_groups = []
    for g in coordination_groups:
        accounts = g.get('accounts', [])
        numeric = [re.findall(r'\d+', acc) for acc in accounts]
        numeric = [int(p[0]) for p in numeric if p]
        if len(numeric) >= 3:
            # FIX: Use 'numeric' instead of 'numbers'
            diffs = [numeric[i+1] - numeric[i] for i in range(len(numeric)-1)]
            if any(d == 1 for d in diffs[:3]):
                cluster_groups.append(g)

    # TTP 11: Content Recycling
    recycle_groups = []
    for g in coordination_groups:
        timestamps = []
        for p in g.get('sample_posts_with_urls', []):
            if p.get('timestamp') and p['timestamp'] != 'N/A':
                try: timestamps.append(datetime.strptime(p['timestamp'], '%Y-%m-%d %H:%M'))
                except: pass
        if len(timestamps) >= 2 and (max(timestamps) - min(timestamps)).days > 7:
            recycle_groups.append(g)
    if recycle_groups:
        evidence = [make_evidence(p, f"Content recycled over extended period.") for p in recycle_groups[0].get('sample_posts_with_urls', [])[:3]]
        ttps.append({
            'name': 'Content Recycling',
            'description': f'{len(recycle_groups)} group(s) reposting identical content over > 7 days.',
            'severity': 'Medium',
            'evidence': 'Long-running recycling keeps narrative alive.',
            'example_posts': evidence
        })

    # TTP 12: Amplification Networks
    amp_groups = [g for g in coordination_groups if len(g.get('sample_posts_with_urls', [])) >= 3]
    if amp_groups:
        evidence = [make_evidence(p, f"Part of source-to-amplifier network.") for p in amp_groups[0].get('sample_posts_with_urls', [])[:3]]
        ttps.append({
            'name': 'Amplification Networks',
            'description': f'{len(amp_groups)} group(s) showing clear source-to-amplifier patterns.',
            'severity': 'Medium',
            'evidence': 'Content originates from few sources, amplified by many.',
            'example_posts': evidence
        })

    return ttps
    
def get_top_pairs(coordination_groups):
    """Get top coordinated account pairs"""
    pairs = []
    for group in coordination_groups[:10]:
        accounts = group['accounts']
        if len(accounts) >= 2:
            pairs.append({
                'accounts': f'{accounts[0][:20]}... ↔ {accounts[1][:20]}...',
                'shared_posts': group['post_count'],
                'platforms': group['platforms']
            })
    return pairs[:10]

# === IMPROVED NETWORK & COORDINATION FUNCTIONS ===
def is_primarily_ethiopia_related(text: str) -> bool:
    """
    Check if post is about Ethiopia elections - EXCLUDES US/Western elections
    """
    if not text or len(text.strip()) < 20:
        return False
    
    text_lower = text.lower()
    
    # === EXCLUSION LIST: US/Western Election Content ===
    us_election_signals = [
        'colorado', 'california', 'texas', 'florida', 'new york', 'georgia',
        'pennsylvania', 'michigan', 'wisconsin', 'arizona', 'nevada',
        'congressional district', 'us election', 'us vote', 'us primary',
        'american election', 'united states election', 'us congress',
        'us house', 'us senate', 'us primary', 'us midterm',
        'democratic primary', 'republican primary', 'us ballot',
        'drop site news', 'melat kiros',  
    ]
    
    # If it contains US election signals, reject it
    if any(signal in text_lower for signal in us_election_signals):
        return False
    
    # === POSITIVE Ethiopia Signals ===
    # High-Confidence Primary Native Script Signals
    native_signals = [
        'ኢትዮጵያ', 'አዲስ አበባ', 'አብይ', 'ብልግና', 'ኖ', 'ሕወሓት', 
        'ህወሃት', 'ኦነ', 'ሸኔ', 'አማራ', 'ሮሚያ', 'ትግራይ', 
        'ክልል', 'ወረዳ', 'ቀበሌ', 'ር', 'ሐበ', 'ሃበሻ', 'ኢሰመቦ'
    ]
    
    if any(ns in text_lower for ns in native_signals):
        return True
    
    # Primary English & Transliterated Ethiopia-Specific Signals
    primary_ethiopia_signals = [
        'ethiopia', 'ethiopian', 'habesha', 'abyssinia', 'birr', 'etb',
        'abiy', 'abiy ahmed', 'pp party', 'prosperity party', 'nebe', 
        'shane', 'ola', 'fano', 'tplf', 'eprdf', 'derg',
        'oromia', 'amhara', 'tigray', 'sidama', 'gambella', 
        'benishangul', 'gumuz', 'afar', 'somali region', 'ogaden', 
        'woreda', 'kebele', 'gott',
        'addis ababa', 'finfinnee', 'mekelle', 'bahir dar', 'gondar', 
        'dessie', 'jimma', 'adama', 'hawassa', 'dire dawa', 'harar', 
        'axum', 'lalibela',
        'oromo', 'amhara', 'tigrayan', 'tegaru', 'gurage', 'wolayta',
    ]
    
    if any(signal in text_lower for signal in primary_ethiopia_signals):
        return True
    
    # Secondary Geopolitical Context (Horn of Africa)
    horn_neighbors = ['eritrea', 'sudan', 'somalia', 'djibouti', 'mogadishu', 'asmara', 'khartoum']
    geopolitical_context = ['gerd', 'nile', 'red sea', 'port access', 'mou', 'somaliland']
    
    has_neighbor = any(neighbor in text_lower for neighbor in horn_neighbors)
    has_context = any(ctx in text_lower for ctx in geopolitical_context)
    
    if has_neighbor and has_context:
        return True
    
    return False
    
def get_coordination_groups(posts_queryset, min_accounts=3, max_groups=15, similarity_threshold=0.85, max_posts=5000):
    """
    COORDINATION GROUP BUILDER (DUAL-SOURCE ARCHITECTURE):
    - Extracts external handles via `RT @user` as explicit External Sources.
    - Preserves the earliest dataset poster as an In-Database Source.
    - Strictly prevents retweeters from being labeled as sources unless they are the true original poster.
    - Constructs graph nodes linking amplifiers to ALL active sources in the cluster.
    """
    
    coordination = []
    
    # 1. Fetch posts and exclude non-text/media channels
    posts_data = list(
        posts_queryset
        .exclude(platform__iexact='TikTok')
        .exclude(platform__iexact='Media')
        .exclude(platform__iexact='News')
        .values('id', 'account_id', 'original_text', 'platform', 'url', 'timestamp_share', 'risk_level')
        .order_by('-timestamp_share')[:max_posts]
    )
    
    if len(posts_data) < min_accounts:
        return []
    
    # 2. Filter valid text entries
    valid_indices = []
    valid_texts = []
    for i, p in enumerate(posts_data):
        text = p.get('original_text', '')
        if text and len(str(text)) > 20:
            valid_indices.append(i)
            valid_texts.append(str(text))
            
    if len(valid_texts) < 2:
        return []
        
    # 3. Vectorized TF-IDF Calculation & Similarity Matching
    vectorizer = TfidfVectorizer(
        max_features=2000, 
        stop_words='english', 
        ngram_range=(1, 2), 
        min_df=2, 
        max_df=0.9
    )
    tfidf_matrix = vectorizer.fit_transform(valid_texts)
    
    chunk_size = 500
    similarity_groups = []
    processed_indices = set()
    
    for chunk_start in range(0, len(valid_texts), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(valid_texts))
        chunk_matrix = tfidf_matrix[chunk_start:chunk_end]
        chunk_similarities = cosine_similarity(chunk_matrix, tfidf_matrix)
        
        for local_i, global_i in enumerate(range(chunk_start, chunk_end)):
            if global_i in processed_indices:
                continue
            similar_indices = [global_i]
            above_threshold = np.where(chunk_similarities[local_i] >= similarity_threshold)[0]
            for j in above_threshold:
                if j != global_i and j not in processed_indices:
                    similar_indices.append(int(j))
                    processed_indices.add(int(j))
            processed_indices.add(global_i)
            
            if len(similar_indices) >= min_accounts:
                group_accounts = set(posts_data[valid_indices[idx]].get('account_id') for idx in similar_indices if posts_data[valid_indices[idx]].get('account_id'))
                if len(group_accounts) >= min_accounts:
                    similarity_groups.append(similar_indices)
                    
    # 4. Compile coordination groups
    for group_indices in similarity_groups[:max_groups]:
        group_posts = [posts_data[valid_indices[idx]] for idx in group_indices]
        
        # Sort chronologically (Earliest First)
        sorted_group_posts = sorted(
            group_posts, 
            key=lambda x: x.get('timestamp_share') or datetime.max
        )
        
        sources_set = set()
        text_sample_raw = str(sorted_group_posts[0].get('original_text', ''))
        
        # --- DUAL SOURCE LOGIC ---
        # 1. Capture external sources via RT / via patterns across the cluster
        for post in sorted_group_posts:
            text_content = str(post.get('original_text', ''))
            rt_match = re.search(r'(?:RT|via)\s+@(\w+)', text_content, re.IGNORECASE)
            if rt_match:
                ext_source = clean_username(rt_match.group(1).strip())
                if ext_source and ext_source != "Unknown":
                    sources_set.add(ext_source)
                    
        # 2. Capture the earliest database poster as an additional in-database source
        earliest_db_account = clean_username(sorted_group_posts[0].get('account_id'))
        if earliest_db_account and earliest_db_account != "Unknown":
            sources_set.add(earliest_db_account)
            
        sources_list = list(sources_set)
        # Primary source assigned for quick access
        primary_source = sources_list[0] if sources_list else "Unknown"

        # --- AMPLIFIER ISOLATION ---
        # Retweeters / DB accounts that are NOT in sources_set become amplifiers
        amplifiers_set = set()
        sources_lower = {s.lower() for s in sources_set}
        
        for post in sorted_group_posts:
            acct = clean_username(post.get('account_id'))
            if acct and acct != "Unknown" and acct.lower() not in sources_lower:
                amplifiers_set.add(acct)
                
        amplifiers_list = list(amplifiers_set)
        ordered_accounts_set = list(dict.fromkeys(sources_list + amplifiers_list))
        
        if len(ordered_accounts_set) < min_accounts:
            continue
            
        # 5. Bot analytics & metadata parsing
        real_posts_for_bot_check = [p for p in sorted_group_posts if clean_username(p.get('account_id')).lower() not in sources_lower]
        if not real_posts_for_bot_check: 
            real_posts_for_bot_check = sorted_group_posts
            
        bot_data = identify_bot_accounts(real_posts_for_bot_check)
        bot_accounts = list(bot_data.keys())
        bot_count = len(bot_accounts)
        bot_percentage = (bot_count / len(amplifiers_list) * 100) if amplifiers_list else 0
        coordination_type = determine_coordination_type(real_posts_for_bot_check, bot_count)
        
        # 6. Post Sample Formatting
        sample_posts_with_urls = []
        all_platforms = set()
        all_hashtags = []
        
        for post in sorted_group_posts[:15]:
            if post.get('platform'):
                all_platforms.add(post['platform'])
            text = str(post.get('original_text', '')).strip()
            found = re.findall(r'#(\w+)', text, re.IGNORECASE)
            all_hashtags.extend([h.lower() for h in found])
            
            ts = post.get('timestamp_share')
            raw_account = post.get('account_id')
            username_clean = clean_username(raw_account)
            bot_reasons = bot_data.get(raw_account, [])
            
            sample_posts_with_urls.append({
                'username': username_clean,
                'platform': post.get('platform', ''),
                'url': post.get('url') if post.get('url') and str(post['url']).startswith('http') else None,
                'timestamp': ts.strftime('%Y-%m-%d %H:%M') if ts else 'N/A',
                'text_preview': text[:150] + '...' if text else '',
                'is_bot': raw_account in bot_accounts,
                'bot_reasons': ", ".join(bot_reasons) if bot_reasons else "",
                'risk_level': post.get('risk_level', 'unknown'),
                'is_source': username_clean.lower() in sources_lower
            })
            
        # 7. Network Graph Topology (Multi-Source Hubs)
        graph_nodes = []
        graph_links = []
        existing_nodes = set()
        
        # Injects ALL identified sources (Blue Nodes)
        for src in sources_list:
            src_key = str(src).strip()
            if src_key.lower() not in existing_nodes:
                graph_nodes.append({
                    'id': src_key, 
                    'label': src_key, 
                    'type': 'source', 
                    'group': 'source',
                    'color': '#1e90ff',  # Force Blue
                    'size': 24,
                    'shape': 'dot',
                    'physics': True
                })
                existing_nodes.add(src_key.lower())
            
        # Cap amplifiers shown in the visualization graph to 15 to fit canvas bounds
        top_graph_amplifiers = amplifiers_list[:15]
        
        for amp in top_graph_amplifiers:
            amp_key = str(amp).strip()
            if amp_key.lower() not in existing_nodes:
                graph_nodes.append({
                    'id': amp_key, 
                    'label': amp_key, 
                    'type': 'amplifier', 
                    'group': 'amplifier',
                    'color': '#e67e22',  # Force Orange
                    'size': 14,
                    'shape': 'dot',
                    'physics': True
                })
                existing_nodes.add(amp_key.lower())
                
            # Create directional links from sources to this amplifier
            for src in sources_list:
                s_id = str(src).strip()
                t_id = str(amp_key).strip()
                
                if s_id.lower() in existing_nodes and t_id.lower() in existing_nodes and s_id.lower() != t_id.lower():
                    graph_links.append({
                        'from': s_id,
                        'to': t_id,
                        'source': s_id,
                        'target': t_id,
                        'length': 110,       
                        'arrows': 'to',
                        'width': 1.5
                    })
                
        unique_urls = list(set(p['url'] for p in real_posts_for_bot_check if p.get('url') and str(p['url']).startswith('http')))[:5]
        text_sample = str(text_sample_raw)[:200] if text_sample_raw else '[Similar content]'
        
        coordination.append({
            'id': len(coordination) + 1,
            'accounts': ordered_accounts_set[:15],  
            'account_count': len(ordered_accounts_set),
            'post_count': len(sorted_group_posts),
            'bot_count': bot_count,
            'bot_percentage': round(bot_percentage, 1),
            'text_sample': text_sample,
            'sample_posts_with_urls': sample_posts_with_urls,  
            'unique_urls': unique_urls,
            'platforms': sorted(list(all_platforms)),
            'coordination_type': coordination_type,
            'similarity_score': f'≥{int(similarity_threshold*100)}%',
            'sub_narrative': extract_sub_narrative(text_sample),
            'hashtags': list(set(all_hashtags))[:10],
            'primary_type': 'amplification_network' if bot_percentage >= 50 else 'coordination',
            'source_node': primary_source,
            'sources': sources_list,                           # All Sources (External + In-DB First Poster)
            'amplifiers': amplifiers_list,                     # Strictly Amplifiers / Retweeters
            'source_count': len(sources_list),
            'amplifier_count': len(amplifiers_list),
            'nodes': graph_nodes,
            'links': graph_links,
        })
        
    coordination.sort(key=lambda x: (-x['bot_percentage'], -x['account_count']))
    return coordination[:max_groups]
    
def identify_bot_accounts(posts):
    """
    Enhanced bot detection with timestamp analysis and behavioral patterns.
    Returns a dict: {account_id: [list of reasons]}
    """
    bot_data = {}  # Changed from set() to dict
    account_posts = defaultdict(list)
    
    # Group posts by account
    for post in posts:
        if post.get('account_id'):
            account_posts[post['account_id']].append(post)
    
    for account_id, account_post_list in account_posts.items():
        bot_signals = 0
        signal_reasons = []
        
        # === STRONG SIGNALS ===
        
        # 1. Check posting frequency patterns (bots post at regular intervals)
        if len(account_post_list) >= 5:
            timestamps = [p.get('timestamp_share') for p in account_post_list if p.get('timestamp_share')]
            if len(timestamps) >= 5:
                timestamps.sort()
                # Calculate time differences between posts
                time_diffs = []
                for i in range(1, len(timestamps)):
                    diff = (timestamps[i] - timestamps[i-1]).total_seconds() / 60  # minutes
                    time_diffs.append(diff)
                
                # Check for suspicious regularity (bots post at exact intervals)
                if time_diffs:
                    avg_interval = sum(time_diffs) / len(time_diffs)
                    # If variance is very low, likely automated
                    variance = sum((x - avg_interval) ** 2 for x in time_diffs) / len(time_diffs)
                    std_dev = variance ** 0.5
                    coefficient_of_variation = std_dev / avg_interval if avg_interval > 0 else 1
                    
                    if coefficient_of_variation < 0.3 and len(account_post_list) >= 10:
                        bot_signals += 3
                        signal_reasons.append(f"Regular posting interval (CV: {coefficient_of_variation:.2f})")
                    
                    # Very high frequency (multiple posts per minute)
                    if avg_interval < 5:  # Less than 5 minutes between posts
                        bot_signals += 2
                        signal_reasons.append(f"Very high frequency ({avg_interval:.1f} min avg)")
        
        # 2. Check for 24/7 activity (bots don't sleep)
        if len(account_post_list) >= 10:
            hours_active = set()
            for post in account_post_list:
                if post.get('timestamp_share'):
                    hours_active.add(post['timestamp_share'].hour)
            
            if len(hours_active) >= 20:  # Active in 20+ hours of the day
                bot_signals += 2
                signal_reasons.append("24/7 activity pattern")
        
        # 3. Check for burst posting (many posts in short time)
        if len(account_post_list) >= 10:
            posts_per_hour = defaultdict(int)
            for post in account_post_list:
                if post.get('timestamp_share'):
                    hour_key = post['timestamp_share'].strftime('%Y-%m-%d %H')
                    posts_per_hour[hour_key] += 1
            
            max_posts_in_hour = max(posts_per_hour.values()) if posts_per_hour else 0
            if max_posts_in_hour >= 20:  # 20+ posts in one hour
                bot_signals += 2
                signal_reasons.append(f"Burst posting ({max_posts_in_hour} posts/hour)")
            elif max_posts_in_hour >= 10:
                bot_signals += 1
                signal_reasons.append(f"High volume burst ({max_posts_in_hour} posts/hour)")
        
        # === MEDIUM SIGNALS ===
        
        # 4. Generic account name patterns (only strong patterns)
        clean_name = clean_username(account_id).lower()
        strong_bot_patterns = ['bot', 'auto', 'auto_', 'auto.', 'daily_', 'daily.', 'news_bot', 'update_bot']
        if any(pattern in clean_name for pattern in strong_bot_patterns):
            bot_signals += 2
            signal_reasons.append("Bot-like name pattern")
        
        # 5. Very short/templated content
        short_posts = sum(1 for p in account_post_list 
                         if p.get('original_text') and len(str(p['original_text']).strip()) < 30)
        if short_posts > len(account_post_list) * 0.7:  # 70%+ very short posts
            bot_signals += 1
            signal_reasons.append(f"Mostly short content ({short_posts}/{len(account_post_list)})")
        
        # 6. Identical content repetition
        if len(account_post_list) >= 5:
            content_hashes = [hash(str(p.get('original_text', ''))[:50]) for p in account_post_list if p.get('original_text')]
            unique_content = len(set(content_hashes))
            if unique_content < len(content_hashes) * 0.3:  # 70%+ duplicate content
                bot_signals += 2
                signal_reasons.append(f"High content repetition ({unique_content}/{len(content_hashes)} unique)")
        
        # === WEAK SIGNALS ===
        
        # 7. Account name with years (only if combined with other signals)
        if re.search(r'(2023|2024|2025)', clean_name):
            bot_signals += 0.5  # Weak signal alone
        
        # 8. High/critical risk level
        high_risk_posts = sum(1 for p in account_post_list 
                             if p.get('risk_level') in ['high', 'critical'])
        if high_risk_posts > 0:
            bot_signals += 0.5
        
        # RETURN DICT WITH REASONS if bot_signals >= 3
        if bot_signals >= 3:
            logger.info(f"BOT DETECTED: {account_id} (score: {bot_signals}) - Reasons: {', '.join(signal_reasons)}")
            bot_data[account_id] = signal_reasons  # Store reasons in dict
    
    return bot_data  

def determine_coordination_type(posts, bot_count):
    """
    Determine the type of coordination based on patterns
    """
    total_accounts = len(set(p['account_id'] for p in posts if p.get('account_id')))
    bot_percentage = (bot_count / total_accounts * 100) if total_accounts > 0 else 0

    # Check for URL sharing
    urls = [p['url'] for p in posts if p.get('url')]
    has_url_coordination = len(urls) > 0 and len(set(urls)) < len(urls)

    # Check for hashtag coordination
    hashtags = []
    for post in posts:
        if post.get('original_text'):
            found = re.findall(r'#\w+', str(post['original_text']))
            hashtags.extend(found)
    has_hashtag_coordination = len(hashtags) > 0 and len(set(hashtags)) < len(hashtags)

    # Determine type
    if bot_percentage >= 50:
        return "🤖 Bot Network"
    elif has_url_coordination and has_hashtag_coordination:
        return "🔗 URL + Hashtag Amplification"
    elif has_url_coordination:
        return "🔗 URL Amplification"
    elif has_hashtag_coordination:
        return "#️ Hashtag Coordination"
    elif bot_percentage >= 25:
        return " Bot-Assisted"
    else:
        return "👥 Human Coordination"

def generate_network_graph_data(posts_queryset, min_connections=2, top_n=50, layout='spring'):
    """
    OPTIMIZED: Pre-fetches all account data in ONE query.
    Eliminates N+1 database queries.
    """
    import networkx as nx
    from django.db.models import Count
    
    G = nx.Graph()
    account_roles = {}
    
    # Group by text using Django ORM (fast, done in SQL)
    text_groups = posts_queryset.values('original_text').annotate(
        account_count=Count('account_id', distinct=True)
    ).filter(account_count__gte=min_connections)
    
    # Collect ALL relevant posts in ONE query
    relevant_texts = [g['original_text'] for g in text_groups if g['original_text']]
    if not relevant_texts:
        return {'nodes': [], 'edges': [], 'stats': {'nodes': 0, 'edges': 0}}
    
    # Fetch ALL posts for these texts in a SINGLE query
    all_posts = list(
        posts_queryset
        .filter(original_text__in=relevant_texts)
        .order_by('timestamp_share')
        .values('account_id', 'platform', 'url', 'timestamp_share', 'original_text')
    )
    
    # Build a lookup dict IN MEMORY (no more DB queries in loops)
    posts_by_text = defaultdict(list)
    for post in all_posts:
        posts_by_text[post['original_text']].append(post)
    
    # Pre-compute account statistics IN MEMORY
    account_stats = defaultdict(lambda: {
        'post_count': 0, 
        'platforms': set(), 
        'sample_url': None,
        'first_post_time': None
    })
    
    for post in all_posts:
        acc = clean_username(post['account_id'])
        if not acc or len(acc) < 2:
            continue
        account_stats[acc]['post_count'] += 1
        if post.get('platform'):
            account_stats[acc]['platforms'].add(post['platform'])
        if not account_stats[acc]['sample_url'] and post.get('url') and str(post['url']).startswith('http'):
            account_stats[acc]['sample_url'] = post['url']
        if not account_stats[acc]['first_post_time'] and post.get('timestamp_share'):
            account_stats[acc]['first_post_time'] = post['timestamp_share']
    
    # Build graph using in-memory data
    for group in text_groups:
        text = group['original_text']
        if not is_primarily_ethiopia_related(text):
            continue
        
        posts_for_text = posts_by_text.get(text, [])
        if not posts_for_text:
            continue
        
        accounts = []
        for idx, post_data in enumerate(posts_for_text):
            username = clean_username(post_data['account_id'])
            if not username or len(username) < 2 or username.lower() in ['twitter', 'facebook', 'tiktok', 'source']:
                continue
            
            if idx == 0:
                account_roles[username] = 'source'
            elif username not in account_roles:
                account_roles[username] = 'amplifier'
            
            accounts.append({
                'id': username,
                'platform': post_data.get('platform', ''),
                'sample_url': post_data.get('url') if post_data.get('url') and str(post_data['url']).startswith('http') else None
            })
        
        for i in range(len(accounts)):
            for j in range(i+1, len(accounts)):
                u_id, v_id = accounts[i]['id'], accounts[j]['id']
                if u_id == v_id:
                    continue
                if G.has_edge(u_id, v_id):
                    G[u_id][v_id]['weight'] += 1
                else:
                    G.add_edge(u_id, v_id, weight=1,
                             platform1=accounts[i]['platform'],
                             platform2=accounts[j]['platform'],
                             sample_url1=accounts[i]['sample_url'],
                             sample_url2=accounts[j]['sample_url'])
    
    if G.number_of_edges() == 0:
        return {'nodes': [], 'edges': [], 'stats': {'nodes': 0, 'edges': 0}}
    
    # Filter and layout
    nodes_to_keep = [n for n, d in G.degree() if d >= min_connections]
    G = G.subgraph(nodes_to_keep).copy()
    
    if G.number_of_edges() == 0:
        return {'nodes': [], 'edges': [], 'stats': {'nodes': 0, 'edges': 0}}
    
    top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:top_n]
    top_node_names = [n for n, _ in top_nodes]
    G_top = G.subgraph(top_node_names).copy()
    
    # Layout computation
    if layout == 'circular':
        pos = nx.circular_layout(G_top)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G_top)
    else:
        pos = nx.spring_layout(G_top, k=0.6, iterations=50, seed=42)
    
    #  Build nodes using PRE-COMPUTED in-memory stats (NO DB queries!)
    nodes = []
    for node in G_top.nodes():
        degree = G_top.degree(node)
        
        # Look up stats from memory instead of database
        stats = account_stats.get(node, {'post_count': 0, 'platforms': set(), 'sample_url': None})
        post_count = stats['post_count']
        platforms = list(stats['platforms'])
        platform = platforms[0] if platforms else 'Unknown'
        sample_url = stats['sample_url']
        
        node_type = account_roles.get(node, 'source')
        node_color = '#3b82f6' if node_type == 'source' else '#f59e0b'
        
        nodes.append({
            'id': node, 'label': node, 'degree': degree,
            'post_count': post_count, 'platform': platform,
            'url': sample_url, 'sample_url': sample_url,
            'x': float(pos[node][0]), 'y': float(pos[node][1]),
            'size': max(15, degree * 3),
            'color': node_color, 'type': node_type
        })
    
    # Build edges
    edges = []
    for u, v, data in G_top.edges(data=True):
        if u in pos and v in pos:
            edges.append({
                'source': u, 'target': v,
                'weight': data.get('weight', 1),
                'source_x': float(pos[u][0]), 'source_y': float(pos[u][1]),
                'target_x': float(pos[v][0]), 'target_y': float(pos[v][1]),
                'sample_url': data.get('sample_url1') or data.get('sample_url2')
            })
    
    return {
        'nodes': nodes, 'edges': edges,
        'stats': {
            'nodes': len(nodes), 'edges': len(edges),
            'density': G_top.number_of_edges() / (G_top.number_of_nodes() * (G_top.number_of_nodes() - 1) / 2) if G_top.number_of_nodes() > 1 else 0
        }
    }
def generate_network_graph_from_groups(coordination_groups, top_n=50, layout='spring'):
    """
    Build network graph from coordination groups.
    FIXED: Properly extracts sources from RT patterns and timestamps.
    """

    
    G = nx.Graph()
    account_roles = {}       # username -> 'source' or 'amplifier'
    account_post_counts = {}
    account_platforms = {}
    account_timestamps = {}  # username -> earliest timestamp (datetime object)
    hub_nodes = set()
    
    # Regex to detect retweets
    rt_pattern = re.compile(r'RT\s+@([A-Za-z0-9_]+)', re.IGNORECASE)
    
    # ─ PHASE 1: Extract ALL accounts and their roles ──────────────────
    for group_idx, group in enumerate(coordination_groups):
        group_accounts = set(group.get('accounts', []))
        sample_posts = group.get('sample_posts_with_urls', [])
        group_id = group.get('id', f'group_{group_idx}')
        
        if len(group_accounts) < 2:
            continue
        
        # ── 1a. Parse timestamps and detect RT patterns ──────────────
        for post in sample_posts:
            username = post.get('username', '')
            if not username:
                continue
            
            # Track post count & platform
            account_post_counts[username] = account_post_counts.get(username, 0) + 1
            platform = post.get('platform', 'Unknown')
            if platform and platform != 'Unknown':
                account_platforms[username] = platform
            elif username not in account_platforms:
                account_platforms[username] = 'Unknown'
            
            # Parse timestamp
            timestamp_str = post.get('timestamp', '')
            if username and timestamp_str and timestamp_str != 'N/A':
                try:
                    ts = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M')
                    if username not in account_timestamps or ts < account_timestamps[username]:
                        account_timestamps[username] = ts
                except:
                    pass
            
            # Detect if this is a retweet
            text = post.get('text_preview', '') or ''
            rt_match = rt_pattern.match(text.strip())
            if rt_match:
                rt_target = rt_match.group(1).strip()
                # Add RT target as a SOURCE candidate (even if not in group_accounts)
                if rt_target and rt_target.lower() not in [acc.lower() for acc in group_accounts]:
                    group_accounts.add(rt_target)
                    # RT targets are sources by definition
                    if rt_target not in account_roles:
                        account_roles[rt_target] = 'source'
        
        # ── 1b. Determine source for this group ───────────────────────
        # Strategy: Find account with EARLIEST timestamp that didn't RT anyone
        group_timestamps = {
            acc: account_timestamps.get(acc) 
            for acc in group_accounts 
            if account_timestamps.get(acc)
        }
        
        if group_timestamps:
            # Sort by timestamp (earliest first)
            sorted_accounts = sorted(group_timestamps.items(), key=lambda x: x[1])
            
            # Find first account that posted original content (not an RT)
            source_account = None
            for acc, ts in sorted_accounts:
                # Check if this account ever posted original content
                is_always_rt = False
                for post in sample_posts:
                    if post.get('username') == acc:
                        text = post.get('text_preview', '') or ''
                        if not rt_pattern.match(text.strip()):
                            # This account posted original content
                            source_account = acc
                            break
                
                if source_account:
                    break
            
            # Fallback: if everyone RT'd, use the earliest poster
            if not source_account and sorted_accounts:
                source_account = sorted_accounts[0][0]
            
            # Assign roles
            if source_account:
                account_roles[source_account] = 'source'
                for acc in group_accounts:
                    if acc != source_account and acc not in account_roles:
                        account_roles[acc] = 'amplifier'
    
    # ── 1c. Build graph edges ───────────────────────────────────────
    for group in coordination_groups:
        group_accounts = list(group.get('accounts', []))
        if len(group_accounts) < 2:
            continue
        
        weight = group.get('post_count', 1)
        
        # Connect all accounts in the group
        for i in range(len(group_accounts)):
            for j in range(i + 1, len(group_accounts)):
                u, v = group_accounts[i], group_accounts[j]
                if G.has_edge(u, v):
                    G[u][v]['weight'] += weight
                else:
                    G.add_edge(u, v, weight=1, type='coordination')
    
    if G.number_of_nodes() == 0:
        return {'nodes': [], 'edges': [], 'stats': {'nodes': 0, 'edges': 0, 'density': 0}}
    
    # ── PHASE 2: Layout computation ───────────────────────────────────
    G_final = G.copy()
    
    # Use better layout parameters to avoid "chunky" graphs
    if layout == 'circular':
        pos = nx.circular_layout(G_final)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G_final)
    else:
        # Spring layout with better parameters
        pos = nx.spring_layout(
            G_final, 
            k=0.7,           # Optimal distance between nodes (higher = more spread)
            iterations=100,  # More iterations for better convergence
            seed=42,         # Reproducible results
            scale=2.0        # Scale factor
        )
    
    # ── Rescale positions to consistent coordinate space ─────────────
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_range = (x_max - x_min) or 1.0
    y_range = (y_max - y_min) or 1.0
    TARGET = 900.0
    
    def _rescale(x, y):
        nx_ = (x - x_min) / x_range * TARGET - TARGET / 2
        ny_ = (y - y_min) / y_range * TARGET - TARGET / 2
        return nx_, ny_
    
    pos = {node: _rescale(*p) for node, p in pos.items()}
    
    # ── PHASE 3: Build JSON for frontend ──────────────────────────────
    nodes = []
    for node in G_final.nodes():
        degree = G_final.degree(node)
        node_type = account_roles.get(node, 'amplifier')  # default
        node_color = '#3b82f6' if node_type == 'source' else '#f59e0b'  # Blue or Orange
        
        post_count = account_post_counts.get(node, 0)
        platform = account_platforms.get(node, 'Unknown')
        
        # Size: sources are larger, scaled by degree
        base_size = 28 if node_type == 'source' else 18
        size = max(base_size, min(55, base_size + degree * 2))
        
        nodes.append({
            'id': node,
            'label': node[:20],  # Truncate long usernames
            'degree': degree,
            'post_count': post_count,
            'platform': platform,
            'x': float(pos[node][0]),
            'y': float(pos[node][1]),
            'size': size,
            'color': node_color,
            'type': node_type,
        })
    
    edges = []
    for u, v, data in G_final.edges(data=True):
        if u in pos and v in pos:
            edges.append({
                'source': u,
                'target': v,
                'weight': data.get('weight', 1),
                'type': data.get('type', 'coordination'),
                'source_x': float(pos[u][0]),
                'source_y': float(pos[u][1]),
                'target_x': float(pos[v][0]),
                'target_y': float(pos[v][1]),
            })
    
    # Stats
    n_nodes = G_final.number_of_nodes()
    n_edges = G_final.number_of_edges()
    density = n_edges / (n_nodes * (n_nodes - 1) / 2) if n_nodes > 1 else 0
    
    source_count = sum(1 for role in account_roles.values() if role == 'source')
    amplifier_count = sum(1 for role in account_roles.values() if role == 'amplifier')
    
    return {
        'nodes': nodes,
        'edges': edges,
        'bounds': {'x_min': -TARGET/2, 'x_max': TARGET/2, 
                   'y_min': -TARGET/2, 'y_max': TARGET/2},
        'stats': {
            'nodes': n_nodes,
            'edges': n_edges,
            'density': round(density, 4),
            'sources': source_count,
            'amplifiers': amplifier_count,
        }
    }
    
def calculate_ethiopia_relevance(text):
    if not text:
        return 0

    text = str(text).lower()

    ethiopia_entities = [
        "ethiopia",
        "ethiopian",
        "abiy",
        "fano",
        "tplf",
        "amhara",
        "oromia",
        "tigray",
        "addis",
        "nebe",
        "prosperity party"
    ]

    matches = sum(
        text.count(entity)
        for entity in ethiopia_entities
    )

    unique_matches = sum(
        1
        for entity in ethiopia_entities
        if entity in text
    )

    return (
        matches * 0.4 +
        unique_matches * 0.6
    )    
def _get_platform_color(platform):
    """Get color hex code for platform"""
    colors = {
        'X': '#1DA1F2', 'Twitter': '#1DA1F2',
        'Facebook': '#1877F2',
        'TikTok': '#000000',
        'Telegram': '#0088cc',
        'Media': '#6B7280', 'News': '#6B7280',
        'Unknown': '#9CA3AF'
    }
    return colors.get(platform, '#9CA3AF')


# === CONFIG: Reuse your Ethiopia lexicon ===
CONFIG = {
    "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
    "bertrend": {"min_cluster_size": 3},
    "analysis": {"time_window": "48H"},
    "coordination_detection": {"threshold": 0.85, "max_features": 5000},
    
    # === ETHIOPIA LEXICON: Category-Term Mapping ===
    "lexicon": {
        "ethnic_identity": {
            #"አማራ": {"severity": "medium", "target_entity": "Amhara", "language": "Amharic"},
            #"amhara": {"severity": "medium", "target_entity": "Amhara", "language": "English"},
            "ነፍጠኛ": {"severity": "high", "target_entity": "Amhara", "language": "Amharic"},
            "ነፍጠኛ አመለካከት": {"severity": "high", "target_entity": "Amhara", "language": "Amharic"},
            "ነፍጠኛ ፋኖ": {"severity": "high", "target_entity": "Amhara", "language": "Amharic"},
            "አስነዋሪ ነፍጠኛ": {"severity": "high", "target_entity": "Amhara", "language": "Amharic"},
            "neftegna": {"severity": "high", "target_entity": "Amhara", "language": "English"},
            "ቆምጬ": {"severity": "high", "target_entity": "Amhara", "language": "Amharic"},
            "አንተ ቆምጬ ፋራው": {"severity": "high", "target_entity": "Amhara", "language": "Amharic"},
            "qomcee": {"severity": "high", "target_entity": "Amhara", "language": "Oromo"},
            "Kutichaa Minilik": {"severity": "high", "target_entity": "Amhara", "language": "Oromo"},
            "Jawsa": {"severity": "high", "target_entity": "Amhara", "language": "Oromo"},
            "Jawisa": {"severity": "high", "target_entity": "Amhara", "language": "Oromo"},
            "Jawisaa": {"severity": "high", "target_entity": "Amhara", "language": "Oromo"},
            "አማራ ጠል": {"severity": "high", "target_entity": "Amhara", "language": "Amharic"},
            "አማራ ስጋ": {"severity": "high", "target_entity": "Amhara", "language": "Amharic"},
            "Fota-wearer": {"severity": "medium", "target_entity": "Amhara", "language": "English"},
            "ፎጣ ለባሽ": {"severity": "medium", "target_entity": "Amhara", "language": "Amharic"},
            
           # "ኦሮሞ": {"severity": "medium", "target_entity": "Oromo", "language": "Amharic"},
            #"oromo": {"severity": "medium", "target_entity": "Oromo", "language": "English"},
            "ጋላ": {"severity": "high", "target_entity": "Oromo", "language": "Amharic"},
            "ጋላው": {"severity": "high", "target_entity": "Oromo", "language": "Amharic"},
            "አንተ ጋላ": {"severity": "high", "target_entity": "Oromo", "language": "Amharic"},
            "ጥምብ ላም": {"severity": "high", "target_entity": "Oromo", "language": "Amharic"},
            "የጋላ ባህሪ": {"severity": "high", "target_entity": "Oromo", "language": "Amharic"},
            "galla": {"severity": "high", "target_entity": "Oromo", "language": "English"},
            "ተረኛ": {"severity": "high", "target_entity": "Oromo", "language": "Amharic"},
            
            "ትግሬ": {"severity": "medium", "target_entity": "Tigrayan", "language": "Amharic"},
            "tigrayan": {"severity": "medium", "target_entity": "Tigrayan", "language": "English"},
            "አጋሜ": {"severity": "high", "target_entity": "Tigrayan", "language": "Amharic"},
            "አጋሜ ወያኔ ትግሬ ነው": {"severity": "high", "target_entity": "Tigrayan", "language": "Amharic"},
            "ማላም ትግሬ": {"severity": "high", "target_entity": "Tigrayan", "language": "Amharic"},
            "ትግሬ ሌባ ነው": {"severity": "high", "target_entity": "Tigrayan", "language": "Amharic"},
            "ቁልቋላም": {"severity": "high", "target_entity": "Tigrayan", "language": "Amharic"},
            "ቁልቋል በሊታ": {"severity": "high", "target_entity": "Tigrayan", "language": "Amharic"},
            "የቀን ጅብ": {"severity": "high", "target_entity": "Tigrayan", "language": "Amharic"},
            "የቀን ጅብ ዘራፊዎች": {"severity": "high", "target_entity": "Tigrayan", "language": "Amharic"},
            "Tigrayaan bofa": {"severity": "high", "target_entity": "Tigrayan", "language": "Oromo"},
            
            "ቅማንት": {"severity": "medium", "target_entity": "Qemant", "language": "Amharic"},
            "qemant": {"severity": "medium", "target_entity": "Qemant", "language": "English"},
            "አገው": {"severity": "medium", "target_entity": "Agew", "language": "Amharic"},
            "agew": {"severity": "medium", "target_entity": "Agew", "language": "English"},
            #"ሶማሌ": {"severity": "medium", "target_entity": "Somali", "language": "Amharic"},
            #"አፋር": {"severity": "medium", "target_entity": "Afar", "language": "Amharic"},
            "ስልጤ": {"severity": "medium", "target_entity": "Silte", "language": "Amharic"},
            #"ጉራጌ": {"severity": "medium", "target_entity": "Gurage", "language": "Amharic"},
            
            "ወላሞ": {"severity": "high", "target_entity": "Wolayta", "language": "Amharic"},
            "ዲቻ": {"severity": "high", "target_entity": "Wolayta", "language": "Amharic"},
            "ሻንቅላ": {"severity": "high", "target_entity": "Benishangul/Gumuz", "language": "Amharic"},
            "ሻንቅሎች": {"severity": "high", "target_entity": "Benishangul/Gumuz", "language": "Amharic"}
        },
        
        "political_groups": {
            "ብልግና": {"severity": "low", "target_entity": "Prosperity Party", "language": "Amharic"},
            "ብልፅግና ታጥቦ ከጭቃ ነው": {"severity": "medium", "target_entity": "Prosperity Party", "language": "Amharic"},
            "prosperity party": {"severity": "low", "target_entity": "Prosperity Party", "language": "English"},
            
            #"ብአዴን": {"severity": "low", "target_entity": "ADP", "language": "Amharic"},
            "adp": {"severity": "low", "target_entity": "ADP", "language": "English"},
            "በስበሰ አዴፓ": {"severity": "high", "target_entity": "ADP", "language": "Amharic"},
            "የአዴፓ አመራሮች ካላለቁ የአማራ ህዝብ አይድንም": {"severity": "critical", "target_entity": "ADP", "language": "Amharic"},
            "ቆሻሻዉ ብአዴን ዋጋ ይከፍልባታል።": {"severity": "high", "target_entity": "ADP", "language": "Amharic"},
            "ብአዴን ይውደም": {"severity": "critical", "target_entity": "ADP", "language": "Amharic"},
            
            "ፋኖ": {"severity": "medium", "target_entity": "Fano", "language": "Amharic"},
            "fano": {"severity": "medium", "target_entity": "Fano", "language": "English"},
            "Faannoon ajjeesaadha": {"severity": "high", "target_entity": "Fano", "language": "Oromo"},
            
            "ኦነግ": {"severity": "high", "target_entity": "OLF", "language": "Amharic"},
            "oneg": {"severity": "high", "target_entity": "OLF", "language": "English"},
            "ሁሉም ኦሮሞ ኦነግ ነው": {"severity": "high", "target_entity": "OLF", "language": "Amharic"},
            "ሸኔ": {"severity": "high", "target_entity": "OLA", "language": "Amharic"},
            
            "ወያኔ": {"severity": "high", "target_entity": "TPLF", "language": "Amharic"},
            "woyane": {"severity": "high", "target_entity": "TPLF", "language": "English"},
            "የወያኔ ተላላኪ": {"severity": "high", "target_entity": "TPLF", "language": "Amharic"},
            "ሕወሓት": {"severity": "high", "target_entity": "TPLF", "language": "Amharic"},
            "tplf": {"severity": "high", "target_entity": "TPLF", "language": "English"},
            "ጁንታ": {"severity": "high", "target_entity": "TPLF", "language": "Amharic"},
            "junta": {"severity": "high", "target_entity": "TPLF", "language": "English"},
            "ጁንታ ቡድን": {"severity": "high", "target_entity": "TPLF", "language": "Amharic"},
            "ጨካኝ ጁንታ": {"severity": "high", "target_entity": "TPLF", "language": "Amharic"},
            
            "ቄሮ": {"severity": "medium", "target_entity": "Qeerroo", "language": "Amharic"},
            "ደርግ": {"severity": "medium", "target_entity": "Derg", "language": "Amharic"},
            
            "አብን": {"severity": "medium", "target_entity": "NAMA", "language": "Amharic"},
            "የአማራ ጠላት አብንና ብሄርተኞች": {"severity": "high", "target_entity": "NAMA", "language": "Amharic"},
            "ዳውን ዳውን የሽምቅላዎች ፓርቲ አብን": {"severity": "high", "target_entity": "NAMA", "language": "Amharic"},
            
            "አብይ ሌባ ነው": {"severity": "high", "target_entity": "Abiy Ahmed", "language": "Amharic"},
            "Abiy ajjeesaadha": {"severity": "critical", "target_entity": "Abiy Ahmed", "language": "Oromo"},
            
            "Mootummaa gantuu": {"severity": "high", "target_entity": "Government", "language": "Oromo"},
            "Mootummaa ejjituu": {"severity": "high", "target_entity": "Government", "language": "Oromo"},
            "Mootummaa harree": {"severity": "high", "target_entity": "Government", "language": "Oromo"},
            "የማይረባ መንግስት": {"severity": "medium", "target_entity": "Government", "language": "Amharic"},
            "እብድ መንግስት": {"severity": "high", "target_entity": "Government", "language": "Amharic"},
            "ከፋፋይ መንግስት": {"severity": "high", "target_entity": "Government", "language": "Amharic"},
            "አምባገነን መንግስት": {"severity": "high", "target_entity": "Government", "language": "Amharic"},
            "ማፍያ ፓርቲ": {"severity": "high", "target_entity": "Government", "language": "Amharic"},
            "አሳዳጅ ፓርቲ": {"severity": "high", "target_entity": "Government", "language": "Amharic"},
            
            "አሃዳዊ": {"severity": "medium", "target_entity": "Unitarian", "language": "Amharic"},
            "አሃዳዊነት": {"severity": "medium", "target_entity": "Unitarian", "language": "Amharic"},
            "Sirna saba tokkichaa": {"severity": "medium", "target_entity": "Unitarian", "language": "Oromo"},
            "former systems": {"severity": "medium", "target_entity": "Historical Systems", "language": "English"},
            
            "ካድሬ": {"severity": "medium", "target_entity": "Cadre", "language": "Amharic"},
            "Kaadiree qoonqoo": {"severity": "high", "target_entity": "Cadre", "language": "Oromo"},
            "ፈሳም መከላከያ": {"severity": "high", "target_entity": "ENDF", "language": "Amharic"}
        },
        
        "violence_incitement": {
            "ግደል": {"severity": "critical", "target_entity": "", "language": "Amharic"},
            "kill": {"severity": "critical", "target_entity": "", "language": "English"},
            "ግደሉ": {"severity": "critical", "target_entity": "", "language": "Amharic"},
            "kill them": {"severity": "critical", "target_entity": "", "language": "English"},
            "ይገደል": {"severity": "critical", "target_entity": "", "language": "Amharic"},
            "ይግደሉ": {"severity": "critical", "target_entity": "", "language": "Amharic"},
            "Isaan ajjeesi": {"severity": "critical", "target_entity": "", "language": "Oromo"},
            "Ajjeefamuu qaba": {"severity": "critical", "target_entity": "", "language": "Oromo"},
            
            "destroy": {"severity": "critical", "target_entity": "", "language": "English"},
            "አጥፋ": {"severity": "critical", "target_entity": "", "language": "Amharic"},
            "አጥፋቸው": {"severity": "critical", "target_entity": "", "language": "Amharic"},
            "እናጥፋቸው": {"severity": "critical", "target_entity": "", "language": "Amharic"},
            "አሳዶ ማጥፋት": {"severity": "critical", "target_entity": "", "language": "Amharic"},
            "Isaan balleessi": {"severity": "critical", "target_entity": "", "language": "Oromo"},
            "ማጥፋት": {"severity": "critical", "target_entity": "", "language": "Amharic"},
            
            "ጦርነት": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "war": {"severity": "high", "target_entity": "", "language": "English"},
            "ጥቃት": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "attack": {"severity": "high", "target_entity": "", "language": "English"},
            "ማጥቃት ይቀጥል": {"severity": "critical", "target_entity": "", "language": "Amharic"},
            "ስጋት": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "threat": {"severity": "medium", "target_entity": "", "language": "English"},
            
            "ዘር ማጥፋት": {"severity": "critical", "target_entity": "", "language": "Amharic"},
            "የዘር ማጥፋት": {"severity": "critical", "target_entity": "", "language": "Amharic"},
            "Genocider": {"severity": "critical", "target_entity": "", "language": "English"},
            "massacre": {"severity": "critical", "target_entity": "", "language": "English"},
            "Fixiinsa jumlaa": {"severity": "critical", "target_entity": "", "language": "Oromo"},
            
            "እርስ በእርስ መጋደል": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ንብረት ማጥፋት": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ይወገዱ": {"severity": "critical", "target_entity": "", "language": "Amharic"},
            "ሁሉም ይወገዱ": {"severity": "critical", "target_entity": "", "language": "Amharic"},
            "Isaan ari'i": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "ይባረሩ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ሁሉም ይፈናቀሉ": {"severity": "critical", "target_entity": "", "language": "Amharic"},
            "በቅርቡ ታልቃላችሁ": {"severity": "critical", "target_entity": "", "language": "Amharic"},
            
            "Tigrayaan ajjeesuu dha": {"severity": "critical", "target_entity": "Tigrayan", "language": "Oromo"},
            "Death to \"Weyane\" (TPLF)": {"severity": "critical", "target_entity": "TPLF", "language": "English"},
            "Oromoon ni jibba": {"severity": "high", "target_entity": "Oromo", "language": "Oromo"},
            "Farra Amaaraa": {"severity": "high", "target_entity": "Amhara", "language": "Oromo"},
            
            "አማፂ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ተዋጋ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ተጋደል": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "Waraanuu": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "Waraani": {"severity": "high", "target_entity": "", "language": "Oromo"},
            
            "Dhiiga dhangalaasuu": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "Dhiig-dhangalaasaa": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "Awwaala gadi buusuu": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "Haa du'an": {"severity": "critical", "target_entity": "", "language": "Oromo"},
            "Haa gubatu": {"severity": "critical", "target_entity": "", "language": "Oromo"},
            "ይታረድ": {"severity": "critical", "target_entity": "", "language": "Amharic"},
            "እርምጃ ይወሰድ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "በሜንጫ ቁረጥ": {"severity": "critical", "target_entity": "", "language": "Amharic"},
            "ሊንች ይደረግ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            
            "እደፍርሻለሁ": {"severity": "critical", "target_entity": "", "language": "Amharic"},
            "መቅበር": {"severity": "critical", "target_entity": "", "language": "Amharic"},
            "አበላሻሻለሁ": {"severity": "critical", "target_entity": "", "language": "Amharic"},
            "አሲድ እንዳልደፋብሽ": {"severity": "critical", "target_entity": "", "language": "Amharic"},
            "ግድያ": {"severity": "critical", "target_entity": "", "language": "Amharic"},
            
            "Mimmilki": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "Grab your stick": {"severity": "high", "target_entity": "", "language": "English"},
            "Fight with sticks and bayonets": {"severity": "critical", "target_entity": "", "language": "English"},
            "Let the struggle intensify": {"severity": "high", "target_entity": "", "language": "English"},
            
            "Foxxoqxuu": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "Foxxoqsituu": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "Buttuu/Haqaaqxuu": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "Dhiibbattuu": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "Saamicha": {"severity": "medium", "target_entity": "", "language": "Oromo"},
            "Gidirsaa": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "Madduu": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "ሀገር ትፍረስ": {"severity": "critical", "target_entity": "", "language": "Amharic"},
            "Nu ajjeesaa nu fixaa": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "Bittiinteen haa baatu": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "Otoo duute siif wayya": {"severity": "high", "target_entity": "", "language": "Oromo"}
        },
        
        # === Dehumanizing & Derogatory Terms ===
        "dehumanizing": {
            "እንስሳ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "animal": {"severity": "high", "target_entity": "", "language": "English"},
            "ከብት": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ኦሮሞ ከብት": {"severity": "high", "target_entity": "Oromo", "language": "Amharic"},
            "cattle": {"severity": "high", "target_entity": "", "language": "English"},
            
            "ውሻ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ውሻ ዘር": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ዝም በይ ውሻ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "dog": {"severity": "high", "target_entity": "", "language": "English"},
            "Ilmoo saree": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "Kutichaa": {"severity": "high", "target_entity": "", "language": "Oromo"},
            
            "አህያ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "አሕያ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "የአህያ ዘር": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "የአህያ ልጅ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            
            "አसाማ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ጅብ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "Waraabessa": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "Waraabessa / Shiftaa": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "የቀን ጅቦች": {"severity": "high", "target_entity": "Tigrayan", "language": "Amharic"},
            
            "ዝንዶ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "እሬሳ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "Carcass": {"severity": "high", "target_entity": "", "language": "English"},
            "ተባይ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "በረሮ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "Tisiisa": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "Bineensa": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "ጨካኝ አውሬ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ጭራቅ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "አራዊት": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "Bofa": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "Karkarroo": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "Maggot / Worm": {"severity": "high", "target_entity": "", "language": "English"},
            "Water hyacinth": {"severity": "medium", "target_entity": "", "language": "English"},
            "Monkey": {"severity": "high", "target_entity": "", "language": "English"},
            "Elephant": {"severity": "high", "target_entity": "", "language": "English"},
            "Sub-human": {"severity": "high", "target_entity": "", "language": "English"},
            
            "ጥንብ አንሳ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ቆሻሻ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "trash": {"severity": "high", "target_entity": "", "language": "English"},
            "Stinky": {"severity": "high", "target_entity": "", "language": "English"},
            "ጥምባታም": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ግማታም": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "የሸተተ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "xirooftuu": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "Dhiqamee dhoqqee": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "Kosii": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "ፋንዲያ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "እበት": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "አር": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "አተላ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            
            "ሌባ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "thief": {"severity": "high", "target_entity": "", "language": "English"},
            "Hattuu": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "ሞላጫ ሌባ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "Plunderer": {"severity": "high", "target_entity": "", "language": "English"},
            "የሌባ ዘር": {"severity": "high", "target_entity": "", "language": "Amharic"},
            
            "ነፍሰ ገዳይ": {"severity": "critical", "target_entity": "", "language": "Amharic"},
            "ገዳይ": {"severity": "critical", "target_entity": "", "language": "Amharic"},
            "killer": {"severity": "critical", "target_entity": "", "language": "English"},
            "Ajjeesaa/Ajjeestuu": {"severity": "critical", "target_entity": "", "language": "Oromo"},
            "Abiy the executioner": {"severity": "high", "target_entity": "Abiy Ahmed", "language": "English"},
            
            "ዘረኛ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ጠባብ ዘረኛ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ጠባብ ብሔርተኛ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            
            "ደደብ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "fool": {"severity": "medium", "target_entity": "", "language": "English"},
            "ጅል": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "ደንቆሮ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "የተማረው ደንቆሮ የቆሎ ተማሪ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "የአንተው ድንቁርናስ ለከት የለውም": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ደንቆሮ አስተሳሰብ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ደንቆሮ ብሄር": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "Doofaa": {"severity": "medium", "target_entity": "", "language": "Oromo"},
            "Gowwaa": {"severity": "low", "target_entity": "", "language": "Oromo"},
            "Empty head": {"severity": "medium", "target_entity": "", "language": "English"},
            "ዶማ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ድንጋይ ራስ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ነፈዝ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "Mataa-gogogaa": {"severity": "medium", "target_entity": "", "language": "Oromo"},
            "Ignorant": {"severity": "medium", "target_entity": "", "language": "English"},
            
            "እብድ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ክፉ እብድ አስተሳሰብ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "Machaa'aa": {"severity": "medium", "target_entity": "", "language": "Oromo"},
            "ሰነፍ": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "ባለጌ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "በጣም ባለጌ አስተሳሰብ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            
            "ከሃዲ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "Traitor": {"severity": "high", "target_entity": "", "language": "English"},
            "Son of a traitor": {"severity": "high", "target_entity": "", "language": "English"},
            "Son of a \"Banda\" / Traitor": {"severity": "high", "target_entity": "", "language": "English"},
            "ባንዳ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ባንዳዎች": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "የባንዳ ዘር": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "Gantuu": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "Gosa gantuu ta'e": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "Goobanaa": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "Galtuu / Gantuu": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "ይሁዳ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            
            "ጨካኝ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ጨካኝ ሰው": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ጨካኝ አስተሳሰብ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ጨካኝ ተንኮለኛ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ጨካኝ ተንኮለኛ ባህሪ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "Gara-jabeessa (Aramanee)": {"severity": "high", "target_entity": "", "language": "Oromo"},
            
            "ተንኮለኛ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "የተንኮለኛ ዘር": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ተንኮል": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            
            "ስግብግብ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ስስታም": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ራስ ወዳድ": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "Garaa isaaniitiif qofa kan jiraatan": {"severity": "medium", "target_entity": "", "language": "Oromo"},
            
            "ጥበኛ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ጥጋባችሁ ይበርዳል": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ዐይን አውጣ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "አይን አውጣ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "አይነ ደረቅ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            
            "ፋራ": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "ግልፍጥ": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "ለፋፊ": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "ገርዳሜ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ቀፋፊ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ፋር": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "ክፍት አፍ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "Hasaa-baay'iftuu / Lallabdii": {"severity": "low", "target_entity": "", "language": "Oromo"},
            
            "ጉረኛ": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "ወሬ ብቻ ጉራ ብቻ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ጉራ ብቻ": {"severity": "low", "target_entity": "", "language": "Amharic"},
            
            "ቆማጣ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ቅማላም": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ተለፊ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ባለጊዜ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ጥረኛ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ጠብ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            
            "አሽቃባጭ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "Bootlicker / Sycophant": {"severity": "medium", "target_entity": "", "language": "English"},
            "ከፋይ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ጥቅመኛ": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "በዝባዥ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ብዝበዛ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "Oppressor": {"severity": "medium", "target_entity": "", "language": "English"},
            "oppressive": {"severity": "medium", "target_entity": "", "language": "English"},
            "oppression": {"severity": "medium", "target_entity": "", "language": "English"},
            
            "ንቃ": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "Ka'i": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "Dadhabaa": {"severity": "low", "target_entity": "", "language": "Oromo"},
            "Yeelaltuu": {"severity": "low", "target_entity": "", "language": "Oromo"},
            "Caccabaa / Of-tuulaa": {"severity": "medium", "target_entity": "", "language": "Oromo"},
            "Hirkattuu/Maxxantuu": {"severity": "medium", "target_entity": "", "language": "Oromo"},
            "Tortortuu": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "Hamii/Odeessa": {"severity": "low", "target_entity": "", "language": "Oromo"},
            "Gossip": {"severity": "low", "target_entity": "", "language": "English"},
            
            "ልጋጋም": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ልሀጫም": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ልሃጫም": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ቅንጣም": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ቅዘናም": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ፈሳም": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ቦቅቧቃ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ፈሪ": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "ዝርጥርጥ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "Dough (Weak / Spineless)": {"severity": "medium", "target_entity": "", "language": "English"},
            "ታስላላችሁ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "Na ciigasisa": {"severity": "medium", "target_entity": "", "language": "Oromo"},
            "Fokkisaa / Jibbisiisaa": {"severity": "medium", "target_entity": "", "language": "Oromo"},
            "አስቀያሚ": {"severity": "low", "target_entity": "", "language": "Amharic"},
            
            "ውሸታም": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ቀጣፊ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "Soba": {"severity": "medium", "target_entity": "", "language": "Oromo"},
            "ውሸት": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "ነጭ ውሸት": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "Full of holes": {"severity": "medium", "target_entity": "", "language": "English"},
            
            "ከንቱ": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "ገለባ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "እርባና ቢስ": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "የማይረባ": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "Crumb": {"severity": "low", "target_entity": "", "language": "English"},
            "ውዳቂ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "Badaa": {"severity": "low", "target_entity": "", "language": "Oromo"},
            "Amala badaa": {"severity": "medium", "target_entity": "", "language": "Oromo"},
            "Salphina": {"severity": "medium", "target_entity": "", "language": "Oromo"},
            "Disgrace": {"severity": "medium", "target_entity": "", "language": "English"},
            
            "ነጭናጫ": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "ተናዳጅ": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "Annoying": {"severity": "low", "target_entity": "", "language": "English"},
            
            "ልታም": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ጡት ነካሽ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "የቁም ሞት": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "የሞተ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "Otoo duute siif wayya": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "Joker": {"severity": "low", "target_entity": "", "language": "English"},
            "You are intentionally playing dead": {"severity": "medium", "target_entity": "", "language": "English"},
            "Doom-monger": {"severity": "medium", "target_entity": "", "language": "English"},
            "Moluu": {"severity": "low", "target_entity": "", "language": "Oromo"},
            
            "ኩላ ራስ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ወሸላ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ተበዳ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ተበዳ እና ሙት": {"severity": "critical", "target_entity": "", "language": "Amharic"},
            "የእናትህ እምስ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "እናትህ ትበዳ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ዳታም": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "የአህያ ጀላ ይግባብህ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "Dhangala'aa": {"severity": "high", "target_entity": "", "language": "Oromo"},
            
            "ትምክተኛ": {"severity": "high", "target_entity": "Amhara", "language": "Amharic"},
            "ባለቤት": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "Harka-faktuu": {"severity": "medium", "target_entity": "", "language": "Oromo"},
            "Oduu qofaa": {"severity": "low", "target_entity": "", "language": "Oromo"},
            "ትረኩት ፈላጊ": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "እንቅልፋም": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "ቀርፋፋ": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "ሉዘር": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "ቅሌት": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ቅሌታም ሽማግሌ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            
            "ወራዳ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ወራዳ መሪ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ጭራ መቁላት": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ልክስክስ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ታሪካዊ ጠላት": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ጠማማ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ለምጻም": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ንፉግ": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "ቀማኛ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ቀጣፊ አጭበርባሪ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ቅጥረኛ ተላላኪ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "በዝባዥ ቡድን": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ከፋይ አካል": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ሴረኛ አካል": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ምድረ ደነዝ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ብልሹ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            
            "የአማራ ጠላት": {"severity": "high", "target_entity": "Amhara", "language": "Amharic"},
            "የሀገር ጠላት": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ጠላት": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ጠላት ነው": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ምቀኛ": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "ተራ": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "ተላላኪ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ይወገዱ": {"severity": "critical", "target_entity": "", "language": "Amharic"},
            
            "Kansarii biyyaa": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "Jifuu": {"severity": "medium", "target_entity": "", "language": "Oromo"},
            "Yakkaamaa": {"severity": "medium", "target_entity": "", "language": "Oromo"},
            "Dhiiga-maleessa / Adii": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "Fakkeessituu": {"severity": "low", "target_entity": "", "language": "Oromo"},
            "Haftee": {"severity": "low", "target_entity": "", "language": "Oromo"},
            
            "Sharp-tongued": {"severity": "low", "target_entity": "", "language": "English"},
            "Where the hell were you?": {"severity": "medium", "target_entity": "", "language": "English"},
            "Backward / Primitive": {"severity": "medium", "target_entity": "", "language": "English"},
            "Primitive / Backward rustic": {"severity": "medium", "target_entity": "", "language": "English"},
            
            "ሰይጣን": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "አጋንንት": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "እርኩስ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "የረከሰ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "Lwusifar (Seexana)": {"severity": "high", "target_entity": "", "language": "Oromo"}
        },
        
        # === Election, Governance & Political Conflict Terms ===
        "election_governance": {
            "ምрጫ": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "election": {"severity": "low", "target_entity": "", "language": "English"},
            "ድምፅ": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "ድምጽ": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "vote": {"severity": "low", "target_entity": "", "language": "English"},
            "Sagalee": {"severity": "low", "target_entity": "", "language": "Oromo"},
            "Filannoo": {"severity": "low", "target_entity": "", "language": "Oromo"},
            
            "ኢሰመቦ": {"severity": "low", "target_entity": "NEBE", "language": "Amharic"},
            "የምርጫ ቦርድ": {"severity": "low", "target_entity": "NEBE", "language": "Amharic"},
            "nebe": {"severity": "low", "target_entity": "NEBE", "language": "English"},
            "Boordii Filannoo": {"severity": "low", "target_entity": "NEBE", "language": "Oromo"},
            
            "የተጭበረበረ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ማጭበርበር": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "rigged": {"severity": "medium", "target_entity": "", "language": "English"},
            "fraud": {"severity": "medium", "target_entity": "", "language": "English"},
            "Hata": {"severity": "medium", "target_entity": "", "language": "Oromo"},
            "Maliंग": {"severity": "medium", "target_entity": "", "language": "Oromo"},
            
            "የይስሙላ ምርጫ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ምርጫ ሳይሆን ድራማ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "Sham election": {"severity": "high", "target_entity": "", "language": "English"},
            "Symbolic exercise": {"severity": "medium", "target_entity": "", "language": "English"},
            "Elite bargain": {"severity": "medium", "target_entity": "", "language": "English"},
            
            "ማዕቀብ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ማግለል": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "Boycott": {"severity": "medium", "target_entity": "", "language": "English"},
            "Exclusion": {"severity": "medium", "target_entity": "", "language": "English"},
            "Lagannaa filannoo": {"severity": "medium", "target_entity": "", "language": "Oromo"},
            
            "አምባገነን": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "አድሎ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "Dictator": {"severity": "medium", "target_entity": "", "language": "English"},
            "Authoritarian": {"severity": "medium", "target_entity": "", "language": "English"},
            "Abbaa-irree": {"severity": "medium", "target_entity": "", "language": "Oromo"},
            
            "ከፋፍለህ ግዛ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "የአፓርታይድ ሥርዓት": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "አፓርታይድ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "Apartheid": {"severity": "high", "target_entity": "", "language": "English"},
            "Qoodanii bulchuu": {"severity": "high", "target_entity": "", "language": "Oromo"},
            
            "ካድሬ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "የካድሬ ፖለቲካ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "Cadre": {"severity": "medium", "target_entity": "", "language": "English"},
            "Kaadiree": {"severity": "medium", "target_entity": "", "language": "Oromo"},
            "Kaadiree qonqoo": {"severity": "high", "target_entity": "", "language": "Oromo"}
        },
        
        # === Cross-Border Geopolitical Narratives ===
        "Cross-Border Geopolitical Narratives": {
            "ግብፅ": {"severity": "low", "target_entity": "Egypt", "language": "Amharic"},
            "egypt": {"severity": "low", "target_entity": "Egypt", "language": "English"},
            "ሱዳን": {"severity": "low", "target_entity": "Sudan", "language": "Amharic"},
            "sudan": {"severity": "low", "target_entity": "Sudan", "language": "English"},
            "ኤርትራ": {"severity": "low", "target_entity": "Eritrea", "language": "Amharic"},
            "eritrea": {"severity": "low", "target_entity": "Eritrea", "language": "English"},
            "ሻብያ": {"severity": "high", "target_entity": "Eritrea/EPLF", "language": "Amharic"},
            "ሻቢያ": {"severity": "high", "target_entity": "Eritrea/EPLF", "language": "Amharic"},
            "አሜሪካ": {"severity": "low", "target_entity": "USA", "language": "Amharic"},
            "america": {"severity": "low", "target_entity": "USA", "language": "English"},
            "ቻይና": {"severity": "low", "target_entity": "China", "language": "Amharic"},
            "china": {"severity": "low", "target_entity": "China", "language": "English"},
            "ኤምሬትስ": {"severity": "low", "target_entity": "UAE", "language": "Amharic"},
            "uae": {"severity": "low", "target_entity": "UAE", "language": "English"},
            
            "foreign interference": {"severity": "medium", "target_entity": "", "language": "English"},
            "foreign agent": {"severity": "medium", "target_entity": "", "language": "English"},
            "foreign backed": {"severity": "medium", "target_entity": "", "language": "English"},
            "foreign funded": {"severity": "medium", "target_entity": "", "language": "English"},
            "foreign meddling": {"severity": "medium", "target_entity": "", "language": "English"},
            "foreign hand": {"severity": "medium", "target_entity": "", "language": "English"},
            "foreign plot": {"severity": "medium", "target_entity": "", "language": "English"},
            "foreign puppet": {"severity": "medium", "target_entity": "", "language": "English"},
            "foreign conspiracy": {"severity": "medium", "target_entity": "", "language": "English"},
            "western puppet": {"severity": "medium", "target_entity": "", "language": "English"},
            "የውጭ ጣልቃ ገብነት": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "የውጭ እጅ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "Proxy war": {"severity": "medium", "target_entity": "", "language": "English"},
            "የproxy ጦርነት": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            
            "መጤ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ገንዳ አፍ መጤ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "immigrant": {"severity": "medium", "target_entity": "", "language": "English"},
            "ሰፋሪ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ሰፋሪዎች": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "settler": {"severity": "high", "target_entity": "", "language": "English"},
            "middle settler": {"severity": "medium", "target_entity": "", "language": "English"},
            
            "ባዳ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "የባዳ ዘር": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ባዳ የባዕድ ዘር": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ወፍ ዘራሽ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "Baqataa": {"severity": "medium", "target_entity": "", "language": "Oromo"},
            "Baqaattuu / Baqataa": {"severity": "medium", "target_entity": "", "language": "Oromo"},
            
            "አግላይ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ተስፋፊ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "Baballiftuu": {"severity": "high", "target_entity": "", "language": "Oromo"},
            "የአካባቢ የበላይነት": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            
            "ቅኝ ገዥ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ቅኝ ገዢ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ቅኝ ገዢ ሀገር": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "እነሱ ቅኝ ገዥዎች ናቸው": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "Italian": {"severity": "medium", "target_entity": "", "language": "English"},
            "Neo-colonialism": {"severity": "medium", "target_entity": "", "language": "English"},
            
            "Invader / Aggressor": {"severity": "high", "target_entity": "", "language": "English"},
            "ወራሪ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "Nomad clan / Vagabond / Pastoralist": {"severity": "medium", "target_entity": "", "language": "English"},
            "Pastoralist": {"severity": "low", "target_entity": "", "language": "English"},
            "Stateless": {"severity": "medium", "target_entity": "", "language": "English"}
        },
        
        # === Religious, Cultural & Extremism Terms ===
        "religious_cultural": {
            "ኦርቶዶክስ": {"severity": "low", "target_entity": "Orthodox", "language": "Amharic"},
            "orthodox": {"severity": "low", "target_entity": "Orthodox", "language": "English"},
            "ቄስ": {"severity": "low", "target_entity": "Orthodox", "language": "Amharic"},
            "ደብተራ": {"severity": "medium", "target_entity": "Orthodox", "language": "Amharic"},
            
            "እስልምና": {"severity": "low", "target_entity": "Islam", "language": "Amharic"},
            "islam": {"severity": "low", "target_entity": "Islam", "language": "English"},
            "እስላማዊ መንግስት": {"severity": "medium", "target_entity": "Islam", "language": "Amharic"},
            "ኢስላሚስት": {"severity": "medium", "target_entity": "Islam", "language": "Amharic"},
            "ጅሀድ": {"severity": "medium", "target_entity": "Islam", "language": "Amharic"},
            "ጅሀዲስት": {"severity": "high", "target_entity": "Islam", "language": "Amharic"},
            "ዋሃቢይ": {"severity": "medium", "target_entity": "Wahhabism", "language": "Amharic"},
            "አህባሽ": {"severity": "medium", "target_entity": "Ahbash", "language": "Amharic"},
            
            "አልሸባብ": {"severity": "high", "target_entity": "Al-Shabaab", "language": "Amharic"},
            "ቦኮ ሀራም": {"severity": "high", "target_entity": "Boko Haram", "language": "Amharic"},
            "ፈላሻ": {"severity": "high", "target_entity": "Beta Israel", "language": "Amharic"},
            
            "መናፍቅ": {"severity": "high", "target_entity": "Protestant/Other", "language": "Amharic"},
            "መናፍቃን": {"severity": "high", "target_entity": "Protestant/Other", "language": "Amharic"},
            "አህዛብ": {"severity": "medium", "target_entity": "Non-believers", "language": "Amharic"},
            "ቃፊር": {"severity": "high", "target_entity": "Non-believers", "language": "Amharic"},
            "ኢ-አማኒ": {"severity": "medium", "target_entity": "Atheist", "language": "Amharic"},
            "sinful": {"severity": "high", "target_entity": "", "language": "English"},
            
            "ጣኦት": {"severity": "medium", "target_entity": "Pagan", "language": "Amharic"},
            "ጣኦት አምላኪ": {"severity": "high", "target_entity": "Pagan", "language": "Amharic"},
            "Waaqeffataa": {"severity": "low", "target_entity": "Waaqeffanna", "language": "Oromo"},
            
            "ቡዳ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "የቡዳ ባህሪ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ጠንቋይ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "መተታም": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "አስማተኛ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ኢሉሚናቲ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "Raajii sobaa": {"severity": "medium", "target_entity": "", "language": "Oromo"},
            
            "ገሀነም": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "ገሀነም ግባ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            
            "ሞጣ": {"severity": "low", "target_entity": "Muslims", "language": "Amharic"},
            "የእርጎ ዝንብ": {"severity": "high", "target_entity": "Muslims", "language": "Amharic"},
            "እሬቻ": {"severity": "low", "target_entity": "Oromo/Irreecha", "language": "Amharic"},
            "ምድረ እሬቻም": {"severity": "high", "target_entity": "Oromo/Irreecha", "language": "Amharic"},
            
            "ጽንፈኛ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "አክራሪ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ፋሽሲዝም": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "ፋሽስት": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "Faashistii": {"severity": "high", "target_entity": "", "language": "Oromo"},
            
            "ሴራ": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "ሴረኛ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "የሴራ ሀሳብ": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "የፖለቲካ ሴራ": {"severity": "medium", "target_entity": "", "language": "Amharic"}
        },
        
        # === Gender-Based & Misogynistic Terms ===
       "gender_misogynistic": {
            "ሸርሙጣ": {"severity": "high", "target_entity": "Women", "language": "Amharic"},
            "whore": {"severity": "high", "target_entity": "Women", "language": "English"},
            "bitch": {"severity": "high", "target_entity": "Women", "language": "English"},
            "sagaagaltuu": {"severity": "high", "target_entity": "Women", "language": "Oromo"},
            
            "የጭን ገረድ": {"severity": "high", "target_entity": "Women", "language": "Amharic"},
            "ገረድ": {"severity": "medium", "target_entity": "Women", "language": "Amharic"},
            
            "ያረጀች": {"severity": "medium", "target_entity": "Women", "language": "Amharic"},
            "ያረጀሽ ነሽ": {"severity": "medium", "target_entity": "Women", "language": "Amharic"},
            "አሮጊት": {"severity": "medium", "target_entity": "Women", "language": "Amharic"},
            "ቅሌታም አሮጊት": {"severity": "high", "target_entity": "Women", "language": "Amharic"},
            "spinster": {"severity": "low", "target_entity": "Women", "language": "English"},
            
            "የተላሸች": {"severity": "high", "target_entity": "Women", "language": "Amharic"},
            "የለቀቀች": {"severity": "medium", "target_entity": "Women", "language": "Amharic"},
            "ልቅ": {"severity": "medium", "target_entity": "Women", "language": "Amharic"},
            "አተማሽ": {"severity": "medium", "target_entity": "Women", "language": "Amharic"},
            "ሴሰኛ": {"severity": "medium", "target_entity": "Women", "language": "Amharic"},
            "ጋለሞታ": {"severity": "medium", "target_entity": "Women", "language": "Amharic"},
            "ቁሌታም": {"severity": "high", "target_entity": "Women", "language": "Amharic"},
            "ክፉ ሴተኛ አዳሪ ዘር": {"severity": "high", "target_entity": "Women", "language": "Amharic"},
            "ራቁት ሴቶች": {"severity": "medium", "target_entity": "Women", "language": "Amharic"},
            
            "የተፈጸመባት ጥቃት በአለባበሷ ነው": {"severity": "high", "target_entity": "Women", "language": "Amharic"},
            "ይሄኔ የሆነ ወንድ ነው የለቀቀብሽ": {"severity": "high", "target_entity": "Women", "language": "Amharic"},
            "በቅርቡ ጋሽን ታገኛለሽ": {"severity": "high", "target_entity": "Women", "language": "Amharic"},
            
            "ሴት በዛ ጎመን ጠነዛ": {"severity": "medium", "target_entity": "Women", "language": "Amharic"},
            "ሴት መምራት አትችልም": {"severity": "high", "target_entity": "Women", "language": "Amharic"},
            "ወንድ ይምራ ሴት ትከተል": {"severity": "medium", "target_entity": "Women", "language": "Amharic"},
            "መጽሐፍ ቅዱስ እንደሚለው ሴት ለባሏ ትገዛ": {"severity": "medium", "target_entity": "Women", "language": "Amharic"},
            "የበታች መሆን አለባቸው": {"severity": "high", "target_entity": "Women", "language": "Amharic"},
            "የበታች": {"severity": "medium", "target_entity": "Women", "language": "Amharic"},
            "ሴቶች ታዛዥ": {"severity": "medium", "target_entity": "Women", "language": "Amharic"},
            "ሴቶች አጋዥ ናቸው እንጂ መሪ መሆን የለባቸውም": {"severity": "high", "target_entity": "Women", "language": "Amharic"},
            "አባታዊነትና የወንድ የበላይነት": {"severity": "medium", "target_entity": "Women", "language": "Amharic"},
            "ሴቶች ድምፃቸውን ከፍ አድርገው መናገር የለባቸውም": {"severity": "high", "target_entity": "Women", "language": "Amharic"},
            "ሴት ወደ ወጥ ቤትሽ": {"severity": "high", "target_entity": "Women", "language": "Amharic"},
            "ወደ ጓዳ ግቢ": {"severity": "medium", "target_entity": "Women", "language": "Amharic"},
            
            "Niitii sooressaa": {"severity": "medium", "target_entity": "Women", "language": "Oromo"},
            "dhaqna-hinqabanne": {"severity": "high", "target_entity": "Women", "language": "Oromo"},
            "Balfamtuu": {"severity": "medium", "target_entity": "Women", "language": "Oromo"},
            "Qoollifatamtuu": {"severity": "medium", "target_entity": "Women", "language": "Oromo"},
            "madaqsoo": {"severity": "high", "target_entity": "Women", "language": "Oromo"},
            
            "የሴት ልጅ": {"severity": "low", "target_entity": "Men", "language": "Amharic"},
            "ሴት ያሳደገው": {"severity": "medium", "target_entity": "Men", "language": "Amharic"},
            "ሴታሴት": {"severity": "medium", "target_entity": "Men", "language": "Amharic"},
            "እንደሴት አትሁን": {"severity": "medium", "target_entity": "Men", "language": "Amharic"},
            "ሴት አትሁን": {"severity": "medium", "target_entity": "Men", "language": "Amharic"},
            "ሴት አውል": {"severity": "medium", "target_entity": "Men", "language": "Amharic"},
            "ሴት አዳኝ": {"severity": "low", "target_entity": "Men", "language": "Amharic"},
            
            "ወንዳገረድ": {"severity": "medium", "target_entity": "Intersex/Women", "language": "Amharic"},
            "ወንድ የመሰለች": {"severity": "medium", "target_entity": "Women", "language": "Amharic"}
        },
        
        # === Discriminatory & Homophobic Terms ===
        "discriminatory_homophobic": {
            "ግብረ ሰዶም": {"severity": "high", "target_entity": "LGBTQ+", "language": "Amharic"},
            "ግብረ ሰዶማዊ": {"severity": "high", "target_entity": "LGBTQ+", "language": "Amharic"},
            "ግብረ ሰዶማዊ አመለካከት": {"severity": "high", "target_entity": "LGBTQ+", "language": "Amharic"},
            "ሰዶማዊ": {"severity": "high", "target_entity": "LGBTQ+", "language": "Amharic"},
            "ሉቲ": {"severity": "high", "target_entity": "LGBTQ+", "language": "Amharic"},
            
            "ሌዝቢያን": {"severity": "medium", "target_entity": "LGBTQ+", "language": "Amharic"},
            "ጌይ": {"severity": "high", "target_entity": "LGBTQ+", "language": "Amharic"},
            "lesbian": {"severity": "low", "target_entity": "LGBTQ+", "language": "English"},
            "gay": {"severity": "low", "target_entity": "LGBTQ+", "language": "English"},
            "homosexual": {"severity": "low", "target_entity": "LGBTQ+", "language": "English"},
            
            "ቡሽቲ": {"severity": "high", "target_entity": "LGBTQ+", "language": "Amharic"},
            "ቡሽቲዎች": {"severity": "high", "target_entity": "LGBTQ+", "language": "Amharic"},
            "faggot": {"severity": "high", "target_entity": "LGBTQ+", "language": "English"}
        },
        
        # === Socio-Economic & Caste-Based Terms ===
        "socio_economic_caste": {
            "beggar": {"severity": "medium", "target_entity": "", "language": "English"},
            "ለምናኝ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ረሀብተኛ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ቁራጭ ፈላጊ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            
            "ገረድ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "አሽከር": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "አገልጋይ": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "ቂጥ አጣቢ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "Dishwasher": {"severity": "medium", "target_entity": "", "language": "English"},
            
            "ባሪያ": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "የባሪያ አመለካከት": {"severity": "high", "target_entity": "", "language": "Amharic"},
            "Garee garbaa": {"severity": "high", "target_entity": "", "language": "Oromo"},
            
            "ፋቂ": {"severity": "high", "target_entity": "Faqi Caste", "language": "Amharic"},
            "ቡዳ": {"severity": "high", "target_entity": "Caste/Artisan", "language": "Amharic"},
            
            "መንጋ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "የመንጋ መሪ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            "ኮተታም": {"severity": "low", "target_entity": "", "language": "Amharic"},
            "ደርቡሽ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            
            "Daakuu-galeessa": {"severity": "medium", "target_entity": "", "language": "Oromo"},
            "Daara-degaa": {"severity": "medium", "target_entity": "", "language": "Oromo"},
            "Kiraa sassaabduu": {"severity": "medium", "target_entity": "", "language": "Oromo"},
            "Yartuu": {"severity": "medium", "target_entity": "", "language": "Oromo"},
            "Sirna dhalbeettii": {"severity": "medium", "target_entity": "", "language": "Oromo"},
            
            "middle settler": {"severity": "medium", "target_entity": "", "language": "English"}
        }
    },
    
    # === Risk Scoring Configuration ===
    "risk_scoring": {
        "severity_weights": {"low": 1, "medium": 2, "high": 3, "critical": 4},
        "category_weights": {
            "ethnic_identity": 1.5, 
            "political_groups": 1.2, 
            "violence_incitement": 1.5,
            "dehumanizing": 1.5, 
            "election_governance": 1.0, 
            "foreign_interference": 1.0, 
            "religious_cultural": 1.0,
            "gender_misogynistic": 1.4,
            "discriminatory_homophobic": 1.4,
            "socio_economic_caste": 1.1
        },
        "risk_thresholds": {"low": 3, "medium": 6, "high": 10, "critical": 15}
    },
    
    # === Display Configuration ===
    "display": {"max_terms_per_category": 50, "show_amharic_first": True, "highlight_critical": True}
}
# === STREAMLIT-STYLE DATA PROCESSING FUNCTIONS ===

def infer_platform_from_url(url):
    """Infer platform from URL (Streamlit logic)"""
    if pd.isna(url) or not isinstance(url, str) or not url.startswith("http"):
        return "Unknown"
    url = url.lower()
    platforms = {
        "tiktok.com": "TikTok", "vt.tiktok.com": "TikTok",
        "facebook.com": "Facebook", "fb.watch": "Facebook",
        "twitter.com": "X", "x.com": "X",
        "youtube.com": "YouTube", "youtu.be": "YouTube",
        "instagram.com": "Instagram",
        "telegram.me": "Telegram", "t.me": "Telegram", "telegram.org": "Telegram"
    }
    for key, val in platforms.items():
        if key in url:
            return val
    if any(d in url for d in ["nytimes.com", "bbc.com", "cnn.com", "reuters.com", "aljazeera.com"]):
        return "News/Media"
    return "Media"


def extract_original_text(text):
    """Extract clean original text from post content"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    cleaned = re.sub(r'^(RT|rt|QT|qt|repost|shared|via|credit)\s*[:@]\s*', '', text, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'@\w+|http\S+|www\S+|https\S+', '', cleaned).strip()
    cleaned = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}\b', '', cleaned)
    return re.sub(r'\s+', ' ', cleaned).strip().lower()


def is_original_post(text):
    """Check if post is original (not a repost/retweet)"""
    if pd.isna(text) or not isinstance(text, str):
        return False
    lower = text.strip().lower()
    if not lower:
        return False
    patterns = [
        r'^🔁.*reposted', r'\b(reposted|reshared|retweeted)\b',
        r'^(rt|qt|repost)\s*[:@\s]', r'^\s*[🔁↪️➡️]\s*@?\w*'
    ]
    if any(re.search(p, lower, flags=re.IGNORECASE) for p in patterns):
        return False
    if len(re.sub(r'http\S+|\@\w+', '', text).strip()) < 15:
        return False
    return len(lower) >= 20 and not re.search(r'^\s*["\u201c]|\s*@\w+\s*[":]', lower)


def parse_timestamp_robust(timestamp):
    """Parse timestamp with multiple format support"""
    if pd.isna(timestamp):
        return pd.NaT
    ts_str = re.sub(r'\s+GMT$', '', str(timestamp).strip(), flags=re.IGNORECASE)
    try:
        parsed = pd.to_datetime(ts_str, errors='coerce', utc=True)
        if pd.notna(parsed):
            return parsed
    except:
        pass
    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%d/%m/%Y %H:%M', '%b %d, %Y %H:%M', '%Y-%m-%d']:
        try:
            parsed = pd.to_datetime(ts_str, format=fmt, errors='coerce', utc=True)
            if pd.notna(parsed):
                return parsed
        except:
            continue
    return pd.NaT


def combine_social_media_data(meltwater_df=None, civicsignals_df=None, tiktok_df=None, openmeasures_df=None, brandwatch_df=None):
    """Combine different platform datasets into unified format (Streamlit logic)"""
    combined = []
    
    def get_col(df, cols):
        """Get column with fallback to normalized names"""
        for col in cols:
            if col in df.columns:
                return df[col]
        df_cols = [c.lower().strip() for c in df.columns]
        for col in cols:
            norm = col.lower().strip()
            if norm in df_cols:
                return df[df.columns[df_cols.index(norm)]]
        return pd.Series([np.nan]*len(df), index=df.index)

    # === BRANDWATCH HANDLER (FIXED) ===
    if brandwatch_df is not None and not brandwatch_df.empty:
        bw = pd.DataFrame()
        
        # Create unified Account column (matches your Colab logic)
        bw['account_id'] = brandwatch_df.get('Weblog Title', pd.Series(dtype='object')).combine_first(
            brandwatch_df.get('Author', pd.Series(dtype='object'))
        ).combine_first(
            brandwatch_df.get('Full Name', pd.Series(dtype='object'))
        ).astype(str).str.strip().replace('nan', '')
        
        # Map core columns (matches your Colab renaming)
        bw['original_text'] = brandwatch_df.get('Full Text', pd.Series(dtype='object')).astype(str).str.strip().replace('nan', '')
        bw['URL'] = brandwatch_df.get('Url', pd.Series(dtype='object'))
        bw['timestamp_share'] = brandwatch_df.get('Date', pd.Series(dtype='object'))
        
        # Platform inference from Page Type
        page_type = brandwatch_df.get('Page Type', pd.Series(dtype='object')).astype(str).str.lower()
        platform_map = {
            'twitter': 'X', 'x': 'X', 'x.com': 'X', 't.co': 'X',
            'facebook': 'Facebook', 'fb': 'Facebook', 'fb.watch': 'Facebook',
            'instagram': 'Instagram', 'tiktok': 'TikTok',
            'youtube': 'YouTube', 'telegram': 'Telegram', 't.me': 'Telegram'
        }
        bw['Platform'] = page_type.map(platform_map).fillna('Unknown')
        
        # Safely generate content_id to prevent DataFrame assignment errors
        # Instead of nested .get() which can return a DataFrame, use combine_first on Series
        resource_ids = brandwatch_df.get('Resource Id', pd.Series(dtype='object'))
        mention_ids = brandwatch_df.get('Mention Id', pd.Series(dtype='object'))
        
        bw['content_id'] = resource_ids.combine_first(mention_ids).combine_first(bw['URL']).astype(str).str.strip()
        
        bw['source_dataset'] = 'Brandwatch'
        combined.append(bw)

    # === MELTWATER HANDLER ===
    if meltwater_df is not None and not meltwater_df.empty:
        mw = pd.DataFrame()
        mw['account_id'] = get_col(meltwater_df, ['influencer'])
        mw['content_id'] = get_col(meltwater_df, ['tweet id', 'post id', 'id']).astype(str).str.strip()
        mw['object_id'] = get_col(meltwater_df, ['hit sentence', 'opening text', 'headline', 'text', 'content'])
        mw['URL'] = get_col(meltwater_df, ['url'])
        mw['timestamp_share'] = get_col(meltwater_df, ['date', 'timestamp', 'alternate date format'])
        mw['source_dataset'] = 'Meltwater'
        combined.append(mw)

    # === CIVICSIGNALS HANDLER ===
    if civicsignals_df is not None and not civicsignals_df.empty:
        cs = pd.DataFrame()
        cs['account_id'] = get_col(civicsignals_df, ['media_name', 'author', 'username'])
        cs['content_id'] = get_col(civicsignals_df, ['stories_id', 'post_id', 'id']).astype(str).str.strip()
        cs['object_id'] = get_col(civicsignals_df, ['title', 'text', 'content', 'body'])
        cs['URL'] = get_col(civicsignals_df, ['url', 'link'])
        cs['timestamp_share'] = get_col(civicsignals_df, ['publish_date', 'timestamp', 'date'])
        cs['source_dataset'] = 'Civicsignal'
        combined.append(cs)

    # === TIKTOK HANDLER ===
    if tiktok_df is not None and not tiktok_df.empty:
        tt = pd.DataFrame()
        tt['object_id'] = get_col(tiktok_df, ['text', 'Transcript', 'caption', 'content'])
        tt['account_id'] = get_col(tiktok_df, ['authorMeta/name', 'username', 'creator'])
        tt['content_id'] = get_col(tiktok_df, ['id', 'video_id', 'itemId']).astype(str).str.strip()
        tt['URL'] = get_col(tiktok_df, ['webVideoUrl', 'TikTok Link', 'url'])
        tt['timestamp_share'] = get_col(tiktok_df, ['createTimeISO', 'timestamp', 'date', 'createTime'])
        tt['source_dataset'] = 'TikTok'
        
        # Preserve engagement metrics
        for col in ['playCount', 'diggCount', 'commentCount', 'shareCount', 'repostCount', 'textLanguage']:
            if col in tiktok_df.columns:
                tt[col] = tiktok_df[col]
                
        # Preserve hashtags
        for i in range(5):
            hashtag_col = f'hashtags/{i}/name'
            if hashtag_col in tiktok_df.columns:
                tt[f'hashtag_{i}'] = tiktok_df[hashtag_col]
                
        combined.append(tt)

    # === OPENMEASURE HANDLER ===
    if openmeasures_df is not None and not openmeasures_df.empty:
        om = pd.DataFrame()
        om['account_id'] = get_col(openmeasures_df, ['context_name', 'channelusername', 'channeltitle'])
        om['content_id'] = get_col(openmeasures_df, ['id', 'url']).astype(str).str.strip()
        om['object_id'] = get_col(openmeasures_df, ['text', 'message', 'body'])
        om['URL'] = get_col(openmeasures_df, ['url'])
        raw_dates = get_col(openmeasures_df, ['created_at', 'date'])
        om['timestamp_share'] = raw_dates.astype(str).str.replace(' @ ', ' ', regex=False)
        om['source_dataset'] = 'OpenMeasure_Telegram'
        combined.append(om)

    # Return combined DataFrame or empty DataFrame if nothing was provided
    if combined:
        return pd.concat(combined, ignore_index=True)
    else:
        return pd.DataFrame()


def final_preprocess_and_map_columns(df, coordination_mode="Text Content"):
    """Final preprocessing and column mapping (Streamlit logic)"""
    if df.empty:
        return pd.DataFrame(columns=['account_id','content_id','object_id','URL','timestamp_share','Platform','original_text','Outlet','Channel','cluster','source_dataset','Sentiment'])
    
    dfp = df.copy()
    
    # Filter by sentiment if present
    if 'Sentiment' in dfp.columns:
        dfp = dfp[dfp['Sentiment'].isin(['Negative', 'Neutral'])]
    
    # Filter to original posts only - ONLY if object_id exists
    if 'object_id' in dfp.columns:
        # Convert to string first to safely handle NaN/float values
        obj_str = dfp['object_id'].astype(str)
        mask = dfp['object_id'].apply(is_original_post) & (~obj_str.str.contains('🔁', na=False)) & (~obj_str.str.startswith('RT @', na=False))
        dfp = dfp[mask].copy()
        
        # Clean object_id
        dfp['object_id'] = dfp['object_id'].astype(str).replace('nan','').fillna('')
        dfp = dfp[dfp['object_id'].str.strip() != ""].reset_index(drop=True)
        
        # Extract original text
        dfp['original_text'] = dfp['object_id'].apply(extract_original_text) if coordination_mode=="Text Content" else dfp['URL'].astype(str).replace('nan','')
    else:
        # Handle case where object_id doesn't exist - use text column instead
        if 'text' in dfp.columns:
            dfp['original_text'] = dfp['text'].apply(extract_original_text) if coordination_mode=="Text Content" else dfp['URL'].astype(str).replace('nan','')
        elif 'original_text' not in dfp.columns:
            # Create empty original_text if no text column exists
            dfp['original_text'] = ''
    
    dfp = dfp[dfp['original_text'].str.strip() != ""].reset_index(drop=True)
    
    # Rest of the function remains the same...
    # Infer platform from URL
    dfp['Platform'] = dfp['URL'].apply(infer_platform_from_url)
    
    # Map source_dataset to Platform
    if 'source_dataset' in dfp.columns:
        dfp['source_dataset'] = dfp['source_dataset'].fillna('')
        # TikTok
        tiktok_mask = dfp['source_dataset'].str.contains('TikTok|tiktok|vt.tiktok', case=False, na=False)
        dfp.loc[tiktok_mask, 'Platform'] = 'TikTok'
        # Telegram
        telegram_mask = dfp['source_dataset'].str.contains('Telegram|telegram|t.me|OpenMeasure', case=False, na=False)
        dfp.loc[telegram_mask, 'Platform'] = 'Telegram'
        # Media/News
        media_mask = dfp['source_dataset'].str.contains('Media|News|Civicsignal', case=False, na=False)
        dfp.loc[media_mask, 'Platform'] = 'Media'
    
    # Fill unknown platforms
    dfp['Platform'] = dfp['Platform'].replace('', 'Unknown').fillna('Unknown')
    
    # Initialize remaining columns
    dfp['Outlet'], dfp['Channel'], dfp['cluster'] = np.nan, np.nan, -1
    if 'Sentiment' not in dfp.columns:
        dfp['Sentiment'] = np.nan
    
    # Return only needed columns
    cols = ['account_id','content_id','object_id','URL','timestamp_share','Platform','original_text','Outlet','Channel','cluster','source_dataset','Sentiment']
    return dfp[[c for c in cols if c in dfp.columns]].copy()

def preprocess_dataframe(df):
    """Generic preprocessing for unknown/custom formats"""
    if df.empty:
        return pd.DataFrame()
    
    # Basic cleaning
    df = df.copy()
    
    # Standardize column names (lowercase, strip)
    df.columns = [c.lower().strip() if isinstance(c, str) else c for c in df.columns]
    
    # Try to map common column names
    column_mapping = {
        'account_id': ['account', 'username', 'user', 'author', 'handle'],
        'content_id': ['id', 'post_id', 'tweet_id', 'video_id'],
        'object_id': ['text', 'content', 'body', 'message', 'post'],
        'url': ['link', 'post_url', 'permalink'],
        'timestamp_share': ['date', 'created_at', 'posted_at', 'time']
    }
    
    for target, sources in column_mapping.items():
        if target not in df.columns:
            for source in sources:
                if source in df.columns:
                    df[target] = df[source]
                    break
    
    return df
    
def get_election_posts_queryset(request):
    """
    Centralized date filtering helper.
    Supports 'view_all' parameter to show all data without date filtering.
    Default: Shows last 3 months of data.
    """
    queryset = ProcessedPost.objects.all()
    
    # Check if user wants to view all data
    view_all = request.GET.get('view_all') == 'true'
    
    if view_all:
        # No date filtering - show all posts
        date_range = queryset.aggregate(
            min_date=Min('timestamp_share'),
            max_date=Max('timestamp_share')
        )
        start_date = date_range['min_date'] or timezone.now()
        end_date = date_range['max_date'] or timezone.now()
    else:
        # Apply date filtering
        start_date = request.GET.get('start_date')
        end_date = request.GET.get('end_date')
        
        if start_date and end_date:
            queryset = queryset.filter(
                timestamp_share__date__range=[start_date, end_date]
            )
            # Convert strings to date objects
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            # DEFAULT: Last 3 months 
            end_dt = timezone.now()
            start_dt = end_dt - timedelta(days=90)  # 3 months = 90 days
            queryset = queryset.filter(
                timestamp_share__gte=start_dt
            )
            start_date = start_dt
            end_date = end_dt
    
    return queryset.order_by('-timestamp_share'), start_date, end_date
    
def get_risk_actors_insight(posts_queryset, limit=8):
    """
    Identify risk actors based on:
    1. High post volume (potential spam/coordination)
    2. Repetitive content patterns
    3. High/critical risk levels
    """
    from collections import Counter
    import re
    
    risky_accounts = []
    
    # Get accounts with high post volume (5+ posts)
    account_stats = posts_queryset.values('account_id').annotate(
        post_count=Count('id'),
        high_risk_count=Count('id', filter=Q(risk_level__in=['high', 'critical'])),
        latest_post=Max('timestamp_share')
    ).filter(
        Q(post_count__gte=5) | Q(high_risk_count__gte=2)  # High volume OR multiple high-risk posts
    ).order_by('-post_count')[:limit]
    
    for acc in account_stats:
        account_id = acc['account_id']
        if not account_id or str(account_id).lower() in ['nan', 'none', '', 'twitter', 'source']:
            continue
            
        # Get account's posts
        account_posts = list(posts_queryset.filter(account_id=account_id))
        
        # Check for content repetition (coordination signal)
        content_hashes = []
        for post in account_posts:
            if post.original_text:
                # Normalize text: remove URLs, mentions, extra spaces
                normalized = re.sub(r'http\S+|@\w+|#\w+', '', post.original_text.lower())
                normalized = re.sub(r'\s+', ' ', normalized).strip()
                if len(normalized) > 20:  # Only consider substantial content
                    content_hashes.append(normalized[:100])  # First 100 chars as hash
        
        # Count duplicate content
        content_counter = Counter(content_hashes)
        duplicate_count = sum(1 for count in content_counter.values() if count > 1)
        repetition_rate = (duplicate_count / len(content_counter) * 100) if content_counter else 0
        
        # Get platforms and sample content
        platforms = list(set(p.platform for p in account_posts if p.platform))[:3]
        sample_post = account_posts[0] if account_posts else None
        
        # Calculate risk score
        risk_score = 0
        if acc['post_count'] >= 10: risk_score += 3
        elif acc['post_count'] >= 5: risk_score += 2
        
        if acc['high_risk_count'] >= 3: risk_score += 3
        elif acc['high_risk_count'] >= 1: risk_score += 1
        
        if repetition_rate > 30: risk_score += 2  # High repetition
        
        # Only include if risk score >= 3
        if risk_score >= 3 and sample_post:
            risky_accounts.append({
                'account_id': str(account_id)[:40],
                'post_count': acc['post_count'],
                'high_risk_count': acc['high_risk_count'],
                'repetition_rate': round(repetition_rate, 1),
                'platforms': ', '.join(platforms) if platforms else 'Unknown',
                'recent_content': sample_post.original_text[:120] + '...' if sample_post.original_text else '—',
                'last_seen': acc['latest_post'].strftime('%Y-%m-%d') if acc['latest_post'] else '—',
                'risk_level': 'high' if acc['high_risk_count'] >= 2 else 'medium',
                'risk_score': risk_score
            })
    
    # Sort by risk score
    risky_accounts.sort(key=lambda x: x['risk_score'], reverse=True)
    return risky_accounts[:limit]
    
def get_platform_distribution(posts_queryset):
    """
    Simplified platform distribution matching Streamlit logic
    """
    # Get raw platform counts directly from database
    platform_counts = posts_queryset.values('platform').annotate(
        count=Count('id')
    ).order_by('-count')
    
    # Build a clean list (similar to Streamlit's value_counts)
    platform_data = []
    for item in platform_counts:
        platform_name = item['platform']
        count = item['count']
        
        # Skip invalid platforms
        if not platform_name or str(platform_name).lower() in ['nan', 'none', '', 'unknown', 'null']:
            continue
        
        # Normalize platform names (same logic as Streamlit's infer_platform_from_url)
        p_lower = str(platform_name).lower().strip()
        
        if p_lower in ['x', 'twitter', 't.co', 'x.com', 'twitter source', 'twitter.com']:
            normalized_name = 'X'
        elif p_lower in ['facebook', 'fb.watch', 'facebook.com', 'fb']:
            normalized_name = 'Facebook'
        elif p_lower in ['telegram', 't.me', 'tg']:
            normalized_name = 'Telegram'
        elif p_lower in ['tiktok', 'tik tok', 'tik-tok']:
            normalized_name = 'TikTok'
        elif p_lower in ['media', 'news', 'news/media', 'civicsignal']:
            normalized_name = 'Media'
        elif p_lower in ['youtube', 'youtu.be', 'yt']:
            normalized_name = 'YouTube'
        elif p_lower in ['instagram', 'insta', 'ig']:
            normalized_name = 'Instagram'
        else:
            normalized_name = str(platform_name).title()
        
        platform_data.append({
            'Platform': normalized_name,
            'Count': count
        })
    
    # Create DataFrame (matching Streamlit structure)
    df = pd.DataFrame(platform_data)
    
    if not df.empty:
        # Group by Platform and sum counts (in case of duplicates after normalization)
        df = df.groupby('Platform', as_index=False)['Count'].sum()
        df = df.sort_values('Count', ascending=False).reset_index(drop=True)
        top_platform = df.iloc[0]['Platform']
    else:
        df = pd.DataFrame(columns=['Platform', 'Count'])
        top_platform = "—"
    
    return df, top_platform
    
def get_top_hashtags(posts_queryset, limit=10):
    """Extract and rank top hashtags from post content"""
    import re
    from collections import Counter
    
    hashtags = []
    for post in posts_queryset:
        if post.original_text:
            # Extract hashtags (case-insensitive, handles #Ethiopia, #ETHIOPIA, etc.)
            found = re.findall(r'#(\w+)', post.original_text)
            hashtags.extend([h.lower() for h in found if len(h) > 1])
    
    # Count and rank
    hashtag_counts = Counter(hashtags)
    return [
        {'tag': tag, 'count': count}
        for tag, count in hashtag_counts.most_common(limit)
        if count >= 2  # Only show hashtags used 2+ times
    ]

def get_pep_analysis_insights(posts_queryset, peps_queryset, extra_officials_list=None, limit=6):
    """
    Enhanced PEP analysis with Groq sentiment analysis and URL links.
    """
    from collections import defaultdict, Counter
    import re
    
    if not extra_officials_list:
        extra_officials_list = []
    
    # Build unified dictionary of names
    officials_to_scan = {}
    for pep in peps_queryset:
        name_lower = pep.name.lower().strip()
        if name_lower:
            officials_to_scan[name_lower] = pep.name
    
    for name in extra_officials_list:
        name_lower = name.lower().strip()
        if name_lower and name_lower not in officials_to_scan:
            officials_to_scan[name_lower] = name
    
    pep_mentions = defaultdict(lambda: {
        'count': 0,
        'platforms': Counter(),
        'hourly_distribution': Counter(),
        'hashtags': Counter(),
        'sample_posts': [],
        'critical_posts': [],  # NEW: Track critical posts separately
        'narrative_clusters': defaultdict(list),
        'sentiment_scores': [],  # NEW: Track sentiment scores
    })
    
    # Scan posts
    for post in posts_queryset[:5000]:
        if not post.original_text:
            continue
        text_lower = post.original_text.lower()
        
        for scan_name, display_name in officials_to_scan.items():
            if scan_name in text_lower:
                data = pep_mentions[display_name]
                data['count'] += 1
                
                platform = post.platform or 'Unknown'
                data['platforms'][platform] += 1
                
                if post.timestamp_share:
                    data['hourly_distribution'][post.timestamp_share.hour] += 1
                
                hashtags = re.findall(r'#(\w+)', post.original_text)
                data['hashtags'].update(hashtags)
                
                # Narrative clustering
                if any(kw in text_lower for kw in ['rigged', 'stolen', 'fraud', 'nebe']):
                    data['narrative_clusters']['Election Integrity'].append(post.original_text[:100])
                if any(kw in text_lower for kw in ['ethnic', 'tribal', 'amhara', 'oromo', 'tigray']):
                    data['narrative_clusters']['Ethnic Dynamics'].append(post.original_text[:100])
                
                # Store sample post with URL
                if len(data['sample_posts']) < 5:
                    data['sample_posts'].append({
                        'text': post.original_text[:150],
                        'platform': platform,
                        'timestamp': post.timestamp_share,
                        'url': post.url if post.url else None,  # Include URL
                        'risk_level': post.risk_level if hasattr(post, 'risk_level') else 'medium'
                    })
                
                # Use Groq to analyze sentiment of this post
                try:
                    from groq import Groq
                    client = Groq(api_key=settings.GROQ_API_KEY)
                    
                    prompt = (
                        f"Analyze the sentiment of this post about {display_name}.\n"
                        f'Text: "{post.original_text[:200]}"\n\n'
                        "Is the sentiment Positive (supportive/praise), Negative (criticism/attack), or Neutral (factual/news)?\n"
                        "Reply with ONLY one word: Positive, Negative, or Neutral."
                    )
                    
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=10
                    )
                    
                    sentiment = response.choices[0].message.content.strip().capitalize()
                    data['sentiment_scores'].append(sentiment)
                    
                    # Track critical posts separately
                    if sentiment == 'Negative' and len(data['critical_posts']) < 3:
                        data['critical_posts'].append({
                            'text': post.original_text[:200],
                            'platform': platform,
                            'timestamp': post.timestamp_share,
                            'url': post.url if post.url else None,
                        })
                    
                except Exception as e:
                    logger.error(f"Groq sentiment analysis failed: {e}")
                    data['sentiment_scores'].append('Neutral')
    
    # Build final results
    results = []
    for display_name, data in sorted(pep_mentions.items(), key=lambda x: x[1]['count'], reverse=True)[:limit]:
        if data['count'] < 2:
            continue
        
        # Calculate overall sentiment
        sentiment_counts = Counter(data['sentiment_scores'])
        total_analyzed = len(data['sentiment_scores'])
        
        if total_analyzed > 0:
            negative_pct = (sentiment_counts.get('Negative', 0) / total_analyzed) * 100
            positive_pct = (sentiment_counts.get('Positive', 0) / total_analyzed) * 100
            
            if negative_pct > 50:
                overall_sentiment = 'Negative'
                sentiment_label = '🔴 High Criticism'
            elif positive_pct > 50:
                overall_sentiment = 'Positive'
                sentiment_label = '🟢 Mostly Positive'
            else:
                overall_sentiment = 'Mixed'
                sentiment_label = '🟡 Mixed Sentiment'
        else:
            overall_sentiment = 'Neutral'
            sentiment_label = '⚪ Neutral'
        
        # Calculate risk score based on negative sentiment percentage
        risk_score = min(10, int(negative_pct / 10))
        
        total_posts = sum(data['platforms'].values())
        platform_breakdown = [
            {'name': plat, 'count': cnt, 'percent': round(cnt/total_posts*100)}
            for plat, cnt in data['platforms'].most_common(3)
        ]
        
        clusters = [{'name': name, 'count': len(posts)} for name, posts in data['narrative_clusters'].items()]
        
        results.append({
            'pep_name': display_name,
            'mention_count': data['count'],
            'platform_breakdown': platform_breakdown,
            'narrative_clusters': clusters,
            'sample_posts': data['sample_posts'],
            'critical_posts': data['critical_posts'],  
            'sentiment': overall_sentiment,
            'sentiment_label': sentiment_label,
            'risk_score': risk_score,
            'negative_percentage': round(negative_pct, 1),
        })
    
    return results 


def extract_first_json_array(text):
    """Extract the first valid JSON array from a string by matching brackets."""
    start = text.find('[')
    if start == -1:
        return None
    
    depth = 0
    in_string = False
    escape_next = False
    
    for i in range(start, len(text)):
        char = text[i]
        
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\':
            escape_next = True
            continue
            
        if char == '"':
            in_string = not in_string
            continue
            
        if in_string:
            continue
            
        if char == '[':
            depth += 1
        elif char == ']':
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    
    return None


def extract_new_trigger_terms_llm(text, existing_matches=None):
    """Use LLM to extract NEW trigger terms not in the lexicon."""
    if not text or len(text.strip()) < 20:
        return []
        
    existing_terms = set()
    if existing_matches:
        existing_terms = set(m['term'].lower() for m in existing_matches)
    for category, terms in CONFIG.get('lexicon', {}).items():
        existing_terms.update(t.lower() for t in terms.keys())
        
    existing_list = ', '.join(list(existing_terms)[:50])
    
    # 🔥 UPDATED PROMPT: Explicitly handles "Slur + Name" cases
    prompt = (
        "You are an expert hate speech analyst. Extract NEW trigger terms (slurs, threats, dehumanizing language, harmful adjectives) "
        "from this text that are NOT already in the lexicon.\n\n"
        "TEXT:\n"
        '"' + text + '"\n\n'
        "EXISTING TERMS (do not extract these):\n"
        + existing_list + "\n\n"
        "CRITICAL RULES:\n"
        "- 🚫 NEVER extract personal names, politicians, or specific individuals (e.g., 'Abiy Ahmed', 'Margaret', 'John Doe').\n"
        "- 🚫 NEVER extract neutral demographic terms like 'youth', 'women', 'soldiers' unless clearly used as a slur.\n"
        "- ✅ SEPARATE SLURS FROM NAMES: If a harmful adjective or slur is attached to a person's name (e.g., 'Fascist Abiy Ahmed', 'Thief John', 'በሺስት አብይ አመድ'), extract ONLY the harmful word or phrase (e.g., 'Fascist', 'Thief', 'ፋሺስት'). Do NOT include the person's name in the extracted term.\n"
        "- ✅ ONLY extract actual slurs, threats, dehumanizing phrases, or weaponized language.\n\n"
        "Return ONLY a valid JSON array. Each object must have:\n"
        '- "term": the exact phrase from the text\n'
        '- "category": one of [ethnic_identity, violence_incitement, dehumanizing, religious_cultural, gender_misogynistic, discriminatory_homophobic, socio_economic_caste, political_groups, foreign_interference, election_governance]\n'
        '- "severity": one of [low, medium, high, critical]\n'
        '- "target_entity": the group being targeted (e.g., Amhara, Oromo, Women) or empty string\n'
        '- "language": one of [Amharic, Oromo, English, Tigrinya, Somali]\n'
        '- "confidence": A number between 0.0 and 1.0 representing your confidence\n\n'
        "EXAMPLE OUTPUT:\n"
        '[\n'
        '  {"term": "example slur", "category": "ethnic_identity", "severity": "high", "target_entity": "Group", "language": "Amharic", "confidence": 0.92}\n'
        ']\n\n'
        "If no new terms found, return: []"
    )
    
    try:
        response = safe_llm_call(prompt, max_tokens=1024)
        if not response:
            return []
            
        json_str = extract_first_json_array(response)
        if not json_str:
            logger.warning(f"No JSON array found in LLM response. Preview: {response[:200]}")
            return []
            
        try:
            extracted_terms = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}. Preview: {json_str[:200]}")
            return []
            
        if not isinstance(extracted_terms, list):
            return []
            
        valid_terms = []
        valid_categories = ['ethnic_identity', 'violence_incitement', 'dehumanizing',
                           'religious_cultural', 'gender_misogynistic', 'discriminatory_homophobic',
                           'socio_economic_caste', 'political_groups', 'foreign_interference',
                           'election_governance']
        valid_severities = ['low', 'medium', 'high', 'critical']
        valid_languages = ['Amharic', 'Oromo', 'English', 'Tigrinya', 'Somali']
        
        for term_data in extracted_terms:
            if not isinstance(term_data, dict):
                continue
                
            term = term_data.get('term', '').strip()
            if not term or len(term) < 2:
                continue
                
            if term.lower() in existing_terms:
                continue
                
            # 🔥 PROGRAMMATIC FILTER: Drop terms that look like standard English proper names
            if re.match(r'^[A-Za-z\s]+$', term) and term.istitle():
                logger.info(f"Filtered out likely name: '{term}'")
                continue
                
            category = term_data.get('category', 'uncategorized')
            if category not in valid_categories:
                category = 'uncategorized'
                
            severity = term_data.get('severity', 'medium')
            if severity not in valid_severities:
                severity = 'medium'
                
            language = term_data.get('language', 'Amharic')
            if language not in valid_languages:
                language = 'Amharic'
                
            confidence = term_data.get('confidence', 0.85)
            try:
                confidence = float(confidence)
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError):
                confidence = 0.85
                
            valid_terms.append({
                'term': term,
                'category': category,
                'severity': severity,
                'target_entity': term_data.get('target_entity', ''),
                'language': language,
                'confidence': round(confidence, 2),
                'source': 'LLM Extraction',
                'context': text[:300]
            })
            
        if valid_terms:
            logger.info(f"LLM extracted {len(valid_terms)} new trigger terms")
        return valid_terms
        
    except Exception as e:
        logger.error(f"Error in LLM trigger term extraction: {e}")
        return []
        
def export_new_terms_csv(request):
    """Export new trigger terms detected by LLM to CSV"""
    # Get terms from session
    new_terms = request.session.get('new_trigger_terms', [])
    
    if not new_terms:
        messages.warning(request, "No new terms to export. Please scan text first.")
        return redirect('lexicon_management')
    
    # Create CSV response
    response = HttpResponse(content_type='text/csv')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    response['Content-Disposition'] = f'attachment; filename="new_trigger_terms_{timestamp}.csv"'
    
    # Write CSV
    writer = csv.writer(response)
    writer.writerow(['Term', 'Category', 'Severity', 'Target Entity', 'Language', 'Confidence'])
    
    for term in new_terms:
        writer.writerow([
            term.get('term', ''),
            term.get('category', ''),
            term.get('severity', ''),
            term.get('target_entity', ''),
            term.get('language', ''),
            term.get('confidence', '')
        ])
    
    messages.success(request, f"Exported {len(new_terms)} new trigger terms to CSV")
    return response
    
@login_required
def export_all_lexicon_terms_csv(request):
    """Export ALL lexicon terms (Database + CONFIG) as CSV"""
    import csv
    from django.http import HttpResponse
    from django.utils import timezone
    
    # 1. Get terms from Database
    db_terms = LexiconTerm.objects.filter(is_election_related=True).values(
        'term', 'category', 'severity', 'target_entity', 'language'
    )
    
    # 2. Get terms from CONFIG
    config_terms = []
    for category, terms in CONFIG.get('lexicon', {}).items():
        for term, metadata in terms.items():
            config_terms.append({
                'term': term,
                'category': category,
                'severity': metadata.get('severity', 'medium'),
                'target_entity': metadata.get('target_entity', ''),
                'language': metadata.get('language', 'english'),
            })
    
    # 3. Combine and deduplicate (DB terms take priority)
    seen_terms = set()
    all_terms = []
    
    # Add DB terms first
    for term in db_terms:
        if term['term'] not in seen_terms:
            seen_terms.add(term['term'])
            all_terms.append({
                'term': term['term'],
                'category': term['category'],
                'severity': term['severity'],
                'target_entity': term['target_entity'],
                'language': term['language'],
                'source': 'Database'
            })
    
    # Add CONFIG terms (skip duplicates)
    for term in config_terms:
        if term['term'] not in seen_terms:
            seen_terms.add(term['term'])
            all_terms.append({
                'term': term['term'],
                'category': term['category'],
                'severity': term['severity'],
                'target_entity': term['target_entity'],
                'language': term['language'],
                'source': 'CONFIG'
            })
    
    # 4. Create CSV response
    response = HttpResponse(content_type='text/csv')
    timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
    response['Content-Disposition'] = f'attachment; filename="all_lexicon_terms_{timestamp}.csv"'
    
    # Write CSV with UTF-8 BOM for Excel compatibility (important for Amharic)
    response.write('\ufeff')
    writer = csv.writer(response)
    writer.writerow(['Term', 'Category', 'Severity', 'Target Entity', 'Language', 'Source'])
    
    for term in all_terms:
        writer.writerow([
            term['term'],
            term['category'],
            term['severity'],
            term['target_entity'],
            term['language'],
            term['source']
        ])
    
    messages.success(request, f"Exported {len(all_terms)} lexicon terms to CSV")
    return response
    
def reports_landing(request):
    """Landing page showing all report categories as cards"""
    baseline_reports = MonitoringReport.objects.filter(report_category='baseline').order_by('-uploaded_at')[:3]
    situational_reports = MonitoringReport.objects.filter(report_category='situational').order_by('-uploaded_at')[:3]
    tiktok_reports = MonitoringReport.objects.filter(report_category='tiktok').order_by('-uploaded_at')[:3]
    
    # All reports for archive section
    all_reports = MonitoringReport.objects.all().order_by('-uploaded_at')[:6]
    
    context = {
        'baseline_reports': baseline_reports,
        'situational_reports': situational_reports,
        'tiktok_reports': tiktok_reports,
        'all_reports': all_reports,
        'active_tab': 'reports',
    }
    return render(request, 'dashboard/reports_landing.html', context)

def report_detail(request, report_id):
    """Individual report detail page"""
    report = get_object_or_404(MonitoringReport, id=report_id)
    
    context = {
        'report': report,
        'active_tab': 'reports',
    }
    return render(request, 'dashboard/report_detail.html', context)

def detect_significant_spikes(daily_data, dates, categories, threshold_std=2.0):
    """
    Detect significant spikes across the ENTIRE analysis period.
    FIXED: Handles std=0 (when previous days had 0 mentions) and adds global fallback.
    """
    spikes_detected = []
    for category in categories:
        values = [daily_data[day].get(category, 0) for day in dates]
        if len(values) < 7:
            continue
        
        # Calculate overall mean and std for the entire period (fallback)
        overall_mean = sum(values) / len(values)
        overall_std = (sum((x - overall_mean) ** 2 for x in values) / len(values)) ** 0.5
        
        rolling_window = min(7, len(values) // 3)
        if rolling_window < 3:
            rolling_window = 3
            
        for i, value in enumerate(values):
            if i < rolling_window:
                continue
            
            prev_values = values[i-rolling_window:i]
            mean = sum(prev_values) / len(prev_values)
            std = (sum((x - mean) ** 2 for x in prev_values) / len(prev_values)) ** 0.5
            
            is_spike = False
            z_score = 0
            magnitude = 0
            
            # 1. Check local spike (rolling window)
            if std > 0:
                z_score = (value - mean) / std
                if z_score >= threshold_std and value >= 5:
                    is_spike = True
                    magnitude = value / mean if mean > 0 else value
            elif value >= 5 and mean < 2:
                # If previous period was near 0 and now it's >= 5, it's a massive spike
                is_spike = True
                z_score = 10.0
                magnitude = value
                
            # 2. Fallback: Check global spike (overall period)
            # If local didn't catch it, but it's a massive spike compared to the whole period
            if not is_spike and overall_std > 0:
                global_z = (value - overall_mean) / overall_std
                if global_z >= 3.0 and value >= 10: # Stricter threshold for global
                    is_spike = True
                    z_score = global_z
                    magnitude = value / overall_mean if overall_mean > 0 else value
                    
            if is_spike:
                spikes_detected.append({
                    'category': category,
                    'date': dates[i],
                    'value': value,
                    'mean': round(mean, 2),
                    'std': round(std, 2),
                    'z_score': round(z_score, 2),
                    'spike_magnitude': round(magnitude, 2)
                })
                
    spikes_detected.sort(key=lambda x: x['z_score'], reverse=True)
    return spikes_detected

def get_category_trend_analysis(posts_queryset, days_back=90, cache_suffix="all"):
    """
    OPTIMIZED: Uses random sampling and caching to prevent crashes on large datasets.
    NOW: Detects spikes across entire timeline, not just last 7 days.
    """
    # 1. CHECK CACHE FIRST
    cache_key = f"trend_analysis_{days_back}_{cache_suffix}"
    cached_result = cache.get(cache_key)
    if cached_result:
        return cached_result
    
    # 2. GET DATE RANGE FROM QUERYSET (Respect user's filter)
    date_range = posts_queryset.aggregate(
        min_date=Min('timestamp_share'),
        max_date=Max('timestamp_share')
    )
    
    # Respect the UI's date filter (posts_queryset is already filtered by the view)
    recent_posts = posts_queryset.filter(
        original_text__isnull=False
    ).exclude(original_text='')
    
    total_available = recent_posts.count()
    
    if total_available < 10:
        return {
            'chart_json': None,
            'trending_categories': [],
            'total_categories_tracked': 0,
            'total_posts_scanned': total_available,
            'spikes_detected': [],
            'spike_alerts': [],
            'message': 'Not enough data'
        }
    
    # 3. RANDOM SAMPLING (Increased to 5000 posts)
    sample_size = min(5000, total_available)  # Changed from 1500 to 5000
    post_ids = list(recent_posts.values_list('id', flat=True))
    sampled_ids = random.sample(post_ids, sample_size)
    
    # Fetch only the sampled posts
    posts_to_scan = ProcessedPost.objects.filter(id__in=sampled_ids).iterator()
    
    # 4. SCAN ONLY THE SAMPLE
    daily_data = defaultdict(lambda: defaultdict(int))
    total_scanned = 0
    for post in posts_to_scan:
        if post.original_text and post.timestamp_share:
            # This now includes both lexicon AND LLM-based detection
            matches = scan_text_for_lexicon_terms(post.original_text)
            day = post.timestamp_share.date()
            for match in matches:
                daily_data[day][match['category']] += 1
            total_scanned += 1
    
    if not daily_data:
        return {
            'chart_json': None,
            'trending_categories': [],
            'total_categories_tracked': 0,
            'total_posts_scanned': total_scanned,
            'spikes_detected': [],
            'spike_alerts': [],
            'message': 'No lexicon matches found'
        }
    
    # 5. BUILD CHART DATA
    dates = sorted(daily_data.keys())
    categories = set()
    for day_data in daily_data.values():
        categories.update(day_data.keys())
    
    traces = []
    category_colors = {
        'violence_incitement': '#dc2626', 'dehumanizing': '#ea580c',
        'ethnic_identity': '#d97706', 'political_groups': '#65a30d',
        'election_governance': '#0284c7', 'foreign_interference': '#7c3aed',
        'religious_cultural': '#db2777'
    }
    
    for category in sorted(categories):
        values = [daily_data[day].get(category, 0) for day in dates]
        traces.append({
            'x': [d.strftime('%Y-%m-%d') for d in dates],
            'y': values,
            'type': 'scatter', 'mode': 'lines+markers',
            'name': category.replace('_', ' ').title(),
            'line': {'color': category_colors.get(category, '#6B7280'), 'width': 2.5},
            'marker': {'size': 4}
        })
    
    chart_json = json.dumps({
        'data': traces,
        'layout': {
            'title': f'Weaponized Language Trends (Sampled {total_scanned} posts)',
            'xaxis': {'title': 'Date', 'tickangle': -45, 'gridcolor': '#E5E7EB'},
            'yaxis': {'title': 'Daily Mentions', 'gridcolor': '#E5E7EB'},
            'hovermode': 'x unified', 'height': 450,
            'margin': {'b': 100, 't': 50, 'l': 60, 'r': 20},
            'plot_bgcolor': '#ffffff', 'paper_bgcolor': '#ffffff',
            'legend': {'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'center', 'x': 0.5}
        }
    })
    
    # 6. DETECT SPIKES ACROSS ENTIRE TIMELINE
    spikes_detected = detect_significant_spikes(daily_data, dates, categories, threshold_std=2.0)
    
    # Generate spike alerts for UI
    spike_alerts = []
    if spikes_detected:
        for spike in spikes_detected[:10]:  # Top 10 spikes
            category_display = spike['category'].replace('_', ' ').title()
            date_str = spike['date'].strftime('%b %d')
            magnitude = spike['spike_magnitude']
            
            if magnitude >= 3.0:
                alert_level = "🚨 Critical"
            elif magnitude >= 2.0:
                alert_level = "⚠️ High"
            else:
                alert_level = "⚡ Moderate"
            
            spike_alerts.append({
                'message': f"{alert_level} {category_display}: {spike['value']} mentions on {date_str} ({magnitude}x above average)",
                'category': category_display,
                'date': date_str,
                'value': spike['value'],
                'magnitude': magnitude,
                'alert_level': alert_level
            })
    
    # 7. CALCULATE TRENDS (Last 7 days vs previous period)
    trending = []
    if len(dates) >= 7:
        recent_7 = dates[-7:]
        previous_period = dates[:-7] if len(dates) > 7 else dates
        
        for category in categories:
            recent_count = sum(daily_data[d].get(category, 0) for d in recent_7)
            recent_avg = recent_count / 7
            
            previous_avg = 0
            if previous_period:
                previous_count = sum(daily_data[d].get(category, 0) for d in previous_period)
                previous_avg = previous_count / len(previous_period)
            
            if previous_avg > 0:
                pct_change = ((recent_avg - previous_avg) / previous_avg) * 100
            elif recent_avg > 0:
                pct_change = 100
            else:
                pct_change = 0
            
            if pct_change > 15 and recent_count >= 2:
                trending.append({
                    'category': category.replace('_', ' ').title(),
                    'pct_change': round(pct_change, 1),
                    'recent_count': recent_count,
                    'severity': _get_category_severity_display(category),
                })
        
        trending.sort(key=lambda x: x['pct_change'], reverse=True)
    
    # 8. BUILD FINAL RESULT
    final_result = {
        'chart_json': chart_json,
        'trending_categories': trending[:5],
        'total_categories_tracked': len(categories),
        'total_posts_scanned': total_scanned,
        'date_range': f"{dates[0].strftime('%b %d')} - {dates[-1].strftime('%b %d, %Y')}" if dates else "",
        'spikes_detected': spikes_detected,
        'spike_alerts': spike_alerts,
        'total_spikes': len(spikes_detected),
        'has_significant_spikes': len(spikes_detected) > 0
    }
    
    # 9. SAVE TO CACHE FOR 60 MINUTES
    cache.set(cache_key, final_result, 3600)
    
    return final_result
       
def _get_category_severity_display(category):
    """Map category to severity level for display"""
    severity_map = {
        'violence_incitement': 'critical',
        'dehumanizing': 'high',
        'ethnic_identity': 'medium',
        'political_groups': 'medium',
        'election_governance': 'low',
        'foreign_interference': 'medium',
        'religious_cultural': 'low'
    }
    return severity_map.get(category, 'medium')
    

def batch_translate_terms_llm(terms_list):
    """
    Use LLM to translate a batch of Amharic/Oromo terms to English with context.
    Uses caching to avoid redundant API calls and improve page load speed.
    """
    if not terms_list:
        return {}
    
    # Check cache first (Cache for 7 days)
    cache_key = "llm_amharic_oromo_translations_v1"
    cached_translations = cache.get(cache_key, {})
    
    # Filter out terms we already translated
    terms_to_translate = [t for t in terms_list if t not in cached_translations]
    
    if not terms_to_translate:
        return cached_translations
    
    # Limit to 50 terms per batch to avoid token limits
    terms_to_translate = terms_to_translate[:50]
    terms_str = "\n".join([f"- {t}" for t in terms_to_translate])
    
    # FIXED PROMPT: Corrected historical context and accurate socio-political/harmful nuance mapping
    prompt = f"""You are an expert linguist specializing in Ethiopian languages (Amharic, Oromo, Tigrinya) and hate speech/information operations analysis.

These terms have been FLAGGED as potentially harmful in social media posts. Translate them into English, focusing on:
1. The literal meaning
2. How it's weaponized as an INSULT, SLUR, or DEROGATORY term in socio-political tensions (this is CRITICAL)
3. The targeted demographic group or impact

DO NOT provide neutral dictionary definitions. These are being used as narrative weapons.

Terms to translate:
{terms_str}

Return ONLY a valid JSON object where:
- Keys are the exact original terms
- Values are translations that accurately explain the DEROGATORY/OFFENSIVE usage

Example Output:
{{
  "ነፍጠኛ": "Literal: Rifleman/historical armed settler. Weaponized use: A highly charged political slur used to target and vilify specific ethnic groups (primarily Amhara), framing them as oppressors or historical expansionists.",
  "ሸርሙጣ": "Literal: Whore / prostitute. Weaponized use: A highly offensive, derogatory misogynistic slur used to demean, humiliate, and target women.",
  "ገረድ": "Literal: Maid / domestic worker / female servant. Weaponized use: Used derogatorily to belittle, diminish status, classist abuse, or imply subjugation.",
  "የጭን ገረድ": "Literal: Concubine / personal maid. Weaponized use: A severely demeaning misogynistic slur denoting subjugation, objectification, or a woman who serves out of subservience."
}}
"""
    try:
        # Use the existing safe_llm_call utility from your codebase
        response = safe_llm_call(prompt, max_tokens=1024)
        if not response:
            return cached_translations
        
        # Strip codeblock wrappers if returned by the LLM
        response_text = response.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
            
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response_text.strip(), re.DOTALL)
        if json_match:
            new_translations = json.loads(json_match.group(0))
            # Update cache
            cached_translations.update(new_translations)
            cache.set(cache_key, cached_translations, 86400 * 7) # Cache for 7 days
            return cached_translations
        else:
            logger.warning("LLM translation response did not contain valid JSON.")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM translation JSON: {e}")
    except Exception as e:
        logger.error(f"LLM batch translation failed: {e}")
        
    return cached_translations
    
def analyze_pep_sentiment_groq(sample_texts, pep_name):
    """Use Groq to analyze the actual sentiment/criticality of posts mentioning a PEP"""
    if not sample_texts:
        return "Neutral"
    
    try:
        from groq import Groq
        client = Groq(api_key=settings.GROQ_API_KEY)
        
        combined_text = " | ".join([t[:200] for t in sample_texts[:3]])
        
        # Focus on whether the posts are critical/negative in tone
        prompt = (
            f"Analyze these social media posts that mention {pep_name}.\n\n"
            f'Posts: "{combined_text}"\n\n'
            "Determine the overall tone:\n"
            "- 'Negative' if the posts contain criticism, accusations, attacks, or negative claims (even if directed at someone else)\n"
            "- 'Positive' if the posts are supportive, praising, or positive about the person mentioned\n"
            "- 'Mixed' if there's both positive and negative content\n"
            "- 'Neutral' if it's just factual reporting with no emotional tone\n\n"
            "Reply with ONLY one word: Positive, Negative, Mixed, or Neutral."
        )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=10
        )
        
        sentiment = response.choices[0].message.content.strip().capitalize()
        if sentiment in ['Positive', 'Negative', 'Mixed', 'Neutral']:
            return sentiment
        return "Neutral"
        
    except Exception as e:
        logger.error(f"Groq sentiment analysis failed for {pep_name}: {e}")
        return "Neutral"

def get_tfgbv_lexicon_terms():
    """
    Load all TFGBV-related terms from the lexicon database.
    Returns a list of (term, severity, target_entity) tuples.
    """
    tfgbv_categories = [
        'gender_misogynistic',
        'discriminatory_homophobic',
    ]
    
    terms = []
    
    # Try loading from Django model first
    try:
        from dashboard.models import LexiconTerm
        tfgbv_terms = LexiconTerm.objects.filter(
            category__in=tfgbv_categories,
            is_active=True
        ).values_list('term', 'severity', 'category')
        
        for term, severity, category in tfgbv_terms:
            terms.append({
                'term': term.lower().strip(),
                'severity': severity,
                'category': category,
            })
    except Exception:
        pass
    
    # Fallback: Load from settings if model is empty
    if not terms:
        lexicon = getattr(settings, 'HATE_SPEECH_LEXICON', {})
        for category in tfgbv_categories:
            category_terms = lexicon.get(category, {})
            for term, meta in category_terms.items():
                terms.append({
                    'term': term.lower().strip(),
                    'severity': meta.get('severity', 'medium'),
                    'category': category,
                })
    
    return terms


def detect_tfgbv_in_text(text, tfgbv_terms):
    """
    Scan text against verified TFGBV lexicon terms.
    Returns list of matched terms with details.
    Only flags ACTUAL hate speech terms, not innocent mentions.
    """
    if not text or not tfgbv_terms:
        return []
    
    text_lower = text.lower()
    matches = []
    
    for entry in tfgbv_terms:
        term = entry['term']
        if not term:
            continue
        
        # Exact match check (term appears in text)
        if term in text_lower:
            matches.append({
                'term': term,
                'severity': entry['severity'],
                'category': entry['category'],
            })
    
    return matches


def get_enhanced_pep_analysis(posts_queryset, peps_queryset, limit=8):
    """
    Enhanced PEP analysis – scans all posts, correct sentiment, real critical posts.
    """
    pep_names = {pep.name.lower().strip(): pep for pep in peps_queryset if pep.name}
    if not pep_names:
        return []
 
    try:
        tfgbv_terms = get_tfgbv_lexicon_terms()
    except Exception:
        tfgbv_terms = []
 
    # Word-boundary aware critical / positive keyword check
    _CRIT_RE = re.compile(
        r'\b(' + '|'.join(re.escape(k) for k in [
            'kill', 'attack', 'hate', 'enemy', 'genocide', 'massacre',
            'slaughter', 'destroy', 'eliminate', 'terrorist', 'extremist',
            'traitor', 'criminal', 'dictator', 'oppressor', 'failed state',
            'woyane', 'junta', 'banda', 'neftegna', 'galla', 'fano', 'ola',
            'corrupt', 'thief', 'betrayal', 'conspiracy', 'rigged', 'fraud',
            'stolen', 'incompetent', 'useless', 'worthless', 'disgrace',
            # Amharic
            'ግድያ', 'ጦርነት', 'ፈጅ', 'ጥቃት', 'ባንዳ', 'ወያኔ', 'ጁንታ',
            'ነፍጠኛ', 'ሌባ', 'ከሃዲ', 'ውሸታም', 'አምባገነን', 'ሙስና',
        ]) + r')\b',
        re.IGNORECASE,
    )
    _POS_RE = re.compile(
        r'\b(' + '|'.join(re.escape(k) for k in [
            'congratulate', 'support', 'praise', 'excellent', 'wonderful',
            'thank you', 'appreciate', 'success', 'victory', 'progress',
            'achievement', 'proud', 'honor', 'respect', 'admire', 'inspire',
            'peace', 'unity', 'hope', 'together', 'strong', 'vision',
            'democracy', 'freedom', 'justice', 'fair', 'honest', 'trust',
        ]) + r')\b',
        re.IGNORECASE,
    )
    _HOSTILE_RE = re.compile(
        r'(down with|ውድቀት ለ|ይውረድ|blood on hands|ደምናችሁ|ደም ያፈሰሰ'
        r'|puppet of|የ\w+ ተላላኪ|destroying the country|አገር አፈራሽ'
        r'|step down|ስልጣን ልቀቅ|ልቀቁ)',
        re.IGNORECASE,
    )
 
    pep_data = defaultdict(lambda: {
        'count': 0,
        'platforms': Counter(),
        'hourly_distribution': Counter(),
        'hashtags': Counter(),
        'is_gendered_target': False,
        'tfgbv_matches': [],
        'tfgbv_post_count': 0,
        'narrative_clusters': defaultdict(list),
        'critical_posts': [],      # up to 20 confirmed negative posts
        'positive_posts': [],
        'sample_texts': [],        # first 50 texts for Groq
        'negative_count': 0,
        'positive_count': 0,
        'neutral_count': 0,
    })
 
    # ── Scan ALL posts with iterator (memory efficient) ───────────────────
    total_scanned = 0
    for post in posts_queryset.iterator(chunk_size=500):
        if not post.original_text:
            continue
        total_scanned += 1
        text = post.original_text
        text_lower = text.lower()
 
        for pep_name, pep_obj in pep_names.items():
            if pep_name not in text_lower:
                continue
 
            d = pep_data[pep_obj.name]
            d['count'] += 1
 
            platform = post.platform or 'Unknown'
            d['platforms'][platform] += 1
            if post.timestamp_share:
                d['hourly_distribution'][post.timestamp_share.hour] += 1
            for h in re.findall(r'#(\w+)', text):
                d['hashtags'][h] += 1
 
            # Narrative clusters
            if any(kw in text_lower for kw in
                   ['rigged', 'stolen', 'fraud', 'nebe', 'ማጭበርበር',
                    'election', 'vote', 'ballot', 'tally']):
                d['narrative_clusters']['Election Integrity'].append(text[:100])
            if any(kw in text_lower for kw in
                   ['amhara', 'oromo', 'tigray', 'ethnic', 'fano', 'ola',
                    'ነፍጠኛ', 'ጁንታ', 'ወያኔ', 'tribal']):
                d['narrative_clusters']['Ethnic Dynamics & Conflict'].append(text[:100])
            if any(kw in text_lower for kw in
                   ['corrupt', 'corruption', 'bribe', 'embezzle', 'steal',
                    'misuse', 'abuse', 'ሙስና', 'ሌባ']):
                d['narrative_clusters']['Corruption & Accountability'].append(text[:100])
 
            # TFGBV check
            if tfgbv_terms:
                hits = detect_tfgbv_in_text(text, tfgbv_terms)
                if hits:
                    d['is_gendered_target'] = True
                    d['tfgbv_post_count'] += 1
                    for hit in hits:
                        if not any(m['term'] == hit['term']
                                   for m in d['tfgbv_matches']):
                            d['tfgbv_matches'].append(hit)
 
            # Sentiment scoring
            crit_hits = len(_CRIT_RE.findall(text))
            pos_hits  = len(_POS_RE.findall(text))
            hostile   = bool(_HOSTILE_RE.search(text))
 
            is_negative = (crit_hits > pos_hits and crit_hits > 0) or hostile
            is_positive = (pos_hits > crit_hits and pos_hits > 0) and not hostile
 
            post_dict = {
                'text':       text[:250],
                'platform':   platform,
                'timestamp':  post.timestamp_share,
                'risk_level': getattr(post, 'risk_level', 'medium') or 'medium',
                'url': (post.url if post.url
                        and str(post.url).startswith('http') else None),
            }
 
            if is_negative:
                d['negative_count'] += 1
                if len(d['critical_posts']) < 20:
                    d['critical_posts'].append(post_dict)
            elif is_positive:
                d['positive_count'] += 1
                if len(d['positive_posts']) < 10:
                    d['positive_posts'].append(post_dict)
            else:
                d['neutral_count'] += 1
 
            # Collect sample texts for Groq (first 50 per PEP)
            if len(d['sample_texts']) < 50:
                d['sample_texts'].append(text[:300])
 
    logger.info(f"get_enhanced_pep_analysis: scanned {total_scanned} posts")
 
    # ── Build results ──────────────────────────────────────────────────────
    results = []
    for pep_name, d in sorted(
            pep_data.items(), key=lambda x: x[1]['count'], reverse=True
    )[:limit]:
        if d['count'] < 2:
            continue
 
        total_analyzed = d['negative_count'] + d['positive_count'] + d['neutral_count']
        if total_analyzed == 0:
            continue
 
        neg_pct = d['negative_count'] / total_analyzed * 100
        pos_pct = d['positive_count'] / total_analyzed * 100
 
        # Call Groq ONCE per PEP using a representative sample
        try:
            groq_sentiment = analyze_pep_sentiment_groq(
                d['sample_texts'][:10], pep_name
            )
        except Exception:
            groq_sentiment = None
 
        # Prefer Groq if available, fall back to lexicon counts
        if groq_sentiment in ('Positive', 'Negative', 'Mixed', 'Neutral'):
            overall_sentiment = groq_sentiment
        else:
            if neg_pct > 40:
                overall_sentiment = 'Negative'
            elif pos_pct > 40:
                overall_sentiment = 'Positive'
            elif neg_pct > 20:
                overall_sentiment = 'Mixed'
            else:
                overall_sentiment = 'Neutral'
 
        # Risk score
        risk_score = {'Negative': 8, 'Mixed': 5,
                      'Positive': 2, 'Neutral': 3}.get(overall_sentiment, 3)
        if d['tfgbv_post_count'] > 0:
            risk_score = min(10, risk_score + 2)
        if len(d['critical_posts']) >= 5:
            risk_score = min(10, risk_score + 1)
 
        sentiment_label = {
            'Negative': f'🔴 High Criticism ({neg_pct:.0f}% negative)',
            'Mixed':    f'🟡 Mixed Sentiment (Neg: {neg_pct:.0f}%, Pos: {pos_pct:.0f}%)',
            'Positive': f'🟢 Mostly Positive ({pos_pct:.0f}% positive)',
            'Neutral':  f'⚪ Mostly Neutral',
        }.get(overall_sentiment, '⚪ Neutral')
 
        total_posts = sum(d['platforms'].values())
        platform_breakdown = [
            {'name': pl, 'count': cnt,
             'percent': round(cnt / total_posts * 100) if total_posts else 0}
            for pl, cnt in d['platforms'].most_common(3)
        ]
 
        peak_hour = d['hourly_distribution'].most_common(1)
        peak_hour = peak_hour[0][0] if peak_hour else 0
 
        # sample_posts: show critical ones first, then pad with others
        sample_posts = []
        for p in d['critical_posts'][:3]:
            sample_posts.append({**p, 'sentiment': 'negative'})
        for p in d['positive_posts'][:2]:
            if len(sample_posts) < 5:
                sample_posts.append({**p, 'sentiment': 'positive'})
 
        results.append({
            'pep_name':          pep_name,
            'mention_count':     d['count'],
            'platform_breakdown': platform_breakdown,
            'velocity_alert':    peak_hour in [0, 1, 2, 3, 4, 5],
            'peak_hour':         peak_hour,
            'top_hashtags':      [
                {'tag': t, 'count': c}
                for t, c in d['hashtags'].most_common(5) if c > 1
            ],
            'bot_score':         0,
            'bot_level':         '🟢 Low (Likely Human)',
            'is_gendered_target': d['is_gendered_target'],
            'tfgbv_matches':     d['tfgbv_matches'],
            'tfgbv_post_count':  d['tfgbv_post_count'],
            'narrative_clusters': [
                {'name': n, 'count': len(posts)}
                for n, posts in d['narrative_clusters'].items()
            ],
            'sample_posts':      sample_posts,
            'critical_posts':    d['critical_posts'][:5],
            'risk_score':        risk_score,
            'sentiment':         overall_sentiment,
            'sentiment_label':   sentiment_label,
            'sentiment_breakdown': {
                'negative': round(neg_pct, 1),
                'positive': round(pos_pct, 1),
                'neutral':  round(100 - neg_pct - pos_pct, 1),
            },
            'negative_post_count': d['negative_count'],
            'positive_post_count': d['positive_count'],
            'neutral_post_count':  d['neutral_count'],
        })
 
    return results

_INNOCUOUS_RE = re.compile(
    r'\b(peace|democracy|award|commend|congratulate|showcase|celebrate'
    r'|civic|positive|progress|development|strong|engage|youth|student'
    r'|burundi|kenya|foreign observer|international|observer|commend'
    r'|general election|voting|polling station|civic spirit|youth engagement'
    r'|step towards democracy|democratic)\b',
    re.IGNORECASE,
)
 
 
def _is_genuine_match(term: str, text_lower: str, severity: str,
                       is_innocuous_context: bool) -> bool:
    """
    Return True only when the term is a genuine harmful match.
 
    Rules:
    1. ASCII terms must appear as whole words (word-boundary check).
       This prevents 'war' matching 'award', 'forward', 'cowardly'.
    2. Low/medium-severity terms are suppressed in innocuous contexts.
    3. Non-ASCII (Amharic/Oromo) terms use substring matching as before.
    """
    # Non-ASCII (Ge'ez / Ethiopic script) — substring OK
    if re.search(r'[^\x00-\x7F]', term):
        if term.lower() not in text_lower:
            return False
        # Still suppress low-severity in innocuous context
        if is_innocuous_context and severity in ('low',):
            return False
        return True
 
    # ASCII term — require word boundary
    if not re.search(r'\b' + re.escape(term.lower()) + r'\b', text_lower):
        return False
 
    # Suppress low/medium severity in clearly positive/neutral posts
    if is_innocuous_context and severity in ('low', 'medium'):
        return False
 
    return True
    
class BaseTabMixin:
    """Adds consistent navigation tabs to any class-based view"""
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Define tabs once, use everywhere
        context['tabs'] = [
            {'name': 'Home', 'url_name': 'home', 'icon': '🏠'},
            {'name': 'PEPs/PIPs Tracker', 'url_name': 'peps', 'icon': '👤'},
            {'name': 'Mapped Lexicons', 'url_name': 'lexicons', 'icon': '🗣️'},
            {'name': 'Trending Narratives', 'url_name': 'narratives', 'icon': '📰'},
            {'name': 'Networks & TTPs', 'url_name': 'networks', 'icon': '🕸️'},
            {'name': 'Lexicon Management', 'url_name': 'lexicon_management', 'icon': '⚙️'},
            {'name': 'Reports', 'url_name': 'reports_landing', 'icon': '📑'},
        ]
        # Set active_tab if not already set
        if 'active_tab' not in context:
            context['active_tab'] = self.request.resolver_match.url_name
        return context

class HomeView(BaseTabMixin, TemplateView):
    """Executive dashboard - election-focused"""
    template_name = 'dashboard/home.html'
    
    def get_context_data(self, **kwargs):
        # ── 1. CHECK CACHE FIRST ──────────────────────────────────────
        view_all = self.request.GET.get('view_all') == 'true'
        req_start = self.request.GET.get('start_date', '')
        req_end = self.request.GET.get('end_date', '')
        
        # Use v2 to invalidate any old broken caches
        cache_key = f"home_dashboard_v2_{req_start}_{req_end}_{view_all}"
        cached_data = cache.get(cache_key)
        
        if cached_data:
            logger.info("✅ HomeView: Serving from cache")
            context = super().get_context_data(**kwargs)
            context.update(cached_data)
            return context
        
        context = super().get_context_data(**kwargs)
        
        # ── 2. GET FILTERED QUERYSET (now defaults to 3 months) ───────
        queryset, start_date, end_date = get_election_posts_queryset(self.request)
        posts = queryset
        total_posts = posts.count()
        
        # ── 3. PLATFORM DISTRIBUTION ──────────────────────────────────
        platform_stats = posts.values('platform').annotate(
            total_count=Count('id')
        ).order_by('-total_count')
        
        labels = []
        values = []
        colors = []
        color_map = {
            'X': '#1DA1F2',
            'Facebook': '#1877F2',
            'Telegram': '#0088cc',
            'TikTok': '#000000',
            'Media': '#6B7280',
            'YouTube': '#FF0000',
            'Instagram': '#E4405F'
        }
        
        for item in platform_stats:
            p_name = str(item.get('platform') or '').strip()
            p_count = item.get('total_count') or 0
            if p_name and p_count > 0:
                labels.append(p_name)
                values.append(int(p_count))
                colors.append(color_map.get(p_name, '#6B7280'))
        
        top_platform = labels[0] if labels else "—"
        charts = {}
        
        # ── 4. CREATE BAR CHART ───────────────────────────────────────
        if labels:
            total_sum = sum(values)
            raw_chart_dict = {
                "data": [{
                    "x": labels,
                    "y": values,
                    "type": "bar",
                    "text": [f"{v:,}" for v in values],
                    "textposition": "auto",
                    "hoverinfo": "x+y",
                    "marker": {
                        "color": colors,
                        "line": {"color": "#ffffff", "width": 1}
                    }
                }],
                "layout": {
                    "title": f'Post Distribution by Platform (Total: {total_sum:,} posts)',
                    "margin": {"b": 40, "t": 50, "l": 50, "r": 20},
                    "height": 400,
                    "xaxis": {"title": "Platform", "tickmode": "array"},
                    "yaxis": {"title": "Number of Posts", "gridcolor": "#E5E7EB"},
                    "plot_bgcolor": "#ffffff"
                }
            }
            charts['platform'] = json.dumps(raw_chart_dict)
        
        # ── 5. METRICS ────────────────────────────────────────────────
        unique_accounts = posts.values('account_id').distinct().count()
        high_risk_count = posts.filter(risk_level__in=['high', 'critical']).count()
        alert_level = '🚨 High' if high_risk_count > 50 else '⚠️ Medium' if high_risk_count > 10 else '✅ Low'
        peps_tracked = PEP.objects.filter(is_active=True).count()
        last_update = timezone.now().strftime('%Y-%m-%d %H:%M UTC')
        
        # ── 6. OTHER CHARTS ───────────────────────────────────────────
        if posts.exists():
            # Top Accounts
            top_accounts_raw = posts.values('account_id').annotate(count=Count('id')).order_by('-count')[:10]
            cleaned_accounts = []
            invalid_accounts = ['twitter', 'source', 'source twitter source', 'nan', 'none', '-', '', 'user', 'author', 'account']
            
            for acc in top_accounts_raw:
                name = str(acc['account_id']) if acc['account_id'] else ''
                name = re.sub(r'Twitter Source\s*', '', name, flags=re.IGNORECASE)
                name = re.sub(r'Source Twitter Source\s*', '', name, flags=re.IGNORECASE)
                name = re.sub(r'@\w+\s*Name:\s*\d+.*', '', name)
                name = re.sub(r'dtype.*', '', name, flags=re.IGNORECASE)
                name = re.sub(r'\s+', ' ', name).strip()
                
                if name.lower() not in invalid_accounts and name and name not in ['-', 'nan', 'None', '']:
                    cleaned_accounts.append({'account_id': name[:50], 'count': acc['count']})
            
            if cleaned_accounts:
                df_accounts = pd.DataFrame(cleaned_accounts)
                fig_accounts = px.bar(df_accounts, x='account_id', y='count', labels={'account_id': 'Account', 'count': 'Posts'},
                                    color='count', color_continuous_scale='Viridis', title='Top 10 Accounts by Activity')
                fig_accounts.update_layout(xaxis_tickangle=-45, margin=dict(b=100, t=50, l=50, r=20), height=400)
                charts['accounts'] = fig_accounts.to_json()
            
            # Risk Distribution
            risk_dist = posts.values('risk_level').annotate(count=Count('id')).order_by('risk_level')
            if risk_dist:
                fig_risk = px.pie(risk_dist, names='risk_level', values='count', title='Risk Level Distribution',
                                color='risk_level', color_discrete_map={'low': '#22c55e', 'medium': '#eab308', 'high': '#f97316', 'critical': '#dc2626'})
                charts['risk'] = fig_risk.to_json()
            
            # Daily Volume
            daily_posts = posts.annotate(day=TruncDay('timestamp_share')).values('day').annotate(count=Count('id')).order_by('day')
            if daily_posts:
                daily_data = list(daily_posts)
                if daily_data:
                    fig_daily = px.line(daily_data, x='day', y='count', labels={'day': 'Date', 'count': 'Posts'},
                                      title='Daily Post Volume', markers=True)
                    fig_daily.update_layout(xaxis_tickangle=-45, margin=dict(b=100, t=50, l=50, r=20), height=400)
                    charts['daily'] = fig_daily.to_json()
        
        # ── 7. UPLOAD SUMMARY ─────────────────────────────────────────
        recent_uploads = DataUpload.objects.filter(status='completed').order_by('-uploaded_at')[:5]
        upload_summary = {
            'show': len(recent_uploads) > 0 and (recent_uploads[0].uploaded_at > timezone.now() - timedelta(hours=2)),
            'files': recent_uploads,  # Contains model instances for template rendering
            'total_records': sum(u.records_processed for u in recent_uploads),
        }
        
        # ── 8. TREND ANALYSIS ─────────────────────────────────────────
        start_str = start_date.date().isoformat() if hasattr(start_date, 'date') else str(start_date)
        end_str = end_date.date().isoformat() if hasattr(end_date, 'date') else str(end_date)
        trend_analysis = get_category_trend_analysis(posts, days_back=90, cache_suffix=f"{start_str}_{end_str}")
        
        # ── 9. BUILD CONTEXT ──────────────────────────────────────────
        context.update({
            'active_tab': 'home',
            'metrics': {
                'total_posts': total_posts,
                'unique_accounts': unique_accounts,
                'top_platform': top_platform,
                'peps_tracked': peps_tracked,
                'alert_level': alert_level,
                'last_update': last_update,
            },
            'charts': charts,
            'upload_summary': upload_summary,
            'risk_actors': get_risk_actors_insight(posts),
            'top_hashtags': get_top_hashtags(posts),
            'trend_analysis': trend_analysis,
            'start_date': start_str,
            'end_date': end_str,
        })
        
        # ── 10. SAVE TO CACHE (30 minutes) ────────────────────────────
        # Create a cacheable copy without unpicklable objects (like Django model instances)
        cacheable_context = {
            'metrics': context['metrics'],
            'charts': context['charts'],
            'risk_actors': context['risk_actors'],
            'top_hashtags': context['top_hashtags'],
            'trend_analysis': context['trend_analysis'],
            'start_date': context['start_date'],
            'end_date': context['end_date'],
            'upload_summary': {
                'show': upload_summary['show'],
                'total_records': upload_summary['total_records'],
                # Convert model instances to dictionaries to prevent pickle errors
                'files': [
                    {
                        'id': u.id,
                        'original_filename': u.original_filename,
                        'uploaded_at': u.uploaded_at.isoformat() if u.uploaded_at else None,
                        'records_processed': u.records_processed,
                        'status': u.status,
                    }
                    for u in recent_uploads
                ]
            }
        }
        
        try:
            cache.set(cache_key, cacheable_context, 1800)
            logger.info(f"💾 HomeView: Cached for 30 minutes (key: {cache_key})")
        except Exception as e:
            logger.warning(f"⚠️ HomeView: Failed to save to cache: {e}")
        
        return context     
        
class NarrativesView(TemplateView):
    template_name = 'dashboard/narratives.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        # Reuse date filtering helper
        queryset, start_date, end_date = get_election_posts_queryset(
            self.request
        )
        
        # Query-aware cached snapshot fetcher
        payload, generated_at = get_analytics_snapshot(
            'narratives', 
            request_params=self.request.GET.dict()
        )

        context['summaries'] = payload.get('summaries', [])
        context['generated_at'] = generated_at
        context['total_posts'] = queryset.count()

        # Date range display
        if start_date and end_date:
            # Handle both datetime objects and string inputs
            start_str = start_date.date().isoformat() if hasattr(start_date, 'date') else start_date
            end_str = end_date.date().isoformat() if hasattr(end_date, 'date') else end_date
            context['date_range'] = f"{start_str} to {end_str}"
        else:
            context['date_range'] = "Last 30 days (default)"

        # Form values
        context['start_date'] = start_date.date().isoformat() if hasattr(start_date, 'date') else start_date
        context['end_date'] = end_date.date().isoformat() if hasattr(end_date, 'date') else end_date

        # Monitoring reports
        context['monitoring_reports'] = (
            MonitoringReport.objects
            .all()
            .order_by('-uploaded_at')[:12]
        )
        
        return context
class LexiconsView(TemplateView):
    template_name = 'dashboard/lexicons.html'
    
    # ── CONFIGURATION ──────────────────────────────────────────────────────
    SCAN_LIMIT = 5000           # Baseline limit for regular queries
    TIMEOUT_SECONDS = 90        # Hard stop after 90 seconds (leaves buffer for 100s Cloudflare limit)
    CHUNK_SIZE = 1000           # Process posts in larger chunks
    CACHE_DURATION = 7200       # Cache for 2 hours (7200 seconds)
    
    # Words that are too generic on their own and require secondary context markers
    GENERIC_TERMS_CONTEXT_MAP = {
        'foreign': ['interference', 'meddling', 'influence', 'election', 'funding', 'sponsored', 'agent', 'pressure'],
        'ውጭ': ['ጣልቃ', 'ተዕኖ', 'ገንዘብ', 'ምርጫ', 'ሴራ', 'እጅ']
    }

    # ─ CATEGORY DISPLAY NAME MAPPING ─────────────────────────────────────
    # Maps internal category keys to user-friendly display names.
    # Note: Using the internal key as the display name prevents any translation mismatches!
    CATEGORY_DISPLAY_NAMES = {
        'foreign_interference': 'Cross-Border Geopolitical Narratives',
        'ethnic_identity': 'ethnic_identity',
        'political_groups': 'political_groups',
        'violence_incitement': 'violence_incitement',
        'dehumanizing': 'dehumanizing',
        'election_governance': 'election_governance',
        'religious_cultural': 'religious_cultural',
        'gender_misogynistic': 'gender_misogynistic',
        'discriminatory_homophobic': 'discriminatory_homophobic',
        'socio_economic_caste': 'socio_economic_caste',
    }
    
    # Reverse mapping to translate UI names back to internal keys
    DISPLAY_TO_INTERNAL = {v: k for k, v in CATEGORY_DISPLAY_NAMES.items()}
    
    # Words that are too generic on their own and require secondary context markers
    GENERIC_TERMS_CONTEXT_MAP = {
        'foreign': ['interference', 'meddling', 'influence', 'election', 'funding', 'sponsored', 'agent', 'pressure'],
        'የውጭ': ['ጣልቃ', 'ጣልቃ-ገብነት', 'ተጽዕኖ', 'ገንዘብ', 'ምርጫ', 'ሴራ', 'እጅ', 'ጫና', 'ወኪል'],
    }

    def _get_lexicon_term_count(self):
        """Count total terms in CONFIG lexicon + Database."""
        try:
            total = sum(len(terms) for terms in CONFIG.get('lexicon', {}).values())
            total += LexiconTerm.objects.count()
            return total
        except Exception:
            return "1000+"

    def _is_valid_context(self, term: str, text_lower: str) -> bool:
        """Enforces that ultra-generic words must appear alongside a secondary context word."""
        term_clean = term.strip().lower()
        if term_clean in self.GENERIC_TERMS_CONTEXT_MAP:
            required_contexts = self.GENERIC_TERMS_CONTEXT_MAP[term_clean]

            # Verify if at least one context-strengthening word is present

            if not any(context in text_lower for context in required_contexts):
                return False
        return True

    def _scan_post(self, text: str, category_filter=None):
        """Scan one post's text using lexicon."""
        if not text or len(text.strip()) < 10:
            return []
        text_lower = text.lower()
        is_innocuous = bool(_INNOCUOUS_RE.search(text_lower))
        raw_matches = scan_text_for_lexicon_terms(text, category_filter=category_filter)
        filtered = []
        for m in raw_matches:
            if len(m['term'].strip()) <= 1:
                continue

            # Context Check: filter out generic keywords lacking supporting intent context


            if not self._is_valid_context(m['term'], text_lower):
                continue
            if _is_genuine_match(m['term'], text_lower, m.get('severity', 'medium'), is_innocuous):
                filtered.append(m)
        return filtered
    
    def _check_timeout(self, start_time):
        """Check if we're approaching the timeout limit."""
        import time
        elapsed = time.time() - start_time
        if elapsed >= self.TIMEOUT_SECONDS:
            logger.warning(f"⏰ TIMEOUT: Scan stopped after {elapsed:.1f}s (limit: {self.TIMEOUT_SECONDS}s)")
            return True
        return False

    def get_context_data(self, **kwargs):
        import time
        start_time = time.time()
        context = super().get_context_data(**kwargs)
        
        # ── 1. URL params ────────────────────────────────────────────────
        selected_category = self.request.GET.get('category', '').strip()

        raw_category = self.request.GET.get('category', '').strip()

        view_all = self.request.GET.get('view_all') == 'true'
        req_start = self.request.GET.get('start_date', '')
        req_end   = self.request.GET.get('end_date', '')
        

        # ── 2. Cache (overview only, keyed to date range) ─────────────────

        # Translate display name to internal key before processing.
        # Because most display names ARE the internal keys, this safely resolves everything.
        selected_category = self.DISPLAY_TO_INTERNAL.get(raw_category, raw_category)
        
        # ─ 2. Cache (overview only, keyed to date range) ─────────────────

        cache_key = f"lexicon_dashboard_v7_{req_start}_{req_end}_{view_all}_{selected_category}"
        cached_data = cache.get(cache_key)
        
        if cached_data and not selected_category:
            logger.info("✅ LexiconsView: Serving from cache (instant load)")
            context.update(cached_data)
            context['lexicon_term_count'] = self._get_lexicon_term_count()
            context['selected_category'] = ''
            context['category_terms']    = []
            context['posts_with_terms']  = []
            context['scan_timed_out']    = False
            return context
        
        # ── 3. Fetch posts ────────────────────────────────────────────────
        try:
            _, start_date, end_date = get_election_posts_queryset(self.request)
            
            # Build database conditions dynamically
            query_filters = {}
            if not view_all:
                query_filters['timestamp_share__range'] = (start_date, end_date)
                
            filtered_posts = ProcessedPost.objects.filter(
                **query_filters
            ).exclude(platform__icontains='media').exclude(platform__icontains='news').order_by('-timestamp_share')
            
            total_posts = filtered_posts.count()
        except Exception as e:
            logger.error(f"LexiconsView error: {e}")
            return context
        
        start_str = (start_date.date().isoformat() if hasattr(start_date, 'date') else str(start_date))
        end_str   = (end_date.date().isoformat() if hasattr(end_date, 'date') else str(end_date))
        
        # Expand limits safely. Category view gets 10k, Overview gets 5k to prevent timeouts.
        effective_limit = 10000 if selected_category else self.SCAN_LIMIT
        scan_pool = filtered_posts[:effective_limit].iterator(chunk_size=2000)
        
        category_terms  = []
        posts_with_terms = []
        all_matches     = []
        posts_scanned   = 0
        scan_timed_out  = False
        
        # ── 4a. CATEGORY VIEW ──────────────────────────────────────────────
        if selected_category:
            logger.info(f"LexiconsView: category view → {selected_category} (limit: {effective_limit} posts)")

            
            db_terms = LexiconTerm.objects.filter(category=selected_category)
            if db_terms.exists():
                category_terms = [{'term': t.term, 'severity': t.severity, 'target_entity': t.target_entity, 'language': t.language} for t in db_terms]
            elif selected_category in CONFIG.get('lexicon', {}):
                category_terms = [{'term': t, 'severity': m.get('severity', 'medium'), 'target_entity': m.get('target_entity', ''), 'language': m.get('language', '')} for t, m in CONFIG['lexicon'][selected_category].items()]
            

            # Initialize the variable to avoid UnboundLocalError
            category_terms = []
            # 1. Look up using case-insensitive check in DB
            db_terms = LexiconTerm.objects.filter(category__iexact=selected_category)
            if db_terms.exists():
                category_terms = [{'term': t.term, 'severity': t.severity, 'target_entity': t.target_entity, 'language': t.language} for t in db_terms]
            else:
                # 2. Case-insensitive fallback lookup in the CONFIG dictionary
                config_lexicon = CONFIG.get('lexicon', {})
                matched_config_key = next((k for k in config_lexicon.keys() if k.lower() == selected_category.lower()), None)
                if matched_config_key:
                    category_terms = [{'term': t, 'severity': m.get('severity', 'medium'), 'target_entity': m.get('target_entity', ''), 'language': m.get('language', '')} for t, m in config_lexicon[matched_config_key].items()]
            # This check is now completely safe!

            if category_terms:
                term_meta = {td['term'].lower(): td for td in category_terms if len(td['term']) > 1}
                # SPEED OPTIMIZATION: Pre-verify matching text layout using regex patterns
                regex_pattern = r'(' + '|'.join(re.escape(t) for t in term_meta.keys()) + r')'
                try:
                    compiled_category_re = re.compile(regex_pattern, re.IGNORECASE)
                except Exception:
                    compiled_category_re = None
                

                # SPEED OPTIMIZATION: Pre-verify matching text layout using regex patterns
                regex_pattern = r'(' + '|'.join(re.escape(t) for t in term_meta.keys()) + r')'
                try:
                    compiled_category_re = re.compile(regex_pattern, re.IGNORECASE)
                except Exception:
                    compiled_category_re = None

                # Track seen posts to avoid duplicates
                seen_post_ids = set()
                seen_post_texts = set()

                
                for post in scan_pool:
                    if self._check_timeout(start_time):
                        scan_timed_out = True
                        break
                    posts_scanned += 1
                    if not post.original_text: 
                        continue
                    

                    posts_scanned += 1
                    if not post.original_text: continue
                    # Skip RT/retweet posts
                    if post.original_text.strip().lower().startswith('rt ') or post.original_text.strip().lower().startswith('rt\n'):
                        continue
                    
                    # Skip if we've already seen this post
                    if post.id in seen_post_ids:
                        continue

                    
                    text       = post.original_text
                    text_lower = text.lower()
                    

                    # Instantly drops unrelated records out of processing loop
                    if compiled_category_re and not compiled_category_re.search(text_lower):
                        continue
                        

                    # Skip if we've already seen this exact text content
                    text_hash = hash(text.strip())
                    if text_hash in seen_post_texts:
                        continue
                    
                    # Instantly drops unrelated records out of processing loop
                    if compiled_category_re and not compiled_category_re.search(text_lower):
                        continue

                    is_inoc    = bool(_INNOCUOUS_RE.search(text_lower))
                    matched_terms = []
                    for term_lower, meta in term_meta.items():
                        if term_lower not in text_lower:
                            continue
                        if not self._is_valid_context(meta['term'], text_lower):
                            continue
                        if not _is_genuine_match(term_lower, text_lower, meta.get('severity', 'medium'), is_inoc):
                            continue

                        


                        matched_terms.append(meta['term'])
                        all_matches.append({'term': meta['term'], 'category': selected_category, 'severity': meta.get('severity', 'medium'), 'target_entity': meta.get('target_entity', ''), 'language': meta.get('language', '')})
                    if matched_terms:


                        # Mark this post as seen
                        seen_post_ids.add(post.id)
                        seen_post_texts.add(text_hash)
                        

                        posts_with_terms.append({
                            'id':            post.id,
                            'text':          text,
                            'platform':      post.platform,
                            'timestamp':     post.timestamp_share,
                            'url':           post.url,
                            'matched_terms': list(set(matched_terms))[:5],
                            'detected_by':   'Lexicon',
                            'confidence':    1.0,
                            'model_category': selected_category,
                        })
                
                # LLM translations block
                unique_foreign_terms = {t for p in posts_with_terms for t in p.get('matched_terms', []) if re.search(r'[^\x00-\x7F]', t)}
                if unique_foreign_terms:
                    translations_map = batch_translate_terms_llm(list(unique_foreign_terms))
                    for p_dict in posts_with_terms:
                        p_dict['english_translations'] = [f"{t}: {translations_map[t]}" for t in p_dict.get('matched_terms', []) if t in translations_map]

        # ── 4b. OVERVIEW SCAN ─────────────────────────────────────────────
        else:
            logger.info(f"LexiconsView: overview scan (limit: {effective_limit} posts)")

            


            for post in scan_pool:
                if self._check_timeout(start_time):
                    scan_timed_out = True
                    break
                
                posts_scanned += 1
                if not post.original_text: continue
                
                try:
                    matches = self._scan_post(post.original_text)
                    if matches: 
                        all_matches.extend(matches)
                except Exception as e: 
                    continue
        
        # ── 5. AGGREGATE ANALYTICS ─────────────────────────────────────────
        try:
            term_counts     = Counter([m['term']     for m in all_matches])
            raw_category_counts = Counter([m['category'] for m in all_matches])
            severity_counts = Counter([m['severity'] for m in all_matches])
            
            # Transform category keys to display names
            category_counts = Counter()
            for cat_key, count in raw_category_counts.items():
                display_name = self.CATEGORY_DISPLAY_NAMES.get(cat_key, cat_key)
                category_counts[display_name] = count
            
            if selected_category and category_terms:
                top_terms_with_meta = sorted([{'term': td['term'], 'count': term_counts.get(td['term'], 0), 'metadata': td} for td in category_terms], key=lambda x: x['count'], reverse=True)
            else:
                top_terms_with_meta = []
                for term, count in term_counts.most_common(15):
                    if len(term.strip()) <= 1: continue
                    metadata = {}
                    for cat, terms in CONFIG.get('lexicon', {}).items():
                        if term in terms: 
                            metadata = terms[term]
                            break
                    if not metadata:
                        db_t = LexiconTerm.objects.filter(term=term).first()
                        if db_t: metadata = {'severity': db_t.severity, 'target_entity': db_t.target_entity, 'language': db_t.language}
                    top_terms_with_meta.append({'term': term, 'count': count, 'metadata': metadata})
        except Exception as e:
            logger.error(f"LexiconsView: aggregation error: {e}")
            top_terms_with_meta = []
            category_counts     = Counter()
            severity_counts     = Counter()
        
        # ─ 6. Word cloud & 7. Targeted entities ───────────────────────────
        wordcloud_base64 = None
        if all_matches and not selected_category:
            try:
                valid_terms = [{'term': t, 'count': c} for t, c in term_counts.most_common(50) if len(t.strip()) > 1]
                wc = generate_trigger_wordcloud({'top_terms': valid_terms})
                if wc: wordcloud_base64 = wordcloud_to_base64(wc)
            except Exception: pass
            
        targeted_entities = []
        if not selected_category:
            try:
                entity_patterns = [r'\b(Abiy\s+Ahmed|Prosperity\s+Party|FANO|NEBE)\b', r'\b(Amhara|Tigray|Oromo|Somali)\b']
                entities_found = Counter()
                for m in all_matches:
                    for pattern in entity_patterns:
                        for match in re.findall(pattern, m['term'], re.IGNORECASE):
                            entities_found[match.strip()] += 1
                targeted_entities = [{'entity': e, 'count': c} for e, c in entities_found.most_common(10)]
            except Exception: pass
        
        # ── 8. Build context ───────────────────────────────────────────────
        shared = {
            'active_tab': 'lexicons', 'top_terms': top_terms_with_meta,
            'category_counts': dict(category_counts), 'severity_counts': dict(severity_counts),
            'total_matches': len(all_matches), 'posts_scanned': posts_scanned,
            'total_posts': total_posts, 'start_date': start_str, 'end_date': end_str,
            'lexicon_term_count': self._get_lexicon_term_count(),
            'scan_timed_out': scan_timed_out, 'wordcloud_base64': wordcloud_base64,
            'targeted_entities': targeted_entities
        }
        
        # ── CACHE (only simple data, no model instances) ───────
        if not selected_category:
            try:
                cache.set(cache_key, shared, self.CACHE_DURATION)
                logger.info(f"💾 LexiconsView cached for {self.CACHE_DURATION}s")
            except Exception as e:
                logger.warning(f"Cache save failed: {e}")
        
        context.update(shared)

        context['selected_category'] = selected_category
        context['selected_category'] = raw_category  # Keep original for UI highlighting

        context['category_terms'] = category_terms
        context['posts_with_terms'] = posts_with_terms[:100]
        
        return context
        
    @staticmethod
    def _run_ai_analysis_background(post_ids, cache_key):
        """Runs AFRO-XLMR model in background thread and saves results to cache."""
        import traceback
        logger.info(f"Background AI analysis STARTED for {len(post_ids)} posts")
        try:
            from .utils.hate_speech_detector import get_hate_speech_detector
            from .models import ProcessedPost
            
            detector = get_hate_speech_detector()
            if detector is None:
                logger.error("AFRO-XLMR detector failed to load.")
                cache.delete("lexicons_ai_running")
                return
                
            posts = ProcessedPost.objects.filter(id__in=post_ids)
            category_counts = Counter()
            total_analyzed  = 0
            
            afro_severity_map = {
                'Violence': 'critical', 'Inciteful': 'critical', 'Call for action': 'critical',
                'Dehumanization': 'critical', 'Extremism': 'high', 'Ethnic slur': 'high',
                'Slur': 'high', 'Misognistic': 'high', 'Deragatory': 'medium',
                'Inflammatory': 'high', 'Gender disinformation': 'high', 'Stereotype': 'high',
                'Homophobic': 'high', 'Ethnicity': 'high', 'Xenophobia': 'high', 'Religion': 'high',
                'Ancestry': 'medium', 'Class': 'low', 'Stractural': 'low',
            }
            
            for idx, post in enumerate(posts):
                if post.original_text and len(post.original_text) > 20:
                    try:
                        result = detector.detect(post.original_text)
                        cat = result.get('category')
                        if cat and cat not in ['Neutral', 'error']:
                            category_counts[cat] += 1
                            total_analyzed += 1
                        if (idx + 1) % 10 == 0:
                            logger.info(f"Progress: {idx + 1}/{posts.count()} posts")
                    except Exception as e:
                        logger.warning(f"Scan failed post {post.id}: {e}")
                        
            total_hateful = sum(category_counts.values())
            ai_results = [{'category': cat.replace('_', ' ').title(), 'count': count, 'percentage': round(count / total_analyzed * 100, 1) if total_analyzed else 0, 'severity': afro_severity_map.get(cat, 'medium')} for cat, count in category_counts.most_common(5)]
            
            cache.set(cache_key, {'results': ai_results, 'total_analyzed': total_analyzed, 'total_hateful': total_hateful}, 86400)
            logger.info(f"Background analysis COMPLETE: {total_hateful} hateful / {total_analyzed} analysed.")
            
        except Exception as e:
            logger.error(f"Background analysis FAILED: {e}")
            logger.error(traceback.format_exc())
            cache.set(cache_key, {'results': [], 'total_analyzed': 0, 'total_hateful': 0, 'error': str(e)}, 3600)
        finally:
            cache.delete("lexicons_analysis_running")
            logger.info("Background thread finished and cleaned up.")
            
class PEPsHubView(TemplateView):
    template_name = 'dashboard/peps_hub.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        #  Clear default ordering before distinct() to prevent duplicates
        context['files'] = ElectionOfficeholder.objects.order_by().values('source_file').annotate(
            total=Count('id')
        ).order_by('source_file')
        return context
        
class PEPsDataView(TemplateView):
    template_name = 'dashboard/peps_data.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        filename = self.request.GET.get('file')
        sheet = self.request.GET.get('sheet', '').strip()
        
        # Base queryset
        qs = ElectionOfficeholder.objects.filter(source_file=filename) if filename else ElectionOfficeholder.objects.none()
        
        # Get DISTINCT sheet names exactly once, sorted alphabetically
        if filename:
            context['sheets'] = list(ElectionOfficeholder.objects.filter(
                source_file=filename
            ).values_list('source_sheet', flat=True).distinct().order_by('source_sheet'))
            # Filter out empty/None values
            context['sheets'] = [s for s in context['sheets'] if s]
        else:
            context['sheets'] = []
            
        # Default to first sheet if none selected
        if not sheet and context['sheets']:
            sheet = context['sheets'][0]
            
        if sheet:
            qs = qs.filter(source_sheet=sheet)
            
        paginator = Paginator(qs.order_by('row_index'), 50)
        page_number = self.request.GET.get('page')
        page_obj = paginator.get_page(page_number)
        
        # Pre-align rows to match Excel column order
        column_order = page_obj[0].column_order if page_obj else []
        column_order = [col for col in column_order if col.lower() != 'researcher']
        aligned_rows = [[row.raw_data.get(col) for col in column_order] for row in page_obj]
        
        context.update({
            'page_obj': page_obj,
            'selected_file': filename,
            'selected_sheet': sheet,
            'total_records': qs.count(),
            'column_order': column_order,
            'aligned_rows': aligned_rows,
            'active_tab': 'peps'  # Keeps PEPs tab highlighted
        })
        return context
        
class PEPsView(TemplateView):
    template_name = 'dashboard/peps.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        #view_all = self.request.GET.get('view_all') == 'true'
        #req_start = self.request.GET.get('start_date', '')
        #req_end = self.request.GET.get('end_date', '')
        
        #cache_key = f"peps_view_v1_{req_start}_{req_end}_{view_all}"
        #cached_data = cache.get(cache_key)
        
        #if cached_data:
        #    logger.info("✅ PEPsView: Serving from cache")
         #   return cached_data
        
       # context = super().get_context_data(**kwargs)
        
        # 1. Force sync standard PEPs if DB is empty (existing logic)
        if not PEP.objects.exists():
            github_url = getattr(settings, 'PEPS_CSV_URL', '')
            if github_url:
                try:
                    resp = requests.get(github_url, timeout=15)
                    resp.raise_for_status()
                    lines = resp.text.split('\n')
                    data_lines = '\n'.join([line for line in lines[1:] if line.strip()])
                    reader = csv.DictReader(StringIO(data_lines))
                    for row in reader:
                        name = row.get('full_name_en', '').strip()
                        if not name: continue
                        def clean_link(val):
                            v = str(val).strip() if val else ''
                            return None if v.lower() in ['n/a', 'none', '', 'no verified personal account found'] else v
                        PEP.objects.update_or_create(
                            name=name,
                            defaults={
                                'title': row.get('role', ''),
                                'affiliation': row.get('party_name_en', ''),
                                'ethnic_group': row.get('region', ''),
                                'x_link': clean_link(row.get('twitter_url')),
                                'facebook_link': clean_link(row.get('fb_url')),
                                'confidence_level': 'medium',
                                'last_updated': timezone.now()
                            }
                        )
                except Exception as e:
                    logger.error(f"PEP GitHub sync failed: {e}")

        # 2. Query Standard PEPs
        active_peps = PEP.objects.filter(is_active=True).order_by('name')
        
        # 3. Extract RC Members Names from the newly uploaded Excel file
        rc_member_names = []
        
        # Query ElectionOfficeholder for the RC_Members.xlsx file
        rc_qs = ElectionOfficeholder.objects.filter(source_file__icontains='RC_Members')
        
        for member in rc_qs:
            rd = member.raw_data or {}
            # Smart extraction: look for any key containing 'name' (e.g., 'Name', 'Full Name', 'Member Name')
            for key, val in rd.items():
                if 'name' in key.lower() and val:
                    name_val = str(val).strip()
                    if len(name_val) > 2 and name_val.lower() not in ['nan', 'none', '']:
                        rc_member_names.append(name_val)
                        break
                        
        # Remove duplicates
        rc_member_names = list(set(rc_member_names))
        
        # 4. Get election-related posts for analysis
        election_posts = ProcessedPost.objects.filter(is_election_related=True).order_by('-timestamp_share')
        
        # 5. Generate ENHANCED PEP analysis with Groq sentiment analysis
        pep_analysis_data = get_enhanced_pep_analysis(
          election_posts, 
          active_peps, 
          limit=8  # Show top 8 mentioned officials
        )
        
        # 6. Build context for the template
        context.update({
            'total_candidates': ElectionOfficeholder.objects.count(),
            'peps': active_peps,  # For the officials table
            'total_peps': active_peps.count(),
            
            # NEW: RC Members data for the UI
            'rc_members_count': len(rc_member_names),
            'rc_members_list': rc_member_names[:20],  # Show first 20 in the UI
            
            'verified_x_count': active_peps.filter(x_verified=True).count(),
            'verified_fb_count': active_peps.filter(facebook_verified=True).count(),
            'last_pep_sync': PEP.objects.aggregate(last=Max('last_updated'))['last'],
            'active_tab': 'peps',
            'pep_analysis': pep_analysis_data,  # For the PEPs analysis tab
        })
        # ── SAVE TO CACHE (20 minutes) ────────────────────────────
        #cache.set(cache_key, context, 1200)
        #logger.info(f"💾 PEPsView: Cached for 20 minutes")
        return context        
        
class NetworksView(TemplateView):
    template_name = 'dashboard/networks.html'

    def get(self, request, *args, **kwargs):

        # ── CHECK CACHE FIRST ──────────────────────────────────────
        #min_connections = int(request.GET.get('min_connections', 3))
        #view_all = request.GET.get('view_all') == 'true'
        #req_start = request.GET.get('start_date', '')
        #req_end = request.GET.get('end_date', '')
        
       # cache_key = f"networks_view_v1_{min_connections}_{req_start}_{req_end}_{view_all}"
        #cached_data = cache.get(cache_key)
        
        #if cached_data:
        #    logger.info("✅ NetworksView: Serving from cache")
        #    return render(request, self.template_name, cached_data)
            
        # 1. Parse connection limits and parameters
        min_connections = int(request.GET.get('min_connections', 3))
        top_n = int(request.GET.get('top_n', 40))
        layout_style = request.GET.get('layout', 'spring')

        # 2. Extract coordination data
        posts_qs = ProcessedPost.objects.filter(is_election_related=True)
        coordination_groups = get_coordination_groups(posts_qs, min_accounts=min_connections)

        # Calculate graph stats
        unique_nodes = set()
        for group in coordination_groups:
            unique_nodes.update(str(acc) for acc in group.get('accounts', []))
        
        graph_stats = {
            'nodes': len(unique_nodes),
            'edges': sum(len(group.get('accounts', [])) * (len(group.get('accounts', [])) - 1) // 2 for group in coordination_groups)
        }
        
        total_coordinated_groups = len(coordination_groups)
        total_coordinated_accounts = graph_stats.get('nodes', 0)
        max_group_size = max([g.get('account_count', 0) for g in coordination_groups]) if coordination_groups else 0
        total_posts_analyzed = posts_qs.count()

        # 3. Generate Network Graph
        network_graph_json = "{}"
        if total_coordinated_groups > 0:
            try:
                G = nx.Graph()
                active_groups_subset = coordination_groups[:top_n]
                
                # ── Build a proper source→amplifier map ──────────────────
                # For each group, find the account with the OLDEST timestamp
                # that is NOT retweeting someone else. That's the source.
                group_sources = {}  # group_id -> source_account
                for group in active_groups_subset:
                    source_account = None
                    earliest_ts = None
                    sample_posts = group.get('sample_posts_with_urls', [])
                    
                    for post in sample_posts:
                        username = post.get('username', '')
                        timestamp_str = post.get('timestamp', '')
                        text = post.get('text_preview', '')
                        
                        # Skip if this post is clearly an RT/QT of someone else
                        if text and re.match(r'^(RT|QT|repost)\s+@\w+', text.strip(), re.IGNORECASE):
                            continue
                        
                        # Parse timestamp
                        ts = None
                        if timestamp_str and timestamp_str != 'N/A':
                            try:
                                ts = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M')
                            except:
                                try:
                                    ts = datetime.fromisoformat(timestamp_str)
                                except:
                                    ts = None
                        
                        if ts is None:
                            continue
                        
                        if earliest_ts is None or ts < earliest_ts:
                            earliest_ts = ts
                            source_account = username
                    
                    if source_account:
                        group_sources[group['id']] = source_account
                    elif sample_posts:
                        # Fallback: use first non-RT account
                        for post in sample_posts:
                            text = post.get('text_preview', '')
                            if not re.match(r'^(RT|QT|repost)\s+@\w+', text.strip(), re.IGNORECASE):
                                group_sources[group['id']] = post.get('username', '')
                                break
                
                # ─ Build graph edges ──────────────────────────────────────────
                for group in active_groups_subset:
                    accounts = group.get('accounts', [])
                    weight = group.get('post_count', 1)
                    for i in range(len(accounts)):
                        for j in range(i + 1, len(accounts)):
                            node_a = str(accounts[i])
                            node_b = str(accounts[j])
                            if G.has_edge(node_a, node_b):
                                G[node_a][node_b]['weight'] += weight
                            else:
                                G.add_edge(node_a, node_b, weight=weight)
                
                # ── Layout ────────────────────────────────────────────────────
                if layout_style == 'circular':
                    pos = nx.circular_layout(G)
                elif layout_style == 'kamada_kawai':
                    pos = nx.kamada_kawai_layout(G)
                else:
                    pos = nx.spring_layout(G, k=0.4, iterations=30)
                
                # ── FIX: Mark nodes as source or amplifier ─────────────────────
                # An account is a "source" if it's the originator in ANY group
                source_accounts = set(group_sources.values())
                
                nodes_list = []
                for node in G.nodes():
                    is_source = node in source_accounts
                    nodes_list.append({
                        'id': node,
                        'label': str(node)[:15],
                        'x': float(pos[node][0]),
                        'y': float(pos[node][1]),
                        'type': 'source' if is_source else 'amplifier',
                        'size': int(G.degree(node))
                    })
                
                edges_list = []
                for u, v, data in G.edges(data=True):
                    edges_list.append({
                        'source': u, 'target': v,
                        'source_x': float(pos[u][0]), 'source_y': float(pos[u][1]),
                        'target_x': float(pos[v][0]), 'target_y': float(pos[v][1]),
                        'weight': int(data.get('weight', 1))
                    })
                
                network_graph_json = json.dumps({'nodes': nodes_list, 'edges': edges_list})
            except Exception as e:
                logger.error(f"Error drawing network: {e}")
        # 4. ACTUAL TTP ANALYSIS (Replaces the broken static reference loop)
        # This analyzes your coordination_groups to find the 12 TTPs and attaches real evidence posts
        ttps = analyze_ttps(coordination_groups, posts_qs)
        
        # Ensure the template gets the posts under the key 'evidence_posts'
        for ttp in ttps:
            if 'example_posts' in ttp and 'evidence_posts' not in ttp:
                ttp['evidence_posts'] = ttp['example_posts']
            elif 'evidence_posts' not in ttp:
                ttp['evidence_posts'] = []

        # 5. Build Context
        context = {
            'min_connections': min_connections,
            'top_n': top_n,
            'layout_style': layout_style,
            'coordination_groups': coordination_groups,
            'coordination_groups_json': json.dumps(coordination_groups, default=str),
            'network_graph_json': network_graph_json,
            'graph_stats': graph_stats,
            'total_coordinated_groups': total_coordinated_groups,
            'total_coordinated_accounts': total_coordinated_accounts,
            'max_group_size': max_group_size,
            'total_posts': total_posts_analyzed,
            'ttps': ttps,  
        }
        # ── SAVE TO CACHE (15 minutes) ────────────────────────────
        #cache.set(cache_key, context, 900)
        #logger.info(f"💾 NetworksView: Cached for 15 minutes")
        
        return render(request, self.template_name, context)
        
class LexiconManagementView(TemplateView):
    template_name = 'dashboard/lexicon_management.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Load lexicon terms from DB, excluding single characters
        lexicon_terms = LexiconTerm.objects.exclude(
            term__regex=r'^.$'
        ).order_by('category', 'severity')



        

        # If DB is empty, seed from CONFIG (one-time migration)
        if not lexicon_terms.exists():
            for category, terms in CONFIG['lexicon'].items():
                for term, metadata in terms.items():
                    if len(term.strip()) > 1:
                        LexiconTerm.objects.get_or_create(
                            term=term,
                            defaults={
                                'category': category,
                                'severity': metadata.get('severity', 'medium'),
                                'target_entity': metadata.get('target_entity', ''),
                                'language': metadata.get('language', 'english'),


                                'justification': '',  # Empty for CONFIG terms

                                'is_election_related': True
                            }
                        )
            lexicon_terms = LexiconTerm.objects.exclude(
                term__regex=r'^.$'
            ).order_by('category', 'severity')


        # RESPECT THE GLOBAL DATE FILTER
        filtered_posts, start_date, end_date = get_election_posts_queryset(self.request)
        total_posts_in_filter = filtered_posts.count()

        # Only scan the most recent 3000 posts instead of all
        posts_to_scan = filtered_posts[:3000]
        

        
        # RESPECT THE GLOBAL DATE FILTER
        filtered_posts, start_date, end_date = get_election_posts_queryset(self.request)
        total_posts_in_filter = filtered_posts.count()
        
        # Only scan the most recent 3000 posts instead of all
        posts_to_scan = filtered_posts[:3000]

        all_matches = []
        posts_scanned = 0
        
        # Scan only the limited dataset
        for post in posts_to_scan.iterator():
            if post.original_text:
                try:
                    matches = scan_text_for_lexicon_terms(post.original_text)
                    if matches:
                        all_matches.extend([m for m in matches if len(m['term'].strip()) > 1])
                        posts_scanned += 1
                except Exception as e:
                    logger.warning(f"Error scanning post {post.id}: {e}")
                    continue


        # Get distinct categories for filter dropdown
        categories = lexicon_terms.values_list('category', flat=True).distinct()

        # Get scan results from session (if any) and clear immediately
        scan_results = self.request.session.pop('scan_results', None)

        # Determine filter state for the template text
        view_all = self.request.GET.get('view_all') == 'true'
        has_custom_date_filter = self.request.GET.get('start_date') and self.request.GET.get('end_date') and not view_all


        
        # Get distinct categories for filter dropdown
        categories = lexicon_terms.values_list('category', flat=True).distinct()
        
        # Get scan results from session (if any) and clear immediately
        scan_results = self.request.session.pop('scan_results', None)
        
        # Determine filter state for the template text
        view_all = self.request.GET.get('view_all') == 'true'
        has_custom_date_filter = self.request.GET.get('start_date') and self.request.GET.get('end_date') and not view_all
        

        context.update({
            'active_tab': 'lexicon_management',
            'lexicon_terms': lexicon_terms,
            'categories': categories,
            'total_terms': lexicon_terms.count(),
            'critical_count': lexicon_terms.filter(severity='critical').count(),
            'amharic_count': lexicon_terms.filter(language='amharic').count(),
            'scan_results': scan_results,
            'total_matches': len(all_matches),
            'posts_scanned': posts_scanned,
            'total_posts': total_posts_in_filter,
            'view_all': view_all,
            'has_custom_date_filter': has_custom_date_filter,
            'start_date': start_date.date().isoformat() if hasattr(start_date, 'date') else start_date,
            'end_date': end_date.date().isoformat() if hasattr(end_date, 'date') else end_date,
        })
        return context

    def post(self, request, *args, **kwargs):
        action = request.POST.get('action')

        # Handle Edit Term
        if action == 'edit_term':
            term_id = request.POST.get('term_id')
            if term_id:
                try:
                    obj = LexiconTerm.objects.get(id=term_id)
                    new_term = request.POST.get('term', '').strip()
                    if new_term and len(new_term) > 1:
                        obj.term = new_term
                        obj.category = request.POST.get('category', obj.category)
                        obj.severity = request.POST.get('severity', obj.severity)
                        obj.target_entity = request.POST.get('target_entity', '')
                        obj.language = request.POST.get('language', 'english')
                        obj.justification = request.POST.get('justification', '')  
                        obj.save()
                        messages.success(request, "Term updated successfully!")
                        cache.delete("lexicon_dashboard_data_v2")
                    else:
                        messages.warning(request, "Term must be at least 2 characters long.")
                except LexiconTerm.DoesNotExist:
                    messages.error(request, "Term not found.")

        # Handle Delete Term
        elif action == 'delete_term':
            term_id = request.POST.get('term_id')
            if term_id:
                try:
                    LexiconTerm.objects.filter(id=term_id).delete()
                    messages.success(request, "Term deleted successfully.")
                    cache.delete("lexicon_dashboard_data_v2")
                except Exception as e:
                    messages.error(request, f"Error: {e}")

        # Handle Add Term
        elif action == 'add_term':
            term = request.POST.get('term', '').strip()
            if term and len(term) > 1:
                LexiconTerm.objects.get_or_create(
                    term=term,
                    defaults={
                        'category': request.POST.get('category', 'uncategorized'),
                        'severity': request.POST.get('severity', 'medium'),
                        'target_entity': request.POST.get('target_entity', ''),
                        'language': request.POST.get('language', 'english'),
                        'justification': request.POST.get('justification', ''),  
                        'is_election_related': True,
                    }
                )
                messages.success(request, "Term added successfully!")
                cache.delete("lexicon_dashboard_data_v2")
            else:
                messages.warning(request, "Term must be at least 2 characters long. Single characters are skipped.")


        

        # Handle Scan Text
        elif action == 'scan_text':
            text = request.POST.get('scan_text', '').strip()
            if text:
                # 1. Lexicon-based detection
                lexicon_matches = scan_text_for_lexicon_terms(text)
                lexicon_risk = calculate_risk_score(lexicon_matches)

                # 2. LLM-based detection
                llm_result = detect_hate_speech_llm(text)


                # 3. Extract NEW trigger terms not in lexicon
                new_terms = extract_new_trigger_terms_llm(text, lexicon_matches)

                # 4. Fine-tuned AFRO-XLMR Model detection (Swapped out Gemma here)

                
                # 3. Extract NEW trigger terms not in lexicon
                new_terms = extract_new_trigger_terms_llm(text, lexicon_matches)
                
                # 4. Fine-tuned AFRO-XLMR Model detection

                try:
                    afro_result = detect_hate_speech_afro_xlmr(text)
                except Exception as e:
                    logger.error(f"AFRO-XLMR detection failed: {e}")
                    afro_result = {'category': 'error', 'confidence': 0.0, 'severity': 'low', 'is_hate_speech': False, 'error': 'Model failed to compute'}
                

                # 5. Determine final verdict - AFRO-XLMR / LLM Priority Check
                is_hate_speech = False
                overall_severity_num = 1
                explanation = ""

                llm_is_hate = llm_result.get('is_hate_speech', False)
                llm_confidence = llm_result.get('confidence', 0)
                llm_explanation = llm_result.get('explanation', '').lower()
                
                afro_is_hate = afro_result.get('is_hate_speech', False)
                afro_confidence = afro_result.get('confidence', 0)

                # 5. Determine final verdict
                is_hate_speech = False
                overall_severity_num = 1
                explanation = ""
                
                llm_is_hate = llm_result.get('is_hate_speech', False)
                llm_confidence = llm_result.get('confidence', 0)
                llm_explanation = llm_result.get('explanation', '').lower()
                
                afro_is_hate = afro_result.get('is_hate_speech', False)
                afro_confidence = afro_result.get('confidence', 0)
                
                # SAFETY NET
                hate_indicators = [
                    'dehumaniz', 'incite', 'violence', 'hatred', 'hate speech', 
                    'derogatory', 'discrimination', 'dangerous', 'threat',
                    'ethnic cleansing', 'genocide', 'kill', 'attack', 'slaughter'
                ]
                
                if not llm_is_hate and any(indicator in llm_explanation for indicator in hate_indicators):
                    llm_is_hate = True
                    llm_confidence = max(llm_confidence, 0.75)
                
                # PRIORITY 1: AFRO-XLMR
                if afro_is_hate and afro_confidence >= 0.6:
                    is_hate_speech = True
                    overall_severity_num = {'low':1, 'medium':2, 'high':3, 'critical':4}.get(afro_result.get('severity', 'medium'), 2)
                    explanation = f"AFRO-XLMR Model detected hate speech ({afro_confidence*100:.0f}% confidence)."
                
                # PRIORITY 2: LLM
                elif llm_is_hate and llm_confidence >= 0.6:
                    is_hate_speech = True
                    overall_severity_num = {'low':1, 'medium':2, 'high':3, 'critical':4}.get(llm_result.get('severity', 'medium'), 2)
                    explanation = f"LLM detected hate speech ({llm_confidence*100:.0f}% confidence). {llm_explanation[:150]}"
                
                # PRIORITY 3: Lexicon
                elif lexicon_risk.get('score', 0) > 3 or any(m.get('severity') in ['high', 'critical'] for m in lexicon_matches):
                    is_hate_speech = True
                    overall_severity_num = {'low':1, 'medium':2, 'high':3, 'critical':4}.get(lexicon_risk.get('level', 'medium'), 2)
                    explanation = f"Lexicon detected {len(lexicon_matches)} high-risk term(s) (score: {lexicon_risk.get('score', 0)})"
                
                # PRIORITY 4: Default
                else:
                    is_hate_speech = False
                    overall_severity_num = 1
                    if lexicon_matches:
                        explanation = f"AI models classify as neutral. Lexicon found {len(lexicon_matches)} term(s), but context appears legitimate."
                    else:
                        explanation = "No hate speech detected by any method."
                
                severity_map = {1:'low', 2:'medium', 3:'high', 4:'critical'}
                
                # Create combined analysis
                analysis_parts = []
                if llm_result.get('explanation'):
                    analysis_parts.append(f"LLM Analysis: {llm_result['explanation']}")
                if lexicon_matches:
                    terms_found = [f"'{m['term']}'" for m in lexicon_matches[:5]]
                    analysis_parts.append(f"Lexicon matched {len(lexicon_matches)} term(s): {', '.join(terms_found)}")
                combined_analysis = ". ".join(analysis_parts) if analysis_parts else "No specific patterns detected"

                
                # SAFETY NET: If LLM explanation indicates hate speech, force flag
                hate_indicators = [
                    'dehumaniz', 'incite', 'violence', 'hatred', 'hate speech', 
                    'derogatory', 'discrimination', 'dangerous', 'threat',
                    'ethnic cleansing', 'genocide', 'kill', 'attack', 'slaughter'
                ]
                
                if not llm_is_hate and any(indicator in llm_explanation for indicator in hate_indicators):
                    llm_is_hate = True
                    llm_confidence = max(llm_confidence, 0.75)
                
                # PRIORITY 1: AFRO-XLMR detects hate speech confidently
                if afro_is_hate and afro_confidence >= 0.6:
                    is_hate_speech = True
                    overall_severity_num = {'low':1, 'medium':2, 'high':3, 'critical':4}.get(afro_result.get('severity', 'medium'), 2)
                    explanation = f"AFRO-XLMR Model detected hate speech ({afro_confidence*100:.0f}% confidence)."
                
                # PRIORITY 2: LLM says hate speech
                elif llm_is_hate and llm_confidence >= 0.6:
                    is_hate_speech = True
                    overall_severity_num = {'low':1, 'medium':2, 'high':3, 'critical':4}.get(llm_result.get('severity', 'medium'), 2)
                    explanation = f"LLM detected hate speech ({llm_confidence*100:.0f}% confidence). {llm_explanation[:150]}"
                
                # PRIORITY 3: Lexicon finds high-risk terms
                elif lexicon_risk.get('score', 0) > 3 or any(m.get('severity') in ['high', 'critical'] for m in lexicon_matches):
                    is_hate_speech = True
                    overall_severity_num = {'low':1, 'medium':2, 'high':3, 'critical':4}.get(lexicon_risk.get('level', 'medium'), 2)
                    explanation = f"Lexicon detected {len(lexicon_matches)} high-risk term(s) (score: {lexicon_risk.get('score', 0)})"
                
                # PRIORITY 4: Default to neutral
                else:
                    is_hate_speech = False
                    overall_severity_num = 1
                    if lexicon_matches:
                        explanation = f"AI models classify as neutral. Lexicon found {len(lexicon_matches)} term(s), but context appears legitimate."
                    else:
                        explanation = "No hate speech detected by any method."
                
                severity_map = {1:'low', 2:'medium', 3:'high', 4:'critical'}
                
                # Create combined analysis field
                analysis_parts = []
                if llm_result.get('explanation'):
                    analysis_parts.append(f"LLM Analysis: {llm_result['explanation']}")
                if lexicon_matches:
                    terms_found = [f"'{m['term']}'" for m in lexicon_matches[:5]]
                    analysis_parts.append(f"Lexicon matched {len(lexicon_matches)} term(s): {', '.join(terms_found)}")
                combined_analysis = ". ".join(analysis_parts) if analysis_parts else "No specific patterns detected"
                
                # Save data safely back to UI template context via session
                request.session['scan_results'] = {
                    'text': text[:200] + '...' if len(text) > 200 else text,
                    'lexicon_matches': lexicon_matches,
                    'lexicon_risk': lexicon_risk,
                    'llm_result': llm_result,
                    'afro_xlmr_result': afro_result,  # Replaces gemma_result key smoothly
                    'is_hate_speech': is_hate_speech,
                    'overall_severity': severity_map[overall_severity_num],
                    'overall_confidence': round(max(llm_confidence, afro_confidence), 2),
                    'overall_confidence_pct': f"{round(max(llm_confidence, afro_confidence) * 100)}%",
                    'all_categories': list(set([m['category'] for m in lexicon_matches] + llm_result.get('categories', []))),
                    'targeted_groups': llm_result.get('targeted_groups', []),
                    'explanation': explanation,
                    'analysis': combined_analysis,
                    'has_lexicon_matches': len(lexicon_matches) > 0,
                    'new_trigger_terms': new_terms,
                    'has_new_terms': len(new_terms) > 0,
                }
                
                if is_hate_speech:
                    messages.warning(request, f"Potential hate speech detected! Severity: {severity_map[overall_severity_num].upper()}")
                    if new_terms:
                        messages.info(request, f"{len(new_terms)} new trigger terms extracted for review.")
                else:
                    if lexicon_matches:
                        messages.info(request, f"No hate speech detected. (Note: {len(lexicon_matches)} sensitive term(s) found, but context is neutral).")
                    else:
                        messages.success(request, "No hate speech detected.")
            else:
                messages.warning(request, "Please enter text to scan")

        return redirect('lexicon_management')
           
class UploadDataView(TemplateView):
    """UI for uploading CSV files - handles both GET and POST"""
    template_name = 'dashboard/upload_data.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['active_tab'] = 'upload'
        context['recent_uploads'] = DataUpload.objects.order_by('-uploaded_at')[:10]
        return context
    
    def post(self, request, *args, **kwargs):
        """Handle file upload via POST"""
        import os
        import uuid
        from django.utils import timezone
        from django.core.cache import cache
        
        logger.info(f"📥 Upload request: data_type={request.POST.get('data_type')}, source={request.POST.get('source_name')}")
        logger.info(f"📁 FILES: {list(request.FILES.keys())}")
        
        uploaded_files = request.FILES.getlist('csv_files')
        if not uploaded_files:
            messages.error(request, "No files received.")
            return redirect('upload_data')
        
        results = []
        for uploaded_file in uploaded_files:
            try:
                # Generate unique filename to avoid conflicts
                unique_id = uuid.uuid4().hex[:8]
                timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
                original_name = uploaded_file.name
                name_without_ext = os.path.splitext(original_name)[0]
                ext = os.path.splitext(original_name)[1]
                
                # Create unique filename: originalname_timestamp_uniqueid.ext
                unique_filename = f"{name_without_ext}_{timestamp}_{unique_id}{ext}"
                
                # Save file with unique name
                file_path = default_storage.save(f'uploads/{unique_filename}', uploaded_file)
                full_path = os.path.join(settings.MEDIA_ROOT, file_path)
                
                logger.info(f"🔄 Processing: {original_name} -> {unique_filename}")
                
                # Create upload record
                upload = DataUpload.objects.create(
                    uploaded_file=file_path,
                    original_filename=original_name,
                    uploaded_by=request.user.username if request.user.is_authenticated else 'anonymous',
                    data_type=request.POST.get('data_type', 'custom'),
                    status='processing'
                )
                
                # Process the file
                success, message, count = process_uploaded_csv(
                    full_path, 
                    upload.data_type, 
                    request.POST.get('source_name', 'User Upload')
                )
                
                # Update record
                upload.status = 'completed' if success else 'failed'
                upload.processing_log = message
                upload.records_processed = count if success else 0
                upload.save()
                
                results.append((original_name, success, message, count))
                logger.info(f"{'✅' if success else '❌'} {original_name}: {message}")
                
            except Exception as e:
                logger.error(f"❌ Upload failed for {uploaded_file.name}: {str(e)}", exc_info=True)
                results.append((uploaded_file.name, False, str(e), 0))
        
        
        
        # Show summary in UI
        success_count = sum(1 for _, s, _, _ in results if s)
        if success_count == len(uploaded_files):
            messages.success(request, f"✅ All {len(uploaded_files)} files processed successfully!")
        elif success_count > 0:
            messages.warning(request, f"⚠️ {success_count}/{len(uploaded_files)} succeeded. Check logs for errors.")
        else:
            messages.error(request, "❌ Failed to process any files. Check terminal logs for details.")
        
        return redirect('upload_data')


class ProcessUploadView(View):
    def post(self, request):
        import os
        import uuid
        from django.utils import timezone
        
        #  Match the HTML form's input names ('files' and 'platform')
        uploaded_files = request.FILES.getlist('files')
        data_type = request.POST.get('platform', 'generic')
        source_name = request.POST.get('source_name', 'User Upload')
        
        logger.info(f"📥 Upload request: data_type={data_type}, source={source_name}")
        logger.info(f"📁 FILES: {list(request.FILES.keys())}")
        
        if not uploaded_files:
            messages.error(request, "No files received. Please check the file input name.")
            return redirect('upload_data')
        
        results = []
        for uploaded_file in uploaded_files:
            try:
                # Generate unique filename
                unique_id = uuid.uuid4().hex[:8]
                timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
                original_name = uploaded_file.name
                name_without_ext = os.path.splitext(original_name)[0]
                ext = os.path.splitext(original_name)[1]
                unique_filename = f"{name_without_ext}_{timestamp}_{unique_id}{ext}"
                
                # Save file
                file_path = default_storage.save(f'uploads/{unique_filename}', uploaded_file)
                full_path = os.path.join(settings.MEDIA_ROOT, file_path)
                logger.info(f"🔄 Processing: {original_name} -> {unique_filename} ({uploaded_file.size / 1024 / 1024:.2f} MB)")
                
                # Create upload record
                upload = DataUpload.objects.create(
                    uploaded_file=file_path,
                    original_filename=original_name,
                    uploaded_by=request.user.username if request.user.is_authenticated else 'anonymous',
                    data_type=data_type,
                    status='processing'
                )
                
                # Use robust inline processing instead of the black-box process_uploaded_csv
                if data_type == 'brandwatch':
                    df = pd.read_csv(full_path, sep=',', low_memory=False, on_bad_lines='skip', encoding_errors='ignore', skiprows=6)
                else:
                    df = load_data_robustly(full_path)
                
                logger.info(f"📊 Initial CSV Shape: {df.shape} | Columns: {list(df.columns)}")
                
                if df.empty:
                    raise ValueError("DataFrame is empty after loading")
                
                # Combine/Map data based on source
                if data_type == 'meltwater':
                    combined_df = combine_social_media_data(meltwater_df=df)
                elif data_type == 'civicsignals':
                    combined_df = combine_social_media_data(civicsignals_df=df)
                elif data_type == 'tiktok':
                    combined_df = combine_social_media_data(tiktok_df=df)
                elif data_type == 'openmeasure':
                    combined_df = combine_social_media_data(openmeasures_df=df)
                elif data_type == 'brandwatch':
                    combined_df = combine_social_media_data(brandwatch_df=df)
                else:
                    combined_df = preprocess_dataframe(df)
                
                # Final preprocessing
                processed_df = final_preprocess_and_map_columns(combined_df)
                logger.info(f"📊 Processed Data Shape: {processed_df.shape} | Columns: {list(processed_df.columns)}")
                
                # Parse timestamps
                if 'timestamp_share' in processed_df.columns:
                    processed_df['timestamp_share'] = processed_df['timestamp_share'].apply(
                        lambda x: parse_timestamp_robust(x) if pd.notna(x) else pd.NaT
                    )
                
                # Save to Database with detailed skip tracking
                count = 0
                urls_saved = 0
                skipped_empty = 0
                skipped_dup = 0
                
                for idx, row in processed_df.iterrows():
                    try:
                        text_val = str(row.get('original_text', '')).strip()
                        if not text_val or text_val.lower() in ['nan', 'none', '']:
                            skipped_empty += 1
                            continue
                        
                        cid = str(row.get('content_id', '')).strip()
                        url_val = str(row.get('url') or row.get('URL', '')).strip()
                        
                        # Check duplicates
                        if cid and cid.lower() != 'nan' and ProcessedPost.objects.filter(content_id=cid).exists():
                            skipped_dup += 1
                            continue
                        if url_val.startswith('http') and ProcessedPost.objects.filter(url=url_val).exists():
                            skipped_dup += 1
                            continue
                        
                        source_obj, _ = DataSource.objects.get_or_create(name=source_name)
                        url_value = url_val[:500] if url_val.startswith('http') else None
                        if url_value:
                            urls_saved += 1
                        # Convert Pandas NaT to None for Django
                        ts_val = row.get('timestamp_share')
                        if pd.isna(ts_val) or str(ts_val) == 'NaT':
                            ts_val = None
                            
                        ProcessedPost.objects.create(
                            account_id=str(row.get('account_id', ''))[:100],
                            content_id=str(cid).strip()[:100] if cid else None,
                            original_text=str(row.get('original_text', '')).strip(),
                            url=url_value,
                            platform=str(row.get('Platform', 'Unknown')),
                            timestamp_share=ts_val,  # <--- NOW SAFE FOR DJANGO
                            source_dataset=source_obj,
                            is_election_related=is_election_related(str(row.get('original_text', '')))
                        )
                        count += 1
                    except Exception as row_error:
                        logger.error(f"❌ Row {idx} error: {row_error}")
                        continue
                
                logger.info(f"✅ Saved: {count} | Skipped Empty: {skipped_empty} | Skipped Duplicates: {skipped_dup}")
                
                # Update record
                upload.status = 'completed'
                upload.processing_log = f"Saved: {count}, Empty: {skipped_empty}, Duplicates: {skipped_dup}"
                upload.records_processed = count
                upload.save()
                
                results.append((original_name, True, f"Processed {count} posts", count))
                
            except Exception as e:
                logger.error(f"❌ Upload failed for {uploaded_file.name}: {str(e)}", exc_info=True)
                if 'upload' in locals():
                    upload.status = 'failed'
                    upload.processing_log = str(e)
                    upload.save()
                results.append((uploaded_file.name, False, str(e), 0))
        
        # Show summary in UI
        success_count = sum(1 for _, s, _, c in results if s and c > 0)
        total_saved = sum(c for _, s, _, c in results if s)
        
        if total_saved > 0:
            messages.success(request, f"✅ Successfully saved {total_saved} new posts!")
        elif not any(s for _, s, _, _ in results):
            messages.error(request, "❌ Failed to process files. Check terminal logs for details.")
        else:
            messages.warning(request, f"⚠️ File processed, but 0 new posts were saved. They may be duplicates or have empty text. Check logs.")
        
        return redirect('upload_data')
        
class ClearDataView(View):
    """Clear all uploaded data from database"""
    def post(self, request):
        # Only clear ProcessedPost, keep other data if needed
        ProcessedPost.objects.all().delete()
        # Optional: Also clear upload history
        # DataUpload.objects.all().delete()
        
        messages.success(request, "✅ All post data cleared successfully. You can now upload fresh data.")
        return redirect('upload_data')

class InvestigativeReportsView(TemplateView):
    """Dedicated view for investigative analysis reports"""
    template_name = 'dashboard/investigative_reports.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get all monitoring reports
        from dashboard.models import MonitoringReport
        reports = MonitoringReport.objects.all().order_by('-uploaded_at')
        
        # Calculate stats
        total_reports = reports.count()
        critical_reports = reports.filter(risk_level='critical').count()
        high_reports = reports.filter(risk_level='high').count()
        this_month = reports.filter(uploaded_at__gte=timezone.now().replace(day=1)).count()
        
        context.update({
            'active_tab': 'investigative_reports',
            'reports': reports,
            'stats': {
                'total': total_reports,
                'critical': critical_reports,
                'high': high_reports,
                'this_month': this_month,
            }
        })
        return context

class PEPsAnalysisView(TemplateView):
    template_name = 'dashboard/peps_analysis.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get election-related posts
        posts = ProcessedPost.objects.filter(is_election_related=True).order_by('-timestamp_share')
        active_peps = PEP.objects.filter(is_active=True).order_by('name')
        
        # Generate enhanced analysis
        context['pep_analysis'] = get_enhanced_pep_analysis(posts, active_peps, limit=8)
        context['active_tab'] = 'peps'
        
        return context
        
# === API Endpoints ===

def scan_text_api(request):
    """API endpoint for real-time hate speech scanning"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    text = request.POST.get('text', '')
    if not text:
        return JsonResponse({'error': 'No text provided'}, status=400)
    
    matches = scan_text_for_lexicon_terms(text)
    risk = calculate_risk_score(matches) if matches else {'score': 0, 'level': 'low'}
    
    return JsonResponse({
        'matches': matches,
        'risk': risk,
        'term_count': len(matches)
    })


def export_posts_api(request):
    """API endpoint to export filtered posts as CSV"""
    queryset, start_date, end_date = get_election_posts_queryset(request)
    
    # Convert to DataFrame
    posts = list(queryset.values())
    df = pd.DataFrame(posts)
    
    # Convert to CSV
    import io
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    response = HttpResponse(csv_buffer.getvalue(), content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="ethiopia_election_posts_{start_date.date()}_{end_date.date()}.csv"'
    
    return response


def generate_network_graph(request):
    """API endpoint to generate coordination network graph smoothly for 14k+ records"""
    import re
    from collections import defaultdict

    # Get configuration parameters
    min_connections = int(request.GET.get('min_connections', 2))
    top_n = int(request.GET.get('top_n', 50))
    
    # Check parameters for specific selected cluster rows
    cluster_filter = request.GET.get('cluster') or request.GET.get('group_id') or request.GET.get('coordination_group')
    
    # Base query for election dataset
    query_kwargs = {'is_election_related': True}
    
    if cluster_filter and cluster_filter.strip():
        min_connections = 1  # Relax constraint for deep-dive group views
        if hasattr(ProcessedPost, 'coordination_group_id'):
            query_kwargs['coordination_group_id'] = cluster_filter
        elif hasattr(ProcessedPost, 'coordination_group'):
            query_kwargs['coordination_group'] = cluster_filter
        else:
            try:
                query_kwargs['cluster'] = int(cluster_filter)
            except ValueError:
                pass
    else:
        if hasattr(ProcessedPost, 'cluster'):
            query_kwargs['cluster__gte'] = 0

    # Pull only necessary fields into memory in one quick query
    posts_data = ProcessedPost.objects.filter(**query_kwargs).values('account_id', 'original_text')
    
    G = nx.Graph()
    rt_pattern = re.compile(r'RT\s+@([a-zA-Z0-9_]+)', re.IGNORECASE)
    
    # Map identical text fields to accounts in a single pass
    text_to_accounts = defaultdict(set)
    
    for post in posts_data:
        acc = post['account_id']
        text = post['original_text'] or ''
        
        if not acc or acc == 'unknown':
            continue
            
        # 1. Immediate Retweet Edge connection logic
        rt_match = rt_pattern.search(text)
        if rt_match:
            target_user = rt_match.group(1)
            if acc != target_user:
                if G.has_edge(acc, target_user):
                    G[acc][target_user]['weight'] += 1
                else:
                    G.add_edge(acc, target_user, weight=1)
                    
        # 2. Collect for identical text grouping analysis
        text_to_accounts[text].add(acc)

    # 3. Process identical text groups to track multi-account co-sharing patterns
    for acc_set in text_to_accounts.values():
        if len(acc_set) >= 2:
            acc_list = list(acc_set)
            for i in range(len(acc_list)):
                for j in range(i + 1, len(acc_list)):
                    if G.has_edge(acc_list[i], acc_list[j]):
                        G[acc_list[i]][acc_list[j]]['weight'] += 1
                    else:
                        G.add_edge(acc_list[i], acc_list[j], weight=1)

    # Filter out nodes based on minimum degree connection limits
    nodes_to_keep = [n for n, d in G.degree() if d >= min_connections]
    G = G.subgraph(nodes_to_keep).copy()
    
    if G.number_of_edges() == 0:
        return JsonResponse({'nodes': [], 'edges': [], 'message': 'No coordination links found for this selection'})
    
    # Isolate top N nodes
    top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:top_n]
    top_node_names = [n for n, _ in top_nodes]
    G_top = G.subgraph(top_node_names).copy()
    
    # Format structural node list response data payload
    node_data = [{'id': node, 'degree': G_top.degree(node)} for node in G_top.nodes()]
    
    # Format structural edge list response data payload
    edge_data = [
        {
            'source': u,
            'target': v,
            'weight': data.get('weight', 1)
        }
        for u, v, data in G_top.edges(data=True)
    ]
    
    return JsonResponse({
        'nodes': node_data,
        'edges': edge_data,
        'stats': {
            'total_nodes': G_top.number_of_nodes(),
            'total_edges': G_top.number_of_edges(),
            'avg_degree': sum(d for _, d in G_top.degree()) / G_top.number_of_nodes() if G_top.number_of_nodes() > 0 else 0
        }
    })

def ttp_radar_data_api(request):
    """Return election-related posts as JSON for the TTP Radar UI"""
    from django.http import JsonResponse
    from dashboard.models import ProcessedPost
    from django.utils import timezone
    
    try:
        country = request.GET.get('country', 'Ethiopia')
        limit = min(int(request.GET.get('limit', 100)), 500)
        
        posts = ProcessedPost.objects.filter(
            is_election_related=True
        ).order_by('-timestamp_share')[:limit]
        
        formatted = []
        for p in posts:
            formatted.append({
                'account_id': p.account_id or 'unknown',
                'content_id': f"post_{p.id}",
                'original_text': p.original_text or '',
                'URL': p.url or '',
                'timestamp_share': p.timestamp_share.isoformat() if p.timestamp_share else None,
                'Platform': p.platform or 'Unknown',
                'target_country': getattr(p, 'target_country', 'Ethiopia') or 'Ethiopia',
            })
        
        return JsonResponse({
            'records': formatted,
            'count': len(formatted),
            'country': country,
            'generated_at': timezone.now().isoformat()
        })
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"ttp_radar_data_api error: {e}", exc_info=True)
        return JsonResponse({
            'error': str(e),
            'records': [],
            'count': 0
        }, status=500)

@login_required
def trigger_llm_scan_api(request):
    """API endpoint to trigger background LLM scan"""
    from .utils.background_tasks import start_background_llm_scan
    from .models import ProcessedPost
    
    # Get recent posts (last 500)
    recent_posts = ProcessedPost.objects.filter(
        is_election_related=True
    ).order_by('-timestamp_share')[:500]
    
    post_ids = list(recent_posts.values_list('id', flat=True))
    
    started = start_background_llm_scan(post_ids, user_id=request.user.id)
    
    if started:
        return JsonResponse({
            'success': True,
            'message': f'Scan started for {len(post_ids)} posts',
            'posts_count': len(post_ids)
        })
    else:
        return JsonResponse({
            'success': False,
            'message': 'Scan already in progress. Please wait.'
        }, status=429)
       
def scan_text_afro_xlmr_api(request):
    """API endpoint for AFRO-XLMR hate speech scanning"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    text = request.POST.get('text', '')
    if not text:
        return JsonResponse({'error': 'No text provided'}, status=400)
    
    # Use AFRO-XLMR for detection
    afro_result = detect_hate_speech_afro_xlmr(text)
    
    # Also run lexicon-based detection for comparison
    lexicon_matches = scan_text_for_lexicon_terms(text)
    lexicon_risk = calculate_risk_score(lexicon_matches)
    
    # Combine results
    return JsonResponse({
        'afro_xlmr': afro_result,
        'lexicon': {
            'matches': lexicon_matches,
            'risk': lexicon_risk
        },
        'model_used': 'AFRO-XLMR',
        'text_length': len(text)
    })
