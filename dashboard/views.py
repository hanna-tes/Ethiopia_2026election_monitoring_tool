"""
Django views for Ethiopia Election Monitor
"""
import json
import logging
import os
import re
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
   




logger = logging.getLogger(__name__)

# Global model cache
_GEMMA_MODEL = None
_GEMMA_TOKENIZER = None

def load_gemma_model():
    """Load the fine-tuned Gemma model from cache using MLX"""
    global _GEMMA_MODEL, _GEMMA_TOKENIZER
    if _GEMMA_MODEL is not None and _GEMMA_TOKENIZER is not None:
        return _GEMMA_MODEL, _GEMMA_TOKENIZER
    try:
        # Point fused MLX model
        model_path = getattr(settings, 'GEMMA_TTP_MODEL_PATH', './model_cache/gemma-merged')
        logger.info(f"Loading Gemma TTP model from {model_path} using MLX...")
        
        # Load model and tokenizer using MLX
        from mlx_lm import load
        _GEMMA_MODEL, _GEMMA_TOKENIZER = load(model_path)
        
        logger.info("Gemma TTP model loaded successfully via MLX")
        return _GEMMA_MODEL, _GEMMA_TOKENIZER
    except Exception as e:
        logger.error(f"Failed to load Gemma model: {e}")
        raise
def export_merged_gephi_csv(request):
    """Export a single, merged Gephi-ready CSV containing edges, tweets, and roles."""
    min_connections = int(request.GET.get('min_connections', 2))
    posts = ProcessedPost.objects.filter(is_election_related=True)
    
    # Get coordination groups
    coordination_groups = get_coordination_groups(posts, min_accounts=min_connections, max_groups=15)
    
    # Debug logging
    logger.info(f"Exporting {len(coordination_groups)} coordination groups to CSV")
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="ethiopia_election_network_merged.csv"'
    writer = csv.writer(response)
    
    # Headers formatted for Gephi's Edge Table importer
    writer.writerow([
        'Source', 'Target', 'Weight', 'Type',
        'Tweet', 'Timestamp', 'Platform', 'Sub_Narrative',
        'Source_Role', 'Target_Role'
    ])
    
    rows_written = 0
    
    for group in coordination_groups:
        text = group.get('text_sample', '')
        if not text:
            logger.warning(f"Group {group.get('id')} has no text_sample")
            continue
        
        # Extract sub-narrative directly from the text
        sub_narrative = extract_sub_narrative(text)
        platforms = ', '.join(group.get('platforms', [])) if group.get('platforms') else 'Unknown'
        
        # Get posts ordered by timestamp to identify source vs amplifier
        account_posts = list(posts.filter(original_text=text).order_by('timestamp_share'))
        
        if len(account_posts) < 2:
            logger.warning(f"Text has only {len(account_posts)} post(s), need at least 2 for edge")
            continue
        
        source_account = None
        first_time = None
        
        for idx, post in enumerate(account_posts):
            username = clean_username(post.account_id)
            if not username or len(username) < 2:
                continue
                
            post_time = post.timestamp_share.strftime('%Y-%m-%d %H:%M') if post.timestamp_share else ''
            
            if idx == 0:
                # First poster is the Source
                source_account = username
                first_time = post_time
            else:
                # Subsequent posters are Targets/Amplifiers
                # Clean tweet text for CSV (escape quotes and newlines)
                tweet_clean = text[:200].replace('\n', ' ').replace('\r', '').replace('"', '""')
                
                writer.writerow([
                    source_account,
                    username,
                    1,  # Weight
                    'Directed',
                    f'"{tweet_clean}"',
                    first_time,
                    platforms,
                    sub_narrative,
                    'Source',
                    'Amplifier'
                ])
                rows_written += 1
    
    logger.info(f"Exported {rows_written} rows to CSV")
    
    if rows_written == 0:
        # Add a sample row for testing if no data found
        logger.warning("No coordination edges found. Adding sample row for testing.")
        writer.writerow([
            'sample_source',
            'sample_target',
            1,
            'Directed',
            '"Sample coordination detected"',
            timezone.now().strftime('%Y-%m-%d %H:%M'),
            'Test',
            'General Coordination',
            'Source',
            'Amplifier'
        ])
    
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
                'date': first_date.strftime('%Y-%m-%d %H:%M') if first_date else '',
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
    """Export Edges CSV for Gephi with tweet content"""
    min_connections = int(request.GET.get('min_connections', 2))
    posts = ProcessedPost.objects.filter(is_election_related=True)
    coordination_groups = get_coordination_groups(posts, min_accounts=min_connections, max_groups=15)
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="gephi_edges.csv"'
    writer = csv.writer(response)
    
    # Write header
    writer.writerow(['Source', 'Target', 'Weight', 'Type', 'Tweet', 'Sub Narrative', 'Timestamp'])
    
    for group in coordination_groups:
        accounts = group.get('accounts', [])
        text_sample = group.get('text_sample', '')
        post_count = group.get('post_count', 0)
        sub_narrative = group.get('sub_narrative', 'General Coordination')
        
        # Get timestamp from sample posts
        timestamp = ''
        if group.get('sample_posts_with_urls'):
            first_post = group['sample_posts_with_urls'][0]
            timestamp = first_post.get('timestamp', '')
        
        # Create edges between all pairs
        for i in range(len(accounts)):
            for j in range(i+1, len(accounts)):
                # Clean tweet text for CSV
                tweet_clean = text_sample[:200].replace('\n', ' ').replace('\r', '').replace('"', '""')
                
                writer.writerow([
                    accounts[i],
                    accounts[j],
                    post_count,
                    'Undirected',
                    f'"{tweet_clean}"',  # Quote the tweet
                    sub_narrative,
                    timestamp
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
                first_time = post.timestamp_share.strftime('%Y-%m-%d %H:%M') if post.timestamp_share else ''
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
    writer.writerow(['Source', 'Target', 'Tweet', 'Date', 'Platform', 'URL'])
    
    for edge in edges:
        # Truncate long tweets to prevent CSV issues
        tweet_text = edge['tweet'][:200].replace('\n', ' ').replace('\r', '')
        writer.writerow([
            edge['source'], 
            edge['target'], 
            tweet_text,
            edge['date'],
            edge['platform'],
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

#  HELPER FUNCTIONS

def clean_username(raw_name):
    """Clean username while preserving full account names"""
    if not raw_name or pd.isna(raw_name):
        return "Unknown"
    
    # Convert to string and strip whitespace
    name = str(raw_name).strip()
    
    # Remove common artifacts from the END only
    name = re.sub(r'\s+(?i)(name|source|nan|none)$', '', name).strip()
    
    # Remove leading/trailing special characters
    name = re.sub(r'^[@\s]+|[\s@]+$', '', name)
    
    # If name is empty after cleaning, return Unknown
    if not name or name.lower() in ['nan', 'none', '-', '', 'unknown']:
        return "Unknown"
    
    return name
    
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
    
@never_cache  
def dashboard_view(request):
    """Main Dashboard View with Sidebar Upload and Stats Reporting"""
    
    # 1. Handle File Uploads via POST
    if request.method == 'POST' and request.FILES.getlist('files'):
        platform_type = request.POST.get('platform')
        uploaded_files = request.FILES.getlist('files')
        
        stats = {
            'files_count': 0,
            'total_rows': 0,
            'saved': 0,
            'duplicates': 0
        }
        
        for f in uploaded_files:
            try:
                # Brandwatch needs specific handling (skiprows=6 for metadata)
                if platform_type == 'brandwatch':
                    df = pd.read_csv(f, sep=',', low_memory=False, skiprows=6, on_bad_lines='skip')
                else:
                    
                    df = load_data_robustly(f) 
                
                stats['total_rows'] += len(df)
                
                # Normalize columns based on platform
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

                # ONE CLEAN LOOP
                for _, row in processed_df.iterrows():
                    cid = row.get('content_id')
                    
                    # Check for duplicates before saving
                    if cid and ProcessedPost.objects.filter(content_id=cid).exists():
                        stats['duplicates'] += 1
                        continue
                        
                    source_name = row.get('source_dataset', platform_type)
                    source_obj, _ = DataSource.objects.get_or_create(name=source_name)
            
                    ProcessedPost.objects.create(
                        account_id=str(row.get('account_id', ''))[:100],
                        content_id=cid,
                        # Use 'original_text' which is the standardized column name
                        original_text=str(row.get('original_text', '')),
                        
                        url=row.get('url') or row.get('URL') or row.get('link') or row.get('Link') or '',
                        platform=row.get('Platform', platform_type.title()),
                        timestamp_share=parse_timestamp_robust(row.get('timestamp_share')),
                        source_dataset=source_obj,
                        is_election_related=is_election_related(str(row.get('original_text', '')))
                    )
                    stats['saved'] += 1
                
                stats['files_count'] += 1
            except Exception as e:
                logger.error(f"Upload error: {e}")
                messages.error(request, f"Error processing {f.name}")

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
    
def scan_text_for_lexicon_terms(text, category_filter=None):
    """Scan text for lexicon matches using CONFIG mapping"""
    if not isinstance(text, str) or not text.strip():
        return []
    
    text_lower = text.lower()
    matches = []
    lexicon = CONFIG.get("lexicon", {})
    categories_to_check = category_filter if category_filter else lexicon.keys()
    
    for category in categories_to_check:
        if category not in lexicon: 
            continue
        for term, metadata in lexicon[category].items():
            # FIX: Skip single-character terms (except for Amharic which can be meaningful)
            if len(term.strip()) < 2 and not re.match(r'^[\u1200-\u137F]+$', term):
                continue
                
            if metadata.get("language") == "amharic" or re.match(r'^[\u1200-\u137F]+$', term):
                # Prevent matching inside larger Amharic words using lookarounds
                pattern = r'(?<![\u1200-\u137F])' + re.escape(term) + r'(?![\u1200-\u137F])'
            else:
                pattern = r'\b' + re.escape(term) + r'\b'
            
            if re.search(pattern, text_lower, re.IGNORECASE):
                matches.append({
                    'term': term, 
                    'category': category,
                    'severity': metadata.get('severity', 'medium'),
                    'target_entity': metadata.get('target_entity', ''),
                    'language': metadata.get('language', 'english')
                })
    return matches


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
    try:
        model, tokenizer = load_gemma_model()
        input_data = format_ttp_input(coordination_groups)
        
        system_prompt = (
            "You are a DISARM TTP adjudicator. Consider T0049, T0049.002, T0049.003, T0049.005, "
            "T0016, T0060, T0119, T0119.001, T0119.002, T0097.102, T0097.202, T0143.002, "
            "T0143.003, T0149.003, and T0084.002. Use only raw observable cues encoded in the dossier. "
            "Output strict JSON only. Prefer false negatives over false positives."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(input_data)}
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        from mlx_lm import generate
        
        # Increase max_tokens to give model room to think AND output JSON
        response_text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=2048  # Increased from 1024
        )
        
        logger.info(f"Raw response length: {len(response_text)}")
        
        # === CRITICAL: Extract JSON from thinking output ===
        # Find the first { and last } to extract JSON
        json_start = response_text.find('{')
        json_end = response_text.rfind('}')
        
        if json_start != -1 and json_end != -1 and json_end > json_start:
            response_text = response_text[json_start:json_end + 1]
            logger.info("✅ Extracted JSON from response")
        else:
            logger.error("❌ No JSON found in response")
            return analyze_ttps(coordination_groups, [])
        
        # Parse JSON
        try:
            result = json.loads(response_text)
            
            ttps = []
            if result.get('qualifies', False):
                techniques = result.get('techniques', [])
                reason = result.get('reason', '')
                for technique in techniques:
                    ttp_data = {
                        'name': technique.get('technique_id', ''),
                        'description': technique.get('description', reason),
                        'severity': _get_ttp_severity(technique.get('technique_id', '')),
                        'evidence': f"Detected via Gemma model. {technique.get('evidence', '')}",
                        'confidence': technique.get('confidence', 0.8),
                        'model_source': 'gemma_finetuned'
                    }
                    ttps.append(ttp_data)
                
                if ttps:
                    logger.info(f"🎯 Gemma detected {len(ttps)} TTPs")
                    return ttps
            
            logger.info("Gemma found no TTPs")
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.debug(f"Response: {response_text[:300]}")
        
    except Exception as e:
        logger.error(f"Gemma detection failed: {e}", exc_info=True)
    
    logger.info("Using fallback TTP analysis")
    return analyze_ttps(coordination_groups, [])
    
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
   
def analyze_ttps(coordination_groups, posts):
    """Analyze Tactics, Techniques, and Procedures - 9 Rule-Based + LLM DISARM Detection"""
    ttps = []
    if not coordination_groups:
        return ttps

    # === 9 RULE-BASED TTPs ===
    
    # TTP 1: Coordinated Inauthentic Behavior (CIB)
    cib_groups = [g for g in coordination_groups if g.get('account_count', 0) >= 5]
    if cib_groups:
        ttps.append({
            'name': 'Coordinated Inauthentic Behavior (CIB)',
            'description': f'Detected {len(cib_groups)} groups with 5+ accounts sharing identical content.',
            'severity': 'High',
            'evidence': f'{sum(g.get("post_count", 0) for g in cib_groups)} total posts across {sum(g.get("account_count", 0) for g in cib_groups)} accounts.',
            'source': 'Rule-Based'
        })

    # TTP 2: Cross-Platform Amplification
    cross_platform_groups = []
    for g in coordination_groups:
        platforms = set()
        for p in g.get('sample_posts_with_urls', []):
            if p.get('platform'):
                platforms.add(p['platform'])
        if len(platforms) > 1:
            cross_platform_groups.append({'group': g, 'platforms': list(platforms)})
    
    if cross_platform_groups:
        all_platforms = set()
        for p in cross_platform_groups:
            all_platforms.update(p['platforms'])
        ttps.append({
            'name': 'Cross-Platform Amplification',
            'description': f'{len(cross_platform_groups)} groups operating across {len(all_platforms)} platforms.',
            'severity': 'Medium',
            'evidence': f"Platforms: {', '.join(sorted(all_platforms))}",
            'source': 'Rule-Based'
        })

    # TTP 3: Rapid Response / Burst Posting
    burst_groups = [g for g in coordination_groups if g.get('post_count', 0) > 10]
    if burst_groups:
        max_posts = max(g.get('post_count', 0) for g in burst_groups)
        ttps.append({
            'name': 'Rapid Response / Burst Posting',
            'description': f'{len(burst_groups)} groups with high-volume posting (max: {max_posts} posts/group).',
            'severity': 'Medium',
            'evidence': f"Identical content bursts across {sum(g.get('account_count', 0) for g in burst_groups)} accounts.",
            'source': 'Rule-Based'
        })

    # TTP 4: Hashtag Manipulation
    hashtag_groups = [g for g in coordination_groups if '#' in g.get('text_sample', '')]
    if hashtag_groups:
        hashtags = []
        for g in hashtag_groups[:5]:
            text = g.get('text_sample', '')
            found = re.findall(r'#\w+', text, re.IGNORECASE)
            hashtags.extend(found)
        if hashtags:
            unique_hashtags = list(set(hashtags))[:5]
            ttps.append({
                'name': 'Hashtag Manipulation',
                'description': f'Coordinated use of {len(unique_hashtags)} hashtags: {", ".join(unique_hashtags)}.',
                'severity': 'Low',
                'evidence': f'Found in {len(hashtag_groups)} coordination groups.',
                'source': 'Rule-Based'
            })

    # TTP 5: URL Amplification
    url_groups = [g for g in coordination_groups if len(g.get('unique_urls', [])) > 1]
    if url_groups:
        total_unique_urls = sum(len(g.get('unique_urls', [])) for g in url_groups)
        ttps.append({
            'name': 'URL Amplification',
            'description': f'{len(url_groups)} groups amplifying {total_unique_urls} URLs.',
            'severity': 'Low',
            'evidence': 'Multiple accounts sharing same external links.',
            'source': 'Rule-Based'
        })

    # TTP 6: Narrative Weaponization
    weaponized_keywords = ['genocide', 'kill', 'attack', 'war', 'slur', 'hate', 'ethnic cleansing', 'massacre']
    weaponized_groups = [g for g in coordination_groups if any(kw in g.get('text_sample', '').lower() for kw in weaponized_keywords)]
    if weaponized_groups:
        ttps.append({
            'name': 'Narrative Weaponization',
            'description': f'{len(weaponized_groups)} groups using high-risk weaponized keywords (violence, hate, genocide).',
            'severity': 'Critical',
            'evidence': 'Coordinated amplification of inflammatory and violent narratives.',
            'source': 'Rule-Based'
        })

    # TTP 7: Temporal Coordination (Synchronized Posting)
    synchronized_groups = 0
    for g in coordination_groups:
        timestamps = []
        for p in g.get('sample_posts_with_urls', []):
            if p.get('timestamp') and p['timestamp'] != 'N/A':
                try:
                    ts = datetime.strptime(p['timestamp'], '%Y-%m-%d %H:%M')
                    timestamps.append(ts)
                except:
                    pass
        if len(timestamps) >= 2:
            timestamps.sort()
            for i in range(len(timestamps) - 1):
                diff = (timestamps[i+1] - timestamps[i]).total_seconds() / 60
                if diff <= 60:
                    synchronized_groups += 1
                    break
    if synchronized_groups > 0:
        ttps.append({
            'name': 'Temporal Coordination (Synchronized Posting)',
            'description': f'{synchronized_groups} groups posted identical content within 1 hour of each other.',
            'severity': 'High',
            'evidence': 'Accounts appear to be coordinated in real-time or using scheduling tools.',
            'source': 'Rule-Based'
        })

    # TTP 8: Multi-Platform Narrative Seeding
    narrative_seeding_groups = [g for g in coordination_groups if len(g.get('platforms', [])) >= 2 and g.get('account_count', 0) >= 3]
    if narrative_seeding_groups:
        ttps.append({
            'name': 'Multi-Platform Narrative Seeding',
            'description': f'{len(narrative_seeding_groups)} groups seeding identical narratives across 2+ platforms simultaneously.',
            'severity': 'High',
            'evidence': 'Coordinated cross-platform manipulation to maximize reach and legitimacy.',
            'source': 'Rule-Based'
        })

    # TTP 9: Bot-like Account Behavior
    bot_like_groups = 0
    for g in coordination_groups:
        generic_names = sum(1 for acc in g.get('accounts', []) if any(x in acc.lower() for x in ['user', 'account', 'test', 'bot', '123', 'news']))
        if generic_names >= 2 or g.get('account_count', 0) >= 8:
            bot_like_groups += 1
    if bot_like_groups > 0:
        ttps.append({
            'name': 'Bot-like Account Behavior',
            'description': f'{bot_like_groups} groups contain accounts with generic naming patterns or high coordination density.',
            'severity': 'Medium',
            'evidence': 'Potential use of automated or inauthentic accounts to amplify content.',
            'source': 'Rule-Based'
        })

    # === LLM-BASED TTP DETECTION ===
    # This will find additional TTPs not covered by the rules above using DISARM framework knowledge
    llm_ttps = detect_llm_ttps(coordination_groups, posts)
    ttps.extend(llm_ttps)

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
    Fast keyword-based filter to check if post is PRIMARILY about Ethiopia.
    Returns False for posts that only mention Ethiopia in passing.
    """
    if not text or len(text.strip()) < 20:
        return False
    
    text_lower = text.lower()
    
    # Strong Ethiopia-specific signals (high confidence)
    strong_signals = [
        'ethiopia', 'ethiopian', 'abiy', 'abiy ahmed', 'fano', 'tplf',
        'nebe', 'oromia', 'amhara', 'tigray', 'addis ababa', 'prosperity party',
        'woreda', 'kebele', 'birr', 'habesha', 'federal government',
        'igad', 'pretoria agreement', 'gerd', 'renaissance dam',
        'mekelle', 'bahir dar', 'gondar', 'dessie', 'jimma', 'adama',
        'hawassa', 'dire dawa', 'harar', 'axum', 'lalibela',
        'eprdf', 'derg', 'haile selassie', 'menelik',
        'eritrea', 'sudan', 'somalia', 'djibouti',  # neighbors in context
    ]
    
    # Weak/passing mentions (low confidence - post might be about something else)
    weak_signals = [
        'ethiopia',  # could be "ambassador to Ethiopia"
    ]
    
    # Count strong signals
    strong_count = sum(1 for signal in strong_signals if signal in text_lower)
    
    # If 2+ strong signals, it's definitely about Ethiopia
    if strong_count >= 2:
        return True
    
    # If only 1 signal, check density (is it the main topic?)
    if strong_count == 1:
        words = len(text.split())
        density = 1 / max(words, 1)
        # Allow if post is short (likely focused) or signal appears multiple times
        signal_count = text_lower.count('ethiopia') + text_lower.count('abiy') + text_lower.count('fano')
        if signal_count >= 2:
            return True
        if words < 100:  # Short posts are more likely focused
            return True
        return False
    
    return False
    
def get_coordination_groups(posts_queryset, min_accounts=3, max_groups=15, similarity_threshold=0.85):
    """
    OPTIMIZED: Uses vectorized sparse matrix operations instead of O(n²) Python loops.
    Processes data in chunks to avoid memory issues.
    """
    
    coordination = []
    
    # Pre-fetch ALL data in ONE query 
    posts_data = list(
        posts_queryset
        .exclude(platform__iexact='TikTok')
        .exclude(platform__iexact='Media')
        .exclude(platform__iexact='News')
        .values(
            'id', 'account_id', 'original_text', 'platform',
            'url', 'timestamp_share', 'risk_level'
        )
        .order_by('-timestamp_share')[:5000]  # Increased limit for fast processing 
    )
    
    if len(posts_data) < min_accounts:
        return []
    
    # Filter valid texts
    valid_indices = []
    valid_texts = []
    for i, p in enumerate(posts_data):
        text = p.get('original_text', '')
        if text and len(str(text)) > 20:
            valid_indices.append(i)
            valid_texts.append(str(text))
    
    if len(valid_texts) < 2:
        return []
    
    #  Vectorized TF-IDF + batch cosine similarity
    logger.info(f"🔍 Computing TF-IDF for {len(valid_texts)} posts...")
    vectorizer = TfidfVectorizer(
        max_features=2000, 
        stop_words='english', 
        ngram_range=(1, 2),
        min_df=2,  # Ignore rare words (speeds up significantly)
        max_df=0.9  # Ignore overly common words
    )
    tfidf_matrix = vectorizer.fit_transform(valid_texts)
    
    # Compute ALL similarities at once using sparse matrix operations
    # This replaces the O(n²) Python loop with a single optimized C call
    logger.info(f"📊 Computing cosine similarity matrix ({tfidf_matrix.shape[0]}x{tfidf_matrix.shape[0]})...")
    
    # Process in chunks to avoid memory explosion on large datasets
    chunk_size = 500
    similarity_groups = []
    processed_indices = set()
    
    for chunk_start in range(0, len(valid_texts), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(valid_texts))
        chunk_matrix = tfidf_matrix[chunk_start:chunk_end]
        
        # computes similarity between chunk and ALL posts at once
        chunk_similarities = cosine_similarity(chunk_matrix, tfidf_matrix)
        
        for local_i, global_i in enumerate(range(chunk_start, chunk_end)):
            if global_i in processed_indices:
                continue
            
            similar_indices = [global_i]
            
            # Check similarities for this post against all others
            similarities = chunk_similarities[local_i]
            
            # Use numpy vectorized threshold instead of Python loop
            above_threshold = np.where(similarities >= similarity_threshold)[0]
            
            for j in above_threshold:
                if j != global_i and j not in processed_indices:
                    similar_indices.append(int(j))
                    processed_indices.add(int(j))
            
            processed_indices.add(global_i)
            
            if len(similar_indices) >= min_accounts:
                # Verify unique accounts
                group_accounts = set()
                for idx in similar_indices:
                    acc = posts_data[valid_indices[idx]].get('account_id')
                    if acc:
                        group_accounts.add(acc)
                
                if len(group_accounts) >= min_accounts:
                    similarity_groups.append(similar_indices)
    
    logger.info(f"✅ Found {len(similarity_groups)} coordination groups")
    
    # Build coordination results using pre-fetched data (no DB queries)
    for group_indices in similarity_groups[:max_groups]:
        group_posts = [posts_data[valid_indices[idx]] for idx in group_indices]
        
        accounts = list(set(p['account_id'] for p in group_posts if p.get('account_id')))
        if len(accounts) < min_accounts:
            continue
        
        # Bot detection (in-memory, no DB)
        bot_accounts = identify_bot_accounts(group_posts)
        bot_count = len(bot_accounts)
        bot_percentage = (bot_count / len(accounts) * 100) if accounts else 0
        
        # Coordination type
        coordination_type = determine_coordination_type(group_posts, bot_count)
        
        # Build sample posts (in-memory)
        sample_posts_with_urls = []
        all_platforms = set()
        all_hashtags = []
        
        for post in group_posts[:15]:
            if post.get('platform'):
                all_platforms.add(post['platform'])
            
            text = str(post.get('original_text', ''))
            found = re.findall(r'#(\w+)', text, re.IGNORECASE)
            all_hashtags.extend([h.lower() for h in found])
            
            if len(sample_posts_with_urls) < 10:
                ts = post.get('timestamp_share')
                sample_posts_with_urls.append({
                    'username': clean_username(post.get('account_id', '')),
                    'platform': post.get('platform', ''),
                    'url': post.get('url') if post.get('url') and str(post['url']).startswith('http') else None,
                    'timestamp': ts.strftime('%Y-%m-%d %H:%M') if ts else 'N/A',
                    'text_preview': text[:150] + '...' if text else '',
                    'is_bot': post.get('account_id') in bot_accounts,
                    'risk_level': post.get('risk_level', 'unknown')
                })
        
        unique_urls = list(set(p['url'] for p in group_posts if p.get('url') and str(p['url']).startswith('http')))[:5]
        text_sample = str(group_posts[0].get('original_text', ''))[:200] if group_posts else '[Similar content]'
        
        coordination.append({
            'id': len(coordination) + 1,
            'accounts': accounts[:10],
            'account_count': len(accounts),
            'post_count': len(group_posts),
            'bot_count': bot_count,
            'bot_percentage': round(bot_percentage, 1),
            'text_sample': text_sample,
            'sample_posts_with_urls': sample_posts_with_urls,
            'unique_urls': unique_urls,
            'platforms': list(all_platforms),
            'coordination_type': coordination_type,
            'similarity_score': f'≥{int(similarity_threshold*100)}%',
            'sub_narrative': extract_sub_narrative(text_sample),
            'hashtags': list(set(all_hashtags))[:10],
            'primary_type': 'amplification_network' if bot_percentage >= 50 else 'coordination',
            'sources': [],
            'amplifiers': [],
            'source_count': 0,
            'amplifier_count': 0,
        })
    
    coordination.sort(key=lambda x: (-x['bot_percentage'], -x['account_count']))
    return coordination[:max_groups]

def identify_bot_accounts(posts):
    """
    Identify bot accounts based on multiple signals:
    1. Account name patterns
    2. Posting frequency (high volume in group)
    3. Content similarity
    4. Risk level
    5. Unusual posting hours
    """
    bot_accounts = set()
    account_posts = defaultdict(list)

    # Group posts by account
    for post in posts:
        if post.get('account_id'):
            account_posts[post['account_id']].append(post)

    for account_id, account_post_list in account_posts.items():
        bot_signals = 0

        # Signal 1: Generic account name patterns
        clean_name = clean_username(account_id).lower()
        if any(pattern in clean_name for pattern in [
            'bot', 'auto', 'news', 'update', 'daily',
            'official', 'real', '2024', '2023', 'ethiopia'
        ]):
            bot_signals += 2

        # Signal 2: High volume posting (same account posting multiple similar posts)
        if len(account_post_list) >= 3:
            bot_signals += 2

        # Signal 3: High/critical risk level
        high_risk_posts = sum(1 for p in account_post_list
                              if p.get('risk_level') in ['high', 'critical'])
        if high_risk_posts > 0:
            bot_signals += 1

        # Signal 4: Very short or templated content
        short_posts = sum(1 for p in account_post_list
                          if p.get('original_text') and len(str(p['original_text'])) < 50)
        if short_posts > 0:
            bot_signals += 1

        # Signal 5: Posting at unusual hours (check timestamps)
        unusual_hours = 0
        for post in account_post_list:
            if post.get('timestamp_share'):
                try:
                    hour = post['timestamp_share'].hour
                    if hour in [0, 1, 2, 3, 4, 5]:  # Late night/early morning
                        unusual_hours += 1
                except:
                    pass

        if unusual_hours >= 2:
            bot_signals += 1

        # If 3+ bot signals, mark as bot
        if bot_signals >= 3:
            bot_accounts.add(account_id)

    return bot_accounts


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
    
    # 🔥 FIX 6: Build nodes using PRE-COMPUTED in-memory stats (NO DB queries!)
    nodes = []
    for node in G_top.nodes():
        degree = G_top.degree(node)
        
        # 🔥 Look up stats from memory instead of database
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
            "አማራ": {"severity": "medium", "target_entity": "Amhara", "language": "Amharic"},
            "amhara": {"severity": "medium", "target_entity": "Amhara", "language": "English"},
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
            "አማራ ጠል": {"severity": "high", "target_entity": "Amhara", "language": "Amharic"},
            "አማራ ስጋ": {"severity": "high", "target_entity": "Amhara", "language": "Amharic"},
            "Fota-wearer": {"severity": "medium", "target_entity": "Amhara", "language": "English"},
            "ፎጣ ለባሽ": {"severity": "medium", "target_entity": "Amhara", "language": "Amharic"},
            
            "ኦሮሞ": {"severity": "medium", "target_entity": "Oromo", "language": "Amharic"},
            "oromo": {"severity": "medium", "target_entity": "Oromo", "language": "English"},
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
            "ሶማሌ": {"severity": "medium", "target_entity": "Somali", "language": "Amharic"},
            "አፋር": {"severity": "medium", "target_entity": "Afar", "language": "Amharic"},
            "ስልጤ": {"severity": "medium", "target_entity": "Silte", "language": "Amharic"},
            "ጉራጌ": {"severity": "medium", "target_entity": "Gurage", "language": "Amharic"},
            
            "ወላሞ": {"severity": "high", "target_entity": "Wolayta", "language": "Amharic"},
            "ዲቻ": {"severity": "high", "target_entity": "Wolayta", "language": "Amharic"},
            "ሻንቅላ": {"severity": "high", "target_entity": "Benishangul/Gumuz", "language": "Amharic"},
            "ሻንቅሎች": {"severity": "high", "target_entity": "Benishangul/Gumuz", "language": "Amharic"}
        },
        
        "political_groups": {
            "ብልግና": {"severity": "low", "target_entity": "Prosperity Party", "language": "Amharic"},
            "ብልፅግና ታጥቦ ከጭቃ ነው": {"severity": "medium", "target_entity": "Prosperity Party", "language": "Amharic"},
            "prosperity party": {"severity": "low", "target_entity": "Prosperity Party", "language": "English"},
            
            "ብአዴን": {"severity": "low", "target_entity": "ADP", "language": "Amharic"},
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
        
        # === Foreign Interference, Borders & Xenophobia ===
        "foreign_interference": {
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
            
            "foreign": {"severity": "medium", "target_entity": "", "language": "English"},
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
            
            "ክርስቲያን": {"severity": "low", "target_entity": "Christian", "language": "Amharic"},
            "christian": {"severity": "low", "target_entity": "Christian", "language": "English"},
            "ፓስተር": {"severity": "low", "target_entity": "Protestant", "language": "Amharic"},
            "ጴንጤ": {"severity": "low", "target_entity": "Protestant", "language": "Amharic"},
            
            "አይሁድ": {"severity": "low", "target_entity": "Judaism", "language": "Amharic"},
            "ይሁዲ": {"severity": "low", "target_entity": "Judaism", "language": "Amharic"},
            "ፈላሻ": {"severity": "high", "target_entity": "Beta Israel", "language": "Amharic"},
            
            "መናፍቅ": {"severity": "high", "target_entity": "Protestant/Other", "language": "Amharic"},
            "መናፍቃን": {"severity": "high", "target_entity": "Protestant/Other", "language": "Amharic"},
            "አህዛብ": {"severity": "medium", "target_entity": "Non-believers", "language": "Amharic"},
            "ቃፊር": {"severity": "high", "target_entity": "Non-believers", "language": "Amharic"},
            "ኢ-አማኒ": {"severity": "medium", "target_entity": "Atheist", "language": "Amharic"},
            "sinful": {"severity": "low", "target_entity": "", "language": "English"},
            
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
            "ገሀነም ግባ": {"severity": "medium", "target_entity": "", "language": "Amharic"},
            
            "ሞጣ": {"severity": "high", "target_entity": "Muslims", "language": "Amharic"},
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
    
    # === BRANDWATCH HANDLER  ===
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
        
        # Content ID fallback (use URL hash if missing)
        bw['content_id'] = brandwatch_df.get('Resource Id', brandwatch_df.get('Mention Id', bw['URL']))
        
        bw['source_dataset'] = 'Brandwatch'
        combined.append(bw)
    
    if meltwater_df is not None and not meltwater_df.empty:
        mw = pd.DataFrame()
        mw['account_id'] = get_col(meltwater_df, ['influencer'])
        mw['content_id'] = get_col(meltwater_df, ['tweet id', 'post id', 'id'])
        mw['object_id'] = get_col(meltwater_df, ['hit sentence', 'opening text', 'headline', 'text', 'content'])
        mw['URL'] = get_col(meltwater_df, ['url'])
        mw['timestamp_share'] = get_col(meltwater_df, ['date', 'timestamp', 'alternate date format'])
        mw['source_dataset'] = 'Meltwater'
        combined.append(mw)
    
    if civicsignals_df is not None and not civicsignals_df.empty:
        cs = pd.DataFrame()
        cs['account_id'] = get_col(civicsignals_df, ['media_name', 'author', 'username'])
        cs['content_id'] = get_col(civicsignals_df, ['stories_id', 'post_id', 'id'])
        cs['object_id'] = get_col(civicsignals_df, ['title', 'text', 'content', 'body'])
        cs['URL'] = get_col(civicsignals_df, ['url', 'link'])
        cs['timestamp_share'] = get_col(civicsignals_df, ['publish_date', 'timestamp', 'date'])
        cs['source_dataset'] = 'Civicsignal'
        combined.append(cs)
    
    if tiktok_df is not None and not tiktok_df.empty:
        tt = pd.DataFrame()
        tt['object_id'] = get_col(tiktok_df, ['text', 'Transcript', 'caption', 'content'])
        tt['account_id'] = get_col(tiktok_df, ['authorMeta/name', 'username', 'creator'])
        tt['content_id'] = get_col(tiktok_df, ['id', 'video_id', 'itemId'])
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
    
    if openmeasures_df is not None and not openmeasures_df.empty:
        om = pd.DataFrame()
        om['account_id'] = get_col(openmeasures_df, ['context_name', 'channelusername', 'channeltitle'])
        om['content_id'] = get_col(openmeasures_df, ['id', 'url'])
        om['object_id'] = get_col(openmeasures_df, ['text', 'message', 'body'])
        om['URL'] = get_col(openmeasures_df, ['url'])
        raw_dates = get_col(openmeasures_df, ['created_at', 'date'])
        om['timestamp_share'] = raw_dates.astype(str).str.replace(' @ ', ' ', regex=False)
        om['source_dataset'] = 'OpenMeasure_Telegram'
        combined.append(om)
    
    return pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()


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
        # ✅ Convert to string first to safely handle NaN/float values
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
    Enhanced PEP analysis that now includes RC Members and other extra officials.
    """
    from collections import defaultdict, Counter
    import re

    if not extra_officials_list:
        extra_officials_list = []

    # 1. Build a unified dictionary of names to scan: { 'lowercase_name': 'Display Name' }
    officials_to_scan = {}
    
    # Add standard PEPs
    for pep in peps_queryset:
        name_lower = pep.name.lower().strip()
        if name_lower:
            officials_to_scan[name_lower] = pep.name

    # Add RC Members / Extra Officials
    for name in extra_officials_list:
        name_lower = name.lower().strip()
        if name_lower and name_lower not in officials_to_scan:
            officials_to_scan[name_lower] = name

    pep_mentions = defaultdict(lambda: {
        'count': 0,
        'platforms': Counter(),
        'hourly_distribution': Counter(),
        'hashtags': Counter(),
        'risk_score': 0,
        'bot_probability': 0,
        'is_rc_member': False,
        'sample_posts': [],
        'narrative_clusters': defaultdict(list),
    })
    
    # 2. Scan posts for mentions
    for post in posts_queryset[:5000]:
        if not post.original_text:
            continue
            
        text_lower = post.original_text.lower()
        
        for scan_name, display_name in officials_to_scan.items():
            if scan_name in text_lower:
                data = pep_mentions[display_name]
                data['count'] += 1
                
                # Mark if it's an RC member (if it wasn't in the original PEP queryset)
                is_pep = any(p.name == display_name for p in peps_queryset)
                if not is_pep:
                    data['is_rc_member'] = True
                
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
                
                if len(data['sample_posts']) < 3:
                    data['sample_posts'].append({
                        'text': post.original_text[:150],
                        'platform': platform,
                        'timestamp': post.timestamp_share,
                        'risk_level': post.risk_level if hasattr(post, 'risk_level') else 'medium'
                    })

    # 3. Build final results
    results = []
    for display_name, data in sorted(pep_mentions.items(), key=lambda x: x[1]['count'], reverse=True)[:limit]:
        if data['count'] < 2:
            continue
            
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
            'is_rc_member': data['is_rc_member'],
            'risk_score': min(10, data['count'] // 3 + len(clusters)),
        })
    
    return results
    
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

def get_category_trend_analysis(posts_queryset, days_back=90, cache_suffix="all"):
    """
    OPTIMIZED: Uses random sampling and caching to prevent crashes on large datasets.
    """
    # 1. CHECK CACHE FIRST (Fixed cache key generation)
    cache_key = f"trend_analysis_{days_back}_{cache_suffix}"
    cached_result = cache.get(cache_key)
    if cached_result:
        return cached_result

    from django.db.models.functions import TruncDay
    from datetime import timedelta
    
    # Limit to recent posts
    cutoff = timezone.now() - timedelta(days=days_back)
    recent_posts = posts_queryset.filter(
        timestamp_share__gte=cutoff,
        original_text__isnull=False
    ).exclude(original_text='')
    
    total_available = recent_posts.count()
    
    if total_available < 10:
        return {
            'chart_json': None,
            'trending_categories': [],
            'total_categories_tracked': 0,
            'total_posts_scanned': total_available,
            'message': 'Not enough data'
        }
    
    # 2. RANDOM SAMPLING (The Performance Fix)
    sample_size = min(1500, total_available)
    post_ids = list(recent_posts.values_list('id', flat=True))
    sampled_ids = random.sample(post_ids, sample_size)
    
    # Fetch only the sampled posts
    posts_to_scan = ProcessedPost.objects.filter(id__in=sampled_ids).iterator()
    
    # 3. SCAN ONLY THE SAMPLE
    daily_data = defaultdict(lambda: defaultdict(int))
    total_scanned = 0
    
    for post in posts_to_scan:
        if post.original_text and post.timestamp_share:
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
            'message': 'No lexicon matches found'
        }
    
    # 4. BUILD CHART
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
    
    # 5. CALCULATE TRENDS
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
    
    final_result = {
        'chart_json': chart_json,
        'trending_categories': trending[:5],
        'total_categories_tracked': len(categories),
        'total_posts_scanned': total_scanned,
        'date_range': f"{dates[0].strftime('%b %d')} - {dates[-1].strftime('%b %d, %Y')}" if dates else ""
    }
    
    # 6. SAVE TO CACHE FOR 60 MINUTES
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
    
def get_enhanced_pep_analysis(posts_queryset, peps_queryset, limit=6):
    """
    Enhanced PEP analysis with improved bot detection based on posting frequency.
    """
    from collections import defaultdict, Counter
    from datetime import datetime, timedelta
    import re
    
    pep_names = {pep.name.lower().strip(): pep for pep in peps_queryset}
    pep_mentions = defaultdict(lambda: {
        'count': 0,
        'platforms': Counter(),
        'hourly_distribution': Counter(),
        'hashtags': Counter(),
        'risk_score': 0,
        'bot_probability': 0,
        'is_gendered_target': False,
        'narrative_clusters': defaultdict(list),
        'cross_platform_signals': [],
        'deepfake_alerts': [],
        'sample_posts': [],
        'top_amplifiers': Counter(),
        'geographic_origin': Counter(),
        'sentiment_trend': [],
        'posting_frequency': {},  # NEW: Track posting patterns
    })
    
    # Analyze posts
    for post in posts_queryset[:5000]:
        if not post.original_text:
            continue
        text_lower = post.original_text.lower()
        
        # Match PEPs
        for pep_name, pep_obj in pep_names.items():
            if pep_name in text_lower or pep_obj.name.lower() in text_lower:
                data = pep_mentions[pep_obj.name]
                data['count'] += 1
                
                # Platform breakdown
                platform = post.platform or 'Unknown'
                data['platforms'][platform] += 1
                
                # Hourly distribution (for velocity)
                if post.timestamp_share:
                    hour = post.timestamp_share.hour
                    data['hourly_distribution'][hour] += 1
                    
                    # NEW: Track posting frequency patterns
                    if post.account_id:
                        if post.account_id not in data['posting_frequency']:
                            data['posting_frequency'][post.account_id] = {
                                'timestamps': [],
                                'total_posts': 0
                            }
                        data['posting_frequency'][post.account_id]['timestamps'].append(post.timestamp_share)
                        data['posting_frequency'][post.account_id]['total_posts'] += 1
                
                # Extract hashtags
                hashtags = re.findall(r'#(\w+)', post.original_text)
                data['hashtags'].update(hashtags)
                
                # Detect gendered attacks (for women)
                if any(word in text_lower for word in ['she', 'her', 'woman', 'female', 'wife', 'daughter']):
                    if any(term in text_lower for term in ['unqualified', 'emotional', 'weak', 'beautiful', 'sexy', 'mother']):
                        data['is_gendered_target'] = True
                
                # NEW: Enhanced Bot Detection based on posting frequency
                if post.account_id and post.timestamp_share:
                    bot_signals = 0
                    
                    # Get all posts for this account to calculate frequency
                    account_posts = list(posts_queryset.filter(account_id=post.account_id)[:100])
                    
                    if len(account_posts) >= 5:  # Need minimum posts to detect patterns
                        timestamps = [p.timestamp_share for p in account_posts if p.timestamp_share]
                        timestamps.sort()
                        
                        # Calculate average time between posts
                        if len(timestamps) >= 2:
                            time_diffs = []
                            for i in range(1, len(timestamps)):
                                diff = (timestamps[i] - timestamps[i-1]).total_seconds()
                                time_diffs.append(diff)
                            
                            if time_diffs:
                                avg_interval = sum(time_diffs) / len(time_diffs)
                                
                                # Bot signal 1: Very high frequency (posting every few seconds/minutes)
                                if avg_interval < 300:  # Less than 5 minutes average
                                    bot_signals += 3
                                elif avg_interval < 1800:  # Less than 30 minutes
                                    bot_signals += 2
                                elif avg_interval < 3600:  # Less than 1 hour
                                    bot_signals += 1
                                
                                # Bot signal 2: Very regular intervals (suspicious consistency)
                                if len(time_diffs) >= 3:
                                    variance = sum((x - avg_interval) ** 2 for x in time_diffs) / len(time_diffs)
                                    std_dev = variance ** 0.5
                                    coefficient_of_variation = std_dev / avg_interval if avg_interval > 0 else 1
                                    
                                    # If posting at very regular intervals (CV < 0.3), likely automated
                                    if coefficient_of_variation < 0.3 and len(account_posts) >= 10:
                                        bot_signals += 2
                                
                                # Bot signal 3: 24/7 activity (posts at all hours)
                                hours_active = set(ts.hour for ts in timestamps)
                                if len(hours_active) >= 20:  # Active in 20+ hours of the day
                                    bot_signals += 2
                                
                                # Bot signal 4: Burst posting (many posts in short time)
                                posts_per_hour = Counter(ts.strftime('%Y-%m-%d %H') for ts in timestamps)
                                max_posts_in_hour = max(posts_per_hour.values()) if posts_per_hour else 0
                                if max_posts_in_hour >= 20:  # 20+ posts in one hour
                                    bot_signals += 2
                                elif max_posts_in_hour >= 10:
                                    bot_signals += 1
                    
                    # Existing bot signals
                    account_age_days = 365  # Default assumption
                    if hasattr(post, 'account_created_at') and post.account_created_at:
                        account_age_days = (post.timestamp_share - post.account_created_at).days if post.timestamp_share else 365
                    
                    if account_age_days < 30:
                        bot_signals += 2
                    if len(post.original_text) < 50:
                        bot_signals += 1
                    if post.original_text.count('http') > 2:
                        bot_signals += 1
                    
                    data['bot_probability'] = min(100, data['bot_probability'] + bot_signals)
                
                # Narrative clustering (simple keyword-based)
                if any(kw in text_lower for kw in ['rigged', 'stolen', 'fraud', 'nebe']):
                    data['narrative_clusters']['Election Integrity'].append(post.original_text[:100])
                if any(kw in text_lower for kw in ['ethnic', 'tribal', 'amhara', 'oromo', 'tigray']):
                    data['narrative_clusters']['Ethnic Dynamics'].append(post.original_text[:100])
                if any(kw in text_lower for kw in ['corrupt', 'theft', 'bribe']):
                    data['narrative_clusters']['Corruption Allegations'].append(post.original_text[:100])
                
                # Sample posts
                if len(data['sample_posts']) < 3:
                    data['sample_posts'].append({
                        'text': post.original_text[:150],
                        'platform': platform,
                        'timestamp': post.timestamp_share,
                        'risk_level': post.risk_level if hasattr(post, 'risk_level') else 'medium'
                    })
    
    # Build final results
    results = []
    for pep_name, data in sorted(pep_mentions.items(), key=lambda x: x[1]['count'], reverse=True)[:limit]:
        if data['count'] < 2:
            continue
        
        # Calculate platform percentages
        total_posts = sum(data['platforms'].values())
        platform_breakdown = [
            {'name': plat, 'count': cnt, 'percent': round(cnt/total_posts*100)}
            for plat, cnt in data['platforms'].most_common(3)
        ]
        
        # Detect velocity spikes
        peak_hour = data['hourly_distribution'].most_common(1)
        peak_hour = peak_hour[0][0] if peak_hour else 0
        velocity_alert = peak_hour in [0, 1, 2, 3, 4, 5]  # Unusual late night activity
        
        # Top hashtags
        top_hashtags = [{'tag': tag, 'count': cnt} for tag, cnt in data['hashtags'].most_common(5) if cnt > 1]
        
        # Narrative clusters summary
        clusters = [{'name': name, 'count': len(posts)} for name, posts in data['narrative_clusters'].items()]
        
        # Bot score interpretation with frequency-based detection
        bot_score = min(100, data['bot_probability'])
        if bot_score >= 60:
            bot_level = '🔴 High (Likely Bot)'
        elif bot_score >= 30:
            bot_level = '🟡 Medium (Suspicious)'
        else:
            bot_level = '🟢 Low (Likely Human)'
        
        # Analyze posting frequency patterns for top amplifiers
        top_amplifiers_analysis = []
        if data['posting_frequency']:
            sorted_accounts = sorted(
                data['posting_frequency'].items(),
                key=lambda x: x[1]['total_posts'],
                reverse=True
            )[:5]
            
            for account_id, freq_data in sorted_accounts:
                timestamps = freq_data['timestamps']
                if len(timestamps) >= 5:
                    timestamps.sort()
                    time_diffs = []
                    for i in range(1, len(timestamps)):
                        diff = (timestamps[i] - timestamps[i-1]).total_seconds() / 60  # in minutes
                        time_diffs.append(diff)
                    
                    avg_interval = sum(time_diffs) / len(time_diffs) if time_diffs else 0
                    
                    # Determine posting pattern
                    if avg_interval < 5:
                        pattern = "🤖 Very High Frequency (Bot-like)"
                    elif avg_interval < 30:
                        pattern = "⚠️ High Frequency (Suspicious)"
                    elif avg_interval < 120:
                        pattern = "⚡ Moderate Frequency"
                    else:
                        pattern = "👤 Normal Frequency"
                    
                    top_amplifiers_analysis.append({
                        'account': account_id[:40],
                        'posts': freq_data['total_posts'],
                        'avg_interval_min': round(avg_interval, 1),
                        'pattern': pattern
                    })
        
        results.append({
            'pep_name': pep_name,
            'mention_count': data['count'],
            'platform_breakdown': platform_breakdown,
            'velocity_alert': velocity_alert,
            'peak_hour': peak_hour,
            'top_hashtags': top_hashtags,
            'bot_score': bot_score,
            'bot_level': bot_level,
            'is_gendered_target': data['is_gendered_target'],
            'narrative_clusters': clusters,
            'sample_posts': data['sample_posts'],
            'risk_score': min(10, data['count'] // 5 + len(clusters)),
            'top_amplifiers_frequency': top_amplifiers_analysis,  # NEW: Frequency analysis
        })
    
    return results
    
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
        context = super().get_context_data(**kwargs)
        
        # 1. GET FILTERED QUERYSET (now defaults to 3 months)
        queryset, start_date, end_date = get_election_posts_queryset(self.request)
        posts = queryset
        total_posts = posts.count()
        
        # 2. PLATFORM DISTRIBUTION - Direct aggregation extraction
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
        
        # Set top platform metrics fallback
        top_platform = labels[0] if labels else "—"
        
        charts = {}
        
        # 3. CREATE BAR CHART
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
                    "xaxis": {
                        "title": "Platform",
                        "tickmode": "array"
                    },
                    "yaxis": {
                        "title": "Number of Posts",
                        "gridcolor": "#E5E7EB"
                    },
                    "plot_bgcolor": "#ffffff"
                }
            }
            charts['platform'] = json.dumps(raw_chart_dict)
        
        # 4. METRICS
        unique_accounts = posts.values('account_id').distinct().count()
        high_risk_count = posts.filter(risk_level__in=['high', 'critical']).count()
        alert_level = '🚨 High' if high_risk_count > 50 else '⚠️ Medium' if high_risk_count > 10 else '✅ Low'
        peps_tracked = PEP.objects.filter(is_active=True).count()
        last_update = timezone.now().strftime('%Y-%m-%d %H:%M UTC')
        
        # 5. OTHER CHARTS
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
        
        # 6. UPLOAD SUMMARY
        recent_uploads = DataUpload.objects.filter(status='completed').order_by('-uploaded_at')[:5]
        upload_summary = {
            'show': len(recent_uploads) > 0 and (recent_uploads[0].uploaded_at > timezone.now() - timedelta(hours=2)),
            'files': recent_uploads,
            'total_records': sum(u.records_processed for u in recent_uploads),
        }
        # 7. TREND ANALYSIS - Track weaponized language categories over time
        start_str = start_date.date().isoformat() if hasattr(start_date, 'date') else str(start_date)
        end_str = end_date.date().isoformat() if hasattr(end_date, 'date') else str(end_date)
        trend_analysis = get_category_trend_analysis(posts, days_back=90, cache_suffix=f"{start_str}_{end_str}")
        
        # 8. BUILD CONTEXT
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
            'start_date': start_date.date().isoformat() if hasattr(start_date, 'date') else start_date,
            'end_date': end_date.date().isoformat() if hasattr(end_date, 'date') else end_date,
        })
        return context     
        
class NarrativesView(TemplateView):
    template_name = 'dashboard/narratives.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        # Reuse date filtering helper
        queryset, start_date, end_date = get_election_posts_queryset(
            self.request
        )

        # Generate narratives from filtered data
        context['summaries'] = get_ethiopia_summaries(queryset)
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

    def _get_lexicon_term_count(self):
        """Count total terms in CONFIG lexicon"""
        try:
            total = 0
            for category, terms in CONFIG.get('lexicon', {}).items():
                total += len(terms)
            return total
        except Exception:
            return "1000+"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # 1. TRY TO LOAD CACHED RESULTS FIRST (Instant load)
        cache_key = "lexicon_dashboard_data_v2"
        cached_data = cache.get(cache_key)
        if cached_data:
            context.update(cached_data)
            context['ai_insights'] = cache.get("lexicons_ai_insights_v1")
            context['ai_is_running'] = cache.get("lexicons_ai_running")
            # Refresh term count even from cache (fast query)
            context['lexicon_term_count'] = self._get_lexicon_term_count()
            return context

        # 2. GET POSTS (Limit to recent 5000 for performance)
        try:
            filtered_posts, start_date, end_date = get_election_posts_queryset(self.request)
            total_posts = filtered_posts.count()
        except Exception as e:
            logger.error(f"Error getting posts: {e}")
            context.update({
                'active_tab': 'lexicons',
                'top_terms': [],
                'category_counts': {},
                'severity_counts': {},
                'total_matches': 0,
                'posts_scanned': 0,
                'total_posts': 0,
                'wordcloud_base64': None,
                'targeted_entities': [],
                'ai_insights': None,
                'ai_is_running': False,
                'lexicon_term_count': self._get_lexicon_term_count(),
            })
            return context

        # Only scan the most recent 5,000 posts to prevent timeout
        posts_to_scan = filtered_posts[:5000]

        # 3. FAST REGEX SCAN
        all_matches = []
        posts_scanned = 0
        try:
            for post in posts_to_scan.iterator():
                if post.original_text:
                    try:
                        matches = scan_text_for_lexicon_terms(post.original_text)
                        if matches:
                            # Filter out single-character terms
                            all_matches.extend([m for m in matches if len(m['term'].strip()) > 1])
                            posts_scanned += 1
                    except Exception as e:
                        logger.warning(f"Error scanning post {post.id}: {e}")
                        continue
        except Exception as e:
            logger.error(f"Error during lexicon scan: {e}")

        # 4. AGGREGATE ANALYTICS
        try:
            from collections import Counter
            term_counts = Counter([m['term'] for m in all_matches])
            category_counts = Counter([m['category'] for m in all_matches])
            severity_counts = Counter([m['severity'] for m in all_matches])
            top_terms = term_counts.most_common(15)
            top_terms_with_meta = []
            for term, count in top_terms:
                if len(term.strip()) <= 1:
                    continue
                metadata = {}
                for cat, terms in CONFIG['lexicon'].items():
                    if term in terms:
                        metadata = terms[term]
                        break
                top_terms_with_meta.append({'term': term, 'count': count, 'metadata': metadata})
        except Exception as e:
            logger.error(f"Error aggregating analytics: {e}")
            top_terms_with_meta = []
            category_counts = Counter()
            severity_counts = Counter()

        # Word Cloud
        wordcloud_base64 = None
        if all_matches:
            try:
                valid_terms = [{'term': t, 'count': c} for t, c in term_counts.most_common(50) if len(t.strip()) > 1]
                wordcloud = generate_trigger_wordcloud({'top_terms': valid_terms})
                if wordcloud:
                    wordcloud_base64 = wordcloud_to_base64(wordcloud)
            except Exception as e:
                logger.warning(f"Word cloud failed: {e}")

        # Targeted Entities
        targeted_entities = []
        try:
            entity_patterns = [
                r'\b(Abiy\s+Ahmed|Prosperity\s+Party|FANO|NEBE|National\s+Election\s+Board)\b',
                r'\b(Amhara|Tigray|Oromo|Somali|Afar|Sidama)\b',
                r'[\u1200-\u137F]{3,}(?:\s+[\u1200-\u137F]{2,}){0,2}',
            ]
            entities_found = Counter()
            for post in filtered_posts[:1000]:
                if post.original_text:
                    for pattern in entity_patterns:
                        matches = re.findall(pattern, post.original_text, re.IGNORECASE)
                        for match in matches:
                            entity = match[0] if isinstance(match, tuple) else match
                            if len(entity.strip()) >= 3:
                                entities_found[entity.strip()] += 1
            targeted_entities = [{'entity': e, 'count': c} for e, c in entities_found.most_common(10)]
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")

        # 5. TRIGGER AI ANALYSIS (Non-blocking)
        cache_key_ai = "lexicons_ai_insights_v1"
        ai_insights = cache.get(cache_key_ai)
        ai_is_running = cache.get("lexicons_ai_running")
        if not ai_insights and not ai_is_running and total_posts > 10:
            cache.set("lexicons_ai_running", True, 300)
            post_ids = list(filtered_posts.values_list('id', flat=True)[:1000])
            sample_ids = random.sample(post_ids, min(50, len(post_ids)))
            thread = threading.Thread(
                target=self._run_ai_analysis_background,
                args=(sample_ids, cache_key_ai)
            )
            thread.daemon = True
            thread.start()
            logger.info("Started background analysis...")

        # 6. PREPARE DATA TO CACHE
        results_to_cache = {
            'active_tab': 'lexicons',
            'top_terms': top_terms_with_meta,
            'category_counts': dict(category_counts),
            'severity_counts': dict(severity_counts),
            'total_matches': len(all_matches),
            'posts_scanned': posts_scanned,
            'total_posts': total_posts,
            'wordcloud_base64': wordcloud_base64,
            'targeted_entities': targeted_entities,
            'start_date': start_date.date().isoformat() if hasattr(start_date, 'date') else start_date,
            'end_date': end_date.date().isoformat() if hasattr(end_date, 'date') else end_date,
            'lexicon_term_count': self._get_lexicon_term_count(),
        }
        cache.set(cache_key, results_to_cache, 3600)
        context.update(results_to_cache)
        context['ai_insights'] = ai_insights
        context['ai_is_running'] = ai_is_running and not ai_insights
        return context

    @staticmethod
    def _run_ai_analysis_background(post_ids, cache_key):
        """Runs the heavy AI model in the background and saves to cache."""
        import traceback
        logger.info(f"Background AI analysis STARTED for {len(post_ids)} posts")
        try:
            from .utils.hate_speech_detector import get_hate_speech_detector
            from .models import ProcessedPost
            from collections import Counter
            logger.info("Loading Gemma model (this may take 15-20 minutes)...")
            detector = get_hate_speech_detector()
            logger.info("Gemma model loaded successfully!")
            posts = ProcessedPost.objects.filter(id__in=post_ids)
            category_counts = Counter()
            total_analyzed = 0
            gemma_severity_map = {
                'violence': 'critical', 'inciteful': 'critical', 'call for action': 'critical', 'dehumanization': 'critical',
                'extremism': 'high', 'ethnic slur': 'high', 'slur': 'high', 'misogynistic': 'high',
                'derogatory': 'medium', 'inflammatory': 'high', 'gender disinformation': 'high',
                'stereotype': 'high', 'homophobic': 'high', 'ethnicity': 'high', 'xenophobia': 'high', 'religion': 'high',
                'ancestry': 'low', 'class': 'low', 'structural': 'low'
            }
            logger.info(f"Analyzing {posts.count()} posts...")
            for idx, post in enumerate(posts):
                if post.original_text and len(post.original_text) > 20:
                    try:
                        result = detector.detect(post.original_text)
                        cat = result.get('category')
                        if cat and cat not in ['neutral', 'error']:
                            category_counts[cat] += 1
                            total_analyzed += 1
                        if (idx + 1) % 10 == 0:
                            logger.info(f"Progress: {idx + 1}/{posts.count()} posts analyzed")
                    except Exception as e:
                        logger.warning(f"Scan failed for post {post.id}: {e}")
                        continue
            total_hateful = sum(category_counts.values())
            ai_results = []
            for cat, count in category_counts.most_common(5):
                pct = (count / total_analyzed * 100) if total_analyzed > 0 else 0
                ai_results.append({
                    'category': cat.replace('_', ' ').title(),
                    'count': count,
                    'percentage': round(pct, 1),
                    'severity': gemma_severity_map.get(cat, 'medium')
                })
            cache.set(cache_key, {
                'results': ai_results,
                'total_analyzed': total_analyzed,
                'total_hateful': total_hateful,
            }, 86400)
            logger.info(f"Background analysis COMPLETE! Found {total_hateful} hateful posts out of {total_analyzed}.")
        except Exception as e:
            logger.error(f"Background analysis FAILED: {e}")
            logger.error(traceback.format_exc())
            cache.set(cache_key, {
                'results': [],
                'total_analyzed': 0,
                'total_hateful': 0,
                'error': str(e)
            }, 3600)
        finally:
            cache.delete("lexicons_ai_running")
            logger.info("Background thread FINISHED and cleaned up")

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
        
        # 5. Generate PEP analysis insights - Pass RC members as extra officials
        pep_analysis_data = get_pep_analysis_insights(
            election_posts, 
            active_peps, 
            extra_officials_list=rc_member_names, 
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
        
        return context        
        
class NetworksView(TemplateView):
    template_name = 'dashboard/networks.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        request = self.request
        
        try:
            min_connections = int(request.GET.get('min_connections') or 2)
            top_n = int(request.GET.get('top_n') or 30)
        except (ValueError, TypeError):
            min_connections, top_n = 2, 30
            
        layout_style = request.GET.get('layout', 'spring') or 'spring'
        
        # Generate a cache key based on parameters
        posts_count = ProcessedPost.objects.filter(is_election_related=True).count()
        cache_key = f"networks_{min_connections}_{top_n}_{layout_style}_{posts_count}"
        
        # Check cache first (instant load on repeat visits!)
        cached = cache.get(cache_key)
        if cached:
            logger.info("⚡ Loading networks from cache (instant)")
            context.update(cached)
            context['active_tab'] = 'networks'
            return context
        
        logger.info("🔄 Computing networks from scratch...")
        posts = ProcessedPost.objects.filter(is_election_related=True).exclude(
            platform__iexact='TikTok'
        ).exclude(platform__iexact='Media').exclude(platform__iexact='News')
        
        graph_data = generate_network_graph_data(posts, min_connections=min_connections, top_n=top_n, layout=layout_style)
        coordination_groups = get_coordination_groups(posts, min_accounts=min_connections, max_groups=15)
        ttps = analyze_ttps(coordination_groups, posts)
        
        try:
            disarm_ttp_reference = get_disarm_ttp_reference()
        except Exception:
            disarm_ttp_reference = []
        
        context_data = {
            'active_tab': 'networks',
            'network_graph_json': json.dumps(graph_data, default=str),
            'coordination_groups': coordination_groups,
            'total_coordinated_groups': len(coordination_groups),
            'total_coordinated_accounts': sum(g.get('account_count', 0) for g in coordination_groups),
            'total_posts': posts.count(),
            'max_group_size': max([g.get('account_count', 0) for g in coordination_groups]) if coordination_groups else 0,
            'min_connections': min_connections,
            'top_n': top_n,
            'layout_style': layout_style,
            'ttps': ttps,
            'disarm_ttp_reference': disarm_ttp_reference,
            'disarm_dataset_size': 80000,
        }
        
        # Cache for 30 minutes (invalidated when new data is uploaded)
        cache.set(cache_key, context_data, 1800)
        
        context.update(context_data)
        return context
        
class LexiconManagementView(TemplateView):
    template_name = 'dashboard/lexicon_management.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        lexicon_terms = LexiconTerm.objects.filter(
            is_election_related=True
        ).exclude(
            term__regex=r'^.$'
        ).order_by('category', 'severity')

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
                                'is_election_related': True
                            }
                        )
            lexicon_terms = LexiconTerm.objects.filter(
                is_election_related=True
            ).exclude(
                term__regex=r'^.$'
            ).order_by('category', 'severity')

        filtered_posts, start_date, end_date = get_election_posts_queryset(self.request)
        total_posts_in_filter = filtered_posts.count()

        all_matches = []
        posts_scanned = 0
        for post in filtered_posts.iterator():
            if post.original_text:
                matches = scan_text_for_lexicon_terms(post.original_text)
                if matches:
                    all_matches.extend([m for m in matches if len(m['term'].strip()) > 1])
                    posts_scanned += 1

        categories = lexicon_terms.values_list('category', flat=True).distinct()
        scan_results = self.request.session.pop('scan_results', None)
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
                        obj.save()
                        messages.success(request, "Term updated successfully!")
                        cache.delete("lexicon_dashboard_data_v2")
                    else:
                        messages.warning(request, "Term must be at least 2 characters long.")
                except LexiconTerm.DoesNotExist:
                    messages.error(request, "Term not found.")

        elif action == 'delete_term':
            term_id = request.POST.get('term_id')
            if term_id:
                try:
                    LexiconTerm.objects.filter(id=term_id).delete()
                    messages.success(request, "Term deleted successfully.")
                    cache.delete("lexicon_dashboard_data_v2")
                except Exception as e:
                    messages.error(request, f"Error: {e}")

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
                        'is_election_related': True,
                    }
                )
                messages.success(request, "Term added successfully!")
                cache.delete("lexicon_dashboard_data_v2")
            else:
                messages.warning(request, "Term must be at least 2 characters long. Single characters are skipped.")

        elif action == 'scan_text':
            text = request.POST.get('scan_text', '').strip()
            if text:
                lexicon_matches = scan_text_for_lexicon_terms(text)
                lexicon_risk = calculate_risk_score(lexicon_matches)

                llm_result = detect_hate_speech_llm(text)

                try:
                    gemma_result = get_hate_speech_detector().detect(text)
                except Exception as e:
                    logger.warning(f"Gemma LoRA detection failed: {e}")
                    gemma_result = {'category': 'error', 'confidence': 0.0, 'severity': 'low'}

                is_hate_speech = False
                overall_severity_num = 1

                llm_is_hate = llm_result.get('is_hate_speech', False)
                llm_confidence = llm_result.get('confidence', 0)
                gemma_category = gemma_result.get('category', 'neutral')
                gemma_confidence = gemma_result.get('confidence', 0)
                lexicon_score = lexicon_risk.get('score', 0)

                if llm_is_hate and llm_confidence > 0.7:
                    is_hate_speech = True
                    overall_severity_num = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}.get(llm_result.get('severity', 'low'), 1)

                if gemma_category != 'neutral' and gemma_confidence > 0.7:
                    is_hate_speech = True
                    gemma_sev = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}.get(gemma_result.get('severity', 'low'), 1)
                    overall_severity_num = max(overall_severity_num, gemma_sev)

                if lexicon_score > 5:
                    is_hate_speech = True
                    lex_sev = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}.get(lexicon_risk['level'], 1)
                    overall_severity_num = max(overall_severity_num, lex_sev)

                severity_map = {1: 'low', 2: 'medium', 3: 'high', 4: 'critical'}

                analysis_parts = []
                if llm_result.get('explanation'):
                    analysis_parts.append(f"LLM Analysis: {llm_result['explanation']}")
                if lexicon_matches:
                    terms_found = [f"'{m['term']}'" for m in lexicon_matches[:5]]
                    analysis_parts.append(f"Lexicon matched {len(lexicon_matches)} term(s): {', '.join(terms_found)}")
                if gemma_result.get('category') and gemma_result.get('category') != 'error':
                    analysis_parts.append(f"Gemma model classified as: {gemma_result['category']} ({gemma_result.get('confidence', 0) * 100:.0f}% confidence)")
                combined_analysis = ". ".join(analysis_parts) if analysis_parts else "No specific patterns detected"

                request.session['scan_results'] = {
                    'text': text[:200] + '...' if len(text) > 200 else text,
                    'lexicon_matches': lexicon_matches,
                    'lexicon_risk': lexicon_risk,
                    'llm_result': llm_result,
                    'gemma_result': gemma_result,
                    'is_hate_speech': is_hate_speech,
                    'overall_severity': severity_map[overall_severity_num],
                    'overall_confidence': round((llm_result.get('confidence', 0) + gemma_result.get('confidence', 0)) / 2, 2),
                    'overall_confidence_pct': f"{round((llm_result.get('confidence', 0) + gemma_result.get('confidence', 0)) / 2 * 100)}%",
                    'all_categories': list(set([m['category'] for m in lexicon_matches] + llm_result.get('categories', []))),
                    'targeted_groups': llm_result.get('targeted_groups', []),
                    'explanation': llm_result.get('explanation', ''),
                    'analysis': combined_analysis,
                    'has_lexicon_matches': len(lexicon_matches) > 0
                }

                if is_hate_speech:
                    messages.warning(request, f"Potential hate speech detected! Severity: {severity_map[overall_severity_num].upper()} (Confidence: {request.session['scan_results']['overall_confidence'] * 100:.0f}%)")
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
        import hashlib
        from django.utils import timezone
        
        logger.info(f"📥 Upload request: data_type={request.POST.get('data_type')}, source={request.POST.get('source_name')}")
        logger.info(f"📁 FILES: {list(request.FILES.keys())}")
        
        uploaded_files = request.FILES.getlist('csv_files')
        if not uploaded_files:
            messages.error(request, "No files received.")
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
                
                logger.info(f"🔄 Processing: {original_name} -> {unique_filename}")
                
                # Create upload record
                upload = DataUpload.objects.create(
                    uploaded_file=file_path,
                    original_filename=original_name,
                    uploaded_by=request.user.username if request.user.is_authenticated else 'anonymous',
                    data_type=request.POST.get('data_type', 'custom'),
                    status='processing'
                )
                
                # === STREAMLIT-STYLE DATA PROCESSING ===
                data_type = upload.data_type
                
                # === LOAD CSV WITH APPROPRIATE HANDLING ===
                if data_type == 'brandwatch':
                    df = pd.read_csv(full_path, sep=',', low_memory=False, on_bad_lines='skip', encoding_errors='ignore')
                else:
                    df = load_data_robustly(full_path)
                
                # 🗑️ DROP 'Sentiment' COLUMN IF PRESENT (aligns with original mapping/schema)
                if 'Sentiment' in df.columns:
                    df = df.drop(columns=['Sentiment'])
                    logger.info(f"🗑️ Dropped 'Sentiment' column to match original processing schema.")
                
                # === DEBUG: Check original CSV columns ===
                logger.info(f"📋 ORIGINAL CSV COLUMNS: {list(df.columns)[:15]}{'...' if len(df.columns) > 15 else ''}")
                logger.info(f"📊 CSV Shape: {df.shape}")
                
                if df.empty:
                    raise ValueError(f"Failed to load data from {original_name}")
                
                # === COMBINE/MAP DATA BASED ON SOURCE TYPE ===
                if data_type == 'meltwater':
                    combined_df = combine_social_media_data(meltwater_df=df, civicsignals_df=None)
                elif data_type == 'civicsignals':
                    combined_df = combine_social_media_data(meltwater_df=None, civicsignals_df=df)
                elif data_type == 'tiktok':
                    combined_df = combine_social_media_data(meltwater_df=None, civicsignals_df=None, tiktok_df=df)
                elif data_type == 'openmeasure':
                    combined_df = combine_social_media_data(meltwater_df=None, civicsignals_df=None, openmeasures_df=df)
                elif data_type == 'brandwatch':
                    # === BRANDWATCH-SPECIFIC MAPPING ===
                    logger.info(f"🔄 Mapping Brandwatch columns for {original_name}")
                    # Case-insensitive column lookup
                    df_cols_lower = {c.lower().strip(): c for c in df.columns}
                    combined_df = pd.DataFrame()
                    
                    # 1. Account ID
                    acc_col = None
                    for col in ['source', 'author', 'full name', 'weblog title', 'account', 'username']:
                        if col in df_cols_lower:
                            acc_col = df_cols_lower[col]
                            break
                    combined_df['account_id'] = df[acc_col].astype(str).str.strip().replace('nan', '') if acc_col else 'Unknown'
                    
                    # 2. Original Text (CRITICAL)
                    text_col = None
                    for col in ['text', 'full text', 'content', 'title', 'hit sentence', 'opening text']:
                        if col in df_cols_lower:
                            text_col = df_cols_lower[col]
                            break
                    combined_df['original_text'] = df[text_col].astype(str).str.strip() if text_col else ''
                    
                    # 3. URL (ensure column exists)
                    url_col = None
                    for col in ['url', 'link', 'post url', 'permalink']:
                        if col in df_cols_lower:
                            url_col = df_cols_lower[col]
                            break
                    combined_df['URL'] = df[url_col] if url_col else ''
                    
                    # 4. Timestamp
                    ts_col = None
                    for col in ['timestamp', 'date', 'created at', 'publish date']:
                        if col in df_cols_lower:
                            ts_col = df_cols_lower[col]
                            break
                    combined_df['timestamp_share'] = df[ts_col] if ts_col else pd.NaT
                    
                    # 5. Platform inference
                    pt_col = None
                    for col in ['platform', 'page type', 'source']:
                        if col in df_cols_lower:
                            pt_col = df_cols_lower[col]
                            break
                    pt_map = {
                        'twitter': 'X', 'x': 'X', 'x.com': 'X', 't.co': 'X',
                        'facebook': 'Facebook', 'fb': 'Facebook', 'fb.watch': 'Facebook',
                        'instagram': 'Instagram', 'tiktok': 'TikTok',
                        'youtube': 'YouTube', 'telegram': 'Telegram', 't.me': 'Telegram'
                    }
                    if pt_col:
                        combined_df['Platform'] = df[pt_col].astype(str).str.lower().map(pt_map).fillna('Brandwatch')
                    else:
                        combined_df['Platform'] = 'Brandwatch'
                    
                    # 6. Content ID (generate if missing)
                    cid_col = None
                    for col in ['url', 'resource id', 'mention id', 'post id', 'id']:
                        if col in df_cols_lower:
                            cid_col = df_cols_lower[col]
                            break
                    if cid_col:
                        combined_df['content_id'] = df[cid_col].astype(str).str.strip()
                    else:
                        # Generate hash-based ID from text + URL
                        combined_df['content_id'] = combined_df.apply(
                            lambda r: hashlib.md5(f"{r['original_text'][:50]}_{r['URL']}".encode()).hexdigest()[:16],
                            axis=1
                        )
                    
                    combined_df['source_dataset'] = 'Brandwatch'
                    # Filter: keep only rows with substantial text
                    initial_count = len(combined_df)
                    combined_df = combined_df[combined_df['original_text'].str.len() > 20]
                    logger.info(f"✅ Brandwatch: filtered {initial_count} → {len(combined_df)} valid rows")
                else:
                    # Custom/unknown format
                    combined_df = preprocess_dataframe(df)
                
                # === DEBUG: Check combined data ===
                logger.info(f"📊 COMBINED DATA COLUMNS: {list(combined_df.columns)}")
                if 'URL' not in combined_df.columns:
                    logger.error("❌ URL COLUMN NOT FOUND after combining!")
                    combined_df['URL'] = ''  # Safety fallback
                
                # === FINAL PREPROCESSING ===
                processed_df = final_preprocess_and_map_columns(combined_df)
                
                # === DEBUG: Check processed data ===
                logger.info(f"📊 PROCESSED DATA COLUMNS: {list(processed_df.columns)}")
                if 'URL' not in processed_df.columns:
                    logger.error("❌ URL COLUMN MISSING after final processing!")
                    processed_df['URL'] = ''
                
                # Parse timestamps
                if 'timestamp_share' in processed_df.columns:
                    processed_df['timestamp_share'] = processed_df['timestamp_share'].apply(parse_timestamp_robust)
                
                # === SAVE TO DATABASE ===
                count = 0
                urls_saved = 0
                for _, row in processed_df.iterrows():
                    # Skip if no content
                    if not row.get('original_text') or pd.isna(row.get('original_text')) or str(row.get('original_text', '')).strip() == '':
                        continue
                    
                    # Check for duplicates
                    cid = row.get('content_id')
                    url_val = row.get('url') or row.get('URL')
                    if cid and ProcessedPost.objects.filter(content_id=cid).exists():
                        continue
                    if url_val and str(url_val).startswith('http') and ProcessedPost.objects.filter(url=url_val).exists():
                        continue
                    
                    # Get or create DataSource
                    source_name = str(row.get('source_dataset', data_type))
                    source_obj, _ = DataSource.objects.get_or_create(name=source_name)
                    
                    # Prepare URL value
                    url_value = str(url_val).strip()[:500] if url_val and str(url_val).startswith('http') else None
                    if url_value:
                        urls_saved += 1
                    
                    # Create post
                    ProcessedPost.objects.create(
                        account_id=str(row.get('account_id', ''))[:100],
                        content_id=str(cid).strip()[:100] if cid else None,
                        original_text=str(row.get('original_text', '')).strip(),
                        url=url_value,
                        platform=str(row.get('Platform', 'Unknown')),
                        timestamp_share=row.get('timestamp_share'),
                        source_dataset=source_obj,
                        is_election_related=is_election_related(str(row.get('original_text', '')))
                    )
                    count += 1
                
                logger.info(f"✅ Saved {count} posts, {urls_saved} with URLs from {original_name}")
                
                # Update record
                upload.status = 'completed'
                upload.processing_log = f"Successfully processed {count} posts ({urls_saved} with URLs)"
                upload.records_processed = count
                upload.save()
                
                results.append((original_name, True, f"Processed {count} posts", count))
                logger.info(f"✅ {original_name}: Processed {count} posts")
                
            except Exception as e:
                logger.error(f"❌ Upload failed for {uploaded_file.name}: {str(e)}", exc_info=True)
                if 'upload' in locals():
                    upload.status = 'failed'
                    upload.processing_log = str(e)
                    upload.save()
                results.append((uploaded_file.name, False, str(e), 0))
        
        # === SHOW SUMMARY (ONLY IF RECORDS WERE SAVED) ===
        success_count = sum(1 for _, s, _, c in results if s and c > 0)
        total_saved = sum(c for _, s, _, c in results if s)
        
        if total_saved > 0:
            if success_count == len([r for r in results if r[1]]):
                messages.success(request, f"✅ All {len(uploaded_files)} files processed successfully! Saved {total_saved} posts.")
            else:
                messages.warning(request, f"⚠️ {success_count}/{len(uploaded_files)} files succeeded. Saved {total_saved} posts total.")
        elif not any(s for _, s, _, _ in results):
            messages.error(request, "❌ Failed to process any files. Check terminal logs for details.")
        else:
            messages.info(request, "ℹ️ Upload completed. No new posts matched criteria (check logs for details).")
            
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
    """API endpoint to generate coordination network graph"""
    # Get parameters
    min_connections = int(request.GET.get('min_connections', 2))
    top_n = int(request.GET.get('top_n', 50))
    
    # Build coordination graph
    queryset = ProcessedPost.objects.filter(
        is_election_related=True,
        cluster__gte=0
    )
    
    G = nx.Graph()
    
    # Group by exact text to find coordination
    for text_group in queryset.values('original_text').annotate(
        accounts=Count('account_id', distinct=True)
    ).filter(accounts__gte=2):
        accounts = queryset.filter(original_text=text_group['original_text']).values_list('account_id', flat=True).distinct()
        
        if len(accounts) >= 2:
            for i in range(len(accounts)):
                for j in range(i+1, len(accounts)):
                    if G.has_edge(accounts[i], accounts[j]):
                        G[accounts[i]][accounts[j]]['weight'] += 1
                    else:
                        G.add_edge(accounts[i], accounts[j], weight=1)
    
    # Filter to nodes with minimum connections
    nodes_to_keep = [n for n, d in G.degree() if d >= min_connections]
    G = G.subgraph(nodes_to_keep).copy()
    
    if G.number_of_edges() == 0:
        return JsonResponse({'nodes': [], 'edges': [], 'message': 'No coordination links found'})
    
    # Get top N nodes by degree
    top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:top_n]
    top_node_names = [n for n, _ in top_nodes]
    G_top = G.subgraph(top_node_names).copy()
    
    # Prepare node data
    node_data = []
    for node in G_top.nodes():
        node_data.append({
            'id': node,
            'degree': G_top.degree(node),
        })
    
    # Prepare edge data
    edge_data = []
    for u, v, data in G_top.edges(data=True):
        edge_data.append({
            'source': u,
            'target': v,
            'weight': data.get('weight', 1)
        })
    
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
