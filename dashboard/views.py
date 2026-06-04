"""
Django views for Ethiopia Election Monitor
"""
import json
import logging
import os
import re
import requests
import csv
from django.utils import timezone
from io import StringIO
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from django.shortcuts import render, redirect
from django.contrib import messages
from django.views.generic import TemplateView, View
from django.http import JsonResponse, HttpResponse
from django.db.models import Count, Q, F, Case, When, Value, CharField, Max, Avg
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
        # Point to your newly fused MLX model
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
    if not raw_name or pd.isna(raw_name):
        return "Unknown"
    # Convert to string and take the first part before any space or "Name" suffix
    name = str(raw_name).split(' ')[0].strip()
    # Remove common artifacts
    name = re.sub(r'(?i)(name|source|nan|none)$', '', name).strip()
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
                    # Fix 2: Use the robust loader to handle UTF-16/Tabs for Meltwater/others
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
                        # Fix 3: Robust URL capture (checks both cases)
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
        if category not in lexicon: continue
        for term, metadata in lexicon[category].items():
            if metadata.get("language") == "amharic" or re.match(r'^[\u1200-\u137F]+$', term):
                pattern = re.escape(term)
            else:
                pattern = r'\b' + re.escape(term) + r'\b'
            
            if re.search(pattern, text_lower, re.IGNORECASE):
                matches.append({
                    'term': term, 'category': category,
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
    
    # Ethiopia-specific prompt (adapted from Côte d'Ivoire version)
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
    
import json
from typing import List, Dict, Any

def detect_ttps_with_gemma(coordination_groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Detect DISARM TTPs using the fine-tuned Gemma model.
    Falls back to the old analyze_ttps function if model fails.
    """
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

        # Generate using MLX (NO temp or verbose arguments to avoid crashes)
        raw_output = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=2048
        )

        logger.debug(f"[Gemma raw output] ({len(raw_output)} chars): {repr(raw_output[:800])}")

        response_text = raw_output
        
        # Strip Gemma thinking tags if present (Gemma 3/4 can emit <think>...</think> or <|channel>thought)
        response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
        response_text = re.sub(r'<\|channel\>thought.*?<\|channel\>final>', '', response_text, flags=re.DOTALL).strip()

        # Strip markdown fences
        fence_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response_text)
        if fence_match:
            response_text = fence_match.group(1).strip()
        else:
            # Fall back to finding the first { ... } block
            brace_match = re.search(r'\{[\s\S]*\}', response_text)
            if brace_match:
                response_text = brace_match.group(0).strip()
            else:
                logger.warning(f"No JSON object found in response. Full output: {repr(raw_output[:1000])}")
                raise ValueError("No JSON object found in model output")

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
                    'evidence': f"Detected via Gemma model analysis. {technique.get('evidence', '')}",
                    'confidence': technique.get('confidence', 0.8),
                    'model_source': 'gemma_finetuned'
                }
                ttps.append(ttp_data)

        if ttps:
            logger.info(f"Gemma model detected {len(ttps)} TTPs")
            return ttps
        else:
            logger.info("Gemma model found no qualifying TTPs, using fallback")

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse Gemma response as JSON: {e}")
    except Exception as e:
        logger.error(f"Gemma TTP detection failed: {e}", exc_info=True)

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
  
def analyze_ttps(coordination_groups, posts):
    """Analyze Tactics, Techniques, and Procedures from coordinated groups - FULLY FIXED"""
    ttps = []
    
    if not coordination_groups:
        return ttps
    
    # TTP 1: Coordinated Inauthentic Behavior (CIB)
    cib_groups = [g for g in coordination_groups if g['account_count'] >= 5]
    if cib_groups:
        ttps.append({
            'name': 'Coordinated Inauthentic Behavior (CIB)',
            'description': f'Detected {len(cib_groups)} groups with 5+ accounts sharing identical content.',
            'severity': 'High',
            'evidence': f'{sum(g["post_count"] for g in cib_groups)} total posts across {sum(g["account_count"] for g in cib_groups)} accounts.'
        })
    
    # TTP 2: Cross-Platform Amplification - FIXED for new data structure
    cross_platform_groups = []
    for g in coordination_groups:
        # Extract platforms from sample_posts_with_urls
        platforms = set(p['platform'] for p in g.get('sample_posts_with_urls', []) if p.get('platform'))
        if len(platforms) > 1:
            cross_platform_groups.append({
                'group': g,
                'platforms': list(platforms)
            })
    
    if cross_platform_groups:
        all_platforms = set(p['platforms'] for p in cross_platform_groups)
        ttps.append({
            'name': 'Cross-Platform Amplification',
            'description': f'{len(cross_platform_groups)} groups operating across {len(all_platforms)} platforms.',
            'severity': 'Medium',
            'evidence': f"Platforms: {', '.join(sorted(all_platforms))}"
        })
    
    # TTP 3: Rapid Response / Burst Posting
    burst_groups = [g for g in coordination_groups if g['post_count'] > 10]
    if burst_groups:
        max_posts = max(g['post_count'] for g in burst_groups)
        ttps.append({
            'name': 'Rapid Response / Burst Posting',
            'description': f'{len(burst_groups)} groups with high-volume posting (max: {max_posts} posts/group).',
            'severity': 'Medium',
            'evidence': f"Identical content bursts across {sum(g['account_count'] for g in burst_groups)} accounts."
        })
    
    # TTP 4: Hashtag Manipulation (Simplified)
    hashtag_groups = [g for g in coordination_groups if '#' in g['text_sample']]
    if hashtag_groups:
        hashtags = []
        for g in hashtag_groups[:5]:  # Check top 5 groups
            text = g['text_sample']
            found = re.findall(r'#\w+', text, re.IGNORECASE)
            hashtags.extend(found)
        
        if hashtags:
            unique_hashtags = list(set(hashtags))[:5]
            ttps.append({
                'name': 'Hashtag Manipulation',
                'description': f'Coordinated use of {len(unique_hashtags)} hashtags: {", ".join(unique_hashtags)}.',
                'severity': 'Low',
                'evidence': f'Found in {len(hashtag_groups)} coordination groups.'
            })
    
    # TTP 5: URL Amplification (NEW - uses your URL data!)
    url_groups = [g for g in coordination_groups if len(g.get('unique_urls', [])) > 1]
    if url_groups:
        total_unique_urls = sum(len(g.get('unique_urls', [])) for g in url_groups)
        ttps.append({
            'name': 'URL Amplification',
            'description': f'{len(url_groups)} groups amplifying {total_unique_urls} URLs.',
            'severity': 'Low',
            'evidence': 'Multiple accounts sharing same external links.'
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
    
def get_coordination_groups(posts_queryset, min_accounts=3, max_groups=10):
    """Find accounts posting identical messages - FIXED to show real usernames and URLs"""
    coordination = []
    
    # Group by exact text
    text_groups = posts_queryset.values('original_text').annotate(
        account_count=Count('account_id', distinct=True),
        post_count=Count('id')
    ).filter(account_count__gte=min_accounts).order_by('-account_count')[:max_groups]
    
    for group in text_groups:
        text = group['original_text']
        # Get DISTINCT accounts with their posts and URLs
        if not is_primarily_ethiopia_related(text):
            continue
        account_posts = posts_queryset.filter(original_text=text).values(
            'account_id', 'platform', 'url', 'timestamp_share'
        ).distinct()
        
        accounts = []
        sample_posts_with_urls = []
        
        for ap in account_posts[:20]:
            username = clean_username(ap['account_id'])
            if username and len(username) > 2:
                if username not in accounts:
                    accounts.append(username)
                
                if len(sample_posts_with_urls) < 5:
                    sample_posts_with_urls.append({
                        'username': username,
                        'platform': ap['platform'],
                        # FIX: Use 'url' as the key and ensure it's a string
                        'url': ap['url'] if ap['url'] and str(ap['url']).startswith('http') else None,
                        'timestamp': ap['timestamp_share'].strftime('%Y-%m-%d %H:%M') if ap['timestamp_share'] else 'N/A',
                        'text_preview': text[:100] + '...'
                    })
        
        # Only include groups that still meet the threshold after cleaning
        if len(accounts) >= min_accounts:
            coordination.append({
                'id': len(coordination) + 1,
                'accounts': accounts[:8],  # Show top 8 cleaned usernames
                'account_count': len(accounts),
                'post_count': group['post_count'],
                'text_sample': text[:200] if text else '[Identical message]',
                'sample_posts_with_urls': sample_posts_with_urls,
                'unique_urls': list(set([p['url'] for p in sample_posts_with_urls if p['url']]))[:5]
            })
    
    return coordination[:max_groups]

def generate_network_graph_data(posts_queryset, min_connections=2, top_n=50, layout='spring'):
    """Generate cleaner network graph - FIXED usernames and platform info"""
    G = nx.Graph()
    
    # Group by exact text to find coordination
    text_groups = posts_queryset.values('original_text').annotate(
        account_count=Count('account_id', distinct=True)
    ).filter(account_count__gte=min_connections)
    
    for group in text_groups:
        text = group['original_text']
        # Get real account data with URLs
        if not is_primarily_ethiopia_related(text):
            continue
            
        accounts_data = list(posts_queryset.filter(original_text=text).values(
            'account_id', 'platform', 'url'
        ).distinct())
        
        accounts = []
        for acc_data in accounts_data:
            # --- UPDATED CLEANING LOGIC ---
            username = clean_username(acc_data['account_id'])
            
            # Filter out generic artifacts that aren't real usernames
            if username and len(username) > 2 and username.lower() not in ['twitter', 'facebook', 'tiktok', 'source']:
                accounts.append({
                    'id': username,
                    'platform': acc_data['platform'],
                    'sample_url': acc_data['url'] if acc_data['url'] and acc_data['url'].startswith('http') else None
                })
        
        # Create edges between coordinated accounts
        for i in range(len(accounts)):
            for j in range(i+1, len(accounts)):
                u_id = accounts[i]['id']
                v_id = accounts[j]['id']
                
                # Ensure we don't link an account to itself
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
        return {'nodes': [], 'edges': [], 'message': 'No coordination detected'}
    
    # Filter low-degree nodes
    nodes_to_keep = [n for n, d in G.degree() if d >= min_connections]
    G = G.subgraph(nodes_to_keep).copy()
    
    if G.number_of_edges() == 0:
        return {'nodes': [], 'edges': [], 'message': 'No significant connections'}
    
    # Top N nodes
    top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:top_n]
    top_node_names = [n for n, _ in top_nodes]
    G_top = G.subgraph(top_node_names).copy()
    
    # Generate positions
    if layout == 'circular':
        pos = nx.circular_layout(G_top)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G_top)
    elif layout == 'spring':
        pos = nx.spring_layout(G_top, k=0.6, iterations=50, seed=42)
    else:
        pos = nx.spring_layout(G_top, seed=42)
    
    # Build clean nodes
    nodes = []
    for node in G_top.nodes():
        degree = G_top.degree(node)
        # Search specifically for this cleaned username
        node_posts = posts_queryset.filter(account_id__icontains=node)
        post_count = node_posts.count()
        
        platforms = list(node_posts.values_list('platform', flat=True).distinct())
        platform = platforms[0] if platforms else 'Unknown'
        
        # Get first valid URL properly
        sample_url_obj = node_posts.exclude(url='').exclude(url__isnull=True).filter(url__icontains='http').first()
        sample_url = sample_url_obj.url if sample_url_obj else None
        
        nodes.append({
            'id': node,
            'label': node,
            'degree': degree,
            'post_count': post_count,
            'platform': platform,
            'url': sample_url,         
            'sample_url': sample_url,  
            'x': float(pos[node][0]),
            'y': float(pos[node][1]),
            'size': max(15, degree * 3),
            'color': _get_platform_color(platform)
        })   
    
    # Build clean edges with URLs
    edges = []
    for u, v, data in G_top.edges(data=True):
        if u in pos and v in pos:
            sample_url = data.get('sample_url1') or data.get('sample_url2')
            edges.append({
                'source': u,
                'target': v,
                'weight': data.get('weight', 1),
                'source_x': float(pos[u][0]),
                'source_y': float(pos[u][1]),
                'target_x': float(pos[v][0]),
                'target_y': float(pos[v][1]),
                'sample_url': sample_url
            })
    
    return {
        'nodes': nodes, 
        'edges': edges,
        'stats': {
            'nodes': len(nodes),
            'edges': len(edges),
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
        # === Ethnic/Identity-Based Terms ===
        "ethnic_identity": {
            "አማራ": {"severity": "medium", "target_entity": "Amhara", "language": "amharic"},
            "amhara": {"severity": "medium", "target_entity": "Amhara", "language": "english"},
            "ነፍኛ": {"severity": "high", "target_entity": "Amhara", "language": "amharic"},
            "neftegna": {"severity": "high", "target_entity": "Amhara", "language": "english"},
            "ኦሮሞ": {"severity": "medium", "target_entity": "Oromo", "language": "amharic"},
            "oromo": {"severity": "medium", "target_entity": "Oromo", "language": "english"},
            "ጋላ": {"severity": "high", "target_entity": "Oromo", "language": "amharic"},
            "galla": {"severity": "high", "target_entity": "Oromo", "language": "english"},
            "ትግሬ": {"severity": "medium", "target_entity": "Tigrayan", "language": "amharic"},
            "tigrayan": {"severity": "medium", "target_entity": "Tigrayan", "language": "english"},
            "ወያኔ": {"severity": "high", "target_entity": "TPLF", "language": "amharic"},
            "woyane": {"severity": "high", "target_entity": "TPLF", "language": "english"},
            "ህወሓት": {"severity": "high", "target_entity": "TPLF", "language": "amharic"},
            "tplf": {"severity": "high", "target_entity": "TPLF", "language": "english"},
            "ቅማንት": {"severity": "medium", "target_entity": "Qemant", "language": "amharic"},
            "qemant": {"severity": "medium", "target_entity": "Qemant", "language": "english"},
            "አገው": {"severity": "medium", "target_entity": "Agew", "language": "amharic"},
            "agew": {"severity": "medium", "target_entity": "Agew", "language": "english"},
            "ሶማሌ": {"severity": "medium", "target_entity": "Somali", "language": "amharic"},
            "አፋር": {"severity": "medium", "target_entity": "Afar", "language": "amharic"},
        },
        
        # === Political Groups & Parties ===
        "political_groups": {
            "ብልግና": {"severity": "low", "target_entity": "Prosperity Party", "language": "amharic"},
            "prosperity party": {"severity": "low", "target_entity": "Prosperity Party", "language": "english"},
            "አዴፓ": {"severity": "low", "target_entity": "ADP", "language": "amharic"},
            "adp": {"severity": "low", "target_entity": "ADP", "language": "english"},
            "ፋኖ": {"severity": "medium", "target_entity": "Fano", "language": "amharic"},
            "fano": {"severity": "medium", "target_entity": "Fano", "language": "english"},
            "ኦነግ": {"severity": "high", "target_entity": "ONEG", "language": "amharic"},
            "oneg": {"severity": "high", "target_entity": "ONEG", "language": "english"},
        },
        
        # === Violence & Incitement Terms ===
        "violence_incitement": {
            "ግል": {"severity": "critical", "target_entity": "", "language": "amharic"},
            "kill": {"severity": "critical", "target_entity": "", "language": "english"},
            "ግሉ": {"severity": "critical", "target_entity": "", "language": "amharic"},
            "kill them": {"severity": "critical", "target_entity": "", "language": "english"},
            "አጥ": {"severity": "critical", "target_entity": "", "language": "amharic"},
            "destroy": {"severity": "critical", "target_entity": "", "language": "english"},
            "ጦርነት": {"severity": "high", "target_entity": "", "language": "amharic"},
            "war": {"severity": "high", "target_entity": "", "language": "english"},
            "ጥቃት": {"severity": "high", "target_entity": "", "language": "amharic"},
            "attack": {"severity": "high", "target_entity": "", "language": "english"},
            "ስጋት": {"severity": "medium", "target_entity": "", "language": "amharic"},
            "threat": {"severity": "medium", "target_entity": "", "language": "english"},
        },
        
        # === Dehumanizing & Derogatory Terms ===
        "dehumanizing": {
            "እንስሳ": {"severity": "high", "target_entity": "", "language": "amharic"},
            "animal": {"severity": "high", "target_entity": "", "language": "english"},
            "ከብት": {"severity": "high", "target_entity": "", "language": "amharic"},
            "cattle": {"severity": "high", "target_entity": "", "language": "english"},
            "ውሻ": {"severity": "high", "target_entity": "", "language": "amharic"},
            "dog": {"severity": "high", "target_entity": "", "language": "english"},
            "ደደብ": {"severity": "medium", "target_entity": "", "language": "amharic"},
            "fool": {"severity": "medium", "target_entity": "", "language": "english"},
            "ቆሻሻ": {"severity": "high", "target_entity": "", "language": "amharic"},
            "trash": {"severity": "high", "target_entity": "", "language": "english"},
            "ሌባ": {"severity": "high", "target_entity": "", "language": "amharic"},
            "thief": {"severity": "high", "target_entity": "", "language": "english"},
            "ገዳይ": {"severity": "critical", "target_entity": "", "language": "amharic"},
            "killer": {"severity": "critical", "target_entity": "", "language": "english"},
        },
        
        # === Election & Governance Terms ===
        "election_governance": {
            "ምርጫ": {"severity": "low", "target_entity": "", "language": "amharic"},
            "election": {"severity": "low", "target_entity": "", "language": "english"},
            "ድምፅ": {"severity": "low", "target_entity": "", "language": "amharic"},
            "vote": {"severity": "low", "target_entity": "", "language": "english"},
            "ነቤ": {"severity": "low", "target_entity": "NEBE", "language": "amharic"},
            "nebe": {"severity": "low", "target_entity": "NEBE", "language": "english"},
            "የተጭበበረ": {"severity": "medium", "target_entity": "", "language": "amharic"},
            "rigged": {"severity": "medium", "target_entity": "", "language": "english"},
            "ማጭበርበር": {"severity": "medium", "target_entity": "", "language": "amharic"},
            "fraud": {"severity": "medium", "target_entity": "", "language": "english"},
        },
        
        # === Foreign Interference & Geopolitics ===
        "foreign_interference": {
            "ግብፅ": {"severity": "low", "target_entity": "Egypt", "language": "amharic"},
            "egypt": {"severity": "low", "target_entity": "Egypt", "language": "english"},
            "ሱዳን": {"severity": "low", "target_entity": "Sudan", "language": "amharic"},
            "sudan": {"severity": "low", "target_entity": "Sudan", "language": "english"},
            "ኤርትራ": {"severity": "low", "target_entity": "Eritrea", "language": "amharic"},
            "eritrea": {"severity": "low", "target_entity": "Eritrea", "language": "english"},
            "አሜሪካ": {"severity": "low", "target_entity": "USA", "language": "amharic"},
            "america": {"severity": "low", "target_entity": "USA", "language": "english"},
            "ቻይና": {"severity": "low", "target_entity": "China", "language": "amharic"},
            "china": {"severity": "low", "target_entity": "China", "language": "english"},
            "ውጭ": {"severity": "medium", "target_entity": "", "language": "amharic"},
            "foreign": {"severity": "medium", "target_entity": "", "language": "english"},
        },
        
        # === Religious & Cultural Terms ===
        "religious_cultural": {
            "ኦርቶዶክስ": {"severity": "low", "target_entity": "Orthodox", "language": "amharic"},
            "orthodox": {"severity": "low", "target_entity": "Orthodox", "language": "english"},
            "እስልምና": {"severity": "low", "target_entity": "Islam", "language": "amharic"},
            "islam": {"severity": "low", "target_entity": "Islam", "language": "english"},
            "ክርስቲያን": {"severity": "low", "target_entity": "Christian", "language": "amharic"},
            "christian": {"severity": "low", "target_entity": "Christian", "language": "english"},
        }
    },
    
    # === Risk Scoring Configuration ===
    "risk_scoring": {
        "severity_weights": {"low": 1, "medium": 2, "high": 3, "critical": 4},
        "category_weights": {
            "ethnic_identity": 1.2, "political_groups": 1.2, "violence_incitement": 1.5,
            "dehumanizing": 1.5, "election_governance": 1.0, "foreign_interference": 1.0, "religious_cultural": 1.0
        },
        "risk_thresholds": {"low": 3, "medium": 6, "high": 10, "critical": 15}
    },
    
    # === Display Configuration ===
    "display": {"max_terms_per_category": 20, "show_amharic_first": True, "highlight_critical": True}
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
    """

    queryset = ProcessedPost.objects.all()

    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')

    if start_date and end_date:
        queryset = queryset.filter(
            timestamp_share__date__range=[start_date, end_date]
        )
    else:
        end_dt = timezone.now()
        start_dt = end_dt - timedelta(days=30)

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
    Accurately count and normalize platforms
    Returns: (df_platforms, top_platform_name)
    """
    from collections import defaultdict
    import pandas as pd
    
    raw_platforms = list(posts_queryset.values_list('platform', flat=True))
    
    platform_counts = defaultdict(int)
    
    for plat in raw_platforms:
        if not plat or plat.lower() in ['nan', 'none', '', 'unknown']:
            continue
            
        p = str(plat).lower().strip()
        
        # Normalize platform names
        if p in ['x', 'twitter', 't.co', 'x.com', 'twitter source', 'twitter.com', 'twitter_source']:
            platform_counts['X'] += 1
        elif p in ['facebook', 'fb.watch', 'facebook.com', 'fb', 'facebook_source', 'fb_source']:
            platform_counts['Facebook'] += 1
        elif p in ['telegram', 't.me', 'tg', 'telegram_source']:
            platform_counts['Telegram'] += 1
        elif p in ['tiktok', 'tik tok', 'tik-tok', 'tiktok_source']:
            platform_counts['TikTok'] += 1
        elif p in ['media', 'news', 'news/media', 'civicsignal', 'civic signal', 'civicsignals']:
            platform_counts['Media'] += 1
        elif p in ['youtube', 'youtu.be', 'yt', 'youtube_source']:
            platform_counts['YouTube'] += 1
        elif p in ['instagram', 'insta', 'ig', 'instagram_source']:
            platform_counts['Instagram'] += 1
        else:
            # Keep original but capitalize properly
            platform_counts[str(plat).title()] += 1
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {'platform': k, 'count': int(v)} 
        for k, v in platform_counts.items() 
        if v > 0  # Only include platforms with actual posts
    ])
    
    if not df.empty:
        df = df.sort_values('count', ascending=False)
        top_platform = df.iloc[0]['platform']
    else:
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
    
def get_enhanced_pep_analysis(posts_queryset, peps_queryset, limit=6):
    """
    Enhanced PEP analysis with platform breakdown, velocity, bot detection,
    gendered attacks, narrative clusters, and cross-platform coordination.
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
                
                # Extract hashtags
                hashtags = re.findall(r'#(\w+)', post.original_text)
                data['hashtags'].update(hashtags)
                
                # Detect gendered attacks (for women)
                if any(word in text_lower for word in ['she', 'her', 'woman', 'female', 'wife', 'daughter']):
                    if any(term in text_lower for term in ['unqualified', 'emotional', 'weak', 'beautiful', 'sexy', 'mother']):
                        data['is_gendered_target'] = True
                
                # Bot detection signals
                account_age_days = 365  # Default assumption
                if hasattr(post, 'account_created_at') and post.account_created_at:
                    account_age_days = (post.timestamp_share - post.account_created_at).days if post.timestamp_share else 365
                
                bot_signals = 0
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
        
        # Bot score interpretation
        bot_score = min(100, data['bot_probability'])
        bot_level = ' Low' if bot_score < 30 else '🟡 Medium' if bot_score < 60 else '🔴 High'
        
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
        
        # 1. GET FILTERED QUERYSET
        queryset, start_date, end_date = get_election_posts_queryset(self.request)
        posts = queryset
        total_posts = posts.count()
        
        # 2. PLATFORM DISTRIBUTION & CHART
        df_platforms, top_platform = get_platform_distribution(posts)
        charts = {}
        
        if not df_platforms.empty:
            fig_platform = px.bar(
                df_platforms, x='platform', y='count',
                labels={'platform': 'Platform', 'count': 'Posts'},
                color='count', color_continuous_scale='Blues',
                title='Post Distribution by Platform'
            )
            fig_platform.update_layout(
                xaxis_tickangle=-45, margin=dict(b=100, t=50, l=50, r=20), height=400,
                xaxis={'categoryorder': 'total descending'}
            )
            charts['platform'] = fig_platform.to_json()
        
        # 3. METRICS
        unique_accounts = posts.values('account_id').distinct().count()
        high_risk_count = posts.filter(risk_level__in=['high', 'critical']).count()
        alert_level = '🚨 High' if high_risk_count > 50 else '⚠️ Medium' if high_risk_count > 10 else '✅ Low'
        peps_tracked = PEP.objects.filter(is_active=True).count()
        last_update = timezone.now().strftime('%Y-%m-%d %H:%M UTC')
        
        # 4. OTHER CHARTS
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
        
        # 5. UPLOAD SUMMARY
        recent_uploads = DataUpload.objects.filter(status='completed').order_by('-uploaded_at')[:5]
        upload_summary = {
            'show': len(recent_uploads) > 0 and (recent_uploads[0].uploaded_at > timezone.now() - timedelta(hours=2)),
            'files': recent_uploads,
            'total_records': sum(u.records_processed for u in recent_uploads),
        }
        
        # 6. BUILD CONTEXT
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
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        posts = ProcessedPost.objects.all()
        total_posts = posts.count()
        
        # Scan for lexicon matches
        all_matches = []
        posts_scanned = 0
        
        for post in posts[:3000]:  # Limit for performance
            if post.original_text:
                matches = scan_text_for_lexicon_terms(post.original_text)
                if matches:
                    all_matches.extend(matches)
                    posts_scanned += 1
        
        # Aggregate analytics
        from collections import Counter
        term_counts = Counter([m['term'] for m in all_matches])
        category_counts = Counter([m['category'] for m in all_matches])
        severity_counts = Counter([m['severity'] for m in all_matches])
        
        # Top terms with metadata
        top_terms = term_counts.most_common(15)
        top_terms_with_meta = []
        for term, count in top_terms:
            metadata = {}
            for cat, terms in CONFIG['lexicon'].items():
                if term in terms:
                    metadata = terms[term]
                    break
            top_terms_with_meta.append({'term': term, 'count': count, 'metadata': metadata})
        
        # === 🎨 WORD CLOUD (Streamlit-style) ===
        wordcloud_base64 = None
        if all_matches:
            try:
                wordcloud = generate_trigger_wordcloud(
                    {'top_terms': [{'term': t, 'count': c} for t, c in term_counts.most_common(50)]}
                )
                if wordcloud:
                    wordcloud_base64 = wordcloud_to_base64(wordcloud)
            except Exception as e:
                logger.warning(f"Word cloud generation failed: {e}")
        
        # === 🎯 TARGETED ENTITIES (Streamlit-style) ===
        targeted_entities = []
        if posts.exists():
            # Entity patterns from your Streamlit app
            entity_patterns = [
                r'\b(Abiy\s+Ahmed|Prosperity\s+Party|FANO|NEBE|National\s+Election\s+Board)\b',
                r'\b(Amhara|Tigray|Oromo|Somali|Afar|Sidama)\b',
                r'[\u1200-\u137F]{3,}(?:\s+[\u1200-\u137F]{2,}){0,2}',  # Amharic names
            ]
            entities_found = Counter()
            for post in posts[:1000]:  # Limit for performance
                if post.original_text:
                    for pattern in entity_patterns:
                        matches = re.findall(pattern, post.original_text, re.IGNORECASE)
                        for match in matches:
                            # Handle tuple returns from regex
                            entity = match[0] if isinstance(match, tuple) else match
                            if len(entity.strip()) >= 3:
                                entities_found[entity.strip()] += 1
            targeted_entities = [{'entity': e, 'count': c} for e, c in entities_found.most_common(10)]
        
        context.update({
            'active_tab': 'lexicons',
            'top_terms': top_terms_with_meta,
            'category_counts': dict(category_counts),
            'severity_counts': dict(severity_counts),
            'total_matches': len(all_matches),
            'posts_scanned': posts_scanned,
            'total_posts': total_posts,
            # NEW: Streamlit-style additions
            'wordcloud_base64': wordcloud_base64,
            'targeted_entities': targeted_entities,
        })
        return context


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
        min_connections = int(request.GET.get('min_connections', 2))
        top_n = int(request.GET.get('top_n', 30))
        layout_style = request.GET.get('layout', 'spring')
        
        # Use election-related posts only
        posts = ProcessedPost.objects.filter(is_election_related=True)
        
        # Generate CLEAN network graph
        graph_data = generate_network_graph_data(posts, min_connections=min_connections, top_n=top_n, layout=layout_style)
        
        # Get coordination groups with FIXED usernames and URLs
        coordination_groups = get_coordination_groups(posts, min_accounts=min_connections, max_groups=15)
        
        # Analyze TTPs using Gemma model (with fallback to old method)
        ttps = detect_ttps_with_gemma(coordination_groups)
        
        context.update({
            'active_tab': 'networks',
            'network_graph_json': json.dumps(graph_data, default=str),
            'coordination_groups': coordination_groups,
            'total_coordinated_groups': len(coordination_groups),
            'total_coordinated_accounts': sum(g['account_count'] for g in coordination_groups),
            'total_posts': posts.count(),
            'max_group_size': max([g['account_count'] for g in coordination_groups]) if coordination_groups else 0,
            # Controls
            'min_connections': min_connections,
            'top_n': top_n,
            'layout_style': layout_style,
            # TTPs
            'ttps': ttps,
        })
        return context        

class LexiconManagementView(TemplateView):
    template_name = 'dashboard/lexicon_management.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Load lexicon terms from DB (user-added + CONFIG defaults)
        lexicon_terms = LexiconTerm.objects.filter(is_election_related=True).order_by('category', 'severity')
        
        # If DB is empty, seed from CONFIG (one-time migration)
        if not lexicon_terms.exists():
            for category, terms in CONFIG['lexicon'].items():
                for term, metadata in terms.items():
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
            lexicon_terms = LexiconTerm.objects.filter(is_election_related=True).order_by('category', 'severity')
        
        # Get distinct categories for filter dropdown
        categories = lexicon_terms.values_list('category', flat=True).distinct()
        
        # Get scan results from session (if any) and clear immediately
        scan_results = self.request.session.pop('scan_results', None)
        
        context.update({
            'active_tab': 'lexicon_management',
            'lexicon_terms': lexicon_terms,
            'categories': categories,
            'total_terms': lexicon_terms.count(),
            'critical_count': lexicon_terms.filter(severity='critical').count(),
            'amharic_count': lexicon_terms.filter(language='amharic').count(),
            'scan_results': scan_results,  # Only pass if exists
        })
        return context

    def post(self, request, *args, **kwargs):
        action = request.POST.get('action')
        
        #  Handle Edit Term
        if action == 'edit_term':
            term_id = request.POST.get('term_id')
            if term_id:
                try:
                    obj = LexiconTerm.objects.get(id=term_id)
                    new_term = request.POST.get('term', '').strip()
                    if new_term:
                        obj.term = new_term
                    obj.category = request.POST.get('category', obj.category)
                    obj.severity = request.POST.get('severity', obj.severity)
                    obj.target_entity = request.POST.get('target_entity', '')
                    obj.language = request.POST.get('language', 'english')
                    obj.save()
                    messages.success(request, "✅ Term updated successfully!")
                except LexiconTerm.DoesNotExist:
                    messages.error(request, "❌ Term not found.")
        
        # Handle Delete Term
        elif action == 'delete_term':
            term_id = request.POST.get('term_id')
            if term_id:
                try:
                    LexiconTerm.objects.filter(id=term_id).delete()
                    messages.success(request, "✅ Term deleted successfully.")
                except Exception as e:
                    messages.error(request, f"❌ Error: {e}")

        # Handle Add Term 
        elif action == 'add_term':
            term = request.POST.get('term')
            if term:
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
                messages.success(request, "✅ Term added successfully!")
        
        #  Handle Scan Text 
        elif action == 'scan_text':
            text = request.POST.get('scan_text', '').strip()
            if text and len(text) > 10:
                # 1. Lexicon-based detection
                lexicon_matches = scan_text_for_lexicon_terms(text)
                lexicon_risk = calculate_risk_score(lexicon_matches)
                
                # 2. LLM-based detection
                llm_result = detect_hate_speech_llm(text)
                
                # 3. Combine results
                is_hate_speech = (
                    lexicon_risk['score'] > 0 or 
                    (llm_result.get('is_hate_speech', False) and llm_result.get('confidence', 0) > 0.6)
                )
                
                overall_severity = max(
                    {'low':1, 'medium':2, 'high':3, 'critical':4}.get(llm_result.get('severity','low'), 1),
                    {'low':1, 'medium':2, 'high':3, 'critical':4}.get(lexicon_risk['level'], 1)
                )
                severity_map = {1:'low', 2:'medium', 3:'high', 4:'critical'}
                
                request.session['scan_results'] = {
                    'text': text[:200] + '...' if len(text) > 200 else text,
                    'lexicon_matches': lexicon_matches,
                    'lexicon_risk': lexicon_risk,
                    'llm_result': llm_result,
                    'is_hate_speech': is_hate_speech,
                    'overall_severity': severity_map[overall_severity],
                    'all_categories': list(set([m['category'] for m in lexicon_matches] + llm_result.get('categories', []))),
                    'targeted_groups': llm_result.get('targeted_groups', []),
                    'explanation': llm_result.get('explanation', '')
                }
                
                if is_hate_speech:
                    messages.warning(request, f"⚠️ Potential hate speech detected! Severity: {severity_map[overall_severity].upper()}")
                else:
                    messages.success(request, "✅ No hate speech detected.")
            else:
                messages.warning(request, "⚠️ Please enter text to scan (minimum 10 characters)")
        
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
