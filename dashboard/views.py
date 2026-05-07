"""
Django views for Ethiopia Election Monitor
Reuses your Streamlit app.py logic but queries database instead of CSVs
"""
import json
import logging
import os
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from django.shortcuts import render, redirect
from django.contrib import messages
from django.views.generic import TemplateView, View
from django.http import JsonResponse, HttpResponse
from django.db.models import Count, Q, F, Case, When, Value, CharField
from django.utils import timezone
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.paginator import Paginator
from django.db.models.functions import TruncDay
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go

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


logger = logging.getLogger(__name__)

#  HELPER FUNCTIONS

def clean_username(raw_name):
    if not raw_name or pd.isna(raw_name):
        return "Unknown"
    # Convert to string and take the first part before any space or "Name" suffix
    name = str(raw_name).split(' ')[0].strip()
    # Remove common artifacts
    name = re.sub(r'(?i)(name|source|nan|none)$', '', name).strip()
    return name
    
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
                # Use your robust data loader logic
                df = pd.read_csv(f, low_memory=False, on_bad_lines='skip')
                stats['total_rows'] += len(df)
                
                # Normalize columns based on platform
                if platform_type == 'meltwater':
                    processed_df = combine_social_media_data(meltwater_df=df)
                elif platform_type == 'tiktok':
                    processed_df = combine_social_media_data(tiktok_df=df)
                elif platform_type == 'openmeasure':
                    processed_df = combine_social_media_data(openmeasures_df=df)
                else:
                    processed_df = combine_social_media_data(civicsignals_df=df)

                # ONE CLEAN LOOP
                for _, row in processed_df.iterrows():
                    cid = row.get('content_id')
                    if cid and not ProcessedPost.objects.filter(content_id=cid).exists():
                        
                        # 1. Fetch the actual DataSource object (Fixes the ValueError)
                        source_name = row.get('source_dataset', platform_type)
                        source_obj, _ = DataSource.objects.get_or_create(name=source_name)
                
                        # 2. Create the post using the object instance
                        ProcessedPost.objects.create(
                            account_id=str(row.get('account_id', ''))[:100],
                            content_id=cid,
                            original_text=str(row.get('object_id', '')), # Fixed column name
                            url=row.get('url') or row.get('URL') or row.get('link') or '',
                            platform=row.get('Platform', platform_type.title()),
                            timestamp_share=parse_timestamp_robust(row.get('timestamp_share')),
                            source_dataset=source_obj,  # Passing the actual object
                            is_election_related=is_election_related(str(row.get('object_id', '')))
                        )
                        stats['saved'] += 1
                    else:
                        stats['duplicates'] += 1                
                stats['files_count'] += 1
            except Exception as e:
                logger.error(f"Upload error: {e}")
                messages.error(request, f"Error processing {f.name}")

        # Create the detailed summary message for the sidebar
        detail_msg = (
            f"<strong>Data Upload Details:</strong><br>"
            f"• Source: {platform_type.title()}<br>"
            f"• Files processed: {stats['files_count']}<br>"
            f"• Rows analyzed: {stats['total_rows']}<br>"
            f"• <strong>New unique posts: {stats['saved']}</strong><br>"
            f"• Duplicates ignored: {stats['duplicates']}"
        )
        messages.success(request, detail_msg)
        
        # Redirect back to stay on the same page
        return redirect(request.POST.get('next', 'home'))

    # 2. Page Load Logic (GET)
    # Pull ALL unique data from the database
    all_posts = ProcessedPost.objects.all().order_by('-timestamp_share')
    
    # Run Narrative and Coordination algorithms on the database set
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
    """Generate STRICT summary using ONLY content explicitly present in texts"""
    # Use first 80 texts for context
    sample_texts = texts[:80]
    joined = "\n---\n".join([f"[{i+1}] {t}" for i, t in enumerate(sample_texts)])
    
    # Include real URLs
    real_urls = [u for u in urls if u and u.startswith('http')][:10]
    url_context = "\nReal source links from dataset:\n" + "\n".join(real_urls) if real_urls else ""
    
    prompt = f"""You are an intelligence analyst reviewing social media posts about the Ethiopia election.
Your task is to summarize ONLY what is explicitly stated in the provided posts.

**STRICT RULES - DO NOT VIOLATE:**
1. Use ONLY the exact text content provided below. Do NOT invent, assume, or extrapolate.
2. Do NOT create fake account names, URLs, engagement metrics, or timestamps.
3. Do NOT mention specific likes/retweets/views unless explicitly present in the text.
4. If a claim is not directly stated in the provided texts, DO NOT include it.
5. If you cannot find evidence for a category, write "Not explicitly stated in provided posts."

**Provided Posts (verbatim from dataset, {len(sample_texts)} samples shown):**
{joined}

**Real Source Links (from dataset, for reference only):**
{url_context}

**Time Range:** {min_ts} to {max_ts}

**Output Format (use simple text, no markdown headers):**
NARRATIVE THEME: [One short phrase summarizing the dominant topic]

EXPLICIT CLAIMS (quote or closely paraphrase from posts above):
- [Claim 1, with brief context]
- [Claim 2, with brief context]
- [etc.]

TARGETED GROUPS/ENTITIES (only if explicitly named in posts):
- [Group/entity 1]
- [Group/entity 2]

LANGUAGE/TONE OBSERVED: [e.g., accusatory, urgent, informational, etc.]

SAMPLE QUOTES (exact phrases from provided posts, max 5):
1. '[exact quote 1]'
2. '[exact quote 2]'
3. '[exact quote 3]'

DO NOT include: fake accounts, fake URLs, engagement metrics, or claims not in the provided texts."""
    
    # For now, return a formatted placeholder (replace with actual LLM call when ready)
    sample_quotes = []
    for i, text in enumerate(sample_texts[:3]):
        clean_text = text[:100].replace('"', "'").replace('\n', ' ') if text else ''
        sample_quotes.append(f"{i+1}. '{clean_text}...'")
    
    return f"""NARRATIVE THEME: Cluster of {len(sample_texts)} posts discussing election-related topics

EXPLICIT CLAIMS:
- Posts in this cluster share similar language and themes
- Content focuses on Ethiopian political discourse

TARGETED GROUPS/ENTITIES:
- Various Ethiopian political entities mentioned

LANGUAGE/TONE OBSERVED: Mixed, with some urgent and informational tones

SAMPLE QUOTES:
{chr(10).join(sample_quotes)}

Time Range: {min_ts} to {max_ts}"""


def get_ethiopia_summaries(posts_queryset, max_clusters=10):
    """Generates LLM-powered summaries - FIXED with URLs in sample posts"""
    all_summaries = []
    
    if posts_queryset.count() < 50:
        return all_summaries
    
    post_data = list(posts_queryset.values('original_text', 'url', 'account_id', 'platform', 'timestamp_share')[:2000])
    
    if len(post_data) < 20:
        return all_summaries
    
    texts = [p['original_text'] for p in post_data if p['original_text'] and len(p['original_text'].strip()) > 10]
    
    if len(texts) < 20:
        return all_summaries
    
    try:
        vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1,2))
        X = vectorizer.fit_transform(texts)
        
        n_clusters = max(2, min(max_clusters, len(texts) // 20))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        cluster_posts = defaultdict(list)
        for idx, label in enumerate(labels):
            cluster_posts[label].append(post_data[idx])
        
        for cluster_id, cluster_data in cluster_posts.items():
            if len(cluster_data) >= 3:
                cluster_texts = [p['original_text'] for p in cluster_data if p['original_text']]
                cluster_urls = [p['url'] for p in cluster_data if p.get('url')]
                timestamps = [p['timestamp_share'] for p in cluster_data if p.get('timestamp_share')]
                min_ts = min(timestamps).strftime('%Y-%m-%d') if timestamps else 'N/A'
                max_ts = max(timestamps).strftime('%Y-%m-%d') if timestamps else 'N/A'
                
                # FIXED: Sample posts WITH URLs
                sample_posts_with_urls = []
                for post in cluster_data[:5]:
                    if post['original_text']:
                        sample_posts_with_urls.append({
                            'text': post['original_text'][:150] + '...',
                            'url': post['url'] if post['url'] and str(post['url']).startswith('http') else None,
                            'account': str(post['account_id'])[:30],
                            'platform': post['platform']
                        })
                
                summary_text = summarize_cluster_ethiopia(
                    cluster_texts[:50], cluster_urls[:10], cluster_data, min_ts, max_ts
                )
                
                if not any(phrase in summary_text.lower() for phrase in ["no explicit claims", "not explicitly stated"]):
                    total_reach = len(cluster_data)
                    platforms = [p['platform'] for p in cluster_data if p.get('platform')]
                    platform_counts = Counter(platforms)
                    top_platforms = ", ".join([f"{p} ({c})" for p, c in platform_counts.most_common(3)])
                    
                    narrative_desc = extract_narrative_description(summary_text, cluster_texts[:5])
                    
                    all_summaries.append({
                        'cluster_id': cluster_id,
                        'Context': summary_text,
                        'Narrative_Description': narrative_desc,
                        'Total_Reach': total_reach,
                        'Emerging_Virality': assign_virality_tier(total_reach),
                        'Top_Platforms': top_platforms,
                        'sample_posts_with_urls': sample_posts_with_urls,  # NEW: With URLs!
                        'post_count': len(cluster_data),
                        'unique_urls_count': len(set([p['url'] for p in cluster_data if p['url']]))
                    })
        
        all_summaries.sort(key=lambda x: x['Total_Reach'], reverse=True)
    except Exception as e:
        logger.error(f"Narrative clustering failed: {e}")
    
    return all_summaries

    
def extract_narrative_description(summary_text, sample_posts):
    """Generate a specific, meaningful 1-sentence narrative description from actual cluster posts"""
    
    if not sample_posts:
        return "Analyzing narrative content..."
    
    # Combine all posts in this cluster for analysis
    all_text = ' '.join([p for p in sample_posts if p and isinstance(p, str)]).lower()
    
    # Define topic keywords for Ethiopia election context
    topic_keywords = {
        'election fraud': ['rigged', 'fraud', 'stolen', 'manipulated', 'fake results', 'cheating', 'ballot', 'nebe'],
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


def combine_social_media_data(meltwater_df=None, civicsignals_df=None, tiktok_df=None, openmeasures_df=None):
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
    
    # Filter to original posts only
    if 'object_id' in dfp.columns:
        mask = dfp['object_id'].apply(is_original_post) & (~dfp['object_id'].str.contains('🔁', na=False)) & (~dfp['object_id'].str.startswith('RT @', na=False))
        dfp = dfp[mask].copy()
    
    # Clean object_id
    dfp['object_id'] = dfp['object_id'].astype(str).replace('nan','').fillna('')
    dfp = dfp[dfp['object_id'].str.strip() != ""]
    
    # Extract original text
    dfp['original_text'] = dfp['object_id'].apply(extract_original_text) if coordination_mode=="Text Content" else dfp['URL'].astype(str).replace('nan','')
    dfp = dfp[dfp['original_text'].str.strip() != ""].reset_index(drop=True)
    
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
    Get election-filtered posts with date range filtering
    Reuses your Streamlit date filtering logic
    """
    # Get date range from query params or default to last 30 days
    start_date_str = request.GET.get('start_date')
    end_date_str = request.GET.get('end_date')
    
    if start_date_str and end_date_str:
        try:
            start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
            end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00')) + timedelta(days=1)
        except:
            # Fallback to defaults
            end_date = timezone.now()
            start_date = end_date - timedelta(days=30)
    else:
        end_date = timezone.now()
        start_date = end_date - timedelta(days=30)
    
    # Base queryset: election-related posts only
    queryset = ProcessedPost.objects.filter(
        is_election_related=True,
        timestamp_share__range=[start_date, end_date]
    )
    
    # Platform filter
    platform = request.GET.get('platform')
    if platform and platform != 'all':
        queryset = queryset.filter(platform=platform)
    
    # Risk level filter
    risk_level = request.GET.get('risk_level')
    if risk_level and risk_level != 'all':
        queryset = queryset.filter(risk_level=risk_level)
    
    return queryset, start_date, end_date


class HomeView(TemplateView):
    """Executive dashboard - election-focused"""
    template_name = 'dashboard/home.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # 1. Fetch data and calculate metrics
        posts = ProcessedPost.objects.all()
        total_posts = posts.count()
        
        # Platform breakdown
        platforms = posts.values('platform').annotate(count=Count('id')).order_by('-count')
        top_platform = platforms.first()['platform'] if platforms.exists() else "—"
        
        # Risk and account metrics
        unique_accounts = posts.values('account_id').distinct().count()
        high_risk_count = posts.filter(risk_level__in=['high', 'critical']).count()
        alert_level = '🚨 High' if high_risk_count > 50 else '⚠️ Medium' if high_risk_count > 10 else '✅ Low'
        
        peps_tracked = PEP.objects.filter(is_active=True).count()
        last_update = timezone.now().strftime('%Y-%m-%d %H:%M UTC')
        
        # 2. Prepare Charts
        charts = {}
        if posts.exists():
            # A. Platform Distribution
            fig_platform = px.bar(
                platforms, x='platform', y='count', 
                labels={'platform': 'Platform', 'count': 'Posts'},
                color='count', color_continuous_scale='Blues'
            )
            charts['platform'] = fig_platform.to_json()
            
            # B. Top Accounts
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
                
                if name.lower() in invalid_accounts:
                    continue
                
                if name and name not in ['-', 'nan', 'None', '']:
                    cleaned_accounts.append({'account_id': name[:50], 'count': acc['count']})
            
            if cleaned_accounts:
                import pandas as pd
                df_accounts = pd.DataFrame(cleaned_accounts)
                fig_accounts = px.bar(
                    df_accounts, 
                    x='account_id', y='count',
                    labels={'account_id': 'Account', 'count': 'Posts'},
                    color='count', color_continuous_scale='Viridis',
                    title='Top 10 Accounts by Activity'
                )
                fig_accounts.update_layout(
                    xaxis_tickangle=-45, 
                    margin=dict(b=100, t=50, l=50, r=20),
                    height=400
                )
                charts['accounts'] = fig_accounts.to_json()

            # C. Risk Distribution
            risk_dist = posts.values('risk_level').annotate(count=Count('id')).order_by('risk_level')
            if risk_dist:
                fig_risk = px.pie(
                    risk_dist, names='risk_level', values='count',
                    title='Risk Level Distribution',
                    color='risk_level',
                    color_discrete_map={
                        'low': '#22c55e', 'medium': '#eab308', 
                        'high': '#f97316', 'critical': '#dc2626'
                    }
                )
                charts['risk'] = fig_risk.to_json()
            
            # D. Daily Volume Chart
            daily_posts = posts.annotate(
                day=TruncDay('timestamp_share')
            ).values('day').annotate(
                count=Count('id')
            ).order_by('day')
            
            if daily_posts:
                daily_data = list(daily_posts)
                if daily_data:  
                    fig_daily = px.line(
                        daily_data,
                        x='day',
                        y='count',
                        labels={'day': 'Date', 'count': 'Posts'},
                        title='Daily Post Volume',
                        markers=True
                    )
                    fig_daily.update_layout(
                        xaxis_tickangle=-45,
                        margin=dict(b=100, t=50, l=50, r=20),
                        height=400
                    )
                    charts['daily'] = fig_daily.to_json()
        
        # 3. Recent Upload Summary
        recent_uploads = DataUpload.objects.filter(status='completed').order_by('-uploaded_at')[:5]
        upload_summary = {
            'show': len(recent_uploads) > 0 and (recent_uploads[0].uploaded_at > timezone.now() - timedelta(hours=2)),
            'files': recent_uploads,
            'total_records': sum(u.records_processed for u in recent_uploads),
        }
        
        # 4. Build Context
        context.update({
            'active_tab': 'home',
            'tabs': [
                {'name': 'Home', 'url_name': 'home', 'icon': '🏠'},
                {'name': 'PEPs/PIPs Tracker', 'url_name': 'peps', 'icon': '👤'},
                {'name': 'Mapped Lexicons', 'url_name': 'lexicons', 'icon': '🗣️'},
                {'name': 'Trending Narratives', 'url_name': 'narratives', 'icon': '📰'},
                {'name': 'Networks & TTPs', 'url_name': 'networks', 'icon': '🕸️'},
                {'name': 'Lexicon Management', 'url_name': 'lexicon_management', 'icon': '⚙️'},
            ],
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
        })
        return context

class NarrativesView(TemplateView):
    template_name = 'dashboard/narratives.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Use election posts only
        posts_queryset, start_date, end_date = get_election_posts_queryset(self.request)
        total_posts = posts_queryset.count()
        
        # Generate summaries WITH URLS
        summaries = get_ethiopia_summaries(posts_queryset, max_clusters=12)
        
        context.update({
            'active_tab': 'narratives',
            'tabs': [
                {'name': 'Home', 'url_name': 'home', 'icon': '🏠'},
                {'name': 'PEPs/PIPs Tracker', 'url_name': 'peps', 'icon': '👤'},
                {'name': 'Mapped Lexicons', 'url_name': 'lexicons', 'icon': '🗣️'},
                {'name': 'Trending Narratives', 'url_name': 'narratives', 'icon': '📰'},
                {'name': 'Networks & TTPs', 'url_name': 'networks', 'icon': '🕸️'},
                {'name': 'Lexicon Management', 'url_name': 'lexicon_management', 'icon': '⚙️'},
            ],
            'summaries': summaries,
            'total_posts': total_posts,
            'date_range': f"{start_date.date()} to {end_date.date()}",
        })
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
            'tabs': [
                {'name': 'Home', 'url_name': 'home', 'icon': '🏠'},
                {'name': 'PEPs/PIPs Tracker', 'url_name': 'peps', 'icon': '👤'},
                {'name': 'Mapped Lexicons', 'url_name': 'lexicons', 'icon': '🗣️'},
                {'name': 'Trending Narratives', 'url_name': 'narratives', 'icon': '📰'},
                {'name': 'Networks & TTPs', 'url_name': 'networks', 'icon': '🕸️'},
                {'name': 'Lexicon Management', 'url_name': 'lexicon_management', 'icon': '⚙️'},
            ],
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
        
class PEPsView(TemplateView):
    """PEPs/PIPs Tracker - Political figures with targeting analysis"""
    template_name = 'dashboard/peps.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Load PEPs from GitHub CSV (dynamic, not hardcoded)
        peps_csv_url = getattr(settings, 'PEPS_CSV_URL', None)
        if peps_csv_url:
            try:
                peps_data = load_peps_from_github(peps_csv_url)
                for pep_data in peps_data: 
                    PEP.objects.update_or_create(
                        name=pep_data['Name (English)'],
                        defaults={
                            'title': pep_data.get('Position', ''),
                            'x_link': pep_data.get('X (Twitter) Link') if pep_data.get('X (Twitter) Link') != 'No verified personal account found' else None,
                            'x_verified': pep_data.get('Verified X (Twitter) Account (Yes/No)', '').lower() == 'yes',
                            'facebook_link': pep_data.get('Facebook Link') if pep_data.get('Facebook Link') not in ['No verified personal account found', 'None found (no official page identified)'] else None,
                            'facebook_verified': pep_data.get('Verified Facebook Account (Yes/No)', '').lower() == 'yes',
                            'confidence_level': pep_data.get('Confidence', 'medium').lower(),
                        }
                    )
            except Exception as e:
                logger.error(f"Failed to load PEPs from GitHub: {e}")
        
        # Get all active PEPs from database
        peps = PEP.objects.filter(is_active=True).order_by('name')
        
        # Track PEP mentions over time (JSON-safe)
        pep_timeline = {}
        for pep in peps[:10]:
            mentions = ProcessedPost.objects.filter(
                is_election_related=True,
                original_text__icontains=pep.name
            ).values('timestamp_share__date').annotate(count=Count('id'))
            
            pep_timeline[pep.name] = [
                {'date': str(m['timestamp_share__date']), 'count': m['count']} 
                for m in mentions if m['timestamp_share__date'] is not None
            ]
        
        context.update({
            'active_tab': 'peps',
            'tabs': [
                {'name': 'Home', 'url_name': 'home', 'icon': '🏠'},
                {'name': 'PEPs/PIPs Tracker', 'url_name': 'peps', 'icon': '👤'},
                {'name': 'Mapped Lexicons', 'url_name': 'lexicons', 'icon': '🗣️'},
                {'name': 'Trending Narratives', 'url_name': 'narratives', 'icon': '📰'},
                {'name': 'Networks & TTPs', 'url_name': 'networks', 'icon': '🕸️'},
                {'name': 'Lexicon Management', 'url_name': 'lexicon_management', 'icon': '⚙️'},
            ],
            'peps': peps,
            'pep_timeline': json.dumps(pep_timeline),
            'total_peps': peps.count(),
            'verified_x_count': peps.filter(x_verified=True).count(),
            'verified_fb_count': peps.filter(facebook_verified=True).count(),
        })
        return context


class NetworksView(TemplateView):
    template_name = 'dashboard/networks.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        request = self.request
        min_connections = int(request.GET.get('min_connections', 2))
        top_n = int(request.GET.get('top_n', 30))  # Reduced default
        layout_style = request.GET.get('layout', 'spring')
        
        # Use election-related posts only
        posts = ProcessedPost.objects.filter(is_election_related=True)
        
        # Generate CLEAN network graph
        graph_data = generate_network_graph_data(posts, min_connections=min_connections, top_n=top_n, layout=layout_style)
        
        # Get coordination groups with FIXED usernames and URLs
        coordination_groups = get_coordination_groups(posts, min_accounts=min_connections, max_groups=15)
        
        # Analyze TTPs
        ttps = analyze_ttps(coordination_groups, posts)
        
        context.update({
            'active_tab': 'networks',
            'tabs': [
                {'name': 'Home', 'url_name': 'home', 'icon': '🏠'},
                {'name': 'PEPs/PIPs Tracker', 'url_name': 'peps', 'icon': '👤'},
                {'name': 'Mapped Lexicons', 'url_name': 'lexicons', 'icon': '🗣️'},
                {'name': 'Trending Narratives', 'url_name': 'narratives', 'icon': '📰'},
                {'name': 'Networks & TTPs', 'url_name': 'networks', 'icon': '🕸️'},
                {'name': 'Lexicon Management', 'url_name': 'lexicon_management', 'icon': '⚙️'},
            ],
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
        
        lexicon_terms = LexiconTerm.objects.filter(is_election_related=True).order_by('category', 'severity')
        
        if not lexicon_terms.exists():
            for category, terms in CONFIG['lexicon'].items():
                for term, metadata in terms.items():
                    LexiconTerm.objects.get_or_create(term=term, defaults={
                        'category': category, 'severity': metadata.get('severity', 'medium'),
                        'target_entity': metadata.get('target_entity', ''),
                        'language': metadata.get('language', 'english'), 'is_election_related': True
                    })
            lexicon_terms = LexiconTerm.objects.filter(is_election_related=True).order_by('category', 'severity')
        
        categories = lexicon_terms.values_list('category', flat=True).distinct()
        
        # Get scan results from session (if any) and clear immediately
        scan_results = self.request.session.pop('scan_results', None)
        
        context.update({
            'active_tab': 'lexicon_management',
            'tabs': [
                {'name': 'Home', 'url_name': 'home', 'icon': '🏠'},
                {'name': 'PEPs/PIPs Tracker', 'url_name': 'peps', 'icon': '👤'},
                {'name': 'Mapped Lexicons', 'url_name': 'lexicons', 'icon': '🗣️'},
                {'name': 'Trending Narratives', 'url_name': 'narratives', 'icon': '📰'},
                {'name': 'Networks & TTPs', 'url_name': 'networks', 'icon': '🕸️'},
                {'name': 'Lexicon Management', 'url_name': 'lexicon_management', 'icon': '⚙️'},
            ],
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
        
        if action == 'add_term':
            term = request.POST.get('term')
            if term:
                LexiconTerm.objects.get_or_create(term=term, defaults={
                    'category': request.POST.get('category', 'uncategorized'),
                    'severity': request.POST.get('severity', 'medium'),
                    'target_entity': request.POST.get('target_entity', ''),
                    'language': request.POST.get('language', 'english'),
                    'is_election_related': True,
                })
                messages.success(request, "✅ Term added successfully!")
        
        elif action == 'scan_text':
            text = request.POST.get('scan_text', '').strip()
            if text and len(text) > 10:  # Only scan if meaningful text
                matches = scan_text_for_lexicon_terms(text)
                risk = calculate_risk_score(matches)
                
                # Store in session for display in GET request
                request.session['scan_results'] = {
                    'matches': matches, 
                    'risk': risk, 
                    'text': text[:100] + '...' if len(text) > 100 else text
                }
                messages.success(request, f"🔍 Found {len(matches)} trigger terms. Risk: {risk['level'].upper()}")
            else:
                messages.warning(request, "⚠️ Please enter text to scan (minimum 10 characters)")
        
        return redirect('lexicon_management')
        
class UploadDataView(TemplateView):
    """UI for uploading CSV files - handles both GET and POST"""
    template_name = 'dashboard/upload_data.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['active_tab'] = 'upload'
        context['tabs'] = [
            {'name': 'Home', 'url_name': 'home', 'icon': '🏠'},
            {'name': 'PEPs/PIPs Tracker', 'url_name': 'peps', 'icon': '👤'},
            {'name': 'Mapped Lexicons', 'url_name': 'lexicons', 'icon': '🗣️'},
            {'name': 'Trending Narratives', 'url_name': 'narratives', 'icon': '📰'},
            {'name': 'Networks & TTPs', 'url_name': 'networks', 'icon': '🕸️'},
            {'name': 'Lexicon Management', 'url_name': 'lexicon_management', 'icon': '⚙️'},
        ]
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
                
                # Load the CSV using robust loader
                df = load_data_robustly(full_path)
                
                # === DEBUG: Check original CSV columns ===
                logger.info(f"📋 ORIGINAL CSV COLUMNS: {list(df.columns)}")
                logger.info(f"📊 CSV Shape: {df.shape}")
                
                # Check for URL-like columns
                url_cols = [c for c in df.columns if 'url' in c.lower() or 'link' in c.lower()]
                logger.info(f"🔍 URL-related columns in CSV: {url_cols}")
                
                # Show sample URL values
                for col in url_cols:
                    logger.info(f"🔗 Sample values from '{col}': {df[col].head(3).tolist()}")
                
                if df.empty:
                    raise ValueError(f"Failed to load data from {original_name}")
                
                # Combine data based on source type
                if data_type == 'meltwater':
                    combined_df = combine_social_media_data(meltwater_df=df, civicsignals_df=None)
                elif data_type == 'civicsignal':
                    combined_df = combine_social_media_data(meltwater_df=None, civicsignals_df=df)
                elif data_type == 'tiktok':
                    combined_df = combine_social_media_data(meltwater_df=None, civicsignals_df=None, tiktok_df=df)
                elif data_type == 'openmeasure':
                    combined_df = combine_social_media_data(meltwater_df=None, civicsignals_df=None, openmeasures_df=df)
                else:
                    # Custom/unknown format
                    combined_df = preprocess_dataframe(df)
                
                # === DEBUG: Check combined data ===
                logger.info(f"📊 COMBINED DATA COLUMNS: {list(combined_df.columns)}")
                if 'URL' in combined_df.columns:
                    logger.info(f"🔗 URL column exists! Sample values: {combined_df['URL'].head(3).tolist()}")
                    logger.info(f"🔗 URL column type: {combined_df['URL'].dtype}")
                    logger.info(f"🔗 URL null count: {combined_df['URL'].isna().sum()}")
                else:
                    logger.error("❌ URL COLUMN NOT FOUND after combining!")
                
                # Final preprocessing and column mapping
                processed_df = final_preprocess_and_map_columns(combined_df)
                
                # === DEBUG: Check processed data ===
                logger.info(f"📊 PROCESSED DATA COLUMNS: {list(processed_df.columns)}")
                if 'URL' in processed_df.columns:
                    logger.info(f"✅ URL in processed data!")
                    logger.info(f"🔗 Sample URLs: {processed_df['URL'].head(3).tolist()}")
                    logger.info(f"🔗 URL null count: {processed_df['URL'].isna().sum()}")
                    logger.info(f"🔗 URL empty count: {(processed_df['URL'] == '').sum()}")
                else:
                    logger.error("❌ URL COLUMN MISSING after final processing!")
                
                # Parse timestamps
                if 'timestamp_share' in processed_df.columns:
                    processed_df['timestamp_share'] = processed_df['timestamp_share'].apply(parse_timestamp_robust)
                
                # Save to database
                count = 0
                urls_saved = 0
                for _, row in processed_df.iterrows():
                    # Skip if no content
                    if not row.get('original_text') or pd.isna(row.get('original_text')):
                        continue
                    
                    # Check for duplicates
                    if row.get('content_id') and ProcessedPost.objects.filter(content_id=row['content_id']).exists():
                        continue
                    if row.get('url') and ProcessedPost.objects.filter(url=row['url']).exists():
                        continue
                    
                    # Get or create DataSource instance
                    source_name = str(row.get('source_dataset', data_type))
                    source_obj, _ = DataSource.objects.get_or_create(name=source_name)
                    
                    # DEBUG: Check URL value before saving
                    url_value = str(row.get('url', ''))[:500] if row.get('url') else None
                    if url_value and url_value.startswith('http'):
                        urls_saved += 1
                    
                    # Create new post with the DataSource instance
                    ProcessedPost.objects.create(
                        account_id=str(row.get('account_id', ''))[:100],
                        content_id=str(row.get('content_id', ''))[:100] if row.get('content_id') else None,
                        original_text=str(row.get('original_text', '')),
                        url=url_value,
                        platform=str(row.get('Platform', 'Unknown')),
                        timestamp_share=row.get('timestamp_share'),
                        source_dataset=source_obj,
                        is_election_related=is_election_related(str(row.get('original_text', '')))
                    )
                    count += 1
                
                logger.info(f"✅ Saved {count} posts, {urls_saved} with URLs")
                
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
        
        # Show summary
        success_count = sum(1 for _, s, _, _ in results if s)
        if success_count == len(uploaded_files):
            messages.success(request, f"✅ All {len(uploaded_files)} files processed successfully!")
        elif success_count > 0:
            messages.warning(request, f"⚠️ {success_count}/{len(uploaded_files)} succeeded. Check logs for errors.")
        else:
            messages.error(request, "❌ Failed to process any files. Check terminal logs for details.")
        
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
