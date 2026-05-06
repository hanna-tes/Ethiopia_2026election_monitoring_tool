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

logger = logging.getLogger(__name__)

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


def assign_virality_tier(n):
    """Assign virality tier based on post count"""
    if n >= 500:
        return "Tier 4: Viral Emergency"
    elif n >= 100:
        return "Tier 3: High Spread"
    elif n >= 20:
        return "Tier 2: Moderate"
    else:
        return "Tier 1: Limited"


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
            # List of known non-user values to filter out
            invalid_accounts = ['twitter', 'source', 'source twitter source', 'nan', 'none', '-', '', 'user', 'author', 'account']

            for acc in top_accounts_raw:
                name = str(acc['account_id']) if acc['account_id'] else ''
                
                # 1. Remove specific artifacts
                name = re.sub(r'Twitter Source\s*', '', name, flags=re.IGNORECASE)
                name = re.sub(r'Source Twitter Source\s*', '', name, flags=re.IGNORECASE)
                name = re.sub(r'@\w+\s*Name:\s*\d+.*', '', name)  # Remove @mentions with metadata
                name = re.sub(r'dtype.*', '', name, flags=re.IGNORECASE)  # Remove pandas dtype info
                
                # 2. Clean whitespace
                name = re.sub(r'\s+', ' ', name).strip()
                
                # 3. Filter out invalid accounts (case-insensitive check)
                if name.lower() in invalid_accounts:
                    continue
                
                # 4. Add to list if valid
                if name and name not in ['-', 'nan', 'None', '']:
                    cleaned_accounts.append({'account_id': name[:50], 'count': acc['count']})
            
            # Create chart if we have data
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
                {'name': 'Upload Data', 'url_name': 'upload_data', 'icon': '📤'},
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
    """Trending Narratives - Perform clustering and show narratives"""
    template_name = 'dashboard/narratives.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get all posts
        posts = ProcessedPost.objects.all()
        total_posts = posts.count()
        
        # Perform clustering on the fly (simplified version)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
        import numpy as np
        
        clusters = []
        narrative_summaries = []
        
        if total_posts > 0:
            # Get post texts
            texts = list(posts.values_list('original_text', flat=True)[:1000])  # Limit for performance
            
            if len(texts) > 10:  # Need minimum posts for clustering
                # Vectorize texts
                vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                try:
                    X = vectorizer.fit_transform(texts)
                    
                    # Determine number of clusters
                    n_clusters = min(10, len(texts) // 10)
                    
                    # Perform clustering
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(X)
                    
                    # Group posts by cluster
                    from collections import defaultdict
                    cluster_posts = defaultdict(list)
                    for idx, label in enumerate(cluster_labels):
                        cluster_posts[label].append(texts[idx])
                    
                    # Create narrative summaries
                    for cluster_id, cluster_texts in cluster_posts.items():
                        if len(cluster_texts) >= 3:  # Only show clusters with 3+ posts
                            # Get top terms for this cluster
                            cluster_terms = ' '.join(cluster_texts[:20])
                            
                            clusters.append({
                                'cluster_id': cluster_id,
                                'total_reach': len(cluster_texts),
                                'virality_tier': self._assign_virality(len(cluster_texts)),
                                'theme': f'Narrative Cluster #{cluster_id}',
                                'llm_summary': f'This cluster contains {len(cluster_texts)} posts discussing similar topics. Top terms: {cluster_terms[:100]}...',
                                'sample_posts': cluster_texts[:5]
                            })
                    
                    # Sort by reach
                    clusters.sort(key=lambda x: x['total_reach'], reverse=True)
                    
                except Exception as e:
                    logger.error(f"Clustering failed: {e}")
        
        context.update({
            'active_tab': 'narratives',
            'tabs': [
                {'name': 'Home', 'url_name': 'home', 'icon': '🏠'},
                {'name': 'Upload Data', 'url_name': 'upload_data', 'icon': '📤'},
                {'name': 'PEPs/PIPs Tracker', 'url_name': 'peps', 'icon': '👤'},
                {'name': 'Mapped Lexicons', 'url_name': 'lexicons', 'icon': '🗣️'},
                {'name': 'Trending Narratives', 'url_name': 'narratives', 'icon': '📰'},
                {'name': 'Networks & TTPs', 'url_name': 'networks', 'icon': '🕸️'},
                {'name': 'Lexicon Management', 'url_name': 'lexicon_management', 'icon': '⚙️'},
            ],
            'clusters': clusters,
            'total_posts': total_posts,
        })
        return context
    
    def _assign_virality(self, n):
        if n >= 500: return "Tier 4: Viral Emergency"
        elif n >= 100: return "Tier 3: High Spread"
        elif n >= 20: return "Tier 2: Moderate"
        else: return "Tier 1: Limited"



class LexiconsView(TemplateView):
    """Mapped Lexicons - Scan uploaded posts for lexicon matches"""
    template_name = 'dashboard/lexicons.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get ALL posts from database
        posts = ProcessedPost.objects.all()
        total_posts = posts.count()
        
        # Scan for lexicon matches
        all_matches = []
        posts_scanned = 0
        
        for post in posts[:2000]:  # Limit for performance
            if post.original_text:
                matches = scan_text_for_lexicon_terms(post.original_text)
                if matches:
                    all_matches.extend(matches)
                    posts_scanned += 1
        
        # Aggregate by term
        from collections import Counter
        term_counts = Counter([m['term'] for m in all_matches])
        top_terms = term_counts.most_common(15)
        
        # Category counts
        category_counts = Counter([m['category'] for m in all_matches])
        
        # Severity distribution
        severity_counts = Counter([m['severity'] for m in all_matches])
        
        context.update({
            'active_tab': 'lexicons',
            'tabs': [
                {'name': 'Home', 'url_name': 'home', 'icon': '🏠'},
                {'name': 'Upload Data', 'url_name': 'upload_data', 'icon': '📤'},
                {'name': 'PEPs/PIPs Tracker', 'url_name': 'peps', 'icon': '👤'},
                {'name': 'Mapped Lexicons', 'url_name': 'lexicons', 'icon': '🗣️'},
                {'name': 'Trending Narratives', 'url_name': 'narratives', 'icon': '📰'},
                {'name': 'Networks & TTPs', 'url_name': 'networks', 'icon': '🕸️'},
                {'name': 'Lexicon Management', 'url_name': 'lexicon_management', 'icon': '⚙️'},
            ],
            'top_terms': [{'term': term, 'count': count, 'metadata': next(
                (t for cat in CONFIG['lexicon'].values() for t in [v for k,v in cat.items() if k==term]),
                {}
            )} for term, count in top_terms],
            'category_counts': dict(category_counts),
            'severity_counts': dict(severity_counts),
            'total_matches': len(all_matches),
            'posts_scanned': posts_scanned,
            'total_posts': total_posts,
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
                {'name': 'Upload Data', 'url_name': 'upload_data', 'icon': '📤'},
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
    """Networks & TTPs - Coordination patterns with interactive visualization"""
    template_name = 'dashboard/networks.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        posts = ProcessedPost.objects.all()
        
        # Find coordination (accounts posting identical text)
        coordination = posts.values('original_text').annotate(
            account_count=Count('account_id', distinct=True),
            post_count=Count('id')
        ).filter(account_count__gte=3).order_by('-account_count')[:20]
        
        # Calculate stats
        total_accounts = posts.values('account_id').distinct().count()
        total_posts_count = posts.count()
        avg_accounts = round(coordination.aggregate(avg=Count('account_id'))['avg'] or 0, 1)
        
        # === GENERATE NETWORK GRAPH DATA ===
        import networkx as nx
        import random
        from django.utils import timezone
        
        G = nx.Graph()
        
        # Build graph from coordination data
        for group in coordination:
            text = group['original_text']
            accounts_with_text = posts.filter(original_text=text).values_list('account_id', flat=True).distinct()
            
            # Add edges between all accounts sharing this text
            accounts_list = list(accounts_with_text)
            for i in range(len(accounts_list)):
                for j in range(i+1, len(accounts_list)):
                    if G.has_edge(accounts_list[i], accounts_list[j]):
                        G[accounts_list[i]][accounts_list[j]]['weight'] += 1
                    else:
                        G.add_edge(accounts_list[i], accounts_list[j], weight=1)
        
        # Generate layout if graph has edges
        coordination_data_json = '{"nodes": [], "edges": []}'
        coordination_groups_list = []
        
        if G.number_of_edges() > 0:
            # Get positions using spring layout
            pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
            
            # Prepare node data
            nodes = []
            for node in G.nodes():
                degree = G.degree(node)
                if degree >= 2:  # Only show nodes with 2+ connections
                    # Get platform distribution for this account
                    node_posts = posts.filter(account_id=node)
                    platforms = node_posts.values_list('platform', flat=True)
                    platform_mode = max(set(platforms), key=platforms.count) if platforms else 'Unknown'
                    
                    nodes.append({
                        'id': str(node)[:50],
                        'label': str(node)[:30],
                        'degree': degree,
                        'post_count': node_posts.count(),
                        'platform': platform_mode,
                        'x': float(pos[node][0]),
                        'y': float(pos[node][1]),
                        'color': self._get_platform_color(platform_mode)
                    })
            
            # Prepare edge data
            edges = []
            for u, v, data in G.edges(data=True):
                if u in pos and v in pos:
                    # Get sample posts for this edge
                    shared_texts = posts.filter(
                        account_id__in=[u, v]
                    ).values('original_text').annotate(
                        count=Count('id')
                    ).filter(count__gte=2)
                    
                    edges.append({
                        'source': str(u)[:50],
                        'target': str(v)[:50],
                        'weight': data.get('weight', 1),
                        'source_x': float(pos[u][0]),
                        'source_y': float(pos[u][1]),
                        'target_x': float(pos[v][0]),
                        'target_y': float(pos[v][1]),
                        'shared_messages': len(shared_texts)
                    })
            
            coordination_data_json = json.dumps({'nodes': nodes, 'edges': edges})
            
            # Prepare coordination groups with sample posts
            for idx, group in enumerate(coordination[:10]):
                text = group['original_text']
                accounts = list(posts.filter(original_text=text).values_list('account_id', flat=True).distinct()[:10])
                sample_posts = posts.filter(original_text=text)[:5]
                
                coordination_groups_list.append({
                    'id': idx + 1,
                    'accounts': accounts,
                    'account_count': group['account_count'],
                    'post_count': group['post_count'],
                    'text_sample': text[:200] if text else '[Identical message]',
                    'platforms': list(posts.filter(original_text=text).values_list('platform', flat=True).distinct()),
                    'sample_posts': [
                        {
                            'account_id': str(p.account_id)[:50],
                            'platform': p.platform,
                            'url': p.url,
                            'timestamp': p.timestamp_share.strftime('%Y-%m-%d %H:%M') if p.timestamp_share else 'N/A',
                            'text': p.original_text[:150]
                        }
                        for p in sample_posts
                    ]
                })
        
        context.update({
            'active_tab': 'networks',
            'tabs': [
                {'name': 'Home', 'url_name': 'home', 'icon': '🏠'},
                {'name': 'Upload Data', 'url_name': 'upload_data', 'icon': '📤'},
                {'name': 'PEPs/PIPs Tracker', 'url_name': 'peps', 'icon': '👤'},
                {'name': 'Mapped Lexicons', 'url_name': 'lexicons', 'icon': '🗣️'},
                {'name': 'Trending Narratives', 'url_name': 'narratives', 'icon': '📰'},
                {'name': 'Networks & TTPs', 'url_name': 'networks', 'icon': '🕸️'},
                {'name': 'Lexicon Management', 'url_name': 'lexicon_management', 'icon': '⚙️'},
            ],
            'coordination_groups': coordination_groups_list,
            'coordination_data_json': coordination_data_json,
            'total_coordinated': coordination.count(),
            'total_accounts': total_accounts,
            'total_posts': total_posts_count,
            'avg_accounts': avg_accounts,
            'max_connections': max([G.degree(n) for n in G.nodes()]) if G.nodes() else 0,
        })
        return context
    
    def _get_platform_color(self, platform):
        """Get color hex code for platform"""
        colors = {
            'X': '#1DA1F2',
            'Twitter': '#1DA1F2',
            'Facebook': '#1877F2',
            'TikTok': '#000000',
            'Telegram': '#0088cc',
            'Media': '#6B7280',
            'News': '#6B7280',
            'Unknown': '#9CA3AF'
        }
        return colors.get(platform, '#9CA3AF')
        

class LexiconManagementView(TemplateView):
    """Lexicon Management - Add/edit hate speech terms"""
    template_name = 'dashboard/lexicon_management.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get lexicon terms from database (or fallback to CONFIG)
        lexicon_terms = LexiconTerm.objects.filter(
            is_election_related=True
        ).order_by('category', 'severity')
        
        # If no terms in DB, use CONFIG as fallback
        if not lexicon_terms.exists():
            # This would populate the DB on first run
            for category, terms in CONFIG['lexicon'].items():
                for term, metadata in terms.items():
                    LexiconTerm.objects.get_or_create(
                        term=term,
                        defaults={
                            'category': category,
                            'severity': metadata.get('severity', 'medium'),
                            'target_entity': metadata.get('target_entity', ''),
                            'language': metadata.get('language', 'english'),
                            'is_election_related': True,
                        }
                    )
            lexicon_terms = LexiconTerm.objects.filter(
                is_election_related=True
            ).order_by('category', 'severity')
        
        categories = lexicon_terms.values_list('category', flat=True).distinct()
        
        context.update({
            'active_tab': 'lexicon_management',
            'tabs': [
                {'name': 'Home', 'url_name': 'home', 'icon': '🏠'},
                {'name': 'Upload Data', 'url_name': 'upload_data', 'icon': '📤'},
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
        })
        return context

    def post(self, request, *args, **kwargs):
        """Handle adding new lexicon terms from the form"""
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
        return redirect('lexicon_management')


class UploadDataView(TemplateView):
    """UI for uploading CSV files"""
    template_name = 'dashboard/upload_data.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['active_tab'] = 'upload'
        context['tabs'] = [
            {'name': 'Home', 'url_name': 'home', 'icon': '🏠'},
            {'name': 'Upload Data', 'url_name': 'upload_data', 'icon': '📤'},
            {'name': 'PEPs/PIPs Tracker', 'url_name': 'peps', 'icon': '👤'},
            {'name': 'Mapped Lexicons', 'url_name': 'lexicons', 'icon': '🗣️'},
            {'name': 'Trending Narratives', 'url_name': 'narratives', 'icon': '📰'},
            {'name': 'Networks & TTPs', 'url_name': 'networks', 'icon': '🕸️'},
            {'name': 'Lexicon Management', 'url_name': 'lexicon_management', 'icon': '⚙️'},
        ]
        context['recent_uploads'] = DataUpload.objects.order_by('-uploaded_at')[:10]
        return context


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
                    original_filename=original_name,  # Keep original name for display
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
        
class ClearDataView(View):
    """Clear all uploaded data from database"""
    def post(self, request):
        ProcessedPost.objects.all().delete()
        DataUpload.objects.all().delete()
        messages.success(request, "✅ All data cleared successfully. You can now upload fresh data.")
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
