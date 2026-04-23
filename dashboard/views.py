from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.views.generic import TemplateView, ListView, DetailView
from django.db.models import Count, Q, F
from django.utils import timezone
from datetime import datetime, timedelta
import pandas as pd
import json

from .models import (
    Post, NarrativeCluster, PEP, PEPMention, 
    LexiconTerm, CoordinationGroup, NetworkNode
)
from .utils.lexicon_engine import scan_for_hate_speech
from .utils.pep_tracker import extract_pep_mentions
from .utils.data_loader import load_election_data, load_peps_from_github
from .utils.election_filter import is_election_related


class HomeView(TemplateView):
    """Executive dashboard home - election-focused"""
    template_name = 'dashboard/home.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get date range from query params or default to last 30 days
        end_date = timezone.now()
        start_date = end_date - timedelta(days=30)
        
        # Election-filtered posts ONLY
        election_posts = Post.objects.filter(
            is_election_related=True,
            timestamp_share__range=[start_date, end_date]
        )
        
        # Key election metrics
        context.update({
            'total_posts': election_posts.count(),
            'hate_speech_count': Post.objects.filter(
                is_election_related=True,
                original_text__icontains='kill'
            ).count(),
            'active_narratives': NarrativeCluster.objects.filter(
                is_election_related=True
            ).count(),
            'peps_tracked': PEP.objects.filter(is_active=True).count(),
            'coordination_groups': CoordinationGroup.objects.filter(
                is_election_related=True
            ).count(),
            'last_update_time': timezone.now().strftime('%Y-%m-%d %H:%M UTC'),
            'tabs': [
                {'name': 'Home', 'url_name': 'home', 'icon': '🏠'},
                {'name': 'Trending Narratives', 'url_name': 'narratives', 'icon': '📰'},
                {'name': 'Mapped Lexicons', 'url_name': 'lexicons', 'icon': '🗣️'},
                {'name': 'PEPs/PIPs Tracker', 'url_name': 'peps', 'icon': '👤'},
                {'name': 'Networks & TTPs', 'url_name': 'networks', 'icon': '🕸️'},
                {'name': 'Lexicon Management', 'url_name': 'lexicon_management', 'icon': '⚙️'},
            ],
            'active_tab': 'home'
        })
        
        return context


class NarrativesView(TemplateView):
    """Trending Narratives - Top narratives with sample posts"""
    template_name = 'dashboard/narratives.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get election-related narratives ONLY
        clusters = NarrativeCluster.objects.filter(
            is_election_related=True
        ).order_by('-total_reach')[:10]
        
        # Get sample posts for each cluster (top 3)
        for cluster in clusters:
            cluster.sample_posts = Post.objects.filter(
                cluster=cluster.cluster_id,
                is_election_related=True
            ).order_by('-timestamp_share')[:3]
        
        context.update({
            'active_tab': 'narratives',
            'tabs': [
                {'name': 'Home', 'url_name': 'home', 'icon': '🏠'},
                {'name': 'Trending Narratives', 'url_name': 'narratives', 'icon': '📰'},
                {'name': 'Mapped Lexicons', 'url_name': 'lexicons', 'icon': '🗣️'},
                {'name': 'PEPs/PIPs Tracker', 'url_name': 'peps', 'icon': '👤'},
                {'name': 'Networks & TTPs', 'url_name': 'networks', 'icon': '🕸️'},
                {'name': 'Lexicon Management', 'url_name': 'lexicon_management', 'icon': '⚙️'},
            ],
            'clusters': clusters,
        })
        
        return context


class LexiconsView(TemplateView):
    """Mapped Lexicons - Hate speech terms with temporal analysis"""
    template_name = 'dashboard/lexicons.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get election-filtered data ONLY
        election_posts = Post.objects.filter(is_election_related=True)
        
        # Scan for lexicon matches in election content only
        all_matches = []
        for post in election_posts:
            matches = scan_for_hate_speech(post.original_text)
            if matches:
                all_matches.extend(matches)
        
        # Temporal trend of hate speech in election content
        hate_timeline = {}
        for match in all_matches:
            post = Post.objects.filter(original_text__contains=match['term']).first()
            if post and post.timestamp_share:
                date_key = post.timestamp_share.strftime('%Y-%m-%d')
                hate_timeline[date_key] = hate_timeline.get(date_key, 0) + 1
        
        context.update({
            'active_tab': 'lexicons',
            'tabs': [
                {'name': 'Home', 'url_name': 'home', 'icon': '🏠'},
                {'name': 'Trending Narratives', 'url_name': 'narratives', 'icon': '📰'},
                {'name': 'Mapped Lexicons', 'url_name': 'lexicons', 'icon': '🗣️'},
                {'name': 'PEPs/PIPs Tracker', 'url_name': 'peps', 'icon': '👤'},
                {'name': 'Networks & TTPs', 'url_name': 'networks', 'icon': '🕸️'},
                {'name': 'Lexicon Management', 'url_name': 'lexicon_management', 'icon': '⚙️'},
            ],
            'hate_timeline': json.dumps(hate_timeline),
            'top_terms': LexiconTerm.objects.filter(
                is_active=True,
                is_election_related=True
            ).values('term', 'category', 'severity').annotate(
                count=Count('id')
            ).order_by('-count')[:10]
        })
        
        return context


class PEPsView(TemplateView):
    """PEPs/PIPs Tracker - Political figures with targeting analysis"""
    template_name = 'dashboard/peps.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Load PEPs from your GitHub CSV (dynamic, not hardcoded)
        peps_csv_url = getattr(settings, 'PEPS_CSV_URL', None)
        if peps_csv_url:
            peps_data = load_peps_from_github(peps_csv_url)
            # Import or update PEPs in database
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
        
        # Get all active PEPs from database
        peps = PEP.objects.filter(is_active=True).order_by('name')
        
        # Track PEP mentions over time in election content ONLY
        pep_timeline = {}
        for pep in peps[:10]:  # Top 10 PEPs
            mentions = PEPMention.objects.filter(
                pep=pep,
                post__is_election_related=True
            ).values('mentioned_at__date').annotate(count=Count('id'))
            
            pep_timeline[pep.name] = list(mentions)
        
        context.update({
            'active_tab': 'peps',
            'tabs': [
                {'name': 'Home', 'url_name': 'home', 'icon': '🏠'},
                {'name': 'Trending Narratives', 'url_name': 'narratives', 'icon': '📰'},
                {'name': 'Mapped Lexicons', 'url_name': 'lexicons', 'icon': '🗣️'},
                {'name': 'PEPs/PIPs Tracker', 'url_name': 'peps', 'icon': '👤'},
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
    """Networks & TTPs - Coordination patterns in election discourse"""
    template_name = 'dashboard/networks.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Election-focused coordination analysis ONLY
        election_groups = CoordinationGroup.objects.filter(
            is_election_related=True
        ).select_related('source', 'target')
        
        context.update({
            'active_tab': 'networks',
            'tabs': [
                {'name': 'Home', 'url_name': 'home', 'icon': '🏠'},
                {'name': 'Trending Narratives', 'url_name': 'narratives', 'icon': '📰'},
                {'name': 'Mapped Lexicons', 'url_name': 'lexicons', 'icon': '🗣️'},
                {'name': 'PEPs/PIPs Tracker', 'url_name': 'peps', 'icon': '👤'},
                {'name': 'Networks & TTPs', 'url_name': 'networks', 'icon': '🕸️'},
                {'name': 'Lexicon Management', 'url_name': 'lexicon_management', 'icon': '⚙️'},
            ],
            'coordination_groups': election_groups,
            'network_stats': {
                'total_nodes': NetworkNode.objects.filter(is_election_related=True).count(),
                'total_edges': NetworkNode.objects.filter(is_election_related=True).count(),
                'avg_connections': 0  # Calculate from actual data
            }
        })
        
        return context


class LexiconManagementView(TemplateView):
    """Lexicon Management - Add/edit hate speech terms"""
    template_name = 'dashboard/lexicon_management.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        context.update({
            'active_tab': 'lexicon_management',
            'tabs': [
                {'name': 'Home', 'url_name': 'home', 'icon': '🏠'},
                {'name': 'Trending Narratives', 'url_name': 'narratives', 'icon': '📰'},
                {'name': 'Mapped Lexicons', 'url_name': 'lexicons', 'icon': '🗣️'},
                {'name': 'PEPs/PIPs Tracker', 'url_name': 'peps', 'icon': '👤'},
                {'name': 'Networks & TTPs', 'url_name': 'networks', 'icon': '🕸️'},
                {'name': 'Lexicon Management', 'url_name': 'lexicon_management', 'icon': '⚙️'},
            ],
            'lexicon_terms': LexiconTerm.objects.filter(
                is_election_related=True
            ).order_by('category', 'severity'),
            'categories': LexiconTerm.objects.filter(
                is_election_related=True
            ).values_list('category', flat=True).distinct()
        })
        
        return context
