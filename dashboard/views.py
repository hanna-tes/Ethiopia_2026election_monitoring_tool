from django.shortcuts import render
from django.views.generic import TemplateView
from django.utils import timezone
from datetime import timedelta
import json

from .models import Post, NarrativeCluster, PEP, LexiconTerm
from .utils.llm_service import safe_llm_call
from .utils.data_loader import load_peps_from_github
from .utils.election_filter import is_election_related


class HomeView(TemplateView):
    """Executive dashboard - election-focused"""
    template_name = 'dashboard/home.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Election-filtered metrics (simplified - add your logic)
        context.update({
            'total_posts': Post.objects.filter(is_election_related=True).count(),
            'active_narratives': NarrativeCluster.objects.filter(is_election_related=True).count(),
            'peps_tracked': PEP.objects.filter(is_active=True).count(),
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
        
        clusters = NarrativeCluster.objects.filter(
            is_election_related=True
        ).order_by('-total_reach')[:10]
        
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
            'top_terms': LexiconTerm.objects.filter(
                is_active=True, is_election_related=True
            ).values('term', 'category', 'severity').order_by('-id')[:10]
        })
        return context


class PEPsView(TemplateView):
    """PEPs/PIPs Tracker - Political figures with targeting analysis"""
    template_name = 'dashboard/peps.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Load PEPs from GitHub CSV (dynamic)
        peps_csv_url = getattr(settings, 'PEPS_CSV_URL', None)
        if peps_csv_url:
            peps_data = load_peps_from_github(peps_csv_url)
            for pep_data in peps_data:
                PEP.objects.update_or_create(
                    name=pep_data['Name (English)'],
                    defaults={
                        'title': pep_data.get('Position', ''),
                        'x_link': pep_data.get('X (Twitter) Link') if pep_data.get('X (Twitter) Link') != 'No verified personal account found' else None,
                        'x_verified': pep_data.get('Verified X (Twitter) Account (Yes/No)', '').lower() == 'yes',
                        'confidence_level': pep_data.get('Confidence', 'medium').lower(),
                    }
                )
        
        peps = PEP.objects.filter(is_active=True).order_by('name')
        
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
            'total_peps': peps.count(),
        })
        return context


class NetworksView(TemplateView):
    """Networks & TTPs - Coordination patterns"""
    template_name = 'dashboard/networks.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
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
        })
        return context
