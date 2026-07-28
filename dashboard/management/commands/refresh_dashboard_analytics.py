import json
import logging
from datetime import timedelta

from django.core.cache import cache
from django.core.management.base import BaseCommand
from django.db.models import Max, Min
from django.utils import timezone

from dashboard.models import PEP, ProcessedPost
from dashboard.utils.analytics_engine import (
    compute_filtered_analytics,
    get_filter_cache_key,
)

logger = logging.getLogger(__name__)


def json_safe(value):
    return json.loads(json.dumps(value, default=str))


class Command(BaseCommand):
    help = "Refresh precomputed dashboard analytics used by slow browsing pages."

    def add_arguments(self, parser):
        parser.add_argument('--days', type=int, default=90, help='Number of recent days to precompute.')
        parser.add_argument('--view-all', action='store_true', help='Precompute using all available posts.')
        parser.add_argument('--skip-home', action='store_true', help='Skip home analytics snapshot.')
        parser.add_argument('--skip-narratives', action='store_true', help='Skip narrative summaries snapshot.')
        parser.add_argument('--skip-peps', action='store_true', help='Skip PEP mention analysis snapshot.')

    def handle(self, *args, **options):
        started_at = timezone.now()
        view_all = options['view_all']
        posts = ProcessedPost.objects.all().order_by('-timestamp_share')

        if view_all:
            date_range = posts.aggregate(min_date=Min('timestamp_share'), max_date=Max('timestamp_share'))
            start_date = date_range['min_date'] or started_at
            end_date = date_range['max_date'] or started_at
        else:
            end_date = started_at
            start_date = end_date - timedelta(days=options['days'])
            posts = posts.filter(timestamp_share__gte=start_date)

        logger.info(f"Refreshing dashboard analytics | view_all={view_all} | days={options['days']}")

        # 1. Refresh Home Snapshot
        if not options['skip_home']:
            self.refresh_home_snapshot(start_date, end_date, view_all)

        # 2. Refresh Narratives Snapshot
        if not options['skip_narratives']:
            self.refresh_narratives_snapshot(start_date, end_date, view_all)

        # 3. Refresh PEPs Snapshot
        if not options['skip_peps']:
            self.refresh_peps_snapshot(start_date, end_date, view_all)

        duration_ms = int((timezone.now() - started_at).total_seconds() * 1000)
        logger.info(f"Dashboard analytics refresh completed in {duration_ms}ms")
        self.stdout.write(self.style.SUCCESS(f"Dashboard analytics refreshed in {duration_ms}ms"))

    def refresh_home_snapshot(self, start_date, end_date, view_all):
        self.stdout.write("Refreshing home analytics snapshot...")
        params = {
            'start_date': start_date.isoformat() if hasattr(start_date, 'isoformat') else str(start_date),
            'end_date': end_date.isoformat() if hasattr(end_date, 'isoformat') else str(end_date),
            'view_all': 'true' if view_all else 'false'
        }
        payload = compute_filtered_analytics('home', params)
        self.save_snapshot('home', params, payload)

    def refresh_narratives_snapshot(self, start_date, end_date, view_all):
        self.stdout.write("Refreshing narrative summaries snapshot...")
        params = {
            'start_date': start_date.isoformat() if hasattr(start_date, 'isoformat') else str(start_date),
            'end_date': end_date.isoformat() if hasattr(end_date, 'isoformat') else str(end_date),
            'view_all': 'true' if view_all else 'false'
        }
        payload = compute_filtered_analytics('narratives', params)
        self.save_snapshot('narratives', params, payload)

    def refresh_peps_snapshot(self, start_date, end_date, view_all):
        self.stdout.write("Refreshing PEP mention analysis snapshot...")
        # Import dynamically if PEP analysis is handled via helper to break cycles
        from dashboard.views import get_enhanced_pep_analysis
        
        active_peps = PEP.objects.filter(is_active=True).order_by('name')
        election_posts = ProcessedPost.objects.filter(is_election_related=True).order_by('-timestamp_share')
        
        if not view_all:
            start_date_calc = timezone.now() - timedelta(days=90)
            election_posts = election_posts.filter(timestamp_share__gte=start_date_calc)

        params = {
            'start_date': start_date.isoformat() if hasattr(start_date, 'isoformat') else str(start_date),
            'end_date': end_date.isoformat() if hasattr(end_date, 'isoformat') else str(end_date),
            'view_all': 'true' if view_all else 'false'
        }
        
        payload = {
            'pep_analysis': get_enhanced_pep_analysis(election_posts, active_peps, limit=8),
        }
        self.save_snapshot('peps', params, payload)

    def save_snapshot(self, kind, params, payload):
        key = get_filter_cache_key(kind, params)
        snapshot_data = {
            'payload': json_safe(payload),
            'generated_at': timezone.now().isoformat(),
        }
        cache.set(key, snapshot_data, timeout=86400)
        self.stdout.write(self.style.SUCCESS(f"Saved {kind} snapshot to cache key: {key}"))
