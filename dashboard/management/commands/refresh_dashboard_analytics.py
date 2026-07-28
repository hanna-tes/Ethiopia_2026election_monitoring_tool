import json
from datetime import timedelta

from django.core.cache import cache
from django.core.management.base import BaseCommand
from django.db.models import Max, Min
from django.utils import timezone

from dashboard.models import PEP, ProcessedPost
from dashboard.utils.app_logging import log_event
from dashboard.views import (
    build_analytics_snapshot_key,
    get_category_trend_analysis,
    get_enhanced_pep_analysis,
    get_ethiopia_summaries,
    get_risk_actors_insight,
    get_top_hashtags,
)


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

        log_event(
            'Refreshing dashboard analytics',
            event_type='analytics_refresh',
            status='started',
            source='refresh_dashboard_analytics',
            metadata={'view_all': view_all, 'days': options['days']},
        )

        if not options['skip_home']:
            self.refresh_home_snapshot(posts, start_date, end_date, view_all)

        if not options['skip_narratives']:
            self.refresh_narratives_snapshot(posts, start_date, end_date, view_all)

        if not options['skip_peps']:
            self.refresh_peps_snapshot(posts, start_date, end_date, view_all)

        duration_ms = int((timezone.now() - started_at).total_seconds() * 1000)
        log_event(
            'Dashboard analytics refresh completed',
            event_type='analytics_refresh',
            status='success',
            source='refresh_dashboard_analytics',
            duration_ms=duration_ms,
        )
        self.stdout.write(self.style.SUCCESS(f"Dashboard analytics refreshed in {duration_ms}ms"))

    def refresh_home_snapshot(self, posts, start_date, end_date, view_all):
        self.stdout.write("Refreshing home analytics snapshot...")
        start_key = start_date.date().isoformat() if hasattr(start_date, 'date') else str(start_date)
        end_key = end_date.date().isoformat() if hasattr(end_date, 'date') else str(end_date)
        payload = {
            'risk_actors': get_risk_actors_insight(posts),
            'top_hashtags': get_top_hashtags(posts),
            'trend_analysis': get_category_trend_analysis(posts, days_back=90, cache_suffix=f"{start_key}_{end_key}"),
        }
        self.save_snapshot('home', start_date, end_date, view_all, payload)

    def refresh_narratives_snapshot(self, posts, start_date, end_date, view_all):
        self.stdout.write("Refreshing narrative summaries snapshot...")
        payload = {
            'summaries': get_ethiopia_summaries(posts),
        }
        self.save_snapshot('narratives', start_date, end_date, view_all, payload)

    def refresh_peps_snapshot(self, posts, start_date, end_date, view_all):
        self.stdout.write("Refreshing PEP mention analysis snapshot...")
        active_peps = PEP.objects.filter(is_active=True).order_by('name')
        election_posts = posts.filter(is_election_related=True).order_by('-timestamp_share')
        payload = {
            'pep_analysis': get_enhanced_pep_analysis(election_posts, active_peps, limit=8),
        }
        self.save_snapshot('peps', start_date, end_date, view_all, payload)

    def save_snapshot(self, kind, start_date, end_date, view_all, payload):
        key = build_analytics_snapshot_key(kind, start_date, end_date, view_all=view_all)
        snapshot_data = {
            'payload': json_safe(payload),
            'generated_at': timezone.now().isoformat(),
        }
        cache.set(key, snapshot_data, timeout=86400)
        self.stdout.write(self.style.SUCCESS(f"Saved {kind} snapshot to cache key: {key}"))
