from dashboard.models import ProcessedPost
from collections import Counter

counts = Counter(ProcessedPost.objects.values_list('platform', flat=True))
print(f"\n{len(counts)} distinct platform value(s):\n")
for val, n in counts.most_common():
    print(f"  {repr(val):30s} count={n}")

try:
    from dashboard.views import get_coordination_groups
    qs = ProcessedPost.objects.filter(is_election_related=True)
    print(f"\nTotal election-related posts: {qs.count()}")

    groups = get_coordination_groups(qs, min_accounts=3, max_groups=50, view_all=True)
    print(f"Coordination groups found: {len(groups)}\n")

    for g in groups[:15]:
        plats = sorted({p.get('platform') for p in g.get('sample_posts_with_urls', [])
                         if p.get('platform')})
        first_user = (g.get('sample_posts_with_urls') or [{}])[0].get('username', '?')
        print(f"  group #{g['id']:<3} accounts={g['account_count']:<4} "
              f"posts={g['post_count']:<5} platforms={plats} "
              f"first_user={first_user}")
except Exception as e:
    print(f"\n[coordination group check skipped: {e}]")
