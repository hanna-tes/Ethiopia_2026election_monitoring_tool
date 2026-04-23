from django.db import models
from django.contrib.postgres.fields import ArrayField

class DataSource(models.Model):
    name = models.CharField(max_length=100)  # Meltwater, TikTok, etc.
    url = models.URLField(blank=True)
    last_updated = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.name

class Post(models.Model):
    account_id = models.CharField(max_length=255)
    content_id = models.CharField(max_length=255, unique=True)
    object_id = models.TextField()  # Post content
    url = models.URLField(blank=True)
    timestamp_share = models.DateTimeField()
    platform = models.CharField(max_length=50)  # X, Telegram, TikTok
    source_dataset = models.ForeignKey(DataSource, on_delete=models.CASCADE)
    original_text = models.TextField()  # Cleaned text for analysis
    sentiment = models.CharField(max_length=20, blank=True)  # Negative, Neutral
    cluster = models.IntegerField(default=-1)  # Narrative cluster ID
    
    # Election-specific fields
    is_election_related = models.BooleanField(default=False)
    election_keywords = ArrayField(models.CharField(max_length=100), blank=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['timestamp_share']),
            models.Index(fields=['platform', 'timestamp_share']),
            models.Index(fields=['cluster', 'timestamp_share']),
        ]

class LexiconTerm(models.Model):
    term = models.CharField(max_length=255, unique=True)
    category = models.CharField(max_length=100)  # ethnic_identity, violence_incitement
    severity = models.CharField(max_length=20)  # low, medium, high, critical
    target_entity = models.CharField(max_length=255, blank=True)
    language = models.CharField(max_length=20)  # amharic, english
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.term} ({self.severity})"

class PEP(models.Model):
    name = models.CharField(max_length=255, unique=True)
    title = models.CharField(max_length=255, blank=True)  # Prime Minister, etc.
    affiliation = models.CharField(max_length=255, blank=True)  # Prosperity Party
    ethnic_group = models.CharField(max_length=100, blank=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        verbose_name = "Political Exposed Person"
        verbose_name_plural = "Political Exposed Persons"

class PEPMention(models.Model):
    pep = models.ForeignKey(PEP, on_delete=models.CASCADE)
    post = models.ForeignKey(Post, on_delete=models.CASCADE)
    mention_context = models.TextField()  # Snippet showing how PEP was mentioned
    sentiment = models.CharField(max_length=20)  # Positive, Negative, Neutral
    is_targeted = models.BooleanField(default=False)  # Is PEP being attacked?
    mentioned_at = models.DateTimeField(auto_now_add=True)

class NarrativeCluster(models.Model):
    cluster_id = models.IntegerField(unique=True)
    theme = models.CharField(max_length=255)  # "Election fraud allegations"
    total_reach = models.IntegerField()
    virality_tier = models.CharField(max_length=50)  # Tier 1, Tier 2, etc.
    first_detected = models.DateTimeField()
    last_updated = models.DateTimeField(auto_now=True)
    llm_summary = models.TextField(blank=True)  # AI-generated summary
    is_election_related = models.BooleanField(default=True)
    
    class Meta:
        ordering = ['-total_reach']

class CoordinationGroup(models.Model):
    shared_text = models.TextField()
    account_count = models.IntegerField()
    post_count = models.IntegerField()
    platforms = ArrayField(models.CharField(max_length=50))
    first_seen = models.DateTimeField()
    last_seen = models.DateTimeField()
    coordination_type = models.CharField(max_length=100)  # "High Text Similarity", etc.
    max_similarity = models.FloatField()
