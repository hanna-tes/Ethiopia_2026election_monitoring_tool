from django.db import models
from django.contrib.postgres.fields import ArrayField
from django.utils import timezone
import uuid


class DataSource(models.Model):
    """Data source for social media monitoring"""
    name = models.CharField(max_length=100, unique=True)
    url = models.URLField(blank=True, null=True)
    description = models.TextField(blank=True)
    last_updated = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        verbose_name_plural = "Data Sources"
        ordering = ['name']
    
    def __str__(self):
        return self.name


class Post(models.Model):
    """Social media post model - election-focused"""
    account_id = models.CharField(max_length=255, db_index=True)
    content_id = models.CharField(max_length=255, unique=True, db_index=True)
    object_id = models.TextField()  # Raw post content
    url = models.URLField(blank=True, null=True)
    timestamp_share = models.DateTimeField(db_index=True)
    platform = models.CharField(max_length=50, db_index=True)
    source_dataset = models.ForeignKey(DataSource, on_delete=models.CASCADE, null=True)
    original_text = models.TextField()  # Cleaned text for analysis
    sentiment = models.CharField(max_length=20, blank=True, null=True)
    cluster = models.IntegerField(default=-1, db_index=True)
    
    # Election-specific fields
    is_election_related = models.BooleanField(default=False)
    election_keywords = ArrayField(
        models.CharField(max_length=100), 
        blank=True, 
        default=list
    )
    
    # TikTok-specific fields (from your CSV)
    play_count = models.BigIntegerField(null=True, blank=True)
    digg_count = models.BigIntegerField(null=True, blank=True)
    comment_count = models.BigIntegerField(null=True, blank=True)
    share_count = models.BigIntegerField(null=True, blank=True)
    hashtags = ArrayField(models.CharField(max_length=100), blank=True, default=list)
    text_language = models.CharField(max_length=10, blank=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['timestamp_share']),
            models.Index(fields=['platform', 'timestamp_share']),
            models.Index(fields=['cluster', 'timestamp_share']),
            models.Index(fields=['is_election_related', 'timestamp_share']),
        ]
        ordering = ['-timestamp_share']
    
    def __str__(self):
        return f"{self.platform} - {self.account_id[:30]}"


class LexiconTerm(models.Model):
    """Hate speech and trigger term lexicon - election-focused"""
    term = models.CharField(max_length=255, unique=True)
    category = models.CharField(max_length=100, db_index=True)
    severity = models.CharField(
        max_length=20, 
        choices=[
            ('low', 'Low'),
            ('medium', 'Medium'),
            ('high', 'High'),
            ('critical', 'Critical')
        ],
        default='medium'
    )
    target_entity = models.CharField(max_length=255, blank=True)
    language = models.CharField(
        max_length=20,
        choices=[
            ('amharic', 'Amharic'),
            ('english', 'English'),
            ('oromo', 'Oromo'),
            ('tigrinya', 'Tigrinya')
        ]
    )
    is_election_related = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        verbose_name_plural = "Lexicon Terms"
        ordering = ['category', 'severity', 'term']
    
    def __str__(self):
        return f"{self.term} ({self.severity})"


class PEP(models.Model):
    """Political Exposed Person - election-focused"""
    name = models.CharField(max_length=255, unique=True)
    title = models.CharField(max_length=255, blank=True)
    affiliation = models.CharField(max_length=255, blank=True)
    ethnic_group = models.CharField(max_length=100, blank=True)
    position_type = models.CharField(
        max_length=100,
        choices=[
            ('government', 'Government'),
            ('opposition', 'Opposition'),
            ('military', 'Military'),
            ('civil_society', 'Civil Society'),
            ('media', 'Media'),
            ('other', 'Other')
        ],
        blank=True
    )
    is_active = models.BooleanField(default=True)
    notes = models.TextField(blank=True)
    
    # Social media fields (from your CSV sample)
    x_handle = models.CharField(max_length=255, blank=True, null=True)
    x_verified = models.BooleanField(default=False)
    x_link = models.URLField(blank=True, null=True)
    
    facebook_handle = models.CharField(max_length=255, blank=True, null=True)
    facebook_verified = models.BooleanField(default=False)
    facebook_link = models.URLField(blank=True, null=True)
    
    confidence_level = models.CharField(
        max_length=20,
        choices=[
            ('high', 'High'),
            ('medium', 'Medium'),
            ('low', 'Low'),
            ('uncertain', 'Uncertain')
        ],
        blank=True
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "Political Exposed Person"
        verbose_name_plural = "Political Exposed Persons"
        ordering = ['name']
    
    def __str__(self):
        return f"{self.name} - {self.title}"


class PEPMention(models.Model):
    """Mention of a PEP in social media posts - election-focused"""
    pep = models.ForeignKey(PEP, on_delete=models.CASCADE, related_name='mentions')
    post = models.ForeignKey(Post, on_delete=models.CASCADE, related_name='pep_mentions')
    mention_context = models.TextField()
    sentiment = models.CharField(
        max_length=20,
        choices=[
            ('positive', 'Positive'),
            ('negative', 'Negative'),
            ('neutral', 'Neutral'),
            ('mixed', 'Mixed')
        ]
    )
    is_targeted = models.BooleanField(default=False)  # Is PEP being attacked?
    mentioned_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['pep', 'mentioned_at']),
            models.Index(fields=['post', 'pep']),
        ]
        ordering = ['-mentioned_at']
    
    def __str__(self):
        return f"{self.pep.name} - {self.sentiment}"


class NarrativeCluster(models.Model):
    """Cluster of related posts forming a narrative - election-focused"""
    cluster_id = models.IntegerField(unique=True, db_index=True)
    theme = models.CharField(max_length=255)
    total_reach = models.IntegerField(default=0)
    virality_tier = models.CharField(
        max_length=50,
        choices=[
            ('Tier 1: Limited', 'Tier 1: Limited'),
            ('Tier 2: Moderate', 'Tier 2: Moderate'),
            ('Tier 3: High Spread', 'Tier 3: High Spread'),
            ('Tier 4: Viral Emergency', 'Tier 4: Viral Emergency')
        ]
    )
    first_detected = models.DateTimeField()
    last_updated = models.DateTimeField(auto_now=True)
    llm_summary = models.TextField(blank=True)
    is_election_related = models.BooleanField(default=True)
    risk_score = models.FloatField(default=0.0)
    
    class Meta:
        ordering = ['-total_reach', '-last_updated']
    
    def __str__(self):
        return f"Cluster {self.cluster_id} - {self.theme}"


class CoordinationGroup(models.Model):
    """Detected coordination between accounts - election-focused"""
    shared_text = models.TextField()
    account_count = models.IntegerField()
    post_count = models.IntegerField()
    platforms = ArrayField(models.CharField(max_length=50))
    first_seen = models.DateTimeField()
    last_seen = models.DateTimeField()
    coordination_type = models.CharField(
        max_length=100,
        choices=[
            ('High Text Similarity', 'High Text Similarity'),
            ('Multi-Account Amplification', 'Multi-Account Amplification'),
            ('Cross-Platform Coordination', 'Cross-Platform Coordination'),
            ('Temporal Coordination', 'Temporal Coordination'),
            ('Potential Coordination', 'Potential Coordination')
        ]
    )
    max_similarity = models.FloatField()
    is_election_related = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-post_count', '-first_seen']
    
    def __str__(self):
        return f"Coordination Group - {self.account_count} accounts"
