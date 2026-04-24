from django.db import models
from django.contrib.postgres.fields import ArrayField


class Post(models.Model):
    """Social media post - election-focused"""
    account_id = models.CharField(max_length=255, db_index=True)
    content_id = models.CharField(max_length=255, unique=True, db_index=True)
    object_id = models.TextField()
    url = models.URLField(blank=True, null=True)
    timestamp_share = models.DateTimeField(db_index=True)
    platform = models.CharField(max_length=50, db_index=True)
    source_dataset = models.CharField(max_length=100, db_index=True)
    original_text = models.TextField()
    sentiment = models.CharField(max_length=20, blank=True, null=True)
    cluster = models.IntegerField(default=-1, db_index=True)
    
    # Election-specific
    is_election_related = models.BooleanField(default=False)
    
    # TikTok-specific
    play_count = models.BigIntegerField(null=True, blank=True)
    digg_count = models.BigIntegerField(null=True, blank=True)
    comment_count = models.BigIntegerField(null=True, blank=True)
    share_count = models.BigIntegerField(null=True, blank=True)
    hashtags = ArrayField(models.CharField(max_length=100), blank=True, default=list)
    text_language = models.CharField(max_length=10, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-timestamp_share']
    
    def __str__(self):
        return f"{self.platform} - {self.account_id[:30]}"


class LexiconTerm(models.Model):
    """Hate speech lexicon - election-focused"""
    term = models.CharField(max_length=255, unique=True)
    category = models.CharField(max_length=100, db_index=True)
    severity = models.CharField(
        max_length=20,
        choices=[('low', 'Low'), ('medium', 'Medium'), ('high', 'High'), ('critical', 'Critical')],
        default='medium'
    )
    target_entity = models.CharField(max_length=255, blank=True)
    language = models.CharField(
        max_length=20,
        choices=[('amharic', 'Amharic'), ('english', 'English'), ('oromo', 'Oromo'), ('tigrinya', 'Tigrinya')]
    )
    is_election_related = models.BooleanField(default=True)
    is_active = models.BooleanField(default=True)
    
    def __str__(self):
        return f"{self.term} ({self.severity})"


class PEP(models.Model):
    """Political Exposed Person - election-focused"""
    name = models.CharField(max_length=255, unique=True)
    title = models.CharField(max_length=255, blank=True)
    affiliation = models.CharField(max_length=255, blank=True)
    ethnic_group = models.CharField(max_length=100, blank=True)
    
    # Social media
    x_link = models.URLField(blank=True, null=True)
    x_verified = models.BooleanField(default=False)
    facebook_link = models.URLField(blank=True, null=True)
    facebook_verified = models.BooleanField(default=False)
    confidence_level = models.CharField(
        max_length=20,
        choices=[('high', 'High'), ('medium', 'Medium'), ('low', 'Low'), ('uncertain', 'Uncertain')],
        blank=True
    )
    
    is_active = models.BooleanField(default=True)
    
    class Meta:
        ordering = ['name']
    
    def __str__(self):
        return f"{self.name} - {self.title}"
