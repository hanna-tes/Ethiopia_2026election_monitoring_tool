from django.contrib import admin
from .models import MonitoringReport

@admin.register(MonitoringReport)
class MonitoringReportAdmin(admin.ModelAdmin):
    list_display = ('title', 'report_category', 'uploaded_at', 'risk_level')
    list_filter = ('report_category', 'risk_level', 'uploaded_at')
    search_fields = ('title', 'summary')
    readonly_fields = ('uploaded_at',)  # Makes timestamp view-only
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('title', 'report_category', 'subtitle', 'source_analyst')
        }),
        ('Content', {
            'fields': ('summary', 'report_includes', 'key_findings')
        }),
        ('Files & Links', {
            'fields': ('report_file', 'full_report_url', 'cover_image')
        }),
        ('Analysis', {
            'fields': ('risk_level', 'mentioned_entities', 'weaponised_narratives', 'actor_spotlight', 'ttp_infrastructure')
        }),
    
    )
