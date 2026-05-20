# dashboard/templatetags/custom_filters.py
from django import template

register = template.Library()

@register.filter
def splitlines(value):
    """Split a string by newlines and return non-empty lines"""
    if not value:
        return []
    return [line.strip() for line in value.split('\n') if line.strip()]
