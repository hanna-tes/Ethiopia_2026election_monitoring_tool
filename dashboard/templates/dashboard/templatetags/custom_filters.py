from django import template
register = template.Library()

@register.filter
def get_attr(obj, attr):
    """Get attribute from Django model instance"""
    try:
        return getattr(obj, attr, None)
    except:
        return None
