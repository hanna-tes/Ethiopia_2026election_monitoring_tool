def is_election_related(text, keywords=None):
    """Check if text is election-related"""
    if not text:
        return False
    
    if keywords is None:
        keywords = [
            'election', 'vote', 'ballot', 'NEBE', 'fraud', 'protest',
            'boycott', 'tally', 'registration', 'FANO', 'Amhara', 'Tigray',
            'Oromo', 'Prosperity Party', 'Abiy Ahmed', 'ethnic tension'
        ]
    
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)
