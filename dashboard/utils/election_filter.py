def is_election_related(text):
    """STUB: Check if text is election-related."""
    election_keywords = ['election', 'vote', 'campaign', 'candidate', 'ballot', 'ምርጫ', 'ድምፅ', 'ወቅት']
    if not text: return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in election_keywords)
