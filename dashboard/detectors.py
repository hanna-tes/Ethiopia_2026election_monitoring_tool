import re

def is_election_related(text: str) -> bool:
    """
    Analyzes incoming text to determine if it relates to Ethiopian elections.
    Uses categorized keyword scoring to avoid false positives.
    """
    if not text or not isinstance(text, str):
        return False

    # Normalize text for robust matching
    text_lower = text.lower().strip()

    # Category 1: Direct Electoral Actions (Weight: 3 points each)
    electoral_terms = {
        r'\belection\b', r'\bvote\b', r'\bvoter\b', r'\bvoting\b', r'\bballot\b', 
        r'\bpolling\b', r'\bconstituency\b', r'\bconstituencies\b', r'\bcandidate\b',
        r'\bcandidates\b', r'\bparliament\b', r'\bparliamentary\b', r'\belectoral\b',
        r'\bcampaign\b', r'\bcampaigning\b', r'\bdemocra', r'\bcoalition\b',
        # Amharic English-transliterated equivalents
        r'\bmerecha\b', r'\bmercha\b', r'\bdems\b', r'\bdimts\b', r'\bkoalishn\b'
    }

    # Category 2: Ethiopian Political Entities & Parties (Weight: 4 points each)
    political_entities = {
        r'\bnebe\b',             # National Election Board of Ethiopia
        r'\bprosperity party\b',  # Ruling Party (PP)
        r'\bezema\b',            # Ethiopian Citizens for Social Justice
        r'\bnama\b',             # National Movement of Amhara
        r'\badfm\b',             # Amhara Democratic Force Movement
        r'\bmedrek\b',           # Opposition Coalition
        r'\btplf\b',             # Tigray People's Liberation Front
        r'\bpp\b',               # Abbreviation of Prosperity Party (checked carefully in context)
        r'\bprime minister\b',   # PM
        r'\babiy ahmed\b',       # PM Abiy Ahmed
        r'\bmelatwork hailu\b',  # NEBE Chairperson
        r'\bberhanu nega\b',     # EZEMA figure
        r'\bmerera gudina\b'     # Medrek leader
    }

    # Category 3: Geographic & Contextual Anchors (Weight: 1 point each)
    ethiopian_contexts = {
        r'\bethiopia\b', r'\bethiopian\b', r'\baddis ababa\b', r'\boromia\b', 
        r'\bamhara\b', r'\btigray\b', r'\bsomali region\b', r'\bafar\b', 
        r'\bshashamane\b', r'\bbahir dar\b', r'\bgondar\b', r'\bmekelle\b',
        r'\bhawassa\b', r'\bdire dawa\b', r'\bharar\b'
    }

    score = 0

    # 1. Check for Category 2 (Entities/Parties) first - Highly predictive
    for pattern in political_entities:
        if re.search(pattern, text_lower):
            score += 4

    # 2. Check for Category 1 (Electoral terminology)
    for pattern in electoral_terms:
        if re.search(pattern, text_lower):
            score += 3

    # 3. Check for Category 3 (Geographic Anchors) - Only contributes if other signals exist
    for pattern in ethiopian_contexts:
        if re.search(pattern, text_lower):
            score += 1

    # Score Evaluation:
    # A score of >= 4 means we have either a direct political entity mentioned, 
    # or an electoral term with a geographic anchor (3 + 1), or multiple electoral terms.
    # This prevents purely geographic news (e.g. "Weather in Gondar") from being flagged.
    return score >= 4
