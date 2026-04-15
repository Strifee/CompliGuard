import re

# ---------------------------------------------------------------------------
# Structural marker patterns — ordered from broadest to most specific.
# Each pattern is matched against the full raw text; we collect all match
# positions and scan linearly to maintain a running hierarchy context.
# ---------------------------------------------------------------------------

_MARKERS = [
    # LIVRE I / II / III / IV  (standalone line)
    ("livre", re.compile(r"(?m)^\s*(LIVRE\s+[IVX]+)\s*$")),

    # Document-level headers: Loi, Dahir, Décret, Arrêté, Règlement, Circulaire
    # We exclude table-of-contents lines (ending with ......N)
    ("document", re.compile(
        r"(?m)^\s*("
        r"Dahir[^\n]{3,120}"
        r"|Loi\s+n[°o]\s*\d[^\n]{3,120}"
        r"|D[eé]cret\s+n[°o]\s*\d[^\n]{3,120}"
        r"|Arr[eê]t[eé]\s+n[°o]\s*\d[^\n]{3,120}"
        r"|R[eè]glement\s+[Gg][eé]n[eé]ral[^\n]{0,120}"
        r"|Circulaire\s+de\s+[^\n]{3,120}"
        r")\s*$"
    )),

    # TITRE I / II / ...  (standalone line, any case)
    ("titre", re.compile(r"(?m)^\s*(TITRE\s+(?:[IVX]+|PREMIER))\s*$", re.IGNORECASE)),

    # CHAPITRE / Chapitre PREMIER / I / II ...  (standalone line)
    ("chapitre", re.compile(r"(?m)^\s*((?:CHAPITRE|Chapitre)\s+(?:[IVX]+|PREMIER))\s*$")),

    # Article — standard "Article 12" and Circulaire-style "Article I.1.1"
    ("article", re.compile(r"(?m)^\s*((?:Article|Art\.)\s+[IVX\d][A-Za-z\.\-\d]*(?:er)?)\b")),
]

# Levels in descending order of breadth — resetting a level clears everything below it
_LEVEL_ORDER = ["livre", "document", "titre", "chapitre"]


def split_by_articles(raw_text: str) -> list[dict]:
    """
    Parse raw text into article-level sections, each carrying its full
    hierarchical context:
        livre → document → titre → chapitre → article_ref → text

    Strategy: collect all structural marker positions, sort by offset,
    then scan linearly maintaining a running context dict.  When a
    higher-level marker appears it resets all levels below it.
    """
    all_matches = []

    for level, rx in _MARKERS:
        for m in rx.finditer(raw_text):
            label = " ".join(m.group(1).split())  # normalise whitespace

            # Drop table-of-contents lines (e.g. "Loi n° 43-12 ........... 15")
            if level == "document" and re.search(r"\.{3,}\s*\d+\s*$", label):
                continue

            all_matches.append((m.start(), m.end(), level, label))

    all_matches.sort(key=lambda x: x[0])

    context: dict = {k: None for k in _LEVEL_ORDER}
    sections: list[dict] = []
    current_article_ref: str | None = None
    current_text_start: int = 0

    for pos, end, level, label in all_matches:
        if level == "article":
            # Flush the previous article before starting a new one
            if current_article_ref is not None:
                article_text = raw_text[current_text_start:pos].strip()
                if article_text:
                    sections.append({**context, "article_ref": current_article_ref, "text": article_text})
            current_article_ref = label
            current_text_start = end
        else:
            # Update context and reset all levels below the one that changed
            level_idx = _LEVEL_ORDER.index(level)
            context[level] = label
            for lower in _LEVEL_ORDER[level_idx + 1:]:
                context[lower] = None

    # Flush the last article
    if current_article_ref is not None:
        article_text = raw_text[current_text_start:].strip()
        if article_text:
            sections.append({**context, "article_ref": current_article_ref, "text": article_text})

    if not sections:
        return [{"livre": None, "document": None, "titre": None,
                 "chapitre": None, "article_ref": None, "text": raw_text.strip()}]

    return sections
