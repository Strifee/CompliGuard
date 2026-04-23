import re

# ---------------------------------------------------------------------------
# Structural marker patterns for the AMF France Règlement Général.
# Hierarchy: LIVRE → TITRE → CHAPITRE → Section → Article
#
# Article numbering in the AMF RG follows the pattern:
#   Article XXX-Y  where the first 3 digits encode LIVRE/TITRE/CHAPITRE
#   e.g. Article 111-1 = Livre I, Titre I, Chapitre I, Art. 1
#        Article 212-4 = Livre II, Titre I, Chapitre II, Art. 4
# ---------------------------------------------------------------------------

_MARKERS = [
    # LIVRE I / II / ... / VI  (standalone line, any case)
    ("livre", re.compile(
        r"(?m)^\s*(LIVRE\s+(?:[IVX]+|PREMIER|UNIQUE))\s*$", re.IGNORECASE
    )),

    # TITRE I / II / ...  (standalone line)
    ("titre", re.compile(
        r"(?m)^\s*(TITRE\s+(?:[IVX]+|PREMIER|UNIQUE))\s*$", re.IGNORECASE
    )),

    # CHAPITRE I / II / ...  (standalone line)
    ("chapitre", re.compile(
        r"(?m)^\s*((?:CHAPITRE|Chapitre)\s+(?:[IVX]+|PREMIER|UNIQUE))\s*$"
    )),

    # Section (e.g. "Section 1 - Les émetteurs", "Sous-section 1")
    ("section", re.compile(
        r"(?m)^\s*((?:Sous-)?[Ss]ection\s+\d+(?:[^\n]{0,80})?)\s*$"
    )),

    # Article — AMF RG format: "Article 111-1", "Article 212-4"
    # Also handles plain "Article 1er", "Article 2 bis", simpler texts
    ("article", re.compile(
        r"(?m)^\s*(Article\s+(?:\d{3}-\d+(?:-\d+)?|\d+(?:er|ème)?(?:\s*(?:bis|ter|quater))?))\b"
    )),
]

# Levels in descending order of breadth — resetting a level clears everything below it
_LEVEL_ORDER = ["livre", "titre", "chapitre", "section"]


def split_by_articles(raw_text: str) -> list[dict]:
    """
    Parse the raw text of an AMF regulatory document into article-level
    sections, each carrying its full hierarchical context:

        livre → titre → chapitre → section → article_ref → text

    Strategy: collect all structural marker positions, sort by offset,
    then scan linearly maintaining a running context dict. When a
    higher-level marker appears it resets all levels below it.
    """
    all_matches = []

    for level, rx in _MARKERS:
        for m in rx.finditer(raw_text):
            label = " ".join(m.group(1).split())  # normalise whitespace
            all_matches.append((m.start(), m.end(), level, label))

    all_matches.sort(key=lambda x: x[0])

    context: dict = {k: None for k in _LEVEL_ORDER}
    sections: list[dict] = []
    current_article_ref: str | None = None
    current_text_start: int = 0

    for pos, end, level, label in all_matches:
        if level == "article":
            # Flush the previous article before opening a new one
            if current_article_ref is not None:
                article_text = raw_text[current_text_start:pos].strip()
                if article_text:
                    sections.append({
                        **context,
                        "article_ref": current_article_ref,
                        "text": article_text,
                    })
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
            sections.append({
                **context,
                "article_ref": current_article_ref,
                "text": article_text,
            })

    if not sections:
        # No article markers found — return the whole text as a single section
        return [{
            "livre": None, "titre": None, "chapitre": None,
            "section": None, "article_ref": None, "text": raw_text.strip(),
        }]

    return sections
