import re
import fitz
import hashlib
from pathlib import Path

_NOISE_PATTERNS = [
    re.compile(r"^\s*\d+\s*$"),                              
    re.compile(r"(?i)règlement\s+général\s+de\s+l['']AMF\s*$"), 
    re.compile(r"(?i)autorité\s+des\s+marchés\s+financiers\s*$"),
    re.compile(r"(?i)www\.amf-france\.org\s*$"),
    re.compile(r"^\s*-\s*\d+\s*-\s*$"),                     
]


def _clean_page(text: str) -> str:
    lines = text.splitlines()
    cleaned = [line for line in lines if not any(p.match(line) for p in _NOISE_PATTERNS)]
    return "\n".join(cleaned)


def pdf_to_text(path: str) -> dict:
    p = Path(path)
    doc = fitz.open(p)
    pages = []
    for i, page in enumerate(doc):
        raw = page.get_text("text") or ""
        cleaned = _clean_page(raw)
        pages.append({"page": i + 1, "text": cleaned})

    raw_text = "\n".join(x["text"] for x in pages)
    content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

    return {"pages": pages, "raw_text": raw_text, "content_hash": content_hash}


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/RG-en-vigueur-du-20250602-au-20251226_notes.pdf"
    result = pdf_to_text(path)
    print(f"Pages: {len(result['pages'])}")
    print(result["raw_text"][:1000])
