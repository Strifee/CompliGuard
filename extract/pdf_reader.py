import fitz
import hashlib
from pathlib import Path

def pdf_to_text(path: str) -> dict:
    p = Path(path)
    doc = fitz.open(p)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({"page": i+1, "text": text or ""})
    raw_text = "\n".join(x["text"] for x in pages)

    if len(raw_text.strip()) < 200:
        pass
    content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    return {"pages": pages, "raw_text": raw_text, "content_hash": content_hash}

if __name__ == "__main__":
    import sys
    path = "data/Recueil_AMMC_VF_ Février 2025.pdf"
    result = pdf_to_text(path)
    print(result["raw_text"][:500])