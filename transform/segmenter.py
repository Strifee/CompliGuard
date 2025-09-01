import re

ARTICLE_RX = re.compile(r"(Article|Art\.)\s+(\d+[A-Za-z\-]*)\b", re.IGNORECASE)

def split_by_articles(raw_text: str):
    matches = list(ARTICLE_RX.finditer(raw_text))
    sections = []
    for idx, m in enumerate(matches):
        start = m.end()
        end = matches[idx+1].start() if idx+1 < len(matches) else len(raw_text)
        sections.append({
            "section_type": "article",
            "section_ref": f"Article {m.group(2)}",
            "text": raw_text[start:end].strip()
        })
    return sections or [{"section_type":"body","section_ref":"global","text":raw_text.strip()}]


if __name__ == "__main__":
    import extract.pdf_reader as pdf_reader
    path = "data/Recueil_AMMC_VF_ Février_2025.pdf"
    result = pdf_reader.pdf_to_text(path)
    raw_text = result["raw_text"]
    sections = split_by_articles(raw_text)
    print(f"Found {len(sections)} sections.")
    print(sections[0])
