import re

_SENTENCE_END = re.compile(r'(?<=[.;])\s+')


def chunk_text(text: str, max_chars: int = 1200, overlap_chars: int = 150) -> list[str]:

    text = text.strip()
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]

    segments: list[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            segments.append(para)
        else:
            sentences = _SENTENCE_END.split(para)
            current = ""
            for sent in sentences:
                if current and len(current) + 1 + len(sent) > max_chars:
                    segments.append(current.strip())
                    current = sent
                else:
                    current = (current + " " + sent).strip() if current else sent
            if current:
                segments.append(current.strip())

    chunks: list[str] = []
    current = ""
    for seg in segments:
        if current and len(current) + 2 + len(seg) > max_chars:
            chunks.append(current.strip())
            # Start next chunk with an overlap tail from the previous one
            overlap_tail = current[-overlap_chars:].strip() if overlap_chars else ""
            current = (overlap_tail + "\n\n" + seg).strip() if overlap_tail else seg
        else:
            current = (current + "\n\n" + seg).strip() if current else seg

    if current:
        chunks.append(current.strip())

    return chunks


if __name__ == "__main__":
    sample = "\n\n".join(
        f"Paragraphe {i}: " + "mot " * 120
        for i in range(10)
    )
    result = chunk_text(sample)
    print(f"Generated {len(result)} chunks.")
    for i, c in enumerate(result):
        print(f"Chunk {i + 1} ({len(c)} chars): {c[:80]}...")
