# transform/chunker.py
def chunk_text(text: str, max_len=1200, overlap=150):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+max_len]
        chunks.append(" ".join(chunk))
        i += max_len - overlap
    return chunks



if __name__ == "__main__":
    sample_text = " ".join(f"word{i}" for i in range(3000))
    chunks = chunk_text(sample_text)
    print(f"Generated {len(chunks)} chunks.")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} (length {len(chunk.split())}): {chunk[:50]}...")