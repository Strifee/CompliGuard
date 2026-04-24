import json
from pathlib import Path

import faiss
import numpy as np

from storage.models import Chunk

EMBEDDING_DIM = 768


def build_store(chunks: list[Chunk], vectors: list[list[float]], store_dir: str) -> None:

    out = Path(store_dir)
    out.mkdir(parents=True, exist_ok=True)

    matrix = np.array(vectors, dtype="float32")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(matrix)

    faiss.write_index(index, str(out / "index.faiss"))

    metadata = [chunk.model_dump() for chunk in chunks]
    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Stored {index.ntotal} vectors → {out}")


def load_store(store_dir: str) -> tuple[faiss.Index, list[Chunk]]:

    out = Path(store_dir)

    index = faiss.read_index(str(out / "index.faiss"))

    with open(out / "metadata.json", encoding="utf-8") as f:
        raw = json.load(f)
    chunks = [Chunk(**item) for item in raw]

    return index, chunks


def search(
    query_vector: list[float],
    index: faiss.Index,
    chunks: list[Chunk],
    top_k: int = 8,
    min_score: float = 0.20,
) -> list[tuple[float, Chunk]]:

    q = np.array([query_vector], dtype="float32")
    scores, indices = index.search(q, top_k)
    return [
        (float(score), chunks[i])
        for score, i in zip(scores[0], indices[0])
        if i < len(chunks) and score >= min_score
    ]
