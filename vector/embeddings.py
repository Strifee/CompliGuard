import numpy as np
from fastembed import TextEmbedding

# Runs via ONNX Runtime — falls back to CPU if CUDA unavailable.
_model = TextEmbedding("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed texts and explicitly L2-normalize the vectors.
    Required for cosine similarity via FAISS IndexFlatIP.
    Newer fastembed versions no longer normalize by default.
    """
    vectors = list(_model.embed(texts))
    result = []
    for v in vectors:
        norm = np.linalg.norm(v)
        result.append((v / norm).tolist() if norm > 0 else v.tolist())
    return result
