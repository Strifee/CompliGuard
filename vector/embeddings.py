import numpy as np
from fastembed import TextEmbedding

_model = TextEmbedding("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")


def embed_batch(texts: list[str]) -> list[list[float]]:
    vectors = list(_model.embed(texts))
    result = []
    for v in vectors:
        norm = np.linalg.norm(v)
        result.append((v / norm).tolist() if norm > 0 else v.tolist())
    return result
