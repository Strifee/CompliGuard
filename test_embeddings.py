"""
Quick smoke test for the multilingual embedder.
Run from project root: python test_embeddings.py
"""
import math
from vector.embeddings import embed_batch

# French regulatory text — representative of AMMC document content
samples = [
    "Les dispositions du présent règlement s'appliquent aux organismes de placement collectif.",
    "Article 12 - Tout émetteur doit déposer un prospectus auprès de l'AMMC.",
    "This is an English sentence to verify multilingual support.",
]

print("Loading model and embedding...")
vectors = embed_batch(samples)

print(f"\nShape: {len(vectors)} vectors x {len(vectors[0])} dims")
assert len(vectors) == 3, "Wrong number of vectors"
assert len(vectors[0]) == 768, f"Expected 768 dims, got {len(vectors[0])}"

# Check normalization — L2 norm of each vector should be ~1.0
for i, v in enumerate(vectors):
    norm = math.sqrt(sum(x ** 2 for x in v))
    print(f"Vector {i+1} norm: {norm:.6f}")
    assert abs(norm - 1.0) < 1e-4, f"Vector {i+1} is not normalized"

print("\nAll checks passed.")
