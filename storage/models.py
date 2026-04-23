from pydantic import BaseModel
from typing import Optional


class Chunk(BaseModel):
    """
    A single retrievable unit of AMF regulatory text.

    Why Pydantic over SQLModel: we don't need a relational DB for RAG.
    FAISS holds the vectors; chunk metadata is persisted as JSON alongside
    the index. Keeps the stack simple — no ORM, no migrations, no server.
    """

    chunk_index: int                    # position within the source document
    source: str                         # original PDF filename
    page: int                           # page number where the chunk starts

    # Hierarchical location within the AMF regulatory corpus
    livre:       Optional[str] = None   # e.g. "LIVRE I"
    titre:       Optional[str] = None   # e.g. "TITRE II"
    chapitre:    Optional[str] = None   # e.g. "CHAPITRE PREMIER"
    section:     Optional[str] = None   # e.g. "Section 1 - Champ d'application"
    article_ref: Optional[str] = None   # e.g. "Article 111-1"

    text: str                           # raw chunk text fed to the embedder
