"""
ingest.py — ETL pipeline entry point.

Runs the full ingestion flow for all AMF regulatory PDFs in a source directory:
  PDF → extract text → segment by article → chunk → embed → FAISS store

Usage:
    python ingest.py                          # defaults below
    python ingest.py --pdf-dir data --store-dir vector_store
"""
import typer
from pathlib import Path
from loguru import logger

from extract.pdf_reader import pdf_to_text
from transform.segmenter import split_by_articles
from transform.chunker import chunk_text
from vector.embeddings import embed_batch
from vector.vector_store import build_store
from storage.models import Chunk

app = typer.Typer()


def ingest_pdf(path: Path, store_dir: str) -> None:
    """
    Full pipeline for a single PDF file.

    Segment first, then chunk each section independently — this keeps
    article boundaries intact so the LLM can cite 'Article 12' accurately
    rather than getting a chunk that straddles two articles.
    """
    logger.info(f"Processing {path.name}")

    # 1. Extract
    result = pdf_to_text(str(path))
    pages = result["pages"]
    raw_text = result["raw_text"]
    logger.info(f"  Extracted {len(pages)} pages")

    # 2. Segment by article (AMF-aware)
    sections = split_by_articles(raw_text)
    logger.info(f"  Found {len(sections)} sections")

    # 3. Chunk each section, preserving metadata
    all_chunks: list[Chunk] = []
    chunk_index = 0

    for section in sections:
        text_chunks = chunk_text(section["text"])
        for text in text_chunks:
            if not text.strip():
                continue
            all_chunks.append(Chunk(
                chunk_index=chunk_index,
                source=path.name,
                page=_estimate_page(text, pages),
                livre=section.get("livre"),
                titre=section.get("titre"),
                chapitre=section.get("chapitre"),
                section=section.get("section"),
                article_ref=section.get("article_ref"),
                text=text,
            ))
            chunk_index += 1

    logger.info(f"  Generated {len(all_chunks)} chunks")

    # 4. Embed in batches of 64 to avoid OOM on large docs
    BATCH_SIZE = 64
    all_vectors: list[list[float]] = []
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = [c.text for c in all_chunks[i:i + BATCH_SIZE]]
        all_vectors.extend(embed_batch(batch))
        logger.info(f"  Embedded {min(i + BATCH_SIZE, len(all_chunks))}/{len(all_chunks)} chunks")

    # 5. Persist to FAISS
    build_store(all_chunks, all_vectors, store_dir)
    logger.success(f"Done: {path.name}")


def _estimate_page(chunk_text: str, pages: list[dict]) -> int:
    """
    Estimate the page number where a chunk originates by finding which
    page contains the most overlap with the chunk's starting words.
    Normalises whitespace before comparing to handle newline differences.
    Falls back to page 1 if no match found.
    """
    probe = " ".join(chunk_text.split()[:15])
    for page in pages:
        normalized = " ".join(page["text"].split())
        if probe in normalized:
            return page["page"]
    return 1


@app.command()
def main(
    pdf_dir: str = typer.Option("data", help="Directory containing PDF files"),
    store_dir: str = typer.Option("vector_store", help="Output directory for FAISS index + metadata"),
):
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDFs found in {pdf_dir}")
        raise typer.Exit(1)

    logger.info(f"Found {len(pdf_files)} PDF(s) in {pdf_dir}")
    for pdf in pdf_files:
        ingest_pdf(pdf, store_dir)


if __name__ == "__main__":
    app()
