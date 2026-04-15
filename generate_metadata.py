"""
generate_metadata.py — Metadata-only pass (no embedding, no FAISS).

Runs: PDF → extract → segment → chunk → saves metadata.json

Usage:
    python generate_metadata.py
    python generate_metadata.py --pdf-dir data --store-dir vector_store
"""
import json
import typer
from pathlib import Path
from loguru import logger

from extract.pdf_reader import pdf_to_text
from transform.segmenter import split_by_articles
from transform.chunker import chunk_text
from storage.models import Chunk

app = typer.Typer()


def _estimate_page(chunk_text: str, pages: list[dict]) -> int:
    probe = " ".join(chunk_text.split()[:10])
    for page in pages:
        if probe in page["text"]:
            return page["page"]
    return 1


@app.command()
def main(
    pdf_dir: str = typer.Option("data", help="Directory containing PDF files"),
    store_dir: str = typer.Option("vector_store", help="Output directory for metadata.json"),
):
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDFs found in {pdf_dir}")
        raise typer.Exit(1)

    for path in pdf_files:
        logger.info(f"Processing {path.name}")

        result = pdf_to_text(str(path))
        pages, raw_text = result["pages"], result["raw_text"]
        logger.info(f"  Extracted {len(pages)} pages")

        sections = split_by_articles(raw_text)
        logger.info(f"  Found {len(sections)} sections")

        chunks: list[Chunk] = []
        chunk_index = 0
        for section in sections:
            for text in chunk_text(section["text"]):
                if not text.strip():
                    continue
                chunks.append(Chunk(
                    chunk_index=chunk_index,
                    source=path.name,
                    page=_estimate_page(text, pages),
                    livre=section.get("livre"),
                    document=section.get("document"),
                    titre=section.get("titre"),
                    chapitre=section.get("chapitre"),
                    article_ref=section.get("article_ref"),
                    text=text,
                ))
                chunk_index += 1

        logger.info(f"  Generated {len(chunks)} chunks")

        out = Path(store_dir)
        out.mkdir(parents=True, exist_ok=True)
        metadata_path = out / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in chunks], f, ensure_ascii=False, indent=2)

        logger.success(f"Saved {len(chunks)} chunks → {metadata_path}")


if __name__ == "__main__":
    app()
