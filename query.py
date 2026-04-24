import typer
from rich.console import Console
from rich.table import Table

from vector.embeddings import embed_batch
from vector.vector_store import load_store, search
from storage.models import Chunk

app = typer.Typer()
console = Console()

_index = None
_chunks = None


def get_store(store_dir: str = "vector_store"):
    global _index, _chunks
    if _index is None:
        _index, _chunks = load_store(store_dir)
    return _index, _chunks


def retrieve(
    question: str,
    top_k: int = 3,
    store_dir: str = "vector_store",
    livre: str | None = None,
) -> list[Chunk]:

    index, chunks = get_store(store_dir)

    query_vector = embed_batch([question])[0]

    if livre:
        filtered = [(i, c) for i, c in enumerate(chunks) if c.livre == livre]
        if not filtered:
            return []
        import faiss, numpy as np
        filtered_indices, filtered_chunks = zip(*filtered)
        sub_matrix = np.array(
            [index.reconstruct(i) for i in filtered_indices], dtype="float32"
        )
        sub_index = faiss.IndexFlatIP(sub_matrix.shape[1])
        sub_index.add(sub_matrix)
        scored = search(query_vector, sub_index, list(filtered_chunks), top_k)
    else:
        scored = search(query_vector, index, chunks, top_k)

    return [(score, chunk) for score, chunk in scored]


@app.command()
def main(
    question: str = typer.Argument(..., help="Question to ask"),
    top_k: int = typer.Option(3, help="Number of results to return"),
    store_dir: str = typer.Option("vector_store", help="FAISS index directory"),
    livre: str = typer.Option(None, help="Filter by LIVRE (e.g. 'LIVRE I')"),
):
    results = retrieve(question, top_k=top_k, store_dir=store_dir, livre=livre)

    if not results:
        console.print("[red]No results found.[/red]")
        raise typer.Exit(1)

    table = Table(title=f'Top {len(results)} results for: "{question}"', show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", style="green", width=6)
    table.add_column("Location", style="cyan", min_width=30)
    table.add_column("Text", min_width=60)

    for i, (score, chunk) in enumerate(results, 1):
        location = "\n".join(filter(None, [
            chunk.livre,
            chunk.document,
            chunk.titre,
            chunk.chapitre,
            chunk.article_ref,
            f"p.{chunk.page}",
        ]))
        table.add_row(str(i), f"{score:.2f}", location, chunk.text[:300] + "...")

    console.print(table)


if __name__ == "__main__":
    app()
