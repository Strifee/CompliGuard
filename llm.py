"""
llm.py — Generation layer.

Supports two backends:
  - Claude API (Anthropic) — default, best quality
  - Ollama (local fallback)

Usage (standalone chat):
    python llm.py
    python llm.py --model claude-haiku-4-5-20251001
"""
import json
import os
import urllib.request
import urllib.error
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule

from query import retrieve
from storage.models import Chunk

load_dotenv()

app = typer.Typer()
console = Console()

OLLAMA_URL = "http://localhost:11434/api/chat"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You are a legal assistant specialised in French financial market regulation (AMF — Autorité des marchés financiers).
You answer in the same language as the user's question:
- If the question is in French, reply entirely in French — including all meta-phrases (e.g. use "extraits fournis" not "excerpts", "informations" not "information").
- If the question is in English, reply entirely in English.
Never mix languages within a response.
You answer ONLY based on the regulatory excerpts provided in each user message.
You may explain, rephrase, give examples, and answer follow-up questions.
Strict rules:
- Never invent or guess the meaning of an acronym unless it is explicitly defined in the provided excerpts. If not defined, write it as-is.
- If the question is in French and information is missing: "Cette information ne figure pas dans les extraits fournis."
- If the question is in English and information is missing: "This information does not appear in the provided excerpts."
- Never add knowledge from outside the excerpts.
- Always cite the source articles at the end of your answer."""

EXPAND_PROMPT = """You are an expert in French financial regulation (AMF).
Generate 3 short reformulations of the following question to improve document retrieval.
Each reformulation must be on a separate line, without numbering or dashes.
Include acronyms AND their full forms when relevant.
If the question is in English, reformulate in both French and English.
Reply ONLY with the 3 reformulations, nothing else."""


# ── Context builders ──────────────────────────────────────────────────────────

def build_context(chunks: list[Chunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        location = " > ".join(filter(None, [c.livre, c.titre, c.chapitre, c.section, c.article_ref]))
        parts.append(f"[Excerpt {i} | {location} | p.{c.page}]\n{c.text}")
    return "\n\n---\n\n".join(parts)


def build_citations(chunks: list[Chunk]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        parts = filter(None, [c.livre, c.titre, c.article_ref, f"p.{c.page}"])
        lines.append(f"[{i}] {' — '.join(parts)}")
    return "\n".join(lines)


# ── Claude API ────────────────────────────────────────────────────────────────

def ask_claude(question: str, context: str, history: list[dict]) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    user_message = f"""Relevant regulatory excerpts:

{context}

Question: {question}"""

    messages = []
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text


def expand_query_claude(question: str) -> list[str]:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=200,
        system=EXPAND_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    lines = [l.strip() for l in response.content[0].text.splitlines() if l.strip()]
    return [question] + lines[:3]


# ── Ollama fallback ───────────────────────────────────────────────────────────

def ask_ollama(question: str, context: str, model: str, history: list[dict]) -> str:
    user_message = f"""Relevant regulatory excerpts:

{context}

Question: {question}"""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    payload = {"model": model, "messages": messages, "stream": False}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["message"]["content"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        if e.code == 404:
            raise RuntimeError(f"Model '{model}' not found. Run: ollama pull {model}")
        raise RuntimeError(f"Ollama returned HTTP {e.code}: {e.reason}\n{body}")
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_URL}. Is it running?\nError: {e}"
        )


def expand_query_ollama(question: str, model: str) -> list[str]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": EXPAND_PROMPT},
            {"role": "user", "content": question},
        ],
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            lines = [l.strip() for l in result["message"]["content"].splitlines() if l.strip()]
            return [question] + lines[:3]
    except Exception:
        return [question]


# ── Main answer function ──────────────────────────────────────────────────────

def answer(
    question: str,
    model: str = "claude",
    top_k: int = 10,
    store_dir: str = "vector_store",
    livre: str | None = None,
    history: list[dict] | None = None,
) -> dict:
    """
    Full RAG pipeline: expand query → retrieve chunks → call LLM → return answer.

    model="claude" uses Claude API (requires ANTHROPIC_API_KEY in .env).
    Any other value is treated as an Ollama model name.
    """
    use_claude = model == "claude" and os.environ.get("ANTHROPIC_API_KEY")

    # Retrieve more candidates than needed, then trim to top_k for the LLM
    scored = retrieve(question, top_k=top_k * 2, store_dir=store_dir, livre=livre)
    chunks = [c for _, c in scored][:top_k]

    if not chunks:
        return {"answer": "No relevant excerpts found. / Aucun extrait pertinent trouvé.", "citations": "", "chunks": []}

    context = build_context(chunks)

    if use_claude:
        response = ask_claude(question, context, history or [])
    else:
        response = ask_ollama(question, context, model, history or [])

    return {"answer": response, "citations": build_citations(chunks), "chunks": chunks}


# ── CLI ───────────────────────────────────────────────────────────────────────

@app.command()
def main(
    model: str = typer.Option("claude", help="'claude' or an Ollama model name"),
    top_k: int = typer.Option(8, help="Number of chunks to retrieve"),
    store_dir: str = typer.Option("vector_store", help="FAISS index directory"),
    livre: str = typer.Option(None, help="Filter by LIVRE (e.g. 'LIVRE I')"),
):
    """Interactive chat loop with the AMF regulatory corpus."""
    console.print(Rule("[bold cyan]CompliGuard-FR — Assistant réglementaire AMF[/bold cyan]"))
    backend = "Claude API" if (model == "claude" and os.environ.get("ANTHROPIC_API_KEY")) else model
    console.print(f"Backend : [green]{backend}[/green]  |  Tapez [red]exit[/red] pour quitter\n")

    history: list[dict] = []

    while True:
        try:
            question = console.input("[bold yellow]Vous >[/bold yellow] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Au revoir.[/dim]")
            break

        if not question or question.lower() in ("exit", "quit", "q"):
            console.print("[dim]Au revoir.[/dim]")
            break

        with console.status("[bold green]Recherche en cours..."):
            result = answer(question, model=model, top_k=top_k, store_dir=store_dir, livre=livre, history=history)

        console.print("\n[bold cyan]Assistant >[/bold cyan]")
        console.print(Markdown(result["answer"]))
        if result["citations"]:
            console.print(f"\n[dim]{result['citations']}[/dim]")
        console.print()

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result["answer"]})


if __name__ == "__main__":
    app()
