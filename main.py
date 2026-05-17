#!/usr/bin/env python3
import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from rag.pipeline import RAGPipeline

console = Console()


@click.group()
def cli():
    """Privacy RAG pipeline — ingest documents and ask questions."""


@cli.command()
@click.option("--reset", is_flag=True, help="Clear existing index before ingesting.")
def ingest(reset: bool):
    """Load documents, chunk, embed, and store in ChromaDB."""
    pipeline = RAGPipeline()
    console.print(f"[bold]Loading from:[/bold] {pipeline.docs_path}")
    with console.status("[bold green]Indexing documents..."):
        stats = pipeline.ingest(reset=reset)
    table = Table(title="Ingest complete")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Documents", str(stats["documents"]))
    table.add_row("Chunks indexed", str(stats["chunks"]))
    console.print(table)
    console.print("[dim]Sources:[/dim]", ", ".join(stats["sources"]))


@cli.command()
@click.argument("question")
@click.option("--no-llm-hint", is_flag=True, help="Hide LLM configuration hint.")
def ask(question: str, no_llm_hint: bool):
    """Ask a question against the indexed knowledge base."""
    pipeline = RAGPipeline()
    if pipeline.chunk_count == 0:
        console.print(
            "[red]Index is empty.[/red] Run [bold]python main.py ingest[/bold] first."
        )
        raise SystemExit(1)

    with console.status("[bold green]Retrieving and generating..."):
        result = pipeline.query(question)

    console.print(Panel(Markdown(result["answer"]), title="Answer", border_style="green"))

    if result["sources"]:
        table = Table(title="Retrieved sources")
        table.add_column("Document", style="cyan")
        table.add_column("Chunk", justify="right")
        table.add_column("Score", justify="right")
        for src in result["sources"]:
            table.add_row(src["source"], str(src["chunk_index"]), f"{src['score']:.3f}")
        console.print(table)


@cli.command()
def chat():
    """Interactive Q&A session."""
    pipeline = RAGPipeline()
    if pipeline.chunk_count == 0:
        console.print(
            "[red]Index is empty.[/red] Run [bold]python main.py ingest[/bold] first."
        )
        raise SystemExit(1)

    console.print("[bold]Privacy RAG[/bold] — type 'exit' or 'quit' to leave.\n")
    while True:
        try:
            question = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye.")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            console.print("Goodbye.")
            break
        result = pipeline.query(question)
        console.print(Panel(Markdown(result["answer"]), title="Answer", border_style="green"))
        console.print()


@cli.command()
def status():
    """Show index and configuration status."""
    import config as cfg

    pipeline = RAGPipeline()
    console.print(f"[bold]Documents path:[/bold] {pipeline.docs_path}")
    console.print(f"[bold]Chroma path:[/bold] {pipeline.chroma_path}")
    console.print(f"[bold]Chunks in index:[/bold] {pipeline.chunk_count}")
    console.print(f"[bold]Embedding model:[/bold] {cfg.EMBEDDING_MODEL}")


if __name__ == "__main__":
    cli()
