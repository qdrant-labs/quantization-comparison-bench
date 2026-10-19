import asyncio
import json
from pathlib import Path
from typing import Annotated

import httpx
import typer

from .config import load_config
from .data import create_fixture, embed_dataset, prepare_dataset
from .evaluation import verify_fixture
from .runner import doctor as check_server
from .runner import run_benchmark, save_ground_truth
from .runner import summarize as summarize_run

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
DATA_REVISION = "f018922a46a51388348a307ec2fef28019a33026"
BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


def display(value: object) -> None:
    typer.echo(json.dumps(value, indent=2))


@app.command()
def plan(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/benchmark.yaml"),
) -> None:
    """Show the declared experiment matrix without contacting external services."""
    settings = load_config(config)
    cells = [
        {
            "method": method.name,
            "datatype": method.datatype,
            "k": k,
            "rescore": rescore,
            "oversampling": oversampling,
        }
        for method in settings.methods
        for k, rescore, oversampling in settings.grid(method)
    ]
    display(
        {
            "datasets": settings.datasets,
            "models": settings.models,
            "builds": settings.builds,
            "cells_per_workload_and_build": len(cells),
            "total_cells": len(cells)
            * settings.builds
            * len(settings.datasets)
            * len(settings.models),
            "cache_state": "warm",
            "matrix": cells,
        }
    )


@app.command()
def prepare(
    dataset: Annotated[str, typer.Option()],
    revision: Annotated[str, typer.Option()] = DATA_REVISION,
    output: Annotated[Path, typer.Option()] = Path("data/raw"),
    dev_queries: Annotated[int, typer.Option(min=1)] = 50,
    seed: Annotated[int, typer.Option()] = 42,
    max_docs: Annotated[int | None, typer.Option(min=1)] = None,
    max_test_queries: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    """Download a frozen BEIR corpus, query splits, and relevance judgments."""
    prepare_dataset(
        dataset, output, revision, dev_queries, seed, max_docs, max_test_queries
    )
    typer.echo(f"Prepared {output / dataset}")


@app.command()
def embed(
    dataset: Annotated[str, typer.Option()],
    model: Annotated[str, typer.Option(help="cohere or bge")],
    data_root: Annotated[Path, typer.Option()] = Path("data"),
    revision: Annotated[str | None, typer.Option()] = None,
    batch_size: Annotated[int, typer.Option(min=1, max=96)] = 64,
) -> None:
    """Generate/checkpoint embeddings; Cohere calls use your paid API account."""
    if model not in {"cohere", "bge"}:
        raise ValueError("model must be cohere or bge")
    output = data_root / "embeddings" / dataset / model
    if model == "bge" and revision is None:
        revision = BGE_REVISION
    asyncio.run(
        embed_dataset(data_root / "raw" / dataset, output, model, batch_size, revision)
    )
    typer.echo(f"Saved embeddings to {output}")


@app.command()
def fixture(
    output: Annotated[Path, typer.Option()] = Path("data/synthetic"),
    dimensions: Annotated[int, typer.Option(min=16)] = 384,
    documents: Annotated[int, typer.Option(min=101)] = 256,
    queries: Annotated[int, typer.Option(min=2)] = 32,
    seed: Annotated[int, typer.Option()] = 42,
) -> None:
    """Create small synthetic embeddings without downloads or API calls."""
    create_fixture(output, dimensions, documents, queries, seed)
    typer.echo(f"Created synthetic fixture at {output}")


@app.command()
def verify(data: Annotated[Path, typer.Option(exists=True)]) -> None:
    """Validate artifacts and exact-search invariants offline."""
    display(verify_fixture(data))


@app.command("ground-truth")
def ground_truth(
    data: Annotated[Path, typer.Option(exists=True)],
    k: Annotated[int, typer.Option(min=10)] = 100,
) -> None:
    """Compute exact float32 neighbors independent of quantized collections."""
    result = save_ground_truth(data, k)
    display({"queries": len(result), "k": k, "path": str(data / "ground_truth.json")})


@app.command()
def doctor() -> None:
    """Check the Qdrant endpoint configured by QDRANT_URL/QDRANT_API_KEY."""
    display(asyncio.run(check_server()))


@app.command()
def run(
    config: Annotated[Path, typer.Option(exists=True)],
    data: Annotated[Path, typer.Option(exists=True)],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Run one model/dataset artifact through the selected warm-search matrix."""
    asyncio.run(run_benchmark(load_config(config), data, output))
    typer.echo(f"Completed benchmark. Results: {output / 'summary.json'}")


@app.command()
def summarize(input: Annotated[Path, typer.Option(exists=True)]) -> None:
    """Aggregate results without combining distinct ablation settings."""
    result = summarize_run(input)
    display({"groups": len(result["groups"]), "path": str(input / "summary.json")})


def main() -> None:
    try:
        app()
    except (ValueError, RuntimeError, OSError, httpx.HTTPError) as error:
        typer.echo(f"{type(error).__name__}: {error}", err=True)
        raise SystemExit(1) from None
