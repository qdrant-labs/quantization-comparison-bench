import asyncio
import json
from pathlib import Path
from typing import Annotated

import httpx
import typer

from .config import load_config, save_config
from .data import (
    DATASETS_IN_ORDER,
    create_fixture,
    embed_dataset,
    prepare_all,
    prepare_dataset,
)
from .evaluation import artifact_status, verify_fixture
from .report import build_report
from .runner import doctor as check_server
from .runner import run_benchmark, save_ground_truth
from .runner import summarize as summarize_run
from .selection import select_from_run

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
DATA_REVISION = "f018922a46a51388348a307ec2fef28019a33026"
BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
BGE_BASE_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"


def display(value: object) -> None:
    typer.echo(json.dumps(value, indent=2))


@app.command()
def plan(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/benchmark.yaml"),
) -> None:
    """Show the declared experiment matrix without contacting external services."""
    settings = load_config(config)
    cells = [
        {"method": method.name, "datatype": method.datatype, **cell.model_dump()}
        for method in settings.methods
        for cell in settings.grid(method)
    ]
    display(
        {
            "datasets": settings.datasets,
            "models": settings.models,
            "builds": settings.builds,
            "split": settings.search.split,
            "max_queries": settings.search.max_queries,
            "repeats": settings.search.repeats,
            "cache_state": settings.cache_state,
            "require_telemetry": settings.require_telemetry,
            "memory_limit_bytes": settings.memory_limit_bytes,
            "cells_per_workload_and_build": len(cells),
            "total_cells": len(cells)
            * settings.builds
            * len(settings.datasets)
            * len(settings.models),
            "selection": settings.selection,
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
    """Freeze one BEIR corpus, its judgments, and its supplied Cohere vectors."""
    prepare_dataset(
        dataset, output, revision, dev_queries, seed, max_docs, max_test_queries
    )
    typer.echo(
        f"Prepared {output / dataset} and {output.parent / 'embeddings' / dataset / 'cohere'}"
    )


@app.command("prepare-all")
def prepare_all_command(
    data_root: Annotated[Path, typer.Option()] = Path("data"),
    revision: Annotated[str, typer.Option()] = DATA_REVISION,
    with_bge: Annotated[bool, typer.Option(help="Also embed with BGE-small")] = False,
    bge_revision: Annotated[str, typer.Option()] = BGE_REVISION,
    with_bge_base: Annotated[
        bool, typer.Option(help="Also embed with BGE-base (768-d)")
    ] = False,
    bge_base_revision: Annotated[str, typer.Option()] = BGE_BASE_REVISION,
    dev_queries: Annotated[int, typer.Option(min=1)] = 50,
    seed: Annotated[int, typer.Option()] = 42,
    batch_size: Annotated[int, typer.Option(min=1)] = 64,
    datasets: Annotated[list[str] | None, typer.Option()] = None,
) -> None:
    """Stage all four corpora, judgments, and Cohere arrays; skip complete ones."""
    report = asyncio.run(
        prepare_all(
            data_root,
            revision,
            with_bge,
            bge_revision,
            dev_queries,
            seed,
            batch_size,
            datasets or list(DATASETS_IN_ORDER),
            {"bge_base": bge_base_revision} if with_bge_base else None,
        )
    )
    display(report)


@app.command()
def embed(
    dataset: Annotated[str, typer.Option()],
    model: Annotated[
        str, typer.Option(help="bge or bge_base (Cohere vectors come from prepare)")
    ],
    data_root: Annotated[Path, typer.Option()] = Path("data"),
    revision: Annotated[str | None, typer.Option()] = None,
    batch_size: Annotated[int, typer.Option(min=1)] = 64,
) -> None:
    """Generate and checkpoint local embeddings for one prepared dataset."""
    if revision is None:
        revision = {"bge": BGE_REVISION, "bge_base": BGE_BASE_REVISION}.get(model)
    output = data_root / "embeddings" / dataset / model
    asyncio.run(
        embed_dataset(data_root / "raw" / dataset, output, model, batch_size, revision)
    )
    typer.echo(f"Saved embeddings to {output}")


@app.command()
def readiness(
    data_root: Annotated[Path, typer.Option()] = Path("data"),
    require_all: Annotated[
        bool, typer.Option(help="Fail unless every dataset/model artifact is ready")
    ] = False,
    datasets: Annotated[list[str] | None, typer.Option()] = None,
    models: Annotated[list[str] | None, typer.Option()] = None,
) -> None:
    """Check hashes, dimensions, IDs, labels, and ground truth for each artifact."""
    rows = [
        artifact_status(data_root / "embeddings" / dataset / model, dataset, model)
        for dataset in (datasets or list(DATASETS_IN_ORDER))
        for model in (models or ["cohere", "bge"])
    ]
    statuses = {row["status"] for row in rows}
    ready = "failed" not in statuses and (not require_all or statuses == {"ready"})
    display({"ready": ready, "require_all": require_all, "artifacts": rows})
    if not ready:
        raise SystemExit(1)


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
    max_queries: Annotated[
        int | None,
        typer.Option(min=0, help="Override search.max_queries; 0 uses the whole split"),
    ] = None,
    concurrency: Annotated[
        int | None, typer.Option(min=1, max=64, help="Override search.concurrency")
    ] = None,
) -> None:
    """Run one model/dataset artifact through the configured search matrix."""
    settings = load_config(config)
    if max_queries is not None:
        settings = settings.model_copy(
            update={
                "search": settings.search.model_copy(
                    update={"max_queries": max_queries or None}
                )
            }
        )
    if concurrency is not None:
        settings = settings.model_copy(
            update={
                "search": settings.search.model_copy(
                    update={"concurrency": concurrency}
                )
            }
        )
    asyncio.run(run_benchmark(settings, data, output))
    typer.echo(f"Completed benchmark. Results: {output / 'summary.json'}")


@app.command()
def summarize(input: Annotated[Path, typer.Option(exists=True)]) -> None:
    """Aggregate results without combining distinct ablation settings."""
    result = summarize_run(input)
    display({"groups": len(result["groups"]), "path": str(input / "summary.json")})


@app.command()
def select(
    input: Annotated[Path, typer.Option(exists=True, help="Complete dev tuning run")],
    output: Annotated[Path, typer.Option(help="Frozen test-split config to write")],
    base_config: Annotated[Path, typer.Option(exists=True)] = Path(
        "configs/benchmark.yaml"
    ),
    recall_floor: Annotated[float, typer.Option(min=0, max=1)] = 0.95,
    ndcg_tolerance: Annotated[float, typer.Option(min=0, max=1)] = 0.01,
    latency_metric: Annotated[str, typer.Option()] = "server_ms_p95",
) -> None:
    """Freeze the fastest eligible dev point per method and k into a test config."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen selection {output}")
    config = select_from_run(
        input, load_config(base_config), recall_floor, ndcg_tolerance, latency_metric
    )
    save_config(output, config)
    assert config.selection is not None
    display(
        {
            "path": str(output),
            "rule": config.selection["rule"],
            "choices": config.selection["choices"],
        }
    )


@app.command()
def report(
    results: Annotated[Path, typer.Option(exists=True)] = Path("results"),
    output: Annotated[Path, typer.Option()] = Path("results/report"),
) -> None:
    """Build article tables with bootstrap intervals from completed runs."""
    display(build_report(results, output))


def main() -> None:
    try:
        app()
    except (ValueError, RuntimeError, OSError, httpx.HTTPError) as error:
        typer.echo(f"{type(error).__name__}: {error}", err=True)
        raise SystemExit(1) from None
