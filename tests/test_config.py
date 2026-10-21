from pathlib import Path

import pytest
from pydantic import ValidationError

from quantization_comparison_bench.config import (
    Benchmark,
    Cell,
    Method,
    collection_config,
    load_config,
    save_config,
)

CONFIGS = sorted(Path("configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=[path.stem for path in CONFIGS])
def test_shipped_configs_load_and_plan(path: Path) -> None:
    config = load_config(path)
    assert config.expected_qdrant_version == "1.19.2"
    for method in config.methods:
        assert config.grid(method)


def test_grid_crosses_effort_for_quantized_methods_only() -> None:
    config = Benchmark(
        methods=[
            Method(name="float32"),
            Method(name="turbo4", quantizer="turbo", bits=4),
        ],
        search={
            "ks": [10, 100],
            "hnsw_ef": [64, 256],
            "rescoring": [False, True],
            "oversampling": [1, 2],
        },
    )
    assert len(config.grid(config.methods[0])) == 4
    assert all(cell.rescore is None for cell in config.grid(config.methods[0]))
    assert len(config.grid(config.methods[1])) == 16


def test_cells_replace_the_grid() -> None:
    method = Method(
        name="scalar8",
        quantizer="scalar",
        cells=[Cell(k=10, hnsw_ef=128, rescore=True, oversampling=2)],
    )
    config = Benchmark(methods=[method])
    assert config.grid(method) == method.cells


def test_cells_require_quantization_parameters_only_when_quantized() -> None:
    with pytest.raises(ValidationError):
        Method(name="scalar8", quantizer="scalar", cells=[Cell(k=10, hnsw_ef=64)])
    with pytest.raises(ValidationError):
        Method(
            name="float32",
            cells=[Cell(k=10, hnsw_ef=64, rescore=True, oversampling=2)],
        )


def test_cold_start_requires_hook_and_telemetry() -> None:
    with pytest.raises(ValidationError):
        Benchmark(methods=[Method(name="float32")], cache_state="cold_start")
    with pytest.raises(ValidationError):
        Benchmark(
            methods=[Method(name="float32")],
            cache_state="cold_start",
            cache_reset_command="scripts/cache-reset.sh",
        )
    Benchmark(
        methods=[Method(name="float32")],
        cache_state="cold_start",
        cache_reset_command="scripts/cache-reset.sh",
        require_telemetry=True,
    )


def test_memory_limit_requires_telemetry() -> None:
    with pytest.raises(ValidationError):
        Benchmark(methods=[Method(name="float32")], memory_limit_bytes=1 << 30)


def test_save_and_reload_round_trip(tmp_path: Path) -> None:
    config = load_config(Path("configs/benchmark.yaml"))
    save_config(tmp_path / "copy.yaml", config)
    assert load_config(tmp_path / "copy.yaml") == config


def test_extended_encodings_map_to_server_names() -> None:
    assert Method(name="t", quantizer="turbo", bits=1.5).quantization() == {
        "turbo": {"bits": "bits1_5", "memory": "pinned"}
    }
    assert Method(name="t", quantizer="turbo", bits=4).quantization() == {
        "turbo": {"bits": "bits4", "memory": "pinned"}
    }
    assert Method(name="b", quantizer="binary", encoding="two_bits").quantization() == {
        "binary": {"encoding": "two_bits", "memory": "pinned"}
    }
    assert Method(
        name="b", quantizer="binary", query_encoding="scalar8bits"
    ).quantization() == {
        "binary": {
            "encoding": "one_bit",
            "memory": "pinned",
            "query_encoding": "scalar8bits",
        }
    }
    with pytest.raises(ValidationError):
        Method(name="t", quantizer="turbo", bits=3)


def test_indexing_threads_are_explicit_and_shared() -> None:
    config = load_config(Path("configs/benchmark.yaml"))
    body = collection_config(config, config.methods[0], 16)
    assert body["hnsw_config"]["max_indexing_threads"] == 8
    for path in CONFIGS:
        if path.stem != "smoke":
            assert load_config(path).hnsw.max_indexing_threads == 8, path


def test_concurrency_is_bounded_and_defaults_to_one() -> None:
    assert Benchmark(methods=[Method(name="float32")]).search.concurrency == 1
    with pytest.raises(ValidationError):
        Benchmark(methods=[Method(name="float32")], search={"concurrency": 0})
    assert (
        load_config(Path("configs/memory-pressure-bge.yaml")).memory_limit_bytes
        == 512 * 1024 * 1024
    )
