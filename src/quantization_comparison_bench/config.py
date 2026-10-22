from itertools import product
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Cell(Settings):
    """One search point: result limit plus explicit, frozen search effort."""

    k: int = Field(ge=10)
    hnsw_ef: int = Field(ge=1)
    rescore: bool | None = None
    oversampling: float | None = Field(default=None, ge=1)


class Method(Settings):
    name: str = Field(pattern=r"^[a-z0-9_]+$")
    datatype: Literal["float32", "turbo4"] = "float32"
    quantizer: Literal["none", "scalar", "binary", "product", "turbo"] = "none"
    bits: Literal["1", "1.5", "2", "4"] | None = None
    compression: Literal["x8", "x16"] | None = None
    quantile: float = Field(default=0.99, ge=0.5, le=1)
    encoding: Literal["one_bit", "one_and_half_bits", "two_bits"] = "one_bit"
    query_encoding: Literal["binary", "scalar4bits", "scalar8bits"] | None = None
    # Frozen development-selected points written by `qbench select`. When set,
    # they replace the search grid for this method.
    cells: list[Cell] | None = Field(default=None, min_length=1)

    @field_validator("bits", mode="before")
    @classmethod
    def coerce_bits(cls, value: object) -> object:
        if isinstance(value, int | float) and not isinstance(value, bool):
            return str(int(value)) if float(value).is_integer() else str(value)
        return value

    @model_validator(mode="after")
    def check_parameters(self) -> Self:
        if (self.quantizer == "turbo") != (self.bits is not None):
            raise ValueError("bits must be set exactly when quantizer is turbo")
        if (self.quantizer == "product") != (self.compression is not None):
            raise ValueError("compression must be set exactly for product quantization")
        for cell in self.cells or []:
            quantized = self.quantizer != "none"
            if quantized != (
                cell.rescore is not None and cell.oversampling is not None
            ):
                raise ValueError(
                    f"{self.name}: rescore/oversampling must be set exactly for "
                    "quantized methods"
                )
        return self

    def quantization(self) -> dict[str, Any] | None:
        match self.quantizer:
            case "none":
                return None
            case "scalar":
                return {
                    "scalar": {
                        "type": "int8",
                        "quantile": self.quantile,
                        "memory": "pinned",
                    }
                }
            case "binary":
                binary: dict[str, Any] = {"encoding": self.encoding, "memory": "pinned"}
                if self.query_encoding is not None:
                    binary["query_encoding"] = self.query_encoding
                return {"binary": binary}
            case "product":
                return {
                    "product": {"compression": self.compression, "memory": "pinned"}
                }
            case "turbo":
                bits = {"1": "bits1", "1.5": "bits1_5", "2": "bits2", "4": "bits4"}
                return {"turbo": {"bits": bits[str(self.bits)], "memory": "pinned"}}


class Hnsw(Settings):
    m: int = Field(default=16, ge=4)
    ef_construct: int = Field(default=100, ge=16)
    # Threads for index construction. Affects build time and build-time CPU,
    # not search results; keep it identical across every reported run.
    max_indexing_threads: int = Field(default=8, ge=1)


class Search(Settings):
    ks: list[int] = Field(default_factory=lambda: [10, 100], min_length=1)
    hnsw_ef: list[int] = Field(default_factory=lambda: [256], min_length=1)
    rescoring: list[bool] = Field(default_factory=lambda: [False, True], min_length=1)
    oversampling: list[float] = Field(
        default_factory=lambda: [1.0, 2.0, 4.0, 8.0], min_length=1
    )
    split: Literal["dev", "test"] = "test"
    warmup_queries: int = Field(default=50, ge=1)
    repeats: int = Field(default=5, ge=1)
    # Seeded query subset taken from the split; None uses the whole split.
    max_queries: int | None = Field(default=None, ge=1)
    # Parallel client connections per measured pass. 1 is the latency protocol;
    # larger values measure throughput and record per-pass queries per second.
    concurrency: int = Field(default=1, ge=1, le=64)

    @model_validator(mode="after")
    def check_grid(self) -> Self:
        if any(k < 10 for k in self.ks):
            raise ValueError("k must be >= 10 to report nDCG@10")
        if any(value < 1 for value in self.hnsw_ef):
            raise ValueError("hnsw_ef must be >= 1")
        if any(value < 1 for value in self.oversampling):
            raise ValueError("oversampling must be >= 1")
        for name in ("ks", "hnsw_ef", "rescoring", "oversampling"):
            values = getattr(self, name)
            if len(set(values)) != len(values):
                raise ValueError(f"search.{name} contains duplicates")
        return self


class Benchmark(Settings):
    expected_qdrant_version: str = "1.19.2"
    datasets: list[str] = Field(
        default_factory=lambda: ["fiqa", "scifact", "nfcorpus", "quora"]
    )
    models: list[str] = Field(default_factory=lambda: ["cohere", "bge"])
    builds: int = Field(default=3, ge=1)
    seed: int = 42
    upload_batch_size: int = Field(default=1024, ge=1)
    indexing_timeout_seconds: float = Field(default=7200, gt=0)
    hnsw: Hnsw = Field(default_factory=Hnsw)
    search: Search = Field(default_factory=Search)
    methods: list[Method] = Field(min_length=1)
    # "warm": warm-up requests precede timing. "cold_start": the cache-reset hook
    # runs before every measured pass and the pass warms as queries proceed.
    cache_state: Literal["warm", "cold_start"] = "warm"
    cache_reset_command: str | None = None
    # Refuse to run unless QDRANT_PID, QDRANT_CGROUP_PATH, and
    # QDRANT_STORAGE_PATH resolve to readable VM resources.
    require_telemetry: bool = False
    # Expected cgroup memory.max of the Qdrant container; the run fails on mismatch.
    memory_limit_bytes: int | None = Field(default=None, ge=1)
    # Memory-pressure protocol: build under the normal cap, then lower the
    # container cap to this value before the measured passes. The commands
    # receive the byte count as their only argument and must leave the same
    # container running (docker update, not recreate).
    search_memory_limit_bytes: int | None = Field(default=None, ge=1)
    search_cap_command: str | None = None
    restore_cap_command: str | None = None
    # Provenance written by `qbench select`; copied into the run manifest.
    selection: dict[str, Any] | None = None

    @model_validator(mode="after")
    def check_run(self) -> Self:
        if len({method.name for method in self.methods}) != len(self.methods):
            raise ValueError("Method names must be unique")
        if self.cache_state == "cold_start":
            if not self.cache_reset_command:
                raise ValueError("cold_start requires cache_reset_command")
            if not self.require_telemetry:
                raise ValueError(
                    "cold_start requires require_telemetry so I/O and page faults "
                    "are recorded for every pass"
                )
        if self.memory_limit_bytes is not None and not self.require_telemetry:
            raise ValueError("memory_limit_bytes requires require_telemetry")
        if self.search_memory_limit_bytes is not None:
            if not (self.search_cap_command and self.restore_cap_command):
                raise ValueError(
                    "search_memory_limit_bytes requires search_cap_command and "
                    "restore_cap_command"
                )
            if not self.require_telemetry:
                raise ValueError("search_memory_limit_bytes requires require_telemetry")
        if self.selection is not None and self.search.split != "test":
            raise ValueError("A selected configuration must evaluate the test split")
        return self

    def grid(self, method: Method) -> list[Cell]:
        if method.cells:
            return list(method.cells)
        if method.quantizer == "none":
            return [
                Cell(k=k, hnsw_ef=hnsw_ef)
                for k, hnsw_ef in product(self.search.ks, self.search.hnsw_ef)
            ]
        return [
            Cell(k=k, hnsw_ef=hnsw_ef, rescore=rescore, oversampling=oversampling)
            for k, hnsw_ef, rescore, oversampling in product(
                self.search.ks,
                self.search.hnsw_ef,
                self.search.rescoring,
                self.search.oversampling,
            )
        ]


def load_config(path: Path) -> Benchmark:
    return Benchmark.model_validate(yaml.safe_load(path.read_text()))


def save_config(path: Path, config: Benchmark) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            config.model_dump(exclude_none=True), sort_keys=False, allow_unicode=True
        )
    )


def collection_config(
    config: Benchmark, method: Method, dimensions: int
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "vectors": {
            "size": dimensions,
            "distance": "Cosine",
            "datatype": method.datatype,
            "memory": "cold",
        },
        "shard_number": 1,
        "replication_factor": 1,
        "hnsw_config": {
            "m": config.hnsw.m,
            "ef_construct": config.hnsw.ef_construct,
            "full_scan_threshold": 10,
            "max_indexing_threads": config.hnsw.max_indexing_threads,
            "memory": "cached",
        },
        "optimizers_config": {
            "default_segment_number": 1,
            "indexing_threshold": 1,
            "max_segment_size": 4_000_000,
        },
    }
    if quantization := method.quantization():
        body["quantization_config"] = quantization
    return body
