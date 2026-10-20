from itertools import product
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OperatingPoint(Settings):
    k: int = Field(ge=10)
    hnsw_ef: int = Field(ge=1)
    rescore: bool | None = None
    oversampling: float | None = Field(default=None, ge=1)


class Method(Settings):
    name: str = Field(pattern=r"^[a-z0-9_]+$")
    datatype: Literal["float32", "turbo4"] = "float32"
    quantizer: Literal["none", "scalar", "binary", "product", "turbo"] = "none"
    bits: Literal[1, 2, 4] | None = None
    compression: Literal["x8", "x16"] | None = None
    quantile: float = Field(default=0.99, ge=0.5, le=1)
    query_encoding: Literal["binary", "scalar4bits", "scalar8bits"] = "binary"
    search_settings: list[OperatingPoint] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def check_parameters(self) -> Self:
        if (self.quantizer == "turbo") != (self.bits is not None):
            raise ValueError("bits must be set exactly when quantizer is turbo")
        if (self.quantizer == "product") != (self.compression is not None):
            raise ValueError("compression must be set exactly for product quantization")
        for setting in self.search_settings or []:
            quantized = self.quantizer != "none"
            if quantized != (setting.rescore is not None):
                raise ValueError("Frozen rescoring must match the quantizer")
            if quantized != (setting.oversampling is not None):
                raise ValueError("Frozen oversampling must match the quantizer")
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
                return {
                    "binary": {
                        "encoding": "one_bit",
                        "query_encoding": self.query_encoding,
                        "memory": "pinned",
                    }
                }
            case "product":
                return {
                    "product": {"compression": self.compression, "memory": "pinned"}
                }
            case "turbo":
                return {"turbo": {"bits": f"bits{self.bits}", "memory": "pinned"}}


class Hnsw(Settings):
    m: int = Field(default=16, ge=4)
    ef_construct: int = Field(default=100, ge=16)
    hnsw_ef: int = Field(default=256, ge=1)


class Search(Settings):
    ks: list[int] = Field(default_factory=lambda: [10, 100], min_length=1)
    rescoring: list[bool] = Field(default_factory=lambda: [False, True], min_length=1)
    oversampling: list[float] = Field(
        default_factory=lambda: [1.0, 2.0, 4.0, 8.0], min_length=1
    )
    split: Literal["dev", "test"] = "test"
    warmup_queries: int = Field(default=50, ge=1)
    repeats: int = Field(default=5, ge=1)
    max_queries: int | None = Field(default=None, ge=1)
    hnsw_efs: list[int] | None = Field(default=None, min_length=1)
    cache_state: Literal["warm", "cold_start"] = "warm"
    cache_reset_command: list[str] | None = Field(default=None, min_length=1)
    cache_reset_timeout_seconds: float = Field(default=120, gt=0)

    @model_validator(mode="after")
    def check_grid(self) -> Self:
        if any(k < 10 for k in self.ks):
            raise ValueError("k must be >= 10 to report nDCG@10")
        if any(value < 1 for value in self.oversampling):
            raise ValueError("oversampling must be >= 1")
        if self.hnsw_efs and any(value < 1 for value in self.hnsw_efs):
            raise ValueError("hnsw_efs must be positive")
        if (self.cache_state == "cold_start") != (self.cache_reset_command is not None):
            raise ValueError("cold_start requires an explicit cache_reset_command")
        return self


class Benchmark(Settings):
    expected_qdrant_version: str = "1.19.2"
    datasets: list[str] = Field(
        default_factory=lambda: ["fiqa", "scifact", "nfcorpus", "quora"]
    )
    models: list[str] = Field(default_factory=lambda: ["cohere", "bge"])
    builds: int = Field(default=3, ge=1)
    seed: int = 42
    upload_batch_size: int = Field(default=256, ge=1)
    indexing_timeout_seconds: float = Field(default=7200, gt=0)
    hnsw: Hnsw = Field(default_factory=Hnsw)
    search: Search = Field(default_factory=Search)
    methods: list[Method] = Field(min_length=1)
    memory_pressure: bool = False
    require_telemetry: bool = False
    tuning_provenance: dict[str, Any] | None = None

    @model_validator(mode="after")
    def unique_methods(self) -> Self:
        if len({method.name for method in self.methods}) != len(self.methods):
            raise ValueError("Method names must be unique")
        return self

    def grid(self, method: Method) -> list[tuple[int, bool | None, float | None, int]]:
        if method.search_settings:
            return [
                (point.k, point.rescore, point.oversampling, point.hnsw_ef)
                for point in method.search_settings
            ]
        efs = self.search.hnsw_efs or [self.hnsw.hnsw_ef]
        if method.quantizer == "none":
            return [(k, None, None, ef) for k, ef in product(self.search.ks, efs)]
        return list(
            product(
                self.search.ks, self.search.rescoring, self.search.oversampling, efs
            )
        )


def load_config(path: Path) -> Benchmark:
    return Benchmark.model_validate(yaml.safe_load(path.read_text()))


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
            "max_indexing_threads": 1,
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
