from quantization_comparison_bench.evaluation import summarize_records
from quantization_comparison_bench.telemetry import parse_metrics

METRICS = """# HELP memory_allocated_bytes Total number of bytes allocated by the application
memory_allocated_bytes 77215536
memory_resident_bytes 190840832
collection_hardware_metric_cpu{id="probe"} 64
collection_hardware_metric_cpu{id="other"} 9
collection_hardware_metric_vector_io_read{id="probe"} 12
app_info{name="qdrant",version="1.19.2"} 1
"""


def test_parse_metrics_selects_memory_and_one_collection() -> None:
    parsed = parse_metrics(METRICS, "probe")
    assert parsed["memory"] == {
        "memory_allocated_bytes": 77215536,
        "memory_resident_bytes": 190840832,
    }
    assert parsed["hardware"] == {"cpu": 64, "vector_io_read": 12}
    assert parse_metrics(METRICS)["hardware"] == {}


def test_summary_reports_hardware_means_or_none() -> None:
    base = {
        "dataset": "d",
        "model": "m",
        "config": "c",
        "split": "test",
        "k": 10,
        "cache_state": "warm",
        "hnsw_ef": 64,
        "rescore": True,
        "oversampling": 2.0,
        "build": 0,
        "ann_recall": 1.0,
        "ndcg_at_10": 1.0,
        "request_ms": 1.0,
        "server_ms": 0.5,
    }
    rows = [
        {**base, "query_id": "a", "cpu": 10, "vector_io_read": 100},
        {**base, "query_id": "b", "cpu": 30, "vector_io_read": 300},
    ]
    group = summarize_records(rows)["groups"][0]
    assert group["mean_cpu_per_query"] == 20.0
    assert group["mean_vector_io_read_per_query"] == 200.0
    assert group["mean_payload_io_read_per_query"] is None
