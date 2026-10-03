"""Unit tests for the latency middleware stats (no model needed)."""

from inferline.middleware import LatencyStats, percentile


def test_percentile_basic():
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([5.0], 99) == 5.0
    assert percentile([], 50) == 0.0


def test_stats_snapshot():
    s = LatencyStats()
    for v in (0.01, 0.02, 0.03, 0.04):
        s.record("/predict", v)
    snap = s.snapshot()["/predict"]
    assert snap["count"] == 4
    assert snap["mean_ms"] == 25.0
    assert snap["p50_ms"] == 25.0
    assert snap["p99_ms"] > snap["p50_ms"]


def test_stats_reset():
    s = LatencyStats()
    s.record("/x", 0.1)
    s.reset()
    assert s.snapshot() == {}
