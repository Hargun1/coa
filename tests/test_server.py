import pytest

from adaptive_memory.server import run_demo


def test_demo_uses_identical_traces_and_all_policies():
    data = run_demo({"requests": 120, "warmup": 20, "seed": 5, "workload": "mixed_scan"})
    assert len(data["runs"]) == len(data["summary"]) == 7
    assert len({r["trace_sha256"] for r in data["runs"]}) == 1
    assert all(r["measured_requests"] == 100 for r in data["runs"])


@pytest.mark.parametrize("payload", [[], {"unknown": 1}, {"requests": True}, {"requests": 50001},
                                    {"warmup": -1}, {"warmup": 15000}, {"seed": -1},
                                    {"l3_lines": 513}, {"horizon": 0}, {"workload": "typo"}])
def test_demo_rejects_invalid_requests(payload):
    with pytest.raises(ValueError):
        run_demo(payload)
