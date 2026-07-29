import importlib.util
from pathlib import Path


def _load_benchmark_module():
    path = Path(__file__).parents[1] / "tools" / "robust_fspl_parallax_benchmark.py"
    spec = importlib.util.spec_from_file_location("robust_fspl_parallax_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_curated_benchmark_schema_and_truth_free_seed_policy():
    benchmark = _load_benchmark_module()

    assert benchmark.SCHEMA_VERSION >= 2
    assert set(benchmark.EVENTS) == {"2_755_3280", "0_599_2302"}
    assert all("ra_deg" in spec and "dec_deg" in spec for spec in benchmark.EVENTS.values())
    assert benchmark.REQUIRED_EVENT_FIELDS
    assert "truth" not in benchmark.EVENTS["2_755_3280"]
