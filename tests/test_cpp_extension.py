from __future__ import annotations


def test_cpp_grid_extension_imports():
    import jacscanomaly._cpp_grid as cpp_grid

    assert hasattr(cpp_grid, "run_grid")
    assert hasattr(cpp_grid, "fit_pspl")
