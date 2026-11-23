"""Smoke tests for M5: the harness runs and writes CSV + PNG files (tiny sizes)."""
import os

import benchmark


def test_experiments_produce_files_and_sane_data(tmp_path):
    out = str(tmp_path)
    e1 = benchmark.run_e1(range(5, 9), repeats=1)
    benchmark.write_csv("e1.csv", e1, out); benchmark.plot_e1(e1, out)
    assert [r["brute_force_ops"] for r in e1] == [2 ** n for n in range(5, 9)]   # exactly 2^n subsets
    assert all(r["tabulation_ops"] == r["n"] * r["W"] for r in e1)

    e2 = benchmark.run_e2([20, 40], W=50, repeats=1); benchmark.plot_e2(e2, out)
    assert all(r["memo_states"] <= r["tab_cells"] for r in e2)

    e3 = benchmark.run_e3([100, 200], n=10, repeats=1); benchmark.plot_e3(e3, out)
    assert [r["cells"] for r in e3] == [1000, 2000]

    e4 = benchmark.run_e4(30); benchmark.plot_e4(e4, out)
    s = benchmark.summarize_e4(e4)
    assert s["instances"] == 30 and 0 <= s["suboptimal_pct"] <= 100 and s["worst_ratio"] <= 1.0

    for name in ("e1.csv", "e1_brute_vs_dp.png", "e2_memo_vs_tab.png", "e3_dp_vs_capacity.png",
                 "e4_greedy_quality.png"):
        assert os.path.exists(os.path.join(out, name))


def test_fit_scale_recovers_constant():
    xs = [1.0, 2.0, 3.0]
    assert abs(benchmark.fit_scale(xs, [2 * x for x in xs]) - 2.0) < 1e-12
