"""Benchmarking harness and experiments E1-E4 (module M5).

Usage:
    python benchmark.py            # run all experiments, write CSVs + PNG plots to results/
    python benchmark.py --quick    # smaller sizes / fewer repeats (fast smoke run)
    python benchmark.py E1 E3      # run only some experiments
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import tracemalloc
from typing import Callable, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from inputs import random_instance  # noqa: E402
from knapsack_algorithms import brute_force, greedy, memoization, tabulation  # noqa: E402

RESULTS_DIR = "results"
REPEATS = 5


# ----------------------------------------------------------------- measuring
def median_runtime(fn: Callable, args, repeats: int = REPEATS):
    """Run fn(*args) ``repeats`` times; return (median seconds, last result)."""
    times, res = [], None
    for _ in range(repeats):
        res = fn(*args)
        times.append(res.elapsed)
    return statistics.median(times), res


def peak_memory(fn: Callable, args) -> int:
    """Peak extra memory (bytes) allocated while running fn(*args), via tracemalloc."""
    tracemalloc.start()
    try:
        fn(*args)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def write_csv(name: str, rows: List[dict], out_dir: str = RESULTS_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def fit_scale(xs_theory: List[float], ys: List[float]) -> float:
    """Least-squares constant c minimising sum (y - c*f(x))^2."""
    num = sum(f * y for f, y in zip(xs_theory, ys))
    den = sum(f * f for f in xs_theory)
    return num / den if den else 0.0


# --------------------------------------------------------------- experiments
def run_e1(sizes=range(5, 21), W: int = 100, repeats: int = REPEATS) -> List[dict]:
    """E1: brute force vs DP, n = 5..20, fixed W."""
    rows = []
    for n in sizes:
        inst = random_instance(n, seed=1000 + n, capacity_fraction=0.5)
        # keep W fixed across n: cap/pad the instance capacity at W
        args = (inst.weights, inst.values, W)
        tb, rb = median_runtime(brute_force, args, repeats)
        tt, rt = median_runtime(tabulation, args, repeats)
        assert rb.total_value == rt.total_value
        rows.append({"n": n, "W": W,
                     "brute_force_s": tb, "tabulation_s": tt,
                     "brute_force_ops": rb.ops["subsets_examined"],
                     "tabulation_ops": rt.ops["cells_filled"],
                     "value": rt.total_value})
    return rows


def run_e2(sizes=range(50, 501, 50), W: int = 500, repeats: int = REPEATS) -> List[dict]:
    """E2: memoization vs tabulation, n = 50..500, fixed W."""
    rows = []
    for n in sizes:
        inst = random_instance(n, seed=2000 + n)
        args = (inst.weights, inst.values, W)
        tm, rm = median_runtime(memoization, args, repeats)
        tt, rt = median_runtime(tabulation, args, repeats)
        assert rm.total_value == rt.total_value
        rows.append({"n": n, "W": W,
                     "memoization_s": tm, "tabulation_s": tt,
                     "memo_calls": rm.ops["calls"], "memo_cache_hits": rm.ops["cache_hits"],
                     "memo_states": rm.ops["cells_filled"], "tab_cells": rt.ops["cells_filled"],
                     "memo_peak_kb": round(peak_memory(memoization, args) / 1024, 1),
                     "tab_peak_kb": round(peak_memory(tabulation, args) / 1024, 1),
                     "value": rt.total_value})
    return rows


def run_e3(capacities=range(500, 5001, 500), n: int = 100, repeats: int = REPEATS) -> List[dict]:
    """E3: DP with fixed n and W = 500..5000 (pseudo-polynomial behaviour)."""
    inst = random_instance(n, seed=3000)
    rows = []
    for W in capacities:
        args = (inst.weights, inst.values, W)
        tt, rt = median_runtime(tabulation, args, repeats)
        rows.append({"n": n, "W": W, "tabulation_s": tt, "cells": rt.ops["cells_filled"],
                     "peak_kb": round(peak_memory(tabulation, args) / 1024, 1),
                     "value": rt.total_value})
    return rows


def run_e4(count: int = 500) -> List[dict]:
    """E4: greedy quality on ``count`` random instances (n = 8..25)."""
    import random
    rng = random.Random(4000)
    rows = []
    for k in range(count):
        n = rng.randint(8, 25)
        inst = random_instance(n, seed=10_000 + k, weight_range=(1, 30), value_range=(1, 60),
                               capacity_fraction=0.5)
        opt = tabulation(inst.weights, inst.values, inst.capacity).total_value
        g = greedy(inst.weights, inst.values, inst.capacity).total_value
        assert g <= opt
        rows.append({"instance": k, "n": n, "capacity": inst.capacity, "optimal": opt,
                     "greedy": g, "ratio": (g / opt) if opt else 1.0, "suboptimal": int(g < opt)})
    return rows


# ------------------------------------------------------------------- plotting
def plot_e1(rows, out_dir=RESULTS_DIR) -> str:
    ns = [r["n"] for r in rows]
    bf = [r["brute_force_s"] for r in rows]
    dp = [r["tabulation_s"] for r in rows]
    c_bf = fit_scale([2.0 ** n for n in ns], bf)
    c_dp = fit_scale([n * rows[0]["W"] for n in ns], dp)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.semilogy(ns, bf, "o-", label="Brute force (measured)", color="#c0392b")
    ax.semilogy(ns, [c_bf * 2.0 ** n for n in ns], "--", label=r"$c\cdot 2^n$ (scaled)", color="#e59866")
    ax.semilogy(ns, dp, "s-", label="Tabulation DP (measured)", color="#1f77b4")
    ax.semilogy(ns, [c_dp * n * rows[0]["W"] for n in ns], "--", label=r"$c\cdot nW$ (scaled)", color="#7fb3d5")
    ax.set_xlabel("number of items n"); ax.set_ylabel("median runtime (s, log scale)")
    ax.set_title(f"E1: brute force vs DP (W = {rows[0]['W']})")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    return _save(fig, "e1_brute_vs_dp.png", out_dir)


def plot_e2(rows, out_dir=RESULTS_DIR) -> str:
    ns = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for key, label, color in (("memoization_s", "Memoization", "#8e44ad"),
                              ("tabulation_s", "Tabulation", "#1f77b4")):
        ys = [r[key] for r in rows]
        c = fit_scale([n * rows[0]["W"] for n in ns], ys)
        ax.plot(ns, ys, "o-", label=f"{label} (measured)", color=color)
        ax.plot(ns, [c * n * rows[0]["W"] for n in ns], "--", label=f"{label}: " + r"$c\cdot nW$", color=color, alpha=0.5)
    ax.set_xlabel("number of items n"); ax.set_ylabel("median runtime (s)")
    ax.set_title(f"E2: memoization vs tabulation (W = {rows[0]['W']})")
    ax.grid(True, alpha=0.3); ax.legend()
    return _save(fig, "e2_memo_vs_tab.png", out_dir)


def plot_e3(rows, out_dir=RESULTS_DIR) -> str:
    Ws = [r["W"] for r in rows]
    ys = [r["tabulation_s"] for r in rows]
    n = rows[0]["n"]
    c = fit_scale([n * W for W in Ws], ys)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(Ws, ys, "o-", label="Tabulation (measured)", color="#1f77b4")
    ax.plot(Ws, [c * n * W for W in Ws], "--", label=r"$c\cdot nW$ (scaled)", color="#7fb3d5")
    ax.set_xlabel("capacity W"); ax.set_ylabel("median runtime (s)")
    ax.set_title(f"E3: DP runtime vs capacity (n = {n})")
    ax.grid(True, alpha=0.3); ax.legend()
    return _save(fig, "e3_dp_vs_capacity.png", out_dir)


def plot_e4(rows, out_dir=RESULTS_DIR) -> str:
    ratios = [r["ratio"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.hist(ratios, bins=25, color="#16a085", edgecolor="white")
    ax.set_xlabel("greedy value / optimal value"); ax.set_ylabel("number of instances")
    ax.set_title(f"E4: greedy quality on {len(rows)} random instances")
    ax.grid(True, alpha=0.3)
    return _save(fig, "e4_greedy_quality.png", out_dir)


def _save(fig, name, out_dir) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    return path


def summarize_e4(rows) -> Dict[str, float]:
    sub = [r for r in rows if r["suboptimal"]]
    ratios = [r["ratio"] for r in rows]
    return {
        "instances": len(rows),
        "suboptimal_count": len(sub),
        "suboptimal_pct": 100.0 * len(sub) / len(rows),
        "mean_ratio_all": statistics.mean(ratios),
        "mean_gap_pct_all": 100.0 * (1 - statistics.mean(ratios)),
        "mean_gap_pct_when_suboptimal": 100.0 * (1 - statistics.mean(r["ratio"] for r in sub)) if sub else 0.0,
        "worst_ratio": min(ratios),
    }


def run_all(which=("E1", "E2", "E3", "E4"), quick: bool = False, out_dir: str = RESULTS_DIR) -> Dict[str, list]:
    reps = 2 if quick else REPEATS
    out: Dict[str, list] = {}
    if "E1" in which:
        out["E1"] = run_e1(range(5, 15) if quick else range(5, 21), repeats=reps)
        write_csv("e1_brute_vs_dp.csv", out["E1"], out_dir); plot_e1(out["E1"], out_dir)
    if "E2" in which:
        out["E2"] = run_e2(range(50, 201, 50) if quick else range(50, 501, 50), repeats=reps)
        write_csv("e2_memo_vs_tab.csv", out["E2"], out_dir); plot_e2(out["E2"], out_dir)
    if "E3" in which:
        out["E3"] = run_e3(range(500, 2501, 500) if quick else range(500, 5001, 500), repeats=reps)
        write_csv("e3_dp_vs_capacity.csv", out["E3"], out_dir); plot_e3(out["E3"], out_dir)
    if "E4" in which:
        out["E4"] = run_e4(100 if quick else 500)
        write_csv("e4_greedy_quality.csv", out["E4"], out_dir); plot_e4(out["E4"], out_dir)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run knapsack benchmark experiments E1-E4")
    ap.add_argument("experiments", nargs="*", default=["E1", "E2", "E3", "E4"])
    ap.add_argument("--quick", action="store_true", help="smaller sizes, fewer repeats")
    ap.add_argument("--out", default=RESULTS_DIR)
    a = ap.parse_args()
    res = run_all(tuple(x.upper() for x in a.experiments), a.quick, a.out)
    for k, rows in res.items():
        print(f"{k}: {len(rows)} rows -> {a.out}/")
    if "E4" in res:
        print("E4 summary:", summarize_e4(res["E4"]))
