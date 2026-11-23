"""Table-building and colouring helpers used by the Streamlit app (part of M3).

Kept separate from ``app.py`` so they can be unit-tested without Streamlit.
"""
from __future__ import annotations

from typing import List, Optional, Set, Tuple

import pandas as pd

from inputs import Instance
from knapsack_algorithms import KnapsackResult

# Colours (background, text)
PATH_CELL = "background-color: #ffe08a; color: #000000"          # visited by backtracking
TAKEN_CELL = "background-color: #5cb85c; color: #000000; font-weight: bold"  # item taken here
SKIPPED_CELL = "background-color: #f4f4f4; color: #bbbbbb"        # memoization: never computed
ANSWER_CELL = "background-color: #2e7d32; color: #ffffff; font-weight: bold"  # V[n][W]


def dp_dataframe(result: KnapsackResult, inst: Instance) -> pd.DataFrame:
    """Return the DP table as a DataFrame with readable row/column labels."""
    if result.table is None:
        raise ValueError(f"{result.algorithm} does not produce a DP table")
    index = ["0 (none)"] + [
        f"{i + 1} ({inst.names[i]}: w{inst.weights[i]}, v{inst.values[i]})" for i in range(inst.n)
    ]
    cols = [f"w={w}" for w in range(inst.capacity + 1)]
    return pd.DataFrame(result.table, index=index, columns=cols, dtype="object").fillna("")


def taken_cells(result: KnapsackResult) -> Set[Tuple[int, int]]:
    """Cells (i, w) on the backtracking path where item i was taken (value differs from the cell above)."""
    cells = set()
    table = result.table
    for i, w in result.path:
        if i >= 1 and table[i][w] != table[i - 1][w]:
            cells.add((i, w))
    return cells


def style_matrix(result: KnapsackResult, inst: Instance) -> pd.DataFrame:
    """A DataFrame of CSS strings (same shape as the table) for ``Styler.apply(axis=None)``."""
    n, W = inst.n, inst.capacity
    css = [["" for _ in range(W + 1)] for _ in range(n + 1)]
    for i in range(n + 1):
        for w in range(W + 1):
            if result.table[i][w] is None:
                css[i][w] = SKIPPED_CELL
    for i, w in result.path:
        css[i][w] = PATH_CELL
    for i, w in taken_cells(result):
        css[i][w] = TAKEN_CELL
    css[n][W] = ANSWER_CELL
    index = dp_dataframe(result, inst).index
    cols = [f"w={w}" for w in range(W + 1)]
    return pd.DataFrame(css, index=index, columns=cols)


def styled_dp_table(result: KnapsackResult, inst: Instance):
    """A pandas Styler: path highlighted yellow, taken-item cells green, answer cell dark green."""
    df = dp_dataframe(result, inst)
    css = style_matrix(result, inst)
    return df.style.apply(lambda _: css, axis=None)


def selected_items_dataframe(result: KnapsackResult, inst: Instance) -> pd.DataFrame:
    rows = [{"Item": inst.names[i], "Weight": inst.weights[i], "Value": inst.values[i]}
            for i in result.selected]
    return pd.DataFrame(rows, columns=["Item", "Weight", "Value"])


def comparison_dataframe(results: dict, optimal_value: Optional[int] = None) -> pd.DataFrame:
    """Side-by-side comparison table (value, weight, runtime, operation count, optimality)."""
    rows: List[dict] = []
    for r in results.values():
        row = {
            "Algorithm": r.algorithm,
            "Value": r.total_value,
            "Weight": r.total_weight,
            "Runtime (ms)": round(r.elapsed * 1000, 4),
            "Operations": r.primary_ops,
            "Operation type": next(iter(k for k in ("subsets_examined", "cells_filled", "calls", "comparisons")
                                        if k in r.ops)),
        }
        if "cache_hits" in r.ops:
            row["Cache hits"] = r.ops["cache_hits"]
        if optimal_value is not None:
            row["Optimal?"] = "yes" if r.total_value == optimal_value else "NO"
            row["Value / optimum"] = round(r.total_value / optimal_value, 4) if optimal_value else 1.0
        rows.append(row)
    return pd.DataFrame(rows)
