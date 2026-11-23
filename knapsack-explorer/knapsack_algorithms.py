"""Core 0/1 Knapsack algorithms (module M1).

Four solvers share one result format, :class:`KnapsackResult`:

* ``brute_force``  - tries all 2^n subsets (ground truth, exponential time)
* ``memoization``  - top-down DP: recursion + cache, O(nW)
* ``tabulation``   - bottom-up DP: fills V[i][w] row by row, then backtracks
* ``greedy``       - sort by value/weight ratio, take what fits (heuristic)

Item indices in results are 0-based positions in the input lists. DP table
rows/columns follow the recurrence in the proposal: ``table[i][w]`` is the best
value using only the first ``i`` items with capacity ``w``.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from functools import cmp_to_key
from typing import Dict, List, Optional, Sequence, Tuple

#: Brute force is exponential, so we refuse to run it beyond this many items.
BRUTE_FORCE_MAX_ITEMS = 20


@dataclass
class KnapsackResult:
    """Shared result format returned by every algorithm."""

    algorithm: str
    selected: List[int]                 # 0-based indices of chosen items (ascending)
    total_weight: int
    total_value: int
    table: Optional[List[List[Optional[int]]]] = None   # (n+1) x (W+1); None cells = never computed
    path: List[Tuple[int, int]] = field(default_factory=list)  # backtracking cells (i, w), from (n, W) down
    ops: Dict[str, int] = field(default_factory=dict)   # operation counters
    elapsed: float = 0.0                # seconds (time.perf_counter)

    @property
    def primary_ops(self) -> int:
        """The single headline operation count used in comparison tables."""
        for key in ("subsets_examined", "cells_filled", "calls", "comparisons"):
            if key in self.ops:
                return self.ops[key]
        return 0


def _check(weights: Sequence[int], values: Sequence[int], capacity: int) -> None:
    if len(weights) != len(values):
        raise ValueError("weights and values must have the same length")
    if capacity < 0:
        raise ValueError("capacity must be non-negative")


def _summarize(weights, values, selected):
    selected = sorted(selected)
    return selected, sum(weights[i] for i in selected), sum(values[i] for i in selected)


# --------------------------------------------------------------------------
# Brute force
# --------------------------------------------------------------------------
def brute_force(weights: Sequence[int], values: Sequence[int], capacity: int) -> KnapsackResult:
    """Examine every one of the 2^n subsets and keep the best one that fits.

    Running weight/value totals are updated incrementally while recursing, so
    each subset costs O(1) amortised and the total time is O(2^n) (within the
    O(n * 2^n) bound that a naive per-subset sum would give). No pruning is
    done on purpose: ``subsets_examined`` is exactly 2^n.
    """
    _check(weights, values, capacity)
    n = len(weights)
    if n > BRUTE_FORCE_MAX_ITEMS:
        raise ValueError(
            f"brute force is limited to {BRUTE_FORCE_MAX_ITEMS} items (got {n}); "
            "it examines 2^n subsets"
        )
    start = time.perf_counter()
    best_value = -1
    best_set: List[int] = []
    chosen: List[int] = []
    examined = 0

    def rec(i: int, w: int, v: int) -> None:
        nonlocal best_value, best_set, examined
        if i == n:
            examined += 1
            if w <= capacity and v > best_value:
                best_value = v
                best_set = list(chosen)
            return
        rec(i + 1, w, v)                       # skip item i
        chosen.append(i)
        rec(i + 1, w + weights[i], v + values[i])  # take item i
        chosen.pop()

    rec(0, 0, 0)
    sel, tw, tv = _summarize(weights, values, best_set)
    return KnapsackResult("Brute force", sel, tw, tv, None, [],
                          {"subsets_examined": examined}, time.perf_counter() - start)


# --------------------------------------------------------------------------
# Top-down DP (memoization)
# --------------------------------------------------------------------------
def memoization(weights: Sequence[int], values: Sequence[int], capacity: int) -> KnapsackResult:
    """Top-down DP: ``best(i, w)`` with a cache so each subproblem is solved once.

    The returned table has ``None`` in every cell the recursion never needed,
    which makes the saving over tabulation visible. Counters: ``calls`` (every
    invocation of ``best``), ``cache_hits`` and ``cells_filled`` (distinct
    subproblems actually computed).
    """
    _check(weights, values, capacity)
    n = len(weights)
    # Recursion depth is n + 1; make sure Python allows it.
    needed = n + 200
    if sys.getrecursionlimit() < needed:
        sys.setrecursionlimit(needed)

    start = time.perf_counter()
    cache: Dict[Tuple[int, int], int] = {}
    calls = 0
    hits = 0

    def best(i: int, w: int) -> int:
        nonlocal calls, hits
        calls += 1
        if i == 0 or w == 0:
            return 0
        key = (i, w)
        if key in cache:
            hits += 1
            return cache[key]
        wi = weights[i - 1]
        res = best(i - 1, w)
        if wi <= w:
            take = values[i - 1] + best(i - 1, w - wi)
            if take > res:
                res = take
        cache[key] = res
        return res

    total = best(n, capacity)
    ops = {"calls": calls, "cache_hits": hits, "cells_filled": len(cache)}

    # Backtrack using the cache (every needed state is already stored).
    table: List[List[Optional[int]]] = [[None] * (capacity + 1) for _ in range(n + 1)]
    for w in range(capacity + 1):
        table[0][w] = 0
    for i in range(n + 1):
        table[i][0] = 0
    for (i, w), val in cache.items():
        table[i][w] = val
    selected, path = _backtrack(table, weights, n, capacity, lookup=lambda i, w: best(i, w))
    sel, tw, tv = _summarize(weights, values, selected)
    assert tv == total
    return KnapsackResult("Memoization (top-down DP)", sel, tw, tv, table, path, ops,
                          time.perf_counter() - start)


# --------------------------------------------------------------------------
# Bottom-up DP (tabulation)
# --------------------------------------------------------------------------
def _backtrack(table, weights, n, capacity, lookup=None):
    """Recover the chosen items. Start at (n, W); if V[i][w] != V[i-1][w] item i was taken."""
    get = lookup if lookup is not None else (lambda i, w: table[i][w])
    selected: List[int] = []
    path: List[Tuple[int, int]] = []
    w = capacity
    for i in range(n, 0, -1):
        path.append((i, w))
        if get(i, w) != get(i - 1, w):
            selected.append(i - 1)
            w -= weights[i - 1]
    path.append((0, w))
    return selected, path


def tabulation(weights: Sequence[int], values: Sequence[int], capacity: int) -> KnapsackResult:
    """Bottom-up DP: fill ``V[i][w]`` row by row, then backtrack.

    Recurrence: ``V[i][w] = V[i-1][w]`` if item i does not fit, otherwise
    ``max(V[i-1][w], v[i] + V[i-1][w - w[i]])``. Row 0 and column 0 are 0.
    Counters: ``cells_filled`` (= n * W) and ``comparisons`` (max() decisions).
    """
    _check(weights, values, capacity)
    n = len(weights)
    start = time.perf_counter()
    table = [[0] * (capacity + 1) for _ in range(n + 1)]
    cells = 0
    comparisons = 0
    for i in range(1, n + 1):
        wi, vi = weights[i - 1], values[i - 1]
        prev, row = table[i - 1], table[i]
        for w in range(1, capacity + 1):
            cells += 1
            if wi > w:
                row[w] = prev[w]
            else:
                comparisons += 1
                take = vi + prev[w - wi]
                row[w] = take if take > prev[w] else prev[w]
    selected, path = _backtrack(table, weights, n, capacity)
    sel, tw, tv = _summarize(weights, values, selected)
    return KnapsackResult("Tabulation (bottom-up DP)", sel, tw, tv, table, path,
                          {"cells_filled": cells, "comparisons": comparisons},
                          time.perf_counter() - start)


# --------------------------------------------------------------------------
# Greedy heuristic
# --------------------------------------------------------------------------
def greedy(weights: Sequence[int], values: Sequence[int], capacity: int) -> KnapsackResult:
    """Sort by value/weight ratio (descending) and take every item that still fits.

    Fast (O(n log n)) but NOT optimal for 0/1 knapsack. Ties in ratio keep the
    original item order. ``comparisons`` counts sort comparisons plus fit checks.
    """
    _check(weights, values, capacity)
    n = len(weights)
    start = time.perf_counter()
    comparisons = 0

    def cmp(a: int, b: int) -> int:
        nonlocal comparisons
        comparisons += 1
        # compare v[a]/w[a] with v[b]/w[b] exactly using cross-multiplication
        left, right = values[a] * weights[b], values[b] * weights[a]
        return -1 if left > right else (1 if left < right else 0)

    order = sorted(range(n), key=cmp_to_key(cmp))
    remaining = capacity
    chosen: List[int] = []
    for i in order:
        comparisons += 1  # fit check
        if weights[i] <= remaining:
            chosen.append(i)
            remaining -= weights[i]
    sel, tw, tv = _summarize(weights, values, chosen)
    return KnapsackResult("Greedy (value/weight)", sel, tw, tv, None, [],
                          {"comparisons": comparisons}, time.perf_counter() - start)


ALGORITHMS = {
    "brute_force": brute_force,
    "memoization": memoization,
    "tabulation": tabulation,
    "greedy": greedy,
}


def solve_all(weights: Sequence[int], values: Sequence[int], capacity: int,
              include_brute_force: bool = True) -> Dict[str, KnapsackResult]:
    """Run every algorithm on one instance (brute force only if it is small enough)."""
    out: Dict[str, KnapsackResult] = {}
    for key, fn in ALGORITHMS.items():
        if key == "brute_force" and (not include_brute_force or len(weights) > BRUTE_FORCE_MAX_ITEMS):
            continue
        out[key] = fn(weights, values, capacity)
    return out
