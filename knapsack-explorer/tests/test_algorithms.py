"""Correctness tests for M1 (known answers, edge cases, randomized cross-checks)."""
import random

import pytest

from inputs import builtin_examples, random_instance
from knapsack_algorithms import (BRUTE_FORCE_MAX_ITEMS, brute_force, greedy,
                                 memoization, solve_all, tabulation)

EXACT = [brute_force, memoization, tabulation]
ALL = EXACT + [greedy]


def check_valid(fn, w, v, W):
    """Solution validity: weight <= W, reported totals match the chosen items."""
    r = fn(w, v, W)
    assert len(set(r.selected)) == len(r.selected), "an item was selected twice"
    assert all(0 <= i < len(w) for i in r.selected)
    assert r.total_weight == sum(w[i] for i in r.selected)
    assert r.total_value == sum(v[i] for i in r.selected)
    assert r.total_weight <= W
    return r


# ---------------------------------------------------------------- known answers
@pytest.mark.parametrize("fn", EXACT)
def test_worked_example(fn):
    r = check_valid(fn, [1, 3, 4, 5], [1, 4, 5, 7], 7)
    assert r.total_value == 9
    assert r.selected == [1, 2]          # items B and C
    assert r.total_weight == 7


def test_worked_example_dp_table_matches_proposal():
    expected = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 4, 5, 5, 5, 5],
        [0, 1, 1, 4, 5, 6, 6, 9],
        [0, 1, 1, 4, 5, 7, 8, 9],
    ]
    r = tabulation([1, 3, 4, 5], [1, 4, 5, 7], 7)
    assert r.table == expected
    assert r.path == [(4, 7), (3, 7), (2, 3), (1, 0), (0, 0)]


def test_greedy_counterexample_worked_example():
    r = check_valid(greedy, [1, 3, 4, 5], [1, 4, 5, 7], 7)
    assert r.selected == [0, 3]          # A and D
    assert r.total_weight == 6 and r.total_value == 8
    assert r.total_value < tabulation([1, 3, 4, 5], [1, 4, 5, 7], 7).total_value


def test_greedy_counterexample_classic():
    w, v, W = [10, 20, 30], [60, 100, 120], 50
    assert greedy(w, v, W).total_value == 160
    for fn in EXACT:
        assert fn(w, v, W).total_value == 220


def test_hand_solved_small_instances():
    cases = [
        ([2, 3, 4, 5], [3, 4, 5, 6], 5, 7),            # items 1+2
        ([5, 4, 6, 3], [10, 40, 30, 50], 10, 90),      # items 2+4
        ([1, 2, 3], [6, 10, 12], 5, 22),               # items 2+3
        ([3, 4, 5], [4, 5, 6], 7, 9),                  # items 1+2
    ]
    for w, v, W, expected in cases:
        for fn in EXACT:
            assert check_valid(fn, w, v, W).total_value == expected, (fn.__name__, w, v, W)


# ------------------------------------------------------------------ edge cases
@pytest.mark.parametrize("fn", ALL)
def test_no_items(fn):
    r = check_valid(fn, [], [], 10)
    assert r.selected == [] and r.total_value == 0 and r.total_weight == 0


@pytest.mark.parametrize("fn", ALL)
def test_capacity_zero(fn):
    r = check_valid(fn, [1, 2, 3], [4, 5, 6], 0)
    assert r.selected == [] and r.total_value == 0


@pytest.mark.parametrize("fn", ALL)
def test_every_item_heavier_than_capacity(fn):
    r = check_valid(fn, [10, 12, 15], [5, 6, 7], 9)
    assert r.selected == [] and r.total_value == 0


@pytest.mark.parametrize("fn", ALL)
def test_all_items_fit(fn):
    r = check_valid(fn, [2, 3, 4], [3, 4, 5], 20)
    assert r.selected == [0, 1, 2] and r.total_value == 12


@pytest.mark.parametrize("fn", ALL)
def test_single_item_fits_and_does_not_fit(fn):
    assert check_valid(fn, [5], [9], 5).selected == [0]       # weight exactly equals capacity
    assert check_valid(fn, [5], [9], 4).selected == []


@pytest.mark.parametrize("fn", EXACT)
def test_identical_weights_and_values(fn):
    r = check_valid(fn, [3, 3, 3, 3], [7, 7, 7, 7], 7)
    assert r.total_value == 14 and len(r.selected) == 2


@pytest.mark.parametrize("fn", EXACT)
def test_identical_weights_different_values(fn):
    r = check_valid(fn, [2, 2, 2, 2], [5, 9, 1, 7], 4)
    assert r.total_value == 16


@pytest.mark.parametrize("fn", ALL)
def test_item_weight_exactly_capacity_among_others(fn):
    r = check_valid(fn, [7, 3, 4], [10, 4, 5], 7)
    assert r.total_value >= 9
    if fn is not greedy:
        assert r.total_value == 10


# ------------------------------------------------------ randomized cross-check
def test_randomized_cross_check_200_instances():
    """brute force, memoization and tabulation must agree on 200+ random instances (n <= 15)."""
    rng = random.Random(2024)
    for trial in range(250):
        n = rng.randint(0, 15)
        w = [rng.randint(1, 20) for _ in range(n)]
        v = [rng.randint(1, 50) for _ in range(n)]
        W = rng.randint(0, max(1, sum(w)))
        b, m, t = (check_valid(fn, w, v, W) for fn in EXACT)
        assert b.total_value == m.total_value == t.total_value, (trial, w, v, W)


def test_solution_validity_all_algorithms_random():
    rng = random.Random(7)
    for _ in range(100):
        n = rng.randint(1, 12)
        w = [rng.randint(1, 15) for _ in range(n)]
        v = [rng.randint(1, 40) for _ in range(n)]
        W = rng.randint(1, sum(w))
        for fn in ALL:
            check_valid(fn, w, v, W)


def test_greedy_never_exceeds_optimum_and_is_sometimes_worse():
    suboptimal = 0
    for seed in range(300):
        inst = random_instance(12, seed)
        g = greedy(inst.weights, inst.values, inst.capacity)
        opt = tabulation(inst.weights, inst.values, inst.capacity)
        assert g.total_value <= opt.total_value
        suboptimal += g.total_value < opt.total_value
    assert suboptimal > 0, "expected greedy to fall short on at least one random instance"


# --------------------------------------------------------------- counters etc.
def test_operation_counters():
    w, v, W = [1, 3, 4, 5], [1, 4, 5, 7], 7
    assert brute_force(w, v, W).ops["subsets_examined"] == 2 ** 4
    assert tabulation(w, v, W).ops["cells_filled"] == 4 * 7
    m = memoization(w, v, W)
    assert m.ops["cells_filled"] <= 4 * 7
    assert m.ops["calls"] >= m.ops["cells_filled"]
    assert greedy(w, v, W).ops["comparisons"] > 0


def test_memoization_table_agrees_with_tabulation_where_computed():
    inst = random_instance(10, 5)
    t = tabulation(inst.weights, inst.values, inst.capacity)
    m = memoization(inst.weights, inst.values, inst.capacity)
    for i in range(inst.n + 1):
        for w in range(inst.capacity + 1):
            if m.table[i][w] is not None:
                assert m.table[i][w] == t.table[i][w]


def test_memoization_handles_deep_recursion():
    n = 1200   # deeper than Python's default recursion limit of 1000
    r = memoization([1] * n, [1] * n, 10)
    assert r.total_value == 10


def test_brute_force_size_limit():
    n = BRUTE_FORCE_MAX_ITEMS + 1
    with pytest.raises(ValueError):
        brute_force([1] * n, [1] * n, 5)


def test_solve_all_skips_brute_force_when_large():
    n = BRUTE_FORCE_MAX_ITEMS + 5
    out = solve_all([1] * n, [1] * n, 5)
    assert "brute_force" not in out and set(out) == {"memoization", "tabulation", "greedy"}


def test_builtin_examples_consistent():
    for inst in builtin_examples().values():
        vals = {fn(inst.weights, inst.values, inst.capacity).total_value for fn in EXACT}
        assert len(vals) == 1
