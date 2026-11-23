"""Tests for the table/highlight helpers (M3) - checks the path is highlighted correctly."""
from inputs import builtin_examples
from knapsack_algorithms import memoization, solve_all, tabulation
from viz import (ANSWER_CELL, PATH_CELL, SKIPPED_CELL, TAKEN_CELL, comparison_dataframe, dp_dataframe,
                 selected_items_dataframe, style_matrix, styled_dp_table, taken_cells)

INST = builtin_examples()["Worked example (W=7, items A-D)"]


def run():
    return tabulation(INST.weights, INST.values, INST.capacity)


def test_taken_cells_on_worked_example():
    assert taken_cells(run()) == {(3, 7), (2, 3)}        # C taken at (3,7), B taken at (2,3)


def test_highlight_path_on_worked_example():
    css = style_matrix(run(), INST)
    assert css.iloc[4, 7] == ANSWER_CELL                  # V[4][7] = 9, the answer
    assert css.iloc[3, 7] == TAKEN_CELL and css.iloc[2, 3] == TAKEN_CELL
    assert css.iloc[1, 0] == PATH_CELL and css.iloc[0, 0] == PATH_CELL
    assert css.iloc[2, 5] == ""                           # off-path cell not highlighted


def test_memoization_unvisited_cells_are_greyed():
    m = memoization(INST.weights, INST.values, INST.capacity)
    css = style_matrix(m, INST)
    assert any(SKIPPED_CELL == c for c in css.to_numpy().ravel())


def test_dataframes_and_styler_render():
    r = run()
    df = dp_dataframe(r, INST)
    assert df.shape == (5, 8) and df.iloc[4, 7] == 9
    assert "background-color" in styled_dp_table(r, INST).to_html()
    sel = selected_items_dataframe(r, INST)
    assert list(sel["Item"]) == ["B", "C"] and sel["Weight"].sum() == 7 and sel["Value"].sum() == 9


def test_comparison_table_three_exact_find_9_greedy_finds_8():
    res = solve_all(INST.weights, INST.values, INST.capacity)
    df = comparison_dataframe(res, optimal_value=9)
    values = dict(zip(df["Algorithm"], df["Value"]))
    assert sorted(values.values()) == [8, 9, 9, 9]
    assert df[df["Value"] == 8]["Optimal?"].iloc[0] == "NO"
