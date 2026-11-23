"""Knapsack Explorer - Streamlit interface (module M3).

Run with:  streamlit run app.py
"""
import os
import tempfile

import pandas as pd
import streamlit as st

import benchmark
from inputs import (Instance, InputError, builtin_examples, check_brute_force_size,
                    parse_instance, random_instance)
from knapsack_algorithms import memoization, solve_all, tabulation, brute_force, greedy
from viz import (comparison_dataframe, dp_dataframe, selected_items_dataframe,
                 styled_dp_table)

MAX_DISPLAY_CELLS = 20_000   # DP tables bigger than this are offered as a CSV download instead

st.set_page_config(page_title="Knapsack Explorer", layout="wide")
st.title("Knapsack Explorer")
st.caption("Visualizing and benchmarking 0/1 Knapsack algorithms - CSCE 4110, Algorithms")

# ------------------------------------------------------------------ input form
st.sidebar.header("1. Choose an instance")
source = st.sidebar.radio("Input source", ["Built-in example", "Type my own", "Random (seeded)"])
inst = None
try:
    if source == "Built-in example":
        examples = builtin_examples()
        label = st.sidebar.selectbox("Example", list(examples.keys()))
        inst = examples[label]
    elif source == "Type my own":
        w_text = st.sidebar.text_input("Weights (comma or space separated)", "1, 3, 4, 5")
        v_text = st.sidebar.text_input("Values (same order)", "1, 4, 5, 7")
        c_text = st.sidebar.text_input("Capacity W", "7")
        inst = parse_instance(w_text, v_text, c_text)
    else:
        n = int(st.sidebar.number_input("Number of items", min_value=0, max_value=200, value=8, step=1))
        seed = int(st.sidebar.number_input("Random seed", min_value=0, value=42, step=1))
        w_lo, w_hi = st.sidebar.slider("Weight range", 1, 100, (1, 20))
        v_lo, v_hi = st.sidebar.slider("Value range", 1, 200, (1, 50))
        frac = st.sidebar.slider("Capacity as fraction of total weight", 0.0, 1.0, 0.5, 0.05)
        inst = random_instance(n, seed, (w_lo, w_hi), (v_lo, v_hi), frac)
except InputError as err:
    st.sidebar.error(f"Invalid input: {err}")
    st.error(f"Invalid input: {err}")
    st.stop()
except Exception as err:  # never show a raw crash to the user
    st.error(f"Could not build the instance: {err}")
    st.stop()

st.subheader("Instance")
st.dataframe(pd.DataFrame({"Item": inst.names, "Weight": inst.weights, "Value": inst.values,
                           "Value / weight": [round(v / w, 3) for w, v in zip(inst.weights, inst.values)]}),
             hide_index=True)
st.write(f"**Capacity W = {inst.capacity}**, **n = {inst.n}** items, total weight = {sum(inst.weights)}")

tab_solve, tab_compare, tab_bench = st.tabs(["Solve and DP table", "Compare algorithms", "Benchmark"])

# --------------------------------------------------------------------- Solve tab
with tab_solve:
    algo = st.radio("Algorithm", ["Tabulation (bottom-up DP)", "Memoization (top-down DP)",
                                  "Brute force", "Greedy (value/weight)"], horizontal=True)
    skip_msg = check_brute_force_size(inst.n) if algo == "Brute force" else None
    if skip_msg:
        st.warning(skip_msg)
    else:
        fn = {"Tabulation (bottom-up DP)": tabulation, "Memoization (top-down DP)": memoization,
              "Brute force": brute_force, "Greedy (value/weight)": greedy}[algo]
        res = fn(inst.weights, inst.values, inst.capacity)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total value", res.total_value)
        c2.metric("Total weight", f"{res.total_weight} / {inst.capacity}")
        c3.metric("Runtime", f"{res.elapsed * 1000:.3f} ms")
        c4.metric("Main operation count", f"{res.primary_ops:,}")
        st.write("**Operation counters:** " + ", ".join(f"{k} = {v:,}" for k, v in res.ops.items()))

        st.subheader("Selected items")
        sel = selected_items_dataframe(res, inst)
        if sel.empty:
            st.info("No items were selected (nothing fits, or capacity is 0).")
        else:
            st.dataframe(sel, hide_index=True)
            st.write(f"Total weight **{res.total_weight}**, total value **{res.total_value}**")

        if res.table is not None:
            st.subheader("Dynamic programming table V[i][w]")
            cells = (inst.n + 1) * (inst.capacity + 1)
            if cells <= MAX_DISPLAY_CELLS:
                st.markdown(
                    "<span style='background:#2e7d32;color:#fff;padding:2px 6px'>answer V[n][W]</span> "
                    "<span style='background:#5cb85c;padding:2px 6px'>item taken at this cell</span> "
                    "<span style='background:#ffe08a;padding:2px 6px'>backtracking path</span> "
                    "<span style='background:#f4f4f4;color:#999;padding:2px 6px'>never computed (memoization)</span>",
                    unsafe_allow_html=True)
                st.dataframe(styled_dp_table(res, inst))
                st.caption("Backtracking: start at (n, W). If V[i][w] differs from the cell above, item i was "
                           "taken and w decreases by its weight; then move up one row.")
            else:
                st.info(f"The table has {cells:,} cells - too large to display. Download it instead.")
                st.download_button("Download DP table (CSV)", dp_dataframe(res, inst).to_csv(),
                                   file_name="dp_table.csv")
            st.write("Backtracking path (row, remaining capacity): "
                     + " -> ".join(f"({i}, {w})" for i, w in res.path))

# ------------------------------------------------------------------ Compare tab
with tab_compare:
    msg = check_brute_force_size(inst.n)
    if msg:
        st.warning(msg)
    results = solve_all(inst.weights, inst.values, inst.capacity)
    optimum = results["tabulation"].total_value
    df = comparison_dataframe(results, optimal_value=optimum)
    st.dataframe(df, hide_index=True)
    g = results["greedy"]
    if g.total_value < optimum:
        st.error(f"Greedy is NOT optimal here: it found value {g.total_value} but the optimum is {optimum} "
                 f"({g.total_value / optimum:.1%} of optimal).")
    else:
        st.success("All algorithms found the optimal value on this instance "
                   "(greedy is optimal here, but it is not guaranteed to be).")
    st.caption("Operations: subsets examined (brute force), recursive calls (memoization), table cells filled "
               "(tabulation), comparisons (greedy).")

# ---------------------------------------------------------------- Benchmark tab
SAVED = {
    "E1 - Brute force vs DP (n = 5..20, fixed W)": ("e1_brute_vs_dp.png", "e1_brute_vs_dp.csv"),
    "E2 - Memoization vs tabulation (n = 50..500)": ("e2_memo_vs_tab.png", "e2_memo_vs_tab.csv"),
    "E3 - DP runtime vs capacity W (fixed n)": ("e3_dp_vs_capacity.png", "e3_dp_vs_capacity.csv"),
    "E4 - Greedy quality on 500 random instances": ("e4_greedy_quality.png", "e4_greedy_quality.csv"),
}


def show_results(folder):
    found = False
    for title, (png, csv_name) in SAVED.items():
        p, c = os.path.join(folder, png), os.path.join(folder, csv_name)
        if os.path.exists(p):
            found = True
            st.subheader(title)
            st.image(p)
            if os.path.exists(c):
                with st.expander("Show data"):
                    st.dataframe(pd.read_csv(c), hide_index=True)
    return found


with tab_bench:
    st.write("Experiments E1-E4 measure runtime (median of repeated runs), operation counts and memory, "
             "and plot measured growth against scaled theoretical curves.")
    if st.button("Run quick benchmark now (smaller sizes, about a minute)"):
        with st.spinner("Running experiments..."):
            tmp = tempfile.mkdtemp(prefix="knapsack_bench_")
            data = benchmark.run_all(quick=True, out_dir=tmp)
        st.success("Quick benchmark finished.")
        show_results(tmp)
        s = benchmark.summarize_e4(data["E4"])
        st.write(f"Greedy was suboptimal on {s['suboptimal_count']} of {s['instances']} instances "
                 f"({s['suboptimal_pct']:.1f}%).")
    else:
        st.write("Saved results from the full run (`python benchmark.py`):")
        if not show_results(benchmark.RESULTS_DIR):
            st.info("No saved results found. Run `python benchmark.py` or use the button above.")
