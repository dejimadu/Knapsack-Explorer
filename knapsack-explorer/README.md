# Knapsack Explorer: Visualizing and Benchmarking 0/1 Knapsack Algorithms

**Course:** CSCE 4110 - Algorithms  **Team:** DCE
**Team members:** Daniel Ejimadu (Team Manager)

A Python application that solves the 0/1 Knapsack problem with four algorithms, shows step by step how
the dynamic programming (DP) table is built and which cells the backtracking path visits, and measures how
each algorithm scales as the input grows.

## Features

- **Four algorithms**, all returning one shared result format (`KnapsackResult`):
  brute force (all 2^n subsets), top-down DP (memoization), bottom-up DP (tabulation + backtracking),
  and a greedy value/weight heuristic.
- **Interactive DP table** (Streamlit): the filled table `V[i][w]`, the backtracking path highlighted,
  the cells where an item is taken, and the selected items with total weight and value.
- **Side-by-side comparison**: value found, runtime, operation count, optimality of each algorithm.
- **Input validation** with clear error messages (negative / zero / non-numeric values, mismatched
  list lengths, oversized instances) and **seeded random instances** for reproducible experiments.
- **Benchmark experiments E1-E4** with CSV export and plots of measured growth vs. scaled theory.
- **Automated tests** (pytest): known answers, edge cases, 250 randomized cross-checks of brute force vs. DP.

## Requirements

Python 3.11 or newer. Libraries (see `requirements.txt`): streamlit, pandas, numpy, matplotlib, jinja2, pytest.
No dataset or API key is needed.

## Setup

```bash
git clone <repository-url>
cd knapsack-explorer
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

(`./run.sh` on macOS/Linux or `run.bat` on Windows does the setup and launches the app in one step.)

## Running

| Task | Command |
| --- | --- |
| Launch the interface | `streamlit run app.py` (opens http://localhost:8501) |
| Run the test suite | `pytest` |
| Run all benchmark experiments (writes `results/`) | `python benchmark.py` |
| Quick benchmark (smaller sizes) | `python benchmark.py --quick` |
| Run only some experiments | `python benchmark.py E1 E3` |

## Example usage

**In the interface:** choose *Built-in example -> Worked example*. The *Solve and DP table* tab shows the
filled table, the highlighted path `(4,7) -> (3,7) -> (2,3) -> (1,0) -> (0,0)`, and the selection
**B and C (weight 7, value 9)**. The *Compare algorithms* tab shows the three exact algorithms finding 9
and greedy finding 8. Choose *Type my own* and enter a negative weight to see a validation message.

**From Python:**

```python
from knapsack_algorithms import tabulation, greedy

weights, values, capacity = [1, 3, 4, 5], [1, 4, 5, 7], 7
r = tabulation(weights, values, capacity)
print(r.selected, r.total_weight, r.total_value)   # [1, 2] 7 9   (items B and C)
print(r.table[4][7], r.path)                        # 9 [(4, 7), (3, 7), (2, 3), (1, 0), (0, 0)]
print(greedy(weights, values, capacity).total_value)  # 8  -> greedy is not optimal
```

## Project structure

| File | Module | Purpose |
| --- | --- | --- |
| `knapsack_algorithms.py` | M1 | The four algorithms, `KnapsackResult`, operation counters |
| `inputs.py` | M2 | Parsing, validation, seeded random generator, built-in examples |
| `app.py`, `viz.py` | M3 | Streamlit interface; `viz.py` holds the table/colour helpers (no Streamlit dependency, so they are unit-tested) |
| `tests/` | M4 | `test_algorithms.py`, `test_inputs.py`, `test_viz.py`, `test_benchmark.py` |
| `benchmark.py` | M5 | Timing harness, experiments E1-E4, plots, CSV export |
| `README.md`, `run.sh`, `run.bat`, `requirements.txt` | M6 | Setup and documentation |
| `results/` | | Saved CSVs and PNG plots from the full benchmark run |
| `docs/` | | Demonstration video script and contribution statement |

## Notes

- Brute force is capped at 20 items (2^20 ~ 1 million subsets); the interface skips it above that.
- The DP running time O(nW) is pseudo-polynomial: it depends on the numeric size of W (experiment E3).
- The interface refuses instances whose DP table would exceed 5,000,000 cells, and shows tables larger than
  20,000 cells as a CSV download instead of on screen.
- Timings depend on the machine; the shapes of the curves (not the absolute times) are what the experiments test.
