"""Input parsing, validation and instance generation (module M2)."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from knapsack_algorithms import BRUTE_FORCE_MAX_ITEMS

#: Safety limit so the interface never tries to build an enormous DP table.
MAX_TABLE_CELLS = 5_000_000
MAX_ITEMS = 2000


class InputError(ValueError):
    """Raised for any invalid user input; the message is safe to show to the user."""


@dataclass
class Instance:
    weights: List[int]
    values: List[int]
    capacity: int
    names: List[str]
    label: str = "custom"

    @property
    def n(self) -> int:
        return len(self.weights)


def item_names(n: int) -> List[str]:
    """A, B, ..., Z, AA, AB, ... (spreadsheet-style names)."""
    names = []
    for k in range(n):
        s, m = "", k
        while True:
            s = chr(ord("A") + m % 26) + s
            m = m // 26 - 1
            if m < 0:
                break
        names.append(s)
    return names


_INT = re.compile(r"^[+-]?\d+$")


def parse_int_list(text: str, field: str) -> List[int]:
    """Parse '1, 3 4;5' into [1, 3, 4, 5]; reject anything that is not a whole number."""
    if text is None:
        raise InputError(f"{field}: no input given")
    tokens = [t for t in re.split(r"[,\s;]+", text.strip()) if t]
    out = []
    for t in tokens:
        if not _INT.match(t):
            raise InputError(f"{field}: '{t}' is not a whole number")
        out.append(int(t))
    return out


def parse_capacity(text: str) -> int:
    t = (text or "").strip()
    if not t:
        raise InputError("capacity: no input given")
    if not _INT.match(t):
        raise InputError(f"capacity: '{t}' is not a whole number")
    return int(t)


def validate_instance(weights: Sequence[int], values: Sequence[int], capacity: int) -> None:
    """Check an instance; raise :class:`InputError` with a clear message if invalid."""
    for name, seq in (("weights", weights), ("values", values)):
        for k, x in enumerate(seq):
            if isinstance(x, bool) or not isinstance(x, int):
                raise InputError(f"{name}[{k + 1}] = {x!r} is not a whole number")
    if isinstance(capacity, bool) or not isinstance(capacity, int):
        raise InputError(f"capacity {capacity!r} is not a whole number")
    if len(weights) != len(values):
        raise InputError(
            f"weights and values must have the same length "
            f"(got {len(weights)} weights and {len(values)} values)"
        )
    if len(weights) > MAX_ITEMS:
        raise InputError(f"too many items ({len(weights)}); the limit is {MAX_ITEMS}")
    for k, x in enumerate(weights):
        if x <= 0:
            raise InputError(f"weight of item {k + 1} must be a positive integer (got {x})")
    for k, x in enumerate(values):
        if x <= 0:
            raise InputError(f"value of item {k + 1} must be a positive integer (got {x})")
    if capacity < 0:
        raise InputError(f"capacity must be non-negative (got {capacity})")
    cells = (len(weights) + 1) * (capacity + 1)
    if cells > MAX_TABLE_CELLS:
        raise InputError(
            f"instance too large: the DP table would have {cells:,} cells "
            f"(limit {MAX_TABLE_CELLS:,}). Reduce the number of items or the capacity."
        )


def parse_instance(weights_text: str, values_text: str, capacity_text: str,
                   names: Optional[Sequence[str]] = None) -> Instance:
    """Parse and validate user-typed input into an :class:`Instance`."""
    weights = parse_int_list(weights_text, "weights")
    values = parse_int_list(values_text, "values")
    capacity = parse_capacity(capacity_text)
    validate_instance(weights, values, capacity)
    return Instance(weights, values, capacity, list(names) if names else item_names(len(weights)))


def check_brute_force_size(n: int) -> Optional[str]:
    """Return a message if brute force should be skipped for n items, else None."""
    if n > BRUTE_FORCE_MAX_ITEMS:
        return (f"Brute force skipped: {n} items would need 2^{n} subsets "
                f"(limit {BRUTE_FORCE_MAX_ITEMS} items).")
    return None


def random_instance(n: int, seed: int, weight_range=(1, 20), value_range=(1, 50),
                    capacity_fraction: float = 0.5) -> Instance:
    """Reproducible random instance; capacity = floor(fraction * total weight)."""
    if n < 0:
        raise InputError("number of items must be non-negative")
    lo_w, hi_w = weight_range
    lo_v, hi_v = value_range
    if not (1 <= lo_w <= hi_w) or not (1 <= lo_v <= hi_v):
        raise InputError("ranges must satisfy 1 <= low <= high")
    if not (0.0 <= capacity_fraction <= 1.0):
        raise InputError("capacity fraction must be between 0 and 1")
    rng = random.Random(seed)
    weights = [rng.randint(lo_w, hi_w) for _ in range(n)]
    values = [rng.randint(lo_v, hi_v) for _ in range(n)]
    capacity = int(capacity_fraction * sum(weights))
    validate_instance(weights, values, capacity)
    return Instance(weights, values, capacity, item_names(n), f"random(n={n}, seed={seed})")


def builtin_examples() -> dict:
    """Hand-built examples, including the worked example from the proposal."""
    def mk(label, w, v, cap):
        return Instance(list(w), list(v), cap, item_names(len(w)), label)
    examples = [
        mk("Worked example (W=7, items A-D)", [1, 3, 4, 5], [1, 4, 5, 7], 7),
        mk("Classic greedy counterexample (W=50)", [10, 20, 30], [60, 100, 120], 50),
        mk("Everything fits", [2, 3, 4], [3, 4, 5], 20),
        mk("Nothing fits", [10, 12, 15], [5, 6, 7], 9),
        mk("Capacity zero", [1, 2, 3], [4, 5, 6], 0),
    ]
    return {e.label: e for e in examples}
