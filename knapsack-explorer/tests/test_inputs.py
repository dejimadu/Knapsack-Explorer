"""Tests for M2: input parsing, validation and the random generator."""
import pytest

from inputs import (InputError, MAX_TABLE_CELLS, item_names, parse_instance,
                    random_instance, validate_instance)


def test_valid_input_parses():
    inst = parse_instance("1, 3, 4, 5", "1 4 5 7", "7")
    assert inst.weights == [1, 3, 4, 5] and inst.values == [1, 4, 5, 7] and inst.capacity == 7
    assert inst.names == ["A", "B", "C", "D"]


def test_empty_item_lists_are_valid():
    assert parse_instance("", "", "5").n == 0


@pytest.mark.parametrize("w,v,c,fragment", [
    ("1,-3", "1,2", "5", "positive"),               # negative weight
    ("1,2", "1,-2", "5", "positive"),               # negative value
    ("0,2", "1,2", "5", "positive"),                # zero weight
    ("1,2,3", "1,2", "5", "same length"),           # mismatched lengths
    ("a,2", "1,2", "5", "not a whole number"),      # non-numeric
    ("1.5,2", "1,2", "5", "not a whole number"),    # non-integer
    ("1,2", "1,2", "-1", "non-negative"),           # negative capacity
    ("1,2", "1,2", "abc", "not a whole number"),    # non-numeric capacity
    ("1,2", "1,2", "", "no input"),                 # missing capacity
])
def test_invalid_input_gives_clear_error(w, v, c, fragment):
    with pytest.raises(InputError) as exc:
        parse_instance(w, v, c)
    assert fragment in str(exc.value)


def test_validate_rejects_non_int_types():
    with pytest.raises(InputError):
        validate_instance([1.5], [2], 3)
    with pytest.raises(InputError):
        validate_instance([True], [2], 3)


def test_oversized_instance_rejected():
    with pytest.raises(InputError):
        validate_instance([1] * 100, [1] * 100, MAX_TABLE_CELLS)


def test_random_instance_is_reproducible():
    a, b = random_instance(10, seed=3), random_instance(10, seed=3)
    assert (a.weights, a.values, a.capacity) == (b.weights, b.values, b.capacity)
    c = random_instance(10, seed=4)
    assert (a.weights, a.values) != (c.weights, c.values)


def test_random_instance_respects_ranges_and_fraction():
    inst = random_instance(30, 1, weight_range=(5, 9), value_range=(10, 12), capacity_fraction=0.25)
    assert all(5 <= x <= 9 for x in inst.weights) and all(10 <= x <= 12 for x in inst.values)
    assert inst.capacity == int(0.25 * sum(inst.weights))


def test_random_instance_bad_arguments():
    with pytest.raises(InputError):
        random_instance(5, 1, weight_range=(0, 5))
    with pytest.raises(InputError):
        random_instance(5, 1, capacity_fraction=1.5)


def test_item_names():
    n = item_names(30)
    assert n[:3] == ["A", "B", "C"] and n[25] == "Z" and n[26] == "AA" and len(set(n)) == 30
