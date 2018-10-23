import pytest
import numpy as np
from spots.utils import is_sorted


@pytest.fixture()
def unsorted_array():
    return np.array([0, 3, 1, 4, -1])


@pytest.fixture()
def sorted_array(unsorted_array):
    return np.sort(unsorted_array)


def test_is_sorted(unsorted_array, sorted_array):
    assert is_sorted(sorted_array) is True
    assert is_sorted(unsorted_array) is False
    assert is_sorted(sorted_array[::-1], ascending=False) is True
    assert is_sorted(sorted_array[::-1], ascending=True) is False
