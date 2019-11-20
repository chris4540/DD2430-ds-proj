from utils import PairIndexSet


def test_pair_idx_set():
    """
    Test the basic functionality of PairIndexSet
    """
    s = PairIndexSet()
    assert len(s) == 0

    s.add((1, 2))
    assert s.contains((1, 2))
    assert (1, 2) in s
    assert s.contains((2, 1))
    assert (2, 1) in s
    assert len(s) == 1

    s.add((100, 50))
    assert s.contains((100, 50))
    assert (100, 50) in s
    assert s.contains((50, 100))
    assert (50, 100) in s
    assert len(s) == 2

    s.clear()
    assert len(s) == 0
