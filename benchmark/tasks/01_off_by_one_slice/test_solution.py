from window import windows


def test_windows_covers_the_whole_sequence():
    assert windows([1, 2, 3, 4], 2) == [[1, 2], [2, 3], [3, 4]]


def test_single_window_when_size_equals_length():
    assert windows([1, 2, 3], 3) == [[1, 2, 3]]


def test_empty_when_size_exceeds_length():
    assert windows([1, 2], 3) == []
