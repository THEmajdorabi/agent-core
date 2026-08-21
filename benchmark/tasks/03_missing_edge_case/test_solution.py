from stats import mean


def test_mean_of_values():
    assert mean([1, 2, 3]) == 2.0


def test_mean_of_empty_is_zero():
    assert mean([]) == 0.0
