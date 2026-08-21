from rounding import round_money


def test_defaults_to_two_places():
    assert round_money(3.14159) == 3.14


def test_explicit_places_still_work():
    assert round_money(3.14159, 3) == 3.142
    assert round_money(3.7, 0) == 4
