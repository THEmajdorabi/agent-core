from accumulator import add_item


def test_each_call_starts_fresh():
    assert add_item("a") == ["a"]
    assert add_item("b") == ["b"]


def test_explicit_basket_accumulates():
    basket = []
    add_item("a", basket)
    add_item("b", basket)
    assert basket == ["a", "b"]
