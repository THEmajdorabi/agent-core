"""Collecting items."""


def add_item(item, basket=[]):
    """Append `item` to `basket` and return it.

    Called without a basket, each call starts from an empty one.
    """
    basket.append(item)
    return basket
