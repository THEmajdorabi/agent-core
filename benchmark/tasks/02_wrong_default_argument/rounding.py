"""Money rounding helpers."""


def round_money(amount, places=0):
    """Round a monetary amount to `places` decimal places.

    Money defaults to 2 decimal places, not 0.
    """
    return round(amount, places)
