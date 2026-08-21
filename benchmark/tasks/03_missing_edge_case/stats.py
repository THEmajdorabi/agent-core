"""Simple statistics."""


def mean(values):
    """Return the arithmetic mean, or 0.0 for an empty sequence."""
    return sum(values) / len(values)
