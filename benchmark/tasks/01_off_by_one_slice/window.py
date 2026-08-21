"""Sliding windows over a sequence."""


def windows(items, size):
    """Return every contiguous window of length `size`, in order."""
    result = []
    for start in range(len(items) - size):
        result.append(items[start : start + size])
    return result
