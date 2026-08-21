"""Deadline checks."""

from datetime import datetime


def hours_remaining(deadline_utc):
    """Hours between now (UTC) and `deadline_utc`, which is timezone-aware UTC.

    Returns a float, negative when the deadline has passed.
    """
    return (deadline_utc - datetime.utcnow()).total_seconds() / 3600
