from datetime import UTC, datetime, timedelta

from deadline import hours_remaining


def test_future_deadline_is_positive():
    later = datetime.now(UTC) + timedelta(hours=5)
    assert 4.9 < hours_remaining(later) < 5.1


def test_past_deadline_is_negative():
    earlier = datetime.now(UTC) - timedelta(hours=2)
    assert -2.1 < hours_remaining(earlier) < -1.9
