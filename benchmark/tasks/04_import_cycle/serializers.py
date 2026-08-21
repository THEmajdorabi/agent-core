"""Serialization helpers."""

from models import User


def serialize_user(user):
    """Render a User as a dict."""
    return {"name": user.name}


def is_user(value):
    return isinstance(value, User)
