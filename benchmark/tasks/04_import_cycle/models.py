"""Data models."""

from serializers import serialize_user


class User:
    def __init__(self, name):
        self.name = name

    def to_json(self):
        return serialize_user(self)
