def test_user_serializes():
    from models import User

    assert User("majd").to_json() == {"name": "majd"}


def test_is_user_still_works():
    from models import User
    from serializers import is_user

    assert is_user(User("x")) is True
    assert is_user("not a user") is False
