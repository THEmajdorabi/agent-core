from tokenize_module import words


def test_latin():
    assert words("hello there") == ["hello", "there"]


def test_arabic():
    assert words("مدة الإشعار") == ["مدة", "الإشعار"]


def test_mixed_script_and_digits():
    assert words("المادة 74 states the rule") == [
        "المادة",
        "states",
        "the",
        "rule",
    ]
