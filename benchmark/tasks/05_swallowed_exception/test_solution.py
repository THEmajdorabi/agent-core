import json

import pytest
from config_loader import load_config


def test_missing_file_returns_empty(tmp_path):
    assert load_config(tmp_path / "nope.json") == {}


def test_valid_file_loads(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")
    assert load_config(path) == {"a": 1}


def test_malformed_json_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)
