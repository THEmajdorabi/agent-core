"""Load JSON configuration."""

import json


def load_config(path):
    """Read a JSON config file.

    A missing file returns an empty dict. Malformed JSON must raise
    ValueError, because silently returning {} hides a broken deploy.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}
