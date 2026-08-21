"""Word tokenization."""

import re

# Intended to match words in any script.
WORD = re.compile(r"[A-Za-z]+")


def words(text):
    """Return the words in `text`, in order.

    Must work for Latin script, Arabic script, and a mix of the two.
    Digits and punctuation are not words.
    """
    return WORD.findall(text)
