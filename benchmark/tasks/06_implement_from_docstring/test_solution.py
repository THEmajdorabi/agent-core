from chunker import chunk_by_sentences


def test_empty():
    assert chunk_by_sentences("", 10) == []


def test_packs_sentences_greedily():
    text = "One. Two. Three."
    assert chunk_by_sentences(text, 10) == ["One. Two.", "Three."]


def test_never_splits_a_sentence():
    text = "A very long single sentence that exceeds the limit."
    assert chunk_by_sentences(text, 10) == [
        "A very long single sentence that exceeds the limit."
    ]


def test_handles_question_and_exclamation():
    text = "Why? Because! Done."
    assert chunk_by_sentences(text, 14) == ["Why? Because!", "Done."]
