"""Text chunking."""


def chunk_by_sentences(text, max_chars):
    """Split `text` into chunks of at most `max_chars` characters.

    Sentences end at '.', '?' or '!' followed by a space or end of string.
    A chunk never splits a sentence: sentences are packed greedily until
    adding the next one would exceed `max_chars`. A single sentence longer
    than `max_chars` becomes a chunk of its own. Leading and trailing
    whitespace is stripped from every chunk. Empty input returns [].
    """
    raise NotImplementedError
