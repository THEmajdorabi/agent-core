from consumer import format_results
from producer import search


def test_pipeline_end_to_end():
    assert format_results(search("notice period")) == [
        "article 74 (0.91)",
        "article 75 (0.88)",
    ]
