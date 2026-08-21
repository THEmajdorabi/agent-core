"""Formats search results for display."""


def format_results(results):
    """Render results as 'article <no> (<score>)' lines, highest score first."""
    lines = []
    for result in sorted(results, key=lambda r: r["relevance"], reverse=True):
        lines.append(f"article {result['article_no']} ({result['relevance']:.2f})")
    return lines
