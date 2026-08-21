"""Calibrate the character-per-token heuristic against real provider counts.

`context.py` falls back to a character heuristic when the provider's counter is
unreachable. Its two constants -- one for Arabic, one for Latin -- are reasoned
starting values, and a reasoned value is not a measured one.

This measures both against Anthropic's `count_tokens` endpoint on real text and
prints the corrections. It needs ANTHROPIC_API_KEY and costs nothing: token
counting is not billed.

Run: python scripts/calibrate_tokens.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agent_core.config import load_settings  # noqa: E402
from agent_core.context import (  # noqa: E402
    CHARS_PER_TOKEN_ARABIC,
    CHARS_PER_TOKEN_LATIN,
    estimate_tokens_heuristic,
)
from agent_core.providers import build_provider  # noqa: E402
from agent_core.providers.base import Message  # noqa: E402

# Real prose, not lorem ipsum: tokenizer efficiency depends on whether the text
# looks like the pretraining distribution. The Arabic samples are legal and
# administrative register, which is what repo 02's corpus is.
ARABIC_SAMPLES = [
    "مدة الإشعار المطلوبة قبل إنهاء عقد العمل غير محدد المدة هي ستون يوماً.",
    "يجب على صاحب العمل أن يدفع للعامل أجره في يوم العمل المحدد للدفع.",
    "الباب الثالث، الفصل الأول، المادة الرابعة والسبعون من نظام العمل السعودي.",
    "لا يجوز تشغيل الأحداث في الأعمال الخطرة أو الصناعات الضارة بالصحة.",
    "تنتهي علاقة العمل في حالة إغلاق المنشأة نهائياً أو إنهاء النشاط الذي يعمل فيه العامل.",
]

LATIN_SAMPLES = [
    "The notice period required before terminating an open-ended contract is sixty days.",
    "The employer must pay the worker their wages on the working day set for payment.",
    "Chapter three, section one, article seventy-four of the Saudi Labor Law.",
    "Minors may not be employed in hazardous work or in industries harmful to health.",
    "The employment relationship ends when the establishment closes permanently.",
]


def measure(provider, samples: list[str], label: str) -> float:
    """Actual characters per token for a set of samples."""
    total_chars = 0
    total_tokens = 0

    for sample in samples:
        tokens = provider.count_tokens([Message.user_text(sample)], "")
        total_chars += len(sample)
        total_tokens += tokens
        print(f"  {len(sample):>4} chars -> {tokens:>4} tokens  ({len(sample) / tokens:.2f} c/t)")

    ratio = total_chars / total_tokens
    print(f"  {label}: {total_chars} chars / {total_tokens} tokens = {ratio:.2f} chars per token")
    return ratio


def main() -> int:
    settings = load_settings()
    spec = settings.models["default"]
    provider = build_provider(spec.provider, spec.id)

    print(f"Counting with {spec.id} via the provider's count_tokens endpoint.\n")

    print("Arabic:")
    arabic_ratio = measure(provider, ARABIC_SAMPLES, "Arabic")
    print("\nLatin:")
    latin_ratio = measure(provider, LATIN_SAMPLES, "Latin")

    # Every count_tokens call includes the fixed per-message overhead, so a
    # short sample over-reports tokens per character. Reported as-is rather
    # than corrected, because that overhead is also present in real use.
    print("\n--- result ---")
    print(
        f"measured  CHARS_PER_TOKEN_ARABIC = {arabic_ratio:.2f}  (currently {CHARS_PER_TOKEN_ARABIC})"
    )
    print(
        f"measured  CHARS_PER_TOKEN_LATIN  = {latin_ratio:.2f}  (currently {CHARS_PER_TOKEN_LATIN})"
    )

    if arabic_ratio >= latin_ratio:
        print(
            "\nWARNING: Arabic did not measure as less token-efficient than Latin. "
            "Check the samples before changing the constants."
        )

    # The heuristic must not under-estimate, or compaction fires too late.
    for label, samples, current in (
        ("Arabic", ARABIC_SAMPLES, CHARS_PER_TOKEN_ARABIC),
        ("Latin", LATIN_SAMPLES, CHARS_PER_TOKEN_LATIN),
    ):
        blob = " ".join(samples)
        estimated = estimate_tokens_heuristic(blob)
        actual = provider.count_tokens([Message.user_text(blob)], "")
        verdict = "safe (over-estimates)" if estimated >= actual else "UNSAFE (under-estimates)"
        print(
            f"{label} heuristic on the full blob: estimated {estimated}, actual {actual} "
            f"-> {verdict}"
        )

    print(
        "\nIf a constant needs changing, edit context.py and add a DECISIONS.md "
        "entry with these numbers and today's date."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
