"""Context compaction.

When a transcript approaches the model's window, the loop has to shrink it or
die. Summarising is the easy half. The hard half is *where* you are allowed to
cut, and this module spends most of its length on that.

A `tool_use` block and its matching `tool_result` must stay together. Cutting
between them leaves the model an unanswered tool call, which the API rejects,
or an orphan result referring to a call that no longer exists. So compaction
never cuts at an arbitrary message index -- it cuts only at a *settled*
boundary, where every tool call raised so far has already been answered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import structlog

from agent_core.errors import ProviderError
from agent_core.providers.base import Message, Provider

log = structlog.get_logger(__name__)

SUMMARY_PROMPT = """\
You are compacting the transcript of an agent run so it fits in a smaller
context window. Summarise everything below into four labelled sections:

DECISIONS MADE: what was chosen and why.
FACTS ESTABLISHED: concrete findings -- values, paths, error messages, results.
FILES TOUCHED: every file read or written, with what changed.
OPEN QUESTIONS: what is still unresolved.

Be specific. Preserve exact identifiers, numbers, file paths and error strings;
they are what the agent needs to continue. Omit conversational filler. Do not
invent anything that is not in the transcript.
"""

# Characters per token, by script. Arabic is tokenized far less efficiently
# than Latin text, so one ratio for both is wrong in whichever direction
# matters. These are the starting values; scripts/calibrate_tokens.py measures
# them against real provider usage counts and prints corrections.
CHARS_PER_TOKEN_LATIN = 4.0
CHARS_PER_TOKEN_ARABIC = 2.0

_ARABIC = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")


def estimate_tokens_heuristic(text: str) -> int:
    """Character-based token estimate, split by script.

    Counts Arabic and non-Arabic characters separately and applies each
    script's ratio, because a single ratio calibrated on English
    under-estimates an Arabic transcript badly enough to blow the window
    before compaction ever triggers.

    This is the last-resort path. Prefer the provider's own counter.
    """
    arabic_chars = len(_ARABIC.findall(text))
    other_chars = len(text) - arabic_chars
    return int(arabic_chars / CHARS_PER_TOKEN_ARABIC + other_chars / CHARS_PER_TOKEN_LATIN) + 1


def message_text(message: Message) -> str:
    """Everything in a message as one string, for counting purposes."""
    parts: list[str] = []
    for block in message.content:
        block_type = block.get("type")
        if block_type == "text":
            parts.append(block.get("text", ""))
        elif block_type == "tool_use":
            parts.append(f"{block.get('name', '')} {block.get('input', {})}")
        elif block_type == "tool_result":
            parts.append(str(block.get("content", "")))
        elif block_type == "thinking":
            parts.append(str(block.get("thinking", "")))
    return "\n".join(parts)


def settled_boundaries(messages: list[Message]) -> list[int]:
    """Indices where the transcript can safely be cut.

    Index `i` is settled when every `tool_use` id appearing in `messages[:i]`
    also has a `tool_result` in `messages[:i]`. Index 0 and len(messages) are
    trivially settled.

    Returning every valid index, rather than just the nearest one, lets the
    caller pick the cut that removes the most while staying safe.
    """
    boundaries = [0]
    pending: set[str] = set()

    for index, message in enumerate(messages):
        for block in message.content:
            if block.get("type") == "tool_use":
                pending.add(block["id"])
            elif block.get("type") == "tool_result":
                pending.discard(block["tool_use_id"])
        if not pending:
            boundaries.append(index + 1)

    return boundaries


@dataclass
class PinnedFacts:
    """Facts the agent marked permanent.

    Re-injected verbatim after every compaction, so the one thing the agent
    explicitly said it must not forget is the one thing summarisation cannot
    eat.
    """

    facts: list[str] = field(default_factory=list)

    def pin(self, fact: str) -> None:
        if fact not in self.facts:
            self.facts.append(fact)

    def as_message(self) -> Message | None:
        if not self.facts:
            return None
        body = "\n".join(f"- {fact}" for fact in self.facts)
        return Message.user_text(
            f"PINNED FACTS (carried across compaction, do not discard):\n{body}"
        )


class Compactor:
    """Shrinks a transcript when it grows past a fraction of the window."""

    def __init__(
        self,
        provider: Provider,
        max_tokens: int,
        trigger_ratio: float = 0.8,
        target_ratio: float = 0.6,
        keep_recent: int = 6,
        max_passes: int = 2,
        pinned: PinnedFacts | None = None,
    ) -> None:
        self.provider = provider
        self.max_tokens = max_tokens
        self.trigger_ratio = trigger_ratio
        self.target_ratio = target_ratio
        self.keep_recent = keep_recent
        self.max_passes = max_passes
        self.pinned = pinned or PinnedFacts()
        self.compactions = 0

    # ---- counting --------------------------------------------------------

    def count_tokens(self, messages: list[Message], system: str = "") -> int:
        """Token count, preferring the provider's own counter.

        Anthropic exposes a counting endpoint and OpenAI is counted locally
        with tiktoken; both live behind `Provider.count_tokens`. The heuristic
        below is only reached when that call fails, and it is deliberately the
        pessimistic estimate so a failure compacts early rather than late.
        """
        try:
            return self.provider.count_tokens(messages, system)
        except (ProviderError, ValueError, KeyError, OSError) as exc:
            # A counting failure must never end a run: it degrades to the
            # heuristic, logged so the fallback is visible rather than silent.
            # ProviderError covers API failures; the rest cover a missing
            # tiktoken encoding or an unreachable cache directory.
            log.warning("compactor.count_fallback", error=str(exc))
            blob = system + "\n".join(message_text(m) for m in messages)
            return estimate_tokens_heuristic(blob)

    # ---- compaction ------------------------------------------------------

    def maybe_compact(
        self, messages: list[Message], system: str = ""
    ) -> tuple[list[Message], bool]:
        """Compact if over the trigger. Returns the transcript and whether it changed."""
        tokens = self.count_tokens(messages, system)
        if tokens < self.max_tokens * self.trigger_ratio:
            return messages, False

        target = int(self.max_tokens * self.target_ratio)
        working = messages
        changed = False

        for pass_number in range(self.max_passes):
            compacted = self._compact_once(working, system)
            if compacted is None:
                break
            working = compacted
            changed = True
            tokens = self.count_tokens(working, system)
            log.info("compactor.pass", pass_number=pass_number + 1, tokens=tokens, target=target)
            if tokens <= target:
                break

        if changed:
            self.compactions += 1
        return working, changed

    def _compact_once(self, messages: list[Message], system: str) -> list[Message] | None:
        """One summarisation pass, or None when there is nothing safe to cut."""
        # Never touch the last `keep_recent` messages: recent context is what
        # the model is actually working with.
        latest_cut = max(0, len(messages) - self.keep_recent)
        candidates = [b for b in settled_boundaries(messages) if 0 < b <= latest_cut]
        if not candidates:
            # Everything recent, or a tool call still open at every candidate
            # point. Refusing to cut is correct; a corrupted transcript is
            # worse than a large one.
            log.info("compactor.no_safe_boundary", messages=len(messages))
            return None

        cut = max(candidates)
        head, tail = messages[:cut], messages[cut:]

        transcript = "\n\n".join(f"[{m.role}] {message_text(m)}" for m in head)
        summary = self._summarise(transcript, system)

        rebuilt: list[Message] = [
            Message.user_text(f"SUMMARY OF EARLIER STEPS:\n{summary}"),
        ]
        pinned_message = self.pinned.as_message()
        if pinned_message is not None:
            rebuilt.append(pinned_message)
            # A summary followed by pinned facts is two user messages in a row.
            # Both providers accept that, and keeping them separate makes the
            # pinned block trivially greppable in a dumped transcript.
        rebuilt.extend(tail)
        return rebuilt

    def _summarise(self, transcript: str, system: str) -> str:
        completion = self.provider.complete(
            messages=[Message.user_text(transcript)],
            tools=[],
            system=SUMMARY_PROMPT,
            max_tokens=1500,
        )
        return completion.text

    def stats(self) -> dict[str, Any]:
        return {"compactions": self.compactions, "pinned_facts": len(self.pinned.facts)}
