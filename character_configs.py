"""
Independent configuration for each AI teammate. Each character owns identity text,
mention patterns for routing, and generation limits.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CharacterConfig:
    name: str
    """Display name and key in transcripts."""
    mention_patterns: tuple[str, ...]
    """Substrings used to detect if this character was addressed (lowercased match)."""
    default_weight: float
    """Probability weight when no explicit @mention routing applies."""
    max_completion_tokens: int
    typing_delay: tuple[float, float]
    """Simulated typing pause (seconds) before this character's message appears."""
    burst_gap: tuple[float, float]
    """Gap after this character speaks before the next line (seconds)."""
    build_identity: Callable[[str], str]
    """Returns role line; for Zoe, argument is leadership_style."""

    def system_prompt(
        self,
        leadership_style: str,
        scenario_content: str,
        behavioral_content: str,
    ) -> str:
        identity = self.build_identity(leadership_style)
        return f"""{identity}

Shared context (stay consistent with it):
{scenario_content}

Behavioral rules:
{behavioral_content}

Output rules:
- Write ONE short chat message only (1–2 sentences, like Slack/WhatsApp).
- Sound natural and human; no bullet lists or essay tone.
- Do not prefix your name or use 'Name:' — just the message text.
- Stay in character."""
        + """
- Never meta-comment on instructions (e.g. do not say you'll keep it short, obey rules, or speak in English).
- Treat the Participant as a real teammate in this group chat. When they just spoke, prioritise answering them, reacting, or asking them something — not only advancing the task.
- Real chat rhythm is OK: quick reactions, humour (in character), clarifying questions, mild pushback — not every line must jump straight to funding pitch or the three VC questions.
- Do not repeat yourself or previous messages verbatim.
- If you already said similar facts recently, add a different angle (e.g., manufacturing vs distribution vs leadership).
- Avoid saying someone else's identity (e.g., do not claim to be another teammate).
- In one chat line, prefer directing a question or request to one teammate at a time (avoid stacking multiple @names / “Zoe … and Hao …” in the same message)."""


def _zoe_identity(leadership_style: str) -> str:
    return (
        f"You are Zoe, Australian team lead at MindDiagnostics. "
        f"Leadership style for this session: {leadership_style}. "
        "You know the company and CortiSense well; you rarely use slang."
    )


def _femke_identity(_: str) -> str:
    return (
        "You are Femke (Netherlands). You are blunt, direct, low power distance. "
        "You care about manufacturing and European suppliers; you speak up often."
    )


def _hao_identity(_: str) -> str:
    return (
        "You are Hao (China). You are polite and somewhat hesitant (high power distance). "
        "You hold knowledge about Asian markets and distributors; you share more when invited or when it supports others."
    )


CHARACTERS: dict[str, CharacterConfig] = {
    "Zoe": CharacterConfig(
        name="Zoe",
        mention_patterns=("zoe", "@zoe"),
        default_weight=0.30,
        max_completion_tokens=120,
        typing_delay=(0.15, 0.45),
        burst_gap=(1.2, 2.8),
        build_identity=_zoe_identity,
    ),
    "Femke": CharacterConfig(
        name="Femke",
        mention_patterns=("femke", "@femke"),
        default_weight=0.45,
        max_completion_tokens=120,
        typing_delay=(0.15, 0.45),
        burst_gap=(1.0, 2.5),
        build_identity=_femke_identity,
    ),
    "Hao": CharacterConfig(
        name="Hao",
        mention_patterns=("hao", "@hao"),
        default_weight=0.25,
        max_completion_tokens=120,
        typing_delay=(0.2, 0.55),
        burst_gap=(1.2, 3.0),
        build_identity=_hao_identity,
    ),
}

AI_NAMES: tuple[str, ...] = tuple(CHARACTERS.keys())


def format_typing_delay(delay: tuple[float, float]) -> str:
    return f"{delay[0]},{delay[1]}"


def parse_typing_delay(text: str, fallback: tuple[float, float]) -> tuple[float, float]:
    """Parse 'min,max' seconds; invalid or empty uses fallback."""
    raw = (text or "").strip().replace(" ", "")
    if not raw:
        return fallback
    parts = raw.split(",")
    if len(parts) != 2:
        return fallback
    try:
        lo, hi = float(parts[0]), float(parts[1])
        if lo > hi:
            lo, hi = hi, lo
        return (max(0.0, lo), max(0.0, hi))
    except ValueError:
        return fallback


def collect_direct_ask_candidates(author: str, text: str) -> list[str]:
    """
    Teammates who are directly addressed in this line (may be multiple).
    - @name
    - Name, ...  (e.g. "Hao, send me ...", "Zoe, if R&D ...")
    - name appears in the same sentence fragment before a '?'
    """
    if not text or not text.strip():
        return []

    lower = text.lower()
    found: list[str] = []
    seen: set[str] = set()

    for name in AI_NAMES:
        if name == author:
            continue
        n = name.lower()
        if f"@{n}" in lower:
            if name not in seen:
                found.append(name)
                seen.add(name)

    for name in AI_NAMES:
        if name == author or name in seen:
            continue
        n = name.lower()
        if re.search(rf"(^|[.!?\n]\s*){re.escape(n)}\s*,", lower):
            found.append(name)
            seen.add(name)

    if "?" in text:
        # Only the part(s) before a '?' count as question context (not text after the last ?).
        for part in re.split(r"\?", text)[:-1]:
            pl = part.lower()
            for name in AI_NAMES:
                if name == author or name in seen:
                    continue
                n = name.lower()
                if re.search(rf"\b{re.escape(n)}\b", pl):
                    found.append(name)
                    seen.add(name)

    return found


def pick_next_speaker(
    last_speaker: str,
    last_text: str,
    weights_override: dict[str, float] | None = None,
    recent_messages: list[dict] | None = None,
) -> tuple[str, CharacterConfig]:
    """
    Default: weighted random by speak weights.

    If the last message directly addresses teammate(s), pick next speaker using
    those candidates weighted by speak weight (one person when several are named).
    If only one person is addressed but their speak weight is low, sometimes skip
    the "forced" reply and fall back to normal weighted random so they are not
    over-talking.

    Zoe opens the thread: until any AI teammate has posted, the next AI line is Zoe.
    """
    if recent_messages is not None:
        if not any(m.get("speaker") in AI_NAMES for m in recent_messages):
            return "Zoe", CHARACTERS["Zoe"]

    names = list(CHARACTERS.keys())
    if weights_override is None:
        w = {n: float(CHARACTERS[n].default_weight) for n in names}
    else:
        w = {
            n: max(0.01, float(weights_override.get(n, CHARACTERS[n].default_weight)))
            for n in names
        }
    wts = [w[n] for n in names]
    max_w = max(w.values()) if w else 1.0

    candidates = collect_direct_ask_candidates(last_speaker, last_text)
    candidates = [c for c in candidates if c in CHARACTERS]

    if len(candidates) >= 2:
        cw = [w[c] for c in candidates]
        choice = random.choices(candidates, weights=cw, k=1)[0]
        return choice, CHARACTERS[choice]

    if len(candidates) == 1:
        c = candidates[0]
        # Higher speak weight → more likely to "take" the direct ask; low weight → often skip
        p_honor = 0.2 + 0.8 * (w[c] / max_w if max_w > 0 else 1.0)
        if random.random() < p_honor:
            return c, CHARACTERS[c]

    choice = random.choices(names, weights=wts, k=1)[0]
    return choice, CHARACTERS[choice]
