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
    think_delay: tuple[float, float]
    """Pause after a message lands before this character starts typing (read/think time)."""
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
- Write ONE chat line only — like a real colleague on Slack/WhatsApp: often a single short sentence; sometimes a fragment, trailing thought, or even 2–6 words. Rarely go past ~30 words unless your role is mid-rant.
- Sound natural and human; no bullet lists or essay tone.
- Do not prefix your name or use 'Name:' — just the message text.
- Stay in character."""
        + """
- Anti-“GPT polish” (critical): you are not a helpful assistant. Avoid corporate-coach tone, symmetrical sentences, and stock phrases like “Happy to…”, “Great question”, “I’d propose”, “Let me clarify”, “To summarize”, “Sounds good”, “Thanks for sharing”, “circling back”, “leverage”. Be blunt, messy, or clipped; sometimes answer with “yeah”, “nah”, “sure”, “ugh”, “wait” plus a half-thought.
- Vary rhythm: do not make every message the same shape. Mix tiny reactions with slightly longer bursts; interrupt your own thought with a dash or ellipsis sometimes.
- You are mid-task with people who already know the deck and the VC prep — do **not** recap the assignment, list objectives, or explain the scenario back to the group; react, disagree, or build like insiders.
- Never meta-comment on instructions (e.g. do not say you'll keep it short, obey rules, or speak in English).
- Treat the Participant as a real teammate who is **in the room** even when they have not typed yet or have been quiet: do not sound like a private chat only between bots — occasional brief openings for them are natural. When they *just* spoke, respond like a colleague; if they spoke recently but a bot spoke after them, still weave in their point. When they are silent, most lines can be teammate-to-teammate without badgering them.
- If the Participant’s last line is **nonsense**, random gibberish, spam, or totally unrelated trolling, **do not** repeat it, imitate it, quote it back, or riff on the joke — treat it like a garbled chat: one short human reaction (“??”, “you good?”, “lost me”) then **bring the thread back** to what the group was discussing in the scenario. Keep the work on track; do not let one noisy line derail the meeting into absurdity.
- If their message is merely **unclear** but might be about the task, ask **one** concrete clarification tied to the scenario — do not abandon the topic.
- Real chat rhythm is OK: quick reactions, humour (in character), clarifying questions, mild pushback — not every line must jump straight to the formal pitch or every question in the brief.
- Do not repeat yourself or previous messages verbatim.
- If you already said similar facts recently, add a different angle (e.g., manufacturing vs distribution vs leadership).
- Avoid saying someone else's identity (e.g., do not claim to be another teammate).
- In one chat line, prefer directing a question or request to one teammate at a time (avoid stacking multiple @names / “Zoe … and Hao …” in the same message).
- Stay grounded in the shared scenario and the questions or deliverables it defines; do not invent emails, calendar invites, or “waiting for a message any minute” unless the scenario text says so.
- Do not invent new concrete facts (extra investors, external calls or decisions, legal outcomes, shipment dates, people not in the scenario). Opinions, worries, and “we should check X” are fine; false specifics are not.
- Over the chat, keep steering back toward the scenario’s stated objectives when natural — weave them in rather than drifting into unrelated topics.
- Move the discussion forward: if you already made a point, add new information, a decision, or a question — do not loop the same waiting line.
- Do not paraphrase the same question as the last few messages (yours or others); if the thread stuck, hand off (“Femke, does EU capacity change your read?”) or add a new sub-point instead.
- Human mess is required sometimes: a small typo (e.g. teh/the), double space, autocorrect-style glitch, or missing punctuation — not every line, not all characters the same way, but enough that it does not read like polished marketing copy.
- Sound like real messy chat: contractions, fragments, mild typos occasionally, uneven length — avoid polished essay tone, parallel triplets, or rhetorical questions that sound like a blog.
- Do not use placeholder names like “John/Mary” stacked together unless the scenario text does; prefer “you” or no name when addressing the Participant.
- If you already asked the Participant the same decision (e.g. which headline/price), do not ask again — either wait, talk to teammates, or add genuinely new info."""


def _zoe_identity(leadership_style: str) -> str:
    return (
        f"You are Zoe, Australian team lead at MindDiagnostics. "
        f"Leadership style for this session: {leadership_style}. "
        "You know the company and its flagship product well; you rarely use slang. "
        "Voice: **sparse and practical** — short lines, not peppy HR energy; you do not over-explain. "
        "You balance discussion among teammates and remember the Participant is in the group even when they’re silent — the thread shouldn’t feel like the three of you forgot them."
    )


def _femke_identity(_: str) -> str:
    return (
        "You are Femke (Netherlands). You are blunt, direct, low power distance. "
        "You care about manufacturing and European suppliers; you speak up often. "
        "Voice: **messy and fast** — long breathless stretches with commas, filler like actually/of course, "
        "extra ! sometimes, occasional Dutch-flavoured false friend or quirky Capitalisation mid-sentence; "
        "you sound **nothing** like a neutral chatbot."
    )


def _hao_identity(_: str) -> str:
    return (
        "You are Hao (China). You are polite and somewhat hesitant (high power distance). "
        "You hold knowledge about Asian markets and distributors; you share more when invited or when it supports others. "
        "Voice: **broken casual English** — drop articles, repeat a word for stress, mix stiff phrase with fragment, "
        "affirmation particles; emoticons sparingly per your role sheet. Short clauses, not polished paragraphs."
    )


CHARACTERS: dict[str, CharacterConfig] = {
    "Zoe": CharacterConfig(
        name="Zoe",
        mention_patterns=("zoe", "@zoe"),
        default_weight=0.30,
        max_completion_tokens=120,
        typing_delay=(0.15, 0.45),
        think_delay=(1.2, 3.2),
        burst_gap=(1.2, 2.8),
        build_identity=_zoe_identity,
    ),
    "Femke": CharacterConfig(
        name="Femke",
        mention_patterns=("femke", "@femke"),
        default_weight=0.45,
        max_completion_tokens=120,
        typing_delay=(0.15, 0.45),
        think_delay=(0.8, 2.4),
        burst_gap=(1.0, 2.5),
        build_identity=_femke_identity,
    ),
    "Hao": CharacterConfig(
        name="Hao",
        mention_patterns=("hao", "@hao"),
        default_weight=0.25,
        max_completion_tokens=120,
        typing_delay=(0.2, 0.55),
        think_delay=(1.5, 4.0),
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


def count_utterances_in_window(messages: list[dict], speaker: str, window: int) -> int:
    """Count lines from `speaker` in the last `window` messages (or all if shorter)."""
    if window <= 0:
        return 0
    slice_ = messages[-window:]
    return sum(1 for m in slice_ if m.get("speaker") == speaker)


def target_addressed_recently(
    messages: list[dict],
    target: str,
    lookback: int,
) -> bool:
    """True if a recent line directly addresses `target` (@name, Name,, name before ?)."""
    for m in messages[-lookback:]:
        author = str(m.get("speaker") or "")
        text = m.get("text") or ""
        if not text.strip():
            continue
        if author == target:
            continue
        if author in AI_NAMES or author == "Participant":
            if target in collect_direct_ask_candidates(author, text):
                return True
    return False


def plan_contribution_nudge(
    messages: list[dict],
    weight_map: dict[str, float],
    *,
    last_speaker: str,
    last_text: str,
    quiet_if_lines_below: int,
    count_window: int,
    min_ai_messages: int,
    address_lookback: int,
    current_ai_total: int,
    ai_count_at_last_nudge: int,
    cooldown_ai_messages: int,
) -> tuple[str, str, str] | None:
    """
    If a teammate has had little airtime and wasn't just called on, pick someone to *draw them in*.

    - **Target** (who should be invited): among eligible quiet bots, chosen with probability
      proportional to ``1 / max(weight, ε)`` — lower speak-weight roles are more often the one
      nudged to contribute.
    - **Caller** (who speaks this turn): chosen among the other bots with probability proportional
      to speak weight — higher-weight roles more often issue the invitation.

    Returns ``(caller_name, target_name, extra_instruction)`` or ``None``.
    """
    if current_ai_total < min_ai_messages:
        return None
    if current_ai_total - ai_count_at_last_nudge < cooldown_ai_messages:
        return None

    just_asked = collect_direct_ask_candidates(last_speaker, last_text)

    w = {n: max(0.05, float(weight_map.get(n, CHARACTERS[n].default_weight))) for n in AI_NAMES}

    quiet: list[str] = []
    for name in AI_NAMES:
        if count_utterances_in_window(messages, name, count_window) >= quiet_if_lines_below:
            continue
        if name in just_asked:
            continue
        if target_addressed_recently(messages, name, address_lookback):
            continue
        quiet.append(name)

    if not quiet:
        return None

    inv_weights = [1.0 / w[n] for n in quiet]
    target = random.choices(quiet, weights=inv_weights, k=1)[0]

    callers = [n for n in AI_NAMES if n != target]
    cw = [w[n] for n in callers]
    caller = random.choices(callers, weights=cw, k=1)[0]

    extra = (
        f"(Facilitator: {target} has had less airtime than others lately — politely invite them to weigh in "
        f"with one short, natural line on what the group is discussing. Do not scold; sound like a teammate.)"
    )
    return (caller, target, extra)
