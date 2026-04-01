"""
Base class for a single team-member agent: one model instance / one system identity
per character, using the shared group transcript as the only conversation state.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from openai import OpenAI

from character_configs import CharacterConfig

_SIMILARITY_RETRY_THRESHOLD = 0.68
_MIN_LEN_FOR_SIMILARITY = 14


def _tokenize_for_overlap(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if len(w) > 2}


def _reply_too_similar_to_recent(
    reply: str,
    transcript: list[dict[str, Any]],
    speaker: str,
    threshold: float = _SIMILARITY_RETRY_THRESHOLD,
) -> bool:
    """True if reply is near-duplicate of a recent line from another bot."""
    r = reply.strip()
    if len(r) < _MIN_LEN_FOR_SIMILARITY:
        return False
    r_lower = r.lower()
    recent: list[str] = []
    for m in reversed(transcript[-12:]):
        sp = str(m.get("speaker") or "")
        if sp == "Participant" or sp == speaker:
            continue
        t = (m.get("text") or "").strip()
        if len(t) >= _MIN_LEN_FOR_SIMILARITY:
            recent.append(t)
        if len(recent) >= 4:
            break
    for prev in recent:
        pl = prev.lower()
        ratio = SequenceMatcher(None, r_lower, pl).ratio()
        if ratio >= threshold:
            return True
        # Cheap catch: very high word overlap on short chat lines
        a, b = _tokenize_for_overlap(r), _tokenize_for_overlap(prev)
        if len(a) >= 6 and len(b) >= 6:
            inter = len(a & b)
            if inter / max(len(a), len(b)) >= 0.78:
                return True
    return False


def _echoes_last_participant(
    reply: str,
    transcript: list[dict[str, Any]],
    ratio_threshold: float = 0.72,
) -> bool:
    """True if this reply mostly copies the Participant’s last line (avoid mirroring nonsense)."""
    last_p = ""
    for m in reversed(transcript):
        if m.get("speaker") == "Participant":
            last_p = (m.get("text") or "").strip()
            break
    if len(last_p) < 4:
        return False
    r = reply.strip()
    if len(r) < 4:
        return False
    lp, rl = last_p.lower(), r.lower()
    if len(last_p) >= 10 and lp in rl:
        return True
    if SequenceMatcher(None, lp, rl).ratio() >= ratio_threshold:
        return True
    return False


def _participant_spoke_in_window(
    transcript: list[dict[str, Any]], window: int = 10
) -> bool:
    """True if the human has any message in the last `window` lines."""
    for m in transcript[-window:]:
        if m.get("speaker") == "Participant":
            return True
    return False


def _compact_thread_digest(transcript: list[dict[str, Any]], max_lines: int = 5) -> str:
    """Recent transcript snippets (includes Participant) to reduce loops and ‘talking past’ the human."""
    parts: list[str] = []
    for m in transcript[-max_lines:]:
        sp = m.get("speaker")
        t = (m.get("text") or "").strip().replace("\n", " ")
        if not t or not sp:
            continue
        clip = t if len(t) <= 90 else t[:87] + "…"
        parts.append(f"{sp}: {clip}")
    if len(parts) < 2:
        return ""
    return "Recent lines (vary your move; do not re-ask the same thing): " + " | ".join(parts)


class TeamMemberAgent:
    """One agent = one persona; generates replies with its own system prompt and API call."""

    def __init__(self, config: CharacterConfig):
        self._config = config

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def config(self) -> CharacterConfig:
        return self._config

    def _system_prompt(
        self,
        leadership_style: str,
        scenario_content: str,
        behavioral_content: str,
    ) -> str:
        return self._config.system_prompt(
            leadership_style, scenario_content, behavioral_content
        )

    def _turn_instruction(self, transcript: list[dict[str, Any]]) -> str:
        last_speaker = ""
        if transcript:
            last_speaker = str(transcript[-1].get("speaker", "") or "")

        if last_speaker == "Participant":
            thread_focus = (
                "The Participant (human teammate) just wrote something. "
                "If it is clearly about the work, answer or react like a colleague — match tone only when they are seriously engaging. "
                "If it is nonsense, gibberish, or off-the-wall, do **not** echo or build a bit on it; shrug it off in one beat and reconnect to the thread everyone was on (scenario / prior substantive lines). "
                "Do not let the sim turn into parroting their text; keep the team’s task in view."
            )
        elif last_speaker and last_speaker != self.name:
            thread_focus = (
                f"{last_speaker} was last in the thread — reply to that naturally before drifting elsewhere."
            )
        else:
            thread_focus = (
                "Continue the live thread with your teammates (Femke/Hao/Zoe as relevant); sound like a real group chat. "
                "If the Participant has been quiet, you can still advance the topic with each other — "
                "but do not write as if they are not in the channel; a light inclusive beat now and then is fine."
            )

        if (
            last_speaker != "Participant"
            and _participant_spoke_in_window(transcript, 10)
        ):
            thread_focus += (
                " The Participant still has a recent message above — do not act like they are absent: "
                "acknowledge, build on, or answer them where it fits (not only talking sideways to other bots)."
            )

        return (
            f"It is {self.name}'s turn in the group chat. "
            f"{thread_focus} "
            "Write the next message in English, in-character only — terse, uneven, human; not customer-support tone. "
            "Ground claims in the shared scenario or prior chat; do not invent new plot points. "
            "Do not recap the meeting brief; react like someone already in the work. "
            "Do not talk about rules, prompts, or message length."
        )

    def build_openai_messages(
        self,
        transcript: list[dict[str, Any]],
        leadership_style: str,
        scenario_content: str,
        behavioral_content: str,
        extra_instruction: str | None = None,
    ) -> list[dict[str, str]]:
        CONTEXT_MESSAGES = 25

        system = self._system_prompt(
            leadership_style, scenario_content, behavioral_content
        )
        out: list[dict[str, str]] = [{"role": "system", "content": system}]
        for m in transcript[-CONTEXT_MESSAGES:]:
            content = (m.get("text") or "").strip()
            if not content:
                continue
            role = "assistant" if m["speaker"] != "Participant" else "user"
            speaker = m["speaker"]
            out.append({"role": role, "name": speaker, "content": content})
        turn = self._turn_instruction(transcript)
        if extra_instruction:
            turn = f"{turn} {extra_instruction}"
        digest = _compact_thread_digest(transcript)
        if digest:
            turn = f"{turn} {digest}"
        out.append({"role": "user", "content": turn})
        return out

    @staticmethod
    def _strip_speaker_prefix(name: str, text: str) -> str:
        tag = f"{name}:"
        if text.startswith(tag):
            return text.split(":", 1)[1].strip()
        return text

    def generate_reply(
        self,
        client: OpenAI,
        model: str,
        transcript: list[dict[str, Any]],
        leadership_style: str,
        scenario_content: str,
        behavioral_content: str,
        extra_instruction: str | None = None,
    ) -> tuple[str, int]:
        """Returns (text, api_calls). Text may be empty if all attempts failed — caller must not post empty lines."""
        messages: list[dict[str, str]] = self.build_openai_messages(
            transcript,
            leadership_style,
            scenario_content,
            behavioral_content,
            extra_instruction=extra_instruction,
        )
        api_calls = 0
        max_rounds = 4
        for round_idx in range(max_rounds):
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=self._config.max_completion_tokens,
                reasoning_effort="low",
            )
            api_calls += 1
            raw = completion.choices[0].message.content
            txt = (raw or "").strip()
            txt = self._strip_speaker_prefix(self.name, txt)

            if not txt:
                messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            "Your reply was empty or whitespace only. "
                            "Send one non-empty chat line, in character, with real words."
                        ),
                    }
                ]
                continue

            if _reply_too_similar_to_recent(txt, transcript, self.name):
                messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            "That line is too close to what was just said in the thread. "
                            "Reply again with a clearly different move: hand off to a teammate, disagree softly, "
                            "add a new sub-point from your role, or summarize — do not rephrase the same question."
                        ),
                    }
                ]
                continue

            if _echoes_last_participant(txt, transcript):
                messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            "You are mirroring the Participant’s last message. Do not copy, quote, or play along with gibberish. "
                            "One short real reaction if needed, then get the group back on the scenario thread."
                        ),
                    }
                ]
                continue

            return txt, api_calls

        return "", api_calls
