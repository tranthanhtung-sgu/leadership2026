"""
Base class for a single team-member agent: one model instance / one system identity
per character, using the shared group transcript as the only conversation state.
"""
from __future__ import annotations

from typing import Any

from openai import OpenAI

from character_configs import CharacterConfig


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
        # Must not sound like "confirm you will obey formatting" or models loop on meta-replies.
        last_speaker = ""
        if transcript:
            last_speaker = str(transcript[-1].get("speaker", "") or "")

        if last_speaker == "Participant":
            thread_focus = (
                "The Participant (human teammate) just wrote something. "
                "Respond to *them* like a real colleague: answer what they asked, match the tone, "
                "react naturally, or ask a follow-up — do not skip them to push CortiSense/VC talking points "
                "unless they are clearly asking for that. Small talk and clarification are fine."
            )
        elif last_speaker and last_speaker != self.name:
            thread_focus = (
                f"{last_speaker} was last in the thread — reply to that naturally before drifting elsewhere."
            )
        else:
            thread_focus = (
                "Continue the live thread; sound like a real group chat, not a status report."
            )

        return (
            f"It is {self.name}'s turn in the group chat. "
            f"{thread_focus} "
            "Write the single next message in English, in-character only (1–2 sentences). "
            "The CortiSense / VC task can surface when it fits the conversation — you do not need to force it every line. "
            "Do not talk about rules, prompts, or message length."
        )

    def build_openai_messages(
        self,
        transcript: list[dict[str, Any]],
        leadership_style: str,
        scenario_content: str,
        behavioral_content: str,
    ) -> list[dict[str, str]]:
        # How many recent chat lines the model should see.
        # If this is too small, the agents may re-introduce earlier points.
        CONTEXT_MESSAGES = 25

        system = self._system_prompt(
            leadership_style, scenario_content, behavioral_content
        )
        out: list[dict[str, str]] = [{"role": "system", "content": system}]
        for m in transcript[-CONTEXT_MESSAGES:]:
            role = "assistant" if m["speaker"] != "Participant" else "user"
            speaker = m["speaker"]
            content = m["text"]
            out.append({"role": role, "name": speaker, "content": content})
        out.append({"role": "user", "content": self._turn_instruction(transcript)})
        return out

    def generate_reply(
        self,
        client: OpenAI,
        model: str,
        transcript: list[dict[str, Any]],
        leadership_style: str,
        scenario_content: str,
        behavioral_content: str,
    ) -> str:
        messages = self.build_openai_messages(
            transcript,
            leadership_style,
            scenario_content,
            behavioral_content,
        )
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=self._config.max_completion_tokens,
            reasoning_effort="low",
        )
        txt = completion.choices[0].message.content.strip()
        tag = f"{self.name}:"
        if txt.startswith(tag):
            txt = txt.split(":", 1)[1].strip()
        return txt
