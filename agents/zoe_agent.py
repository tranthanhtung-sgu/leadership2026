"""Zoe: team lead agent; identity is tied to inclusive vs non-inclusive condition."""
from character_configs import CHARACTERS

from .base import TeamMemberAgent


class ZoeAgent(TeamMemberAgent):
    def __init__(self):
        super().__init__(CHARACTERS["Zoe"])
