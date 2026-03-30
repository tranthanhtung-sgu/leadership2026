"""Femke: manufacturing / low power-distance agent."""
from character_configs import CHARACTERS

from .base import TeamMemberAgent


class FemkeAgent(TeamMemberAgent):
    def __init__(self):
        super().__init__(CHARACTERS["Femke"])
