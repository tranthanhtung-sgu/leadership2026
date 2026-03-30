"""Hao: Asian markets / higher power-distance, more hesitant agent."""
from character_configs import CHARACTERS

from .base import TeamMemberAgent


class HaoAgent(TeamMemberAgent):
    def __init__(self):
        super().__init__(CHARACTERS["Hao"])
