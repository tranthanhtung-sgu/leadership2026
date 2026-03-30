"""Three agents (Zoe, Femke, Hao); each is a separate class and own API caller."""
from .base import TeamMemberAgent
from .zoe_agent import ZoeAgent
from .femke_agent import FemkeAgent
from .hao_agent import HaoAgent

AGENTS: dict[str, TeamMemberAgent] = {
    "Zoe": ZoeAgent(),
    "Femke": FemkeAgent(),
    "Hao": HaoAgent(),
}


def get_agent(name: str) -> TeamMemberAgent:
    return AGENTS[name]
