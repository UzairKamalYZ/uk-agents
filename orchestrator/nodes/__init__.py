"""
orchestrator/nodes/ — LangGraph node classes for the multi-agent pipeline.

Public API (imported by graph.py)
----------------------------------
    NormaliserNode    — cleans the message
    ConfirmationNode  — pauses (interrupt) for the user's answer to a pending item
    SupervisorNode    — routes between specialist agents
    SynthesiserNode   — merges all agent outputs into one final answer
    AgentNode         — wraps a single specialist agent
"""
from .agent import AgentNode
from .confirmation import ConfirmationNode
from .normaliser import NormaliserNode
from .supervisor import SupervisorNode
from .synthesiser import SynthesiserNode

__all__ = ["NormaliserNode", "ConfirmationNode", "SupervisorNode", "SynthesiserNode", "AgentNode"]
