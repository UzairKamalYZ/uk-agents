"""
orchestrator/nodes/ — LangGraph node classes for the multi-agent pipeline.

Public API (imported by graph.py)
----------------------------------
    NormaliserNode    — cleans the message and resolves pending confirmations
    SupervisorNode    — routes between specialist agents
    SynthesiserNode   — merges all agent outputs into one final answer
    AgentNode         — wraps a single specialist agent
"""
from .agent import AgentNode
from .normaliser import NormaliserNode
from .supervisor import SupervisorNode
from .synthesiser import SynthesiserNode

__all__ = ["NormaliserNode", "SupervisorNode", "SynthesiserNode", "AgentNode"]
