"""
orchestrator/nodes/ — LangGraph node classes for the multi-agent pipeline.

Public API (imported by graph.py)
----------------------------------
    RouterNode        — cleans the message and plans the agents (one LLM call)
    ConfirmationNode  — pauses (interrupt) for the user's answer to a pending item
    SynthesiserNode   — merges all agent outputs into one final answer
    AgentNode         — wraps a single specialist agent
"""
from .agent import AgentNode
from .confirmation import ConfirmationNode
from .router import RouterNode
from .synthesiser import SynthesiserNode

__all__ = ["RouterNode", "ConfirmationNode", "SynthesiserNode", "AgentNode"]
