from core.src.base_agent import BaseAgent


class WhatsAppAgent(BaseAgent):
    """Short, mobile-friendly property answers (no tables or heavy formatting).

    Uses the property listings (RAG) when "rag" is true in agents.json.
    """
