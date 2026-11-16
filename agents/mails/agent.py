from core.src.base_agent import BaseAgent


class MailAgent(BaseAgent):
    """Answers questions about emails and drafts replies, from the prompt and conversation only.

    No RAG, no tools.  The inbox digest is a separate service (agents/mails/main.py).
    """
