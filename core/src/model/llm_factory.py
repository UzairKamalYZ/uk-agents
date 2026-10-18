from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

from ..config.config import Config
from .claude_code_chat import ChatClaudeCode


def create_llm(provider: str, model_name: str) -> BaseChatModel:
    """
    Factory function to create an LLM instance based on the provider.

    :param provider: The name of the LLM provider (e.g., 'anthropic').
    :param model_name: The name of the model to use (e.g., 'claude-opus-5-5').
    :return: An instance of a LangChain BaseChatModel.
    :raises ValueError: If the provider is not supported.
    """
    if provider == "anthropic" and not Config.ANTHROPIC_API:
        # Local Claude Code CLI (uses its logged-in account, no API key)
        return ChatClaudeCode(model=model_name, effort=Config.LLM_EFFORT)
    if provider == "anthropic":
        return ChatAnthropic(
            model=model_name,
            api_key=Config.ANTHROPIC_API_KEY,
            max_tokens=Config.LLM_MAX_TOKENS,
            output_config={"effort": Config.LLM_EFFORT},
            # On a safety refusal the API re-runs the request on a fallback model
            betas=["server-side-fallback-2026-07-01"],
            model_kwargs={"fallbacks": "default"},
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")
