import os
import sys
from pathlib import Path
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).parents[3]

# Always load from the project root .env, regardless of where the process is launched
load_dotenv(_PROJECT_ROOT / ".env")


def _user_data_dir() -> Path:
    """Per-user data directory for uk-agents, outside the project (survives re-clones)."""
    if os.name == "nt":
        base = Path(os.getenv("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.getenv("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "uk-agents"


class Config:
    """Configuration class for the application."""
    STORE_TYPE = os.getenv("STORE_TYPE", "pinecone")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "claude-opus-5-5")
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
    # Thinking depth / token spend: low | medium | high | xhigh | max
    LLM_EFFORT = os.getenv("LLM_EFFORT", "medium")
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "16000"))
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    # true → Anthropic API (ANTHROPIC_API_KEY); false → local Claude Code CLI and its login
    ANTHROPIC_API = os.getenv("ANTHROPIC_API", "true").lower() == "true"
    SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL")
    MEMORY_FILE = os.getenv("MEMORY_FILE")
    URLS_FILE = os.getenv("URLS_FILE")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
    PROMPT_FILE = os.path.join(os.getenv("PROMPT_FILE"))
    INTERACTION_FILE = os.path.join(os.getenv("INTERACTION_FILE"))
    REFORMULATION_PROMPT = os.path.join(os.getenv("REFORMULATION_PROMPT"))
    # Telegram / LangSmith integrations
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    # Telegram user id of the owner; only they can use the expense tracker
    TELEGRAM_OWNER_ID = os.getenv("TELEGRAM_OWNER_ID", "")
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "uk-agents")
    # API security — leave empty to disable auth
    API_KEY = os.getenv("API_KEY", "")

    # SQLite file for persistent session history
    SESSION_DB_FILE = os.getenv("SESSION_DB_FILE", "sessions.db")
    # Selenium scraping browser connection string
    SBR_WEBDRIVER = os.getenv("SBR_WEBDRIVER", "")
    # Number of listings to retrieve per RAG search
    RAG_K = int(os.getenv("RAG_K", "5"))
    MAX_HISTORY_TURNS= os.getenv("MAX_HISTORY_TURNS", 10)
    # Orchestrator prompt files
    SUPERVISOR_PROMPT_FILE = os.getenv("SUPERVISOR_PROMPT_FILE", "orchestrator/prompts/supervisor.txt")
    SYNTHESISER_PROMPT_FILE = os.getenv("SYNTHESISER_PROMPT_FILE", "orchestrator/prompts/synthesiser.txt")
    NORMALISER_PROMPT_FILE = os.getenv("NORMALISER_PROMPT_FILE", "orchestrator/prompts/normaliser.txt")
    # Registry config files — resolved to absolute paths from the project root
    AGENTS_FILE = str(_PROJECT_ROOT / os.getenv("AGENTS_FILE", "agents.json"))
    # Expense ledger + input log — kept outside the project in the user data dir;
    # a relative EXPENSES_DIR is resolved from the project root, "~" is expanded
    EXPENSES_DIR = str(_PROJECT_ROOT / Path(os.getenv("EXPENSES_DIR") or _user_data_dir() / "expenses").expanduser())
    # Local faster-whisper model size for voice transcription
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
    MCP_FILE    = str(_PROJECT_ROOT / os.getenv("MCP_FILE",    "mcp.json"))

    # ---------------------------------------------------------------------------
    # WhatsApp (Green API — https://green-api.com)
    # Register at green-api.com, create an instance, scan the QR code with your
    # phone, then copy INSTANCE_ID and INSTANCE_TOKEN into .env.
    # ---------------------------------------------------------------------------
    WHATSAPP_INSTANCE_ID    = os.getenv("WHATSAPP_INSTANCE_ID", "")
    WHATSAPP_INSTANCE_TOKEN = os.getenv("WHATSAPP_INSTANCE_TOKEN", "")
    # Chat ID of the group to monitor (format: 120363xxxxxxxxx@g.us).
    # Run agents/whatsapp/green_api_client.GreenAPIClient().get_contacts() to find it.
    WHATSAPP_GROUP_ID       = os.getenv("WHATSAPP_GROUP_ID", "")
    # Name used in the mention-detection prompt — the LLM checks whether this
    # name appears in a message in a way that requires a reply.
    WHATSAPP_USER_NAME      = os.getenv("WHATSAPP_USER_NAME", "Uzair")
    # Message sent to the group when the user is mentioned and appears to be away.
    WHATSAPP_AWAY_MESSAGE   = os.getenv(
        "WHATSAPP_AWAY_MESSAGE",
        "Uzair is away from his phone right now. He'll get back to you soon!",
    )
    # Seconds between Green API polls when the notification queue is empty.
    WHATSAPP_POLL_INTERVAL  = os.getenv("WHATSAPP_POLL_INTERVAL", "2")

    # ---------------------------------------------------------------------------
    # Mail monitor
    # ---------------------------------------------------------------------------
    # JSON array of IMAP account configs, e.g.:
    # [{"email":"uzair@gmail.com","imap_host":"imap.gmail.com","imap_port":993,"password":"app-pw"},
    #  {"email":"uzair@outlook.com","imap_host":"outlook.office365.com","password":"app-pw"}]
    MAIL_ACCOUNTS                  = os.getenv("MAIL_ACCOUNTS", "")
    # How often to pull new emails from each inbox (hours, default 2)
    MAIL_CHECK_INTERVAL_HOURS      = os.getenv("MAIL_CHECK_INTERVAL_HOURS", "2")
    # How often to send the accumulated summary (hours, default 5)
    MAIL_SUMMARY_INTERVAL_HOURS    = os.getenv("MAIL_SUMMARY_INTERVAL_HOURS", "5")
    # WhatsApp group chat ID where the summary is delivered (format: 120363xxx@g.us).
    # Leave empty to write the summary to the application log only.
    MAIL_SUMMARY_WHATSAPP_GROUP_ID = os.getenv("MAIL_SUMMARY_WHATSAPP_GROUP_ID", "")
    # SQLite file that tracks which Message-IDs have already been summarised
    MAIL_SEEN_DB                   = os.getenv("MAIL_SEEN_DB", "mail_seen.db")
