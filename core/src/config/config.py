import os
import sys
from pathlib import Path
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).parents[3]

# Always load from the project root .env, regardless of where the process is launched
load_dotenv(_PROJECT_ROOT / ".env")


def _project_path(env_var: str, default: str) -> str:
    """Path setting: relative values (and the default) resolve from the project root."""
    return str(_PROJECT_ROOT / (os.getenv(env_var) or default))


def _data_path(env_var: str, default: str | Path) -> str:
    """Like _project_path, but "~" is expanded and the default may be an absolute path."""
    return str(_PROJECT_ROOT / Path(os.getenv(env_var) or default).expanduser())


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
    URLS_FILE = _project_path("URLS_FILE", "agents/property/urls.txt")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
    # Property listing crawler (python -m agents.property.crawler.main)
    CRAWLER_SITES_FILE = _project_path("CRAWLER_SITES_FILE", "agents/property/crawler/sites.json")
    CRAWLER_INTERVAL_MINUTES = int(os.getenv("CRAWLER_INTERVAL_MINUTES", "60"))
    # SQLite record of listings already indexed — only new or changed listings are embedded
    CRAWLER_DB_FILE = _data_path("CRAWLER_DB_FILE", _user_data_dir() / "crawler" / "seen.db")
    CRAWLER_HEADLESS = os.getenv("CRAWLER_HEADLESS", "true").lower() == "true"
    CRAWLER_TIMEOUT_SECONDS = float(os.getenv("CRAWLER_TIMEOUT_SECONDS", "30"))
    # Random wait between page loads: CRAWLER_PAGE_DELAY_SECONDS … twice that
    CRAWLER_PAGE_DELAY_SECONDS = float(os.getenv("CRAWLER_PAGE_DELAY_SECONDS", "8"))
    # A site that refuses the crawler is skipped for this long, doubled per repeated block, up to the max
    CRAWLER_BACK_OFF_MINUTES = int(os.getenv("CRAWLER_BACK_OFF_MINUTES", "60"))
    CRAWLER_MAX_BACK_OFF_HOURS = int(os.getenv("CRAWLER_MAX_BACK_OFF_HOURS", "24"))
    CRAWLER_RESPECT_ROBOTS = os.getenv("CRAWLER_RESPECT_ROBOTS", "true").lower() == "true"
    # Chromium binary; empty → the one `playwright install chromium` downloaded
    CRAWLER_BROWSER_PATH = os.getenv("CRAWLER_BROWSER_PATH", "")
    # Property agent prompts
    PROMPT_FILE = _project_path("PROMPT_FILE", "agents/property/prompts/System_Prompt.txt")
    # Telegram / LangSmith integrations
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "uk-agents")
    # API security — leave empty to disable auth
    API_KEY = os.getenv("API_KEY", "")
    # Where the Streamlit UI finds the REST API
    REST_API_URL = os.getenv("REST_API_URL", "http://localhost:8000")

    # SQLite file for persistent session history (relative → from the project root)
    SESSION_DB_FILE = _project_path("SESSION_DB_FILE", "sessions.db")
    # Prompt the scheduled cron client sends on every run
    CRON_SEARCH_PROMPT = os.getenv("CRON_SEARCH_PROMPT", "Find new 2-bedroom apartments for rent")
    # SQLite file where the orchestrator keeps conversations paused for a confirmation
    CHECKPOINT_DB_FILE = _project_path("CHECKPOINT_DB_FILE", "checkpoints.db")
    # Selenium scraping browser connection string
    SBR_WEBDRIVER = os.getenv("SBR_WEBDRIVER", "")
    # Number of listings to retrieve per RAG search
    RAG_K = int(os.getenv("RAG_K", "5"))
    # Turns of an agent's own chat history sent with each request
    MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "10"))
    # Orchestrator prompt files
    ROUTER_PROMPT_FILE = _project_path("ROUTER_PROMPT_FILE", "orchestrator/prompts/router.txt")
    SYNTHESISER_PROMPT_FILE = _project_path("SYNTHESISER_PROMPT_FILE", "orchestrator/prompts/synthesiser.txt")
    CONFIRMATION_PROMPT_FILE = _project_path("CONFIRMATION_PROMPT_FILE", "orchestrator/prompts/confirmation.txt")
    # Registries: which agents exist, which MCP servers they can use
    AGENTS_FILE = _project_path("AGENTS_FILE", "agents.json")
    MCP_FILE = _project_path("MCP_FILE", "mcp.json")
    # Expense ledger + input log — kept outside the project in the user data dir;
    # a relative EXPENSES_DIR is resolved from the project root, "~" is expanded
    EXPENSES_DIR = _data_path("EXPENSES_DIR", _user_data_dir() / "expenses")
    # The Streamlit UI's saved chat (resumed by every new browser session until "New chat")
    STREAMLIT_CHAT_FILE = _data_path("STREAMLIT_CHAT_FILE", _user_data_dir() / "streamlit" / "chat.json")
    # FBR (Pakistan) tax filing — the project holding the fbr-iris-filing skill and its snapshots/;
    # a relative path is resolved from the project root, "~" is expanded
    FBR_TAX_DIR = _data_path("FBR_TAX_DIR", "~/work/fbr-tax-filing")
    # The FBR agent's own records (filings.db: each tax year's submitted/draft status)
    FBR_TAX_DATA_DIR = _data_path("FBR_TAX_DATA_DIR", _user_data_dir() / "fbr_tax")
    # Tool calls one filing step may take (reading a whole return is many browser actions)
    FBR_TAX_MAX_TURNS = int(os.getenv("FBR_TAX_MAX_TURNS", "200"))
    # Local faster-whisper model size for voice transcription
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

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
    MAIL_SEEN_DB                   = _project_path("MAIL_SEEN_DB", "mail_seen.db")


# LangSmith reads its settings from the environment; values already exported win.
os.environ.setdefault("LANGCHAIN_TRACING_V2", Config.LANGCHAIN_TRACING_V2)
os.environ.setdefault("LANGCHAIN_PROJECT", Config.LANGCHAIN_PROJECT)
if Config.LANGCHAIN_API_KEY:
    os.environ.setdefault("LANGCHAIN_API_KEY", Config.LANGCHAIN_API_KEY)
