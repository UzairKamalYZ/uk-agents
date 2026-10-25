import json
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).parents[2]

_KEYS = ("PROMPT_FILE", "URLS_FILE", "HOUSING_DATASET_DIR",
         "SESSION_DB_FILE", "CRON_SEARCH_PROMPT")


def test_config_imports_with_working_defaults_when_unset():
    # Fresh interpreter, .env loading disabled, the keys removed — the real Config, not a test stub.
    env = {k: v for k, v in os.environ.items() if k not in _KEYS}
    code = (
        "import dotenv; dotenv.load_dotenv = lambda *a, **k: None; import json; "
        "from core.src.config.config import Config as C; "
        f"print(json.dumps({{k: getattr(C, k) for k in {list(_KEYS)!r}}}))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=_ROOT, env=env,
                         capture_output=True, text=True, check=True).stdout
    cfg = json.loads(out.strip().splitlines()[-1])
    assert cfg["PROMPT_FILE"] == str(_ROOT / "agents/property/prompts/System_Prompt.txt")
    assert Path(cfg["PROMPT_FILE"]).exists()
    assert Path(cfg["URLS_FILE"]).exists()
    assert cfg["HOUSING_DATASET_DIR"] == str(_ROOT / "agents/property/datasets/pl-housing")
    assert cfg["SESSION_DB_FILE"] == str(_ROOT / "sessions.db")
    assert cfg["CRON_SEARCH_PROMPT"]
