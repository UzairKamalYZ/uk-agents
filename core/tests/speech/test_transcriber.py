import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from core.src.speech.transcriber import Transcriber


def _model(texts):
    model = MagicMock()
    model.transcribe.return_value = (iter([SimpleNamespace(text=t) for t in texts]), MagicMock())
    return model


def test_joins_segments():
    with patch("core.src.speech.transcriber.WhisperModel", return_value=_model([" Colruyt", " ninety euros "])) as cls:
        assert Transcriber().transcribe("a.ogg") == "Colruyt ninety euros"
    cls.return_value.transcribe.assert_called_once_with("a.ogg", vad_filter=True)


def test_empty_audio_returns_empty_string():
    with patch("core.src.speech.transcriber.WhisperModel", return_value=_model([])):
        assert Transcriber().transcribe("a.ogg") == ""


def test_model_loaded_once_with_cpu_int8():
    model = MagicMock()
    model.transcribe.side_effect = lambda *a, **k: (iter([]), MagicMock())
    with patch("core.src.speech.transcriber.WhisperModel", return_value=model) as cls, \
         patch("core.src.speech.transcriber.Config") as cfg:
        cfg.WHISPER_MODEL = "small"
        t = Transcriber()
        t.transcribe("a.ogg")
        t.transcribe("b.ogg")
    cls.assert_called_once_with("small", device="cpu", compute_type="int8")


def test_explicit_model_size_overrides_config():
    model = MagicMock()
    model.transcribe.return_value = (iter([]), MagicMock())
    with patch("core.src.speech.transcriber.WhisperModel", return_value=model) as cls, \
         patch("core.src.speech.transcriber.Config") as cfg:
        cfg.WHISPER_MODEL = "small"
        Transcriber("tiny").transcribe("a.ogg")
    cls.assert_called_once_with("tiny", device="cpu", compute_type="int8")


def test_config_default_whisper_model_is_small():
    # Subprocess: other tests stub Config process-wide, so check the real default in isolation.
    # dotenv is neutralised so a developer .env cannot leak WHISPER_MODEL into the child.
    env = {k: v for k, v in os.environ.items() if k != "WHISPER_MODEL"}
    env.update(PROMPT_FILE="p", INTERACTION_FILE="i", REFORMULATION_PROMPT="r")
    code = (
        "import dotenv; dotenv.load_dotenv = lambda *a, **k: None; "
        "from core.src.config.config import Config; print(Config.WHISPER_MODEL)"
    )
    root = Path(__file__).parents[3]
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "small"


def test_load_is_idempotent_and_reused_by_transcribe():
    with patch("core.src.speech.transcriber.WhisperModel", return_value=_model([])) as cls:
        t = Transcriber("tiny")
        cls.assert_not_called()
        t.load()
        t.load()
        t.transcribe("a.ogg")
    cls.assert_called_once_with("tiny", device="cpu", compute_type="int8")
