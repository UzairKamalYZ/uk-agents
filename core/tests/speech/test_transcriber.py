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
