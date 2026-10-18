from pathlib import Path

from faster_whisper import WhisperModel

from ..config.config import Config


class Transcriber:
    """Local speech-to-text via faster-whisper. Audio never leaves the machine."""

    def __init__(self, model_size: str | None = None) -> None:
        # Resolved at construction time (not import time) so Config overrides apply
        self._model_size = model_size or Config.WHISPER_MODEL
        self._model: WhisperModel | None = None

    def _get_model(self) -> WhisperModel:
        if self._model is None:
            self._model = WhisperModel(self._model_size, device="cpu", compute_type="int8")
        return self._model

    def transcribe(self, audio_path: str | Path) -> str:
        segments, _info = self._get_model().transcribe(str(audio_path), vad_filter=True)
        return " ".join(s.text.strip() for s in segments).strip()
