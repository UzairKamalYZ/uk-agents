from logging_config import setup_logging
setup_logging()

import asyncio
import functools
import logging
import random
import tempfile
import time
from pathlib import Path

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from orchestrator.agent_interface import OrchestratorAgent
from core.src.config.config import Config
from core.src.speech.transcriber import Transcriber
from clients.base import BaseClient

ROUTING_TEXT = "🔀 Finding the right agent for your query…"
WORKING_TEXT = "🤖 {agent} is working on your query…"

logger = logging.getLogger(__name__)


class _Bot:

    def __init__(self):
        self.local_agent = OrchestratorAgent()
        self.transcriber = Transcriber()
        self.app = ApplicationBuilder().token(Config.TELEGRAM_BOT_TOKEN).build()
        self.app.add_handler(CommandHandler("start", self._handle_start))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))
        self.app.add_handler(MessageHandler(filters.VOICE, self._handle_voice))

    @staticmethod
    def _is_owner(user_id: int) -> bool:
        return bool(Config.TELEGRAM_OWNER_ID) and str(user_id) == Config.TELEGRAM_OWNER_ID

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "Hello! I'm your personal assistant. Send me a message or a voice note."
        )

    async def _answer(self, update: Update, text: str, header: str = "") -> None:
        sent = await update.message.reply_text(f"{header}{ROUTING_TEXT}")

        session_id = str(update.effective_chat.id)
        is_owner = self._is_owner(update.effective_user.id)
        loop = asyncio.get_running_loop()

        def on_route(entry) -> None:
            # Runs on the executor thread as soon as the orchestrator picks an agent —
            # hop back onto the event loop to edit the placeholder, and wait so this
            # edit always lands before the final answer.
            status = self._working_text(entry)
            asyncio.run_coroutine_threadsafe(sent.edit_text(f"{header}{status}"), loop).result(timeout=10)

        try:
            fn = functools.partial(
                self.local_agent.ask, text, stream=False, session_id=session_id,
                is_owner=is_owner, on_route=on_route,
            )
            started = time.monotonic()
            response = await loop.run_in_executor(None, fn)
            logger.info("Answered chat %s in %.1fs (%d chars)", session_id, time.monotonic() - started, len(response or ""))
            await sent.edit_text(f"{header}{response or 'No response generated.'}")

        except Exception:
            logger.exception("Error answering chat %s", session_id)
            await sent.edit_text("Sorry, something went wrong. Please try again.")

    @staticmethod
    def _working_text(entry) -> str:
        status = WORKING_TEXT.format(agent=entry.label)
        if entry.remarks:
            status += f"\n\n😄 {random.choice(entry.remarks)}"
        return status

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_text = update.message.text
        logger.info(f"Received message from {update.effective_user.id}: {user_text}")
        await self._answer(update, user_text)

    async def _handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.info(f"Received voice note from {update.effective_user.id}")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "voice.ogg"
                tg_file = await context.bot.get_file(update.message.voice.file_id)
                await tg_file.download_to_drive(path)
                transcript = await asyncio.get_running_loop().run_in_executor(
                    None, self.transcriber.transcribe, path
                )
        except Exception:
            logger.exception("Voice transcription failed")
            await update.message.reply_text(
                "Sorry, I couldn't transcribe that voice note. Please try again or type it."
            )
            return

        logger.info("Transcribed voice note from %s: %r", update.effective_user.id, transcript)
        if not transcript:
            await update.message.reply_text("🎤 I couldn't hear anything — please try again or type it.")
            return

        await self._answer(update, transcript, header=f"🎤 You said: {transcript}\n\n")

    def run(self):
        self.transcriber.load()  # load Whisper once at startup, not on the first voice note
        self.app.run_polling()


class TelegramClient(BaseClient):
    """Runs the Telegram bot via long-polling."""

    def __init__(self):
        self._bot = _Bot()

    def start(self) -> None:
        logger.info("Starting Telegram client...")
        self._bot.run()


if __name__ == "__main__":
    TelegramClient().start()
