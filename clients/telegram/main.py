from logging_config import setup_logging
setup_logging()

import asyncio
import logging
import random
import tempfile
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
        # Chats are answered concurrently; the orchestrator keeps each chat's messages in order
        self.app = ApplicationBuilder().token(Config.TELEGRAM_BOT_TOKEN).concurrent_updates(True).build()
        self.app.add_handler(CommandHandler("start", self._handle_start))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))
        self.app.add_handler(MessageHandler(filters.VOICE, self._handle_voice))

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.effective_message.reply_text(
            "Hello! I'm your personal assistant. Send me a message or a voice note."
        )

    async def _answer(self, update: Update, text: str, header: str = "") -> None:
        sent = await update.effective_message.reply_text(f"{header}{ROUTING_TEXT}")

        session_id = str(update.effective_chat.id)
        loop = asyncio.get_running_loop()
        status_edits = []

        def on_route(entry) -> None:
            # Called on the orchestrator's loop as an agent is picked — edit the
            # placeholder on this loop; the edits are awaited before the answer.
            edit = sent.edit_text(f"{header}{self._working_text(entry)}")
            status_edits.append(asyncio.wrap_future(asyncio.run_coroutine_threadsafe(edit, loop)))

        try:
            response = await self.local_agent.aask(
                text, session_id=session_id, on_route=on_route
            )
            for result in await asyncio.gather(*status_edits, return_exceptions=True):
                if isinstance(result, Exception):
                    logger.warning("Could not show the working agent: %s", result)
            await sent.edit_text(f"{header}{response or 'No response generated.'}")

        except Exception as e:
            logger.error(f"Error from LocalAgent: {e}")
            await sent.edit_text("Sorry, something went wrong. Please try again.")

    @staticmethod
    def _working_text(entry) -> str:
        status = WORKING_TEXT.format(agent=entry.label)
        if entry.remarks:
            status += f"\n\n😄 {random.choice(entry.remarks)}"
        return status

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_text = update.effective_message.text
        logger.info("Received message in chat %s: %s", update.effective_chat.id, user_text)
        await self._answer(update, user_text)

    async def _handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.info("Received voice note in chat %s", update.effective_chat.id)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "voice.ogg"
                tg_file = await context.bot.get_file(update.effective_message.voice.file_id)
                await tg_file.download_to_drive(path)
                transcript = await asyncio.get_running_loop().run_in_executor(
                    None, self.transcriber.transcribe, path
                )
        except Exception as e:
            logger.error(f"Voice transcription failed: {e}")
            await update.effective_message.reply_text(
                "Sorry, I couldn't transcribe that voice note. Please try again or type it."
            )
            return

        if not transcript:
            await update.effective_message.reply_text("🎤 I couldn't hear anything — please try again or type it.")
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
