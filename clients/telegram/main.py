from logging_config import setup_logging
setup_logging()

import asyncio
import functools
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

WAITING_JOKES = [
    "Why did the scarecrow win an award? Because he was outstanding in his field — much like your future home!",
    "Why do real estate agents make great comedians? Because they always know how to close!",
    "What do you call a haunted property? A buy-one-get-one-scare deal.",
    "Why did the house go to therapy? It had too many issues with its foundation.",
    "What did the ocean say to the beach house? Nothing, it just waved.",
    "Why don't houses ever get lonely? Because they're always in a good neighbourhood.",
    "What's a real estate agent's favourite type of music? House!",
]

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
        joke = random.choice(WAITING_JOKES)
        sent = await update.message.reply_text(
            f"{header}Thinking...\n\n💡 While you wait: {joke}"
        )

        session_id = str(update.effective_chat.id)
        is_owner = self._is_owner(update.effective_user.id)

        try:
            fn = functools.partial(
                self.local_agent.ask, text, stream=False, session_id=session_id, is_owner=is_owner
            )
            response = await asyncio.get_running_loop().run_in_executor(None, fn)
            await sent.edit_text(f"{header}{response or 'No response generated.'}")

        except Exception as e:
            logger.error(f"Error from LocalAgent: {e}")
            await sent.edit_text("Sorry, something went wrong. Please try again.")

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
        except Exception as e:
            logger.error(f"Voice transcription failed: {e}")
            await update.message.reply_text(
                "Sorry, I couldn't transcribe that voice note. Please try again or type it."
            )
            return

        if not transcript:
            await update.message.reply_text("🎤 I couldn't hear anything — please try again or type it.")
            return

        await self._answer(update, transcript, header=f"🎤 You said: {transcript}\n\n")

    def run(self):
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
