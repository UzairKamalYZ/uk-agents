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
        # Chats are answered concurrently; the orchestrator keeps each chat's messages in order
        self.app = ApplicationBuilder().token(Config.TELEGRAM_BOT_TOKEN).concurrent_updates(True).build()
        self.app.add_handler(CommandHandler("start", self._handle_start))
        self.app.add_handler(CommandHandler("whoami", self._handle_whoami))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))
        self.app.add_handler(MessageHandler(filters.VOICE, self._handle_voice))

    @staticmethod
    def _owner_ids() -> set[str]:
        """TELEGRAM_OWNER_ID as a set: comma/space separated ids or @usernames, quotes ignored."""
        raw = (Config.TELEGRAM_OWNER_ID or "").replace(",", " ")
        return {part.strip("'\"").lstrip("@").lower() for part in raw.split() if part.strip("'\"@")}

    @staticmethod
    def _sender_keys(update: Update) -> set[str]:
        """Ids and usernames that identify who sent this update — the user, or the
        channel/group they posted as (channel posts and anonymous admins have no user)."""
        keys = set()
        for sender in (update.effective_user, update.effective_sender):
            if sender is None:
                continue
            keys.add(str(sender.id))
            if getattr(sender, "username", None):
                keys.add(sender.username.lower())
        return keys

    def _is_owner(self, update: Update) -> bool:
        return bool(self._owner_ids() & self._sender_keys(update))

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.effective_message.reply_text(
            "Hello! I'm your personal assistant. Send me a message or a voice note."
        )

    async def _handle_whoami(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Shows the ids to put in TELEGRAM_OWNER_ID."""
        keys = ", ".join(sorted(self._sender_keys(update))) or "unknown"
        status = "✅ recognised as the owner" if self._is_owner(update) else "❌ not the owner"
        await update.effective_message.reply_text(
            f"Your Telegram ids: {keys}\nThis chat: {update.effective_chat.id}\n{status}\n\n"
            "Put one of your ids in TELEGRAM_OWNER_ID in .env and restart the bot."
        )

    async def _answer(self, update: Update, text: str, header: str = "") -> None:
        joke = random.choice(WAITING_JOKES)
        sent = await update.effective_message.reply_text(
            f"{header}Thinking...\n\n💡 While you wait: {joke}"
        )

        session_id = str(update.effective_chat.id)
        is_owner = self._is_owner(update)
        logger.info("chat=%s sender=%s owner=%s", session_id, sorted(self._sender_keys(update)), is_owner)

        try:
            response = await self.local_agent.aask(text, session_id=session_id, is_owner=is_owner)
            await sent.edit_text(f"{header}{response or 'No response generated.'}")

        except Exception as e:
            logger.error(f"Error from LocalAgent: {e}")
            await sent.edit_text("Sorry, something went wrong. Please try again.")

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
        if not self._owner_ids():
            logger.warning("TELEGRAM_OWNER_ID is empty — owner-only agents will refuse everyone. Send /whoami to the bot to get your id.")
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
