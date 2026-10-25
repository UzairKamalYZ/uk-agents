import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from clients.telegram.main import TelegramClient


class TestTelegramClient(unittest.TestCase):

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @patch("clients.telegram.main.OrchestratorAgent")
    @patch("clients.telegram.main._Bot")
    def test_should_create_bot_instance_when_client_is_created(self, MockBot, _MockAgent):
        """TelegramClient instantiates _Bot during __init__."""
        TelegramClient()
        MockBot.assert_called_once()

    @patch("clients.telegram.main.OrchestratorAgent")
    @patch("clients.telegram.main._Bot")
    def test_should_store_bot_as_attribute_when_client_is_created(self, MockBot, _MockAgent):
        """TelegramClient stores the _Bot instance as self._bot."""
        mock_bot = MagicMock()
        MockBot.return_value = mock_bot
        client = TelegramClient()
        self.assertIs(client._bot, mock_bot)

    # ------------------------------------------------------------------
    # start()
    # ------------------------------------------------------------------

    @patch("clients.telegram.main.OrchestratorAgent")
    @patch("clients.telegram.main._Bot")
    def test_should_call_bot_run_when_start_is_called(self, MockBot, _MockAgent):
        """start() delegates to _bot.run() which starts long-polling."""
        mock_bot = MagicMock()
        MockBot.return_value = mock_bot
        client = TelegramClient()
        client.start()
        mock_bot.run.assert_called_once()

    @patch("clients.telegram.main.OrchestratorAgent")
    @patch("clients.telegram.main._Bot")
    def test_should_not_call_bot_run_before_start_is_called(self, MockBot, _MockAgent):
        """_bot.run() is never called during construction — only on start()."""
        mock_bot = MagicMock()
        MockBot.return_value = mock_bot
        TelegramClient()  # construction only
        mock_bot.run.assert_not_called()

    # ------------------------------------------------------------------
    # BaseClient conformance
    # ------------------------------------------------------------------

    @patch("clients.telegram.main.OrchestratorAgent")
    @patch("clients.telegram.main._Bot")
    def test_should_implement_base_client_interface(self, _MockBot, _MockAgent):
        """TelegramClient is a concrete implementation of BaseClient."""
        from clients.base import BaseClient
        self.assertIsInstance(TelegramClient(), BaseClient)

    @patch("clients.telegram.main.OrchestratorAgent")
    @patch("clients.telegram.main._Bot")
    def test_should_not_raise_when_stop_is_called(self, _MockBot, _MockAgent):
        """stop() is a safe no-op on TelegramClient."""
        client = TelegramClient()
        try:
            client.stop()
        except Exception as exc:
            self.fail(f"stop() raised unexpectedly: {exc}")

    @patch("clients.telegram.main.OrchestratorAgent")
    @patch("clients.telegram.main._Bot")
    def test_should_call_stop_via_context_manager_when_block_exits(self, MockBot, _MockAgent):
        """Context-manager exit triggers stop() without raising."""
        client = TelegramClient()
        with client:
            pass


class TestBot(unittest.TestCase):
    """Tests for the internal _Bot helper class."""

    @patch("clients.telegram.main.ApplicationBuilder")
    @patch("clients.telegram.main.OrchestratorAgent")
    def test_should_create_local_agent_when_bot_is_created(self, MockAgent, _MockBuilder):
        """_Bot instantiates OrchestratorAgent during construction."""
        from clients.telegram.main import _Bot
        _Bot()
        MockAgent.assert_called_once()

    @patch("clients.telegram.main.OrchestratorAgent")
    def test_should_build_telegram_app_with_bot_token_when_created(self, _MockAgent):
        """_Bot calls ApplicationBuilder().token(Config.TELEGRAM_BOT_TOKEN).build()."""
        from clients.telegram.main import _Bot
        from core.src.config.config import Config
        with patch("clients.telegram.main.ApplicationBuilder") as MockBuilder:
            _Bot()
            MockBuilder.return_value.token.assert_called_once_with(Config.TELEGRAM_BOT_TOKEN)
            MockBuilder.return_value.token.return_value.concurrent_updates.assert_called_once_with(True)


class TestVoiceAndAnswers(unittest.TestCase):

    def _make(self, transcript="", transcribe_exc=None):
        from clients.telegram.main import _Bot
        with patch("clients.telegram.main.OrchestratorAgent") as MockAgent, \
                patch("clients.telegram.main.ApplicationBuilder"), \
                patch("clients.telegram.main.Transcriber") as MockTr, \
                patch("clients.telegram.main.Config") as MockCfg:
            MockCfg.TELEGRAM_BOT_TOKEN = "t"
            bot = _Bot()
        bot.local_agent.aask = AsyncMock(return_value="Logged €90…")
        tr = bot.transcriber
        if transcribe_exc:
            tr.transcribe = MagicMock(side_effect=transcribe_exc)
        else:
            tr.transcribe = MagicMock(return_value=transcript)
        sent = MagicMock()
        sent.edit_text = AsyncMock()
        update = MagicMock()
        update.effective_chat.id = 7
        update.effective_message.text = "hello"
        update.effective_message.reply_text = AsyncMock(return_value=sent)
        context = MagicMock()
        file = MagicMock()
        file.download_to_drive = AsyncMock()
        context.bot.get_file = AsyncMock(return_value=file)
        return bot, update, context, sent

    def _run(self, coro_fn, update, context):
        asyncio.run(coro_fn(update, context))

    def test_run_loads_whisper_model_before_polling(self):
        bot, *_ = self._make()
        order = MagicMock()
        bot.transcriber.load = order.load
        bot.app.run_polling = order.run_polling
        bot.run()
        self.assertEqual([c[0] for c in order.mock_calls], ["load", "run_polling"])

    def test_construction_does_not_load_whisper_model(self):
        bot, *_ = self._make()
        bot.transcriber.load.assert_not_called()

    def test_voice_shows_transcript_then_answer(self):
        bot, update, context, sent = self._make(transcript="Colruyt 90 euros today")
        self._run(bot._handle_voice, update, context)
        sent.edit_text.assert_awaited_with("🎤 You said: Colruyt 90 euros today\n\nLogged €90…")
        self.assertEqual(bot.local_agent.aask.call_args.args[0], "Colruyt 90 euros today")

    def test_empty_transcript_asks_to_retry(self):
        bot, update, context, sent = self._make(transcript="")
        self._run(bot._handle_voice, update, context)
        bot.local_agent.aask.assert_not_called()
        update.effective_message.reply_text.assert_awaited_once_with(
            "🎤 I couldn't hear anything — please try again or type it.")

    def test_transcription_error_replies_sorry(self):
        bot, update, context, sent = self._make(transcribe_exc=RuntimeError("boom"))
        self._run(bot._handle_voice, update, context)
        bot.local_agent.aask.assert_not_called()
        update.effective_message.reply_text.assert_awaited_once_with(
            "Sorry, I couldn't transcribe that voice note. Please try again or type it.")

    def test_message_is_asked_with_chat_session_and_no_owner_flag(self):
        bot, update, context, sent = self._make()
        self._run(bot._handle_message, update, context)
        bot.local_agent.aask.assert_awaited_once()
        self.assertEqual(bot.local_agent.aask.call_args.kwargs["session_id"], "7")
        self.assertNotIn("is_owner", bot.local_agent.aask.call_args.kwargs)

    def test_ask_error_still_replies_sorry(self):
        bot, update, context, sent = self._make()
        bot.local_agent.aask = AsyncMock(side_effect=RuntimeError("x"))
        self._run(bot._handle_message, update, context)
        sent.edit_text.assert_awaited_with("Sorry, something went wrong. Please try again.")

    def test_announces_picked_agent_with_remark_before_answer(self):
        from types import SimpleNamespace
        bot, update, context, sent = self._make()
        entry = SimpleNamespace(label="Expenditure Tracker", remarks=("Counting your coins…",))

        async def aask(text, **kwargs):
            kwargs["on_route"](entry)  # the orchestrator picks an agent mid-request
            return "✅ Expense saved"

        bot.local_agent.aask = AsyncMock(side_effect=aask)
        self._run(bot._handle_message, update, context)
        update.effective_message.reply_text.assert_awaited_once_with("🔀 Finding the right agent for your query…")
        self.assertEqual([c.args[0] for c in sent.edit_text.await_args_list], [
            "🤖 Expenditure Tracker is working on your query…\n\n😄 Counting your coins…",
            "✅ Expense saved",
        ])


if __name__ == "__main__":
    unittest.main()
