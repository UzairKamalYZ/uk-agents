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


class TestVoiceAndOwner(unittest.TestCase):

    def _make(self, transcript="", owner_id="", user_id=42, transcribe_exc=None, username=None):
        from clients.telegram.main import _Bot
        with patch("clients.telegram.main.OrchestratorAgent") as MockAgent, \
                patch("clients.telegram.main.ApplicationBuilder"), \
                patch("clients.telegram.main.Transcriber") as MockTr, \
                patch("clients.telegram.main.Config") as MockCfg:
            MockCfg.TELEGRAM_OWNER_ID = owner_id
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
        update.effective_user.id = user_id
        update.effective_user.username = username
        update.effective_sender = update.effective_user
        update.effective_chat.id = 7
        update.effective_message.text = "hello"
        update.effective_message.reply_text = AsyncMock(return_value=sent)
        context = MagicMock()
        file = MagicMock()
        file.download_to_drive = AsyncMock()
        context.bot.get_file = AsyncMock(return_value=file)
        return bot, update, context, sent, owner_id

    def _run(self, coro_fn, bot, update, context, owner_id):
        with patch("clients.telegram.main.Config") as MockCfg:
            MockCfg.TELEGRAM_OWNER_ID = owner_id
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
        bot, update, context, sent, o = self._make(transcript="Colruyt 90 euros today")
        self._run(bot._handle_voice, bot, update, context, o)
        sent.edit_text.assert_awaited_with("🎤 You said: Colruyt 90 euros today\n\nLogged €90…")
        self.assertEqual(bot.local_agent.aask.call_args.args[0], "Colruyt 90 euros today")

    def test_empty_transcript_asks_to_retry(self):
        bot, update, context, sent, o = self._make(transcript="")
        self._run(bot._handle_voice, bot, update, context, o)
        bot.local_agent.aask.assert_not_called()
        update.effective_message.reply_text.assert_awaited_once_with(
            "🎤 I couldn't hear anything — please try again or type it.")

    def test_transcription_error_replies_sorry(self):
        bot, update, context, sent, o = self._make(transcribe_exc=RuntimeError("boom"))
        self._run(bot._handle_voice, bot, update, context, o)
        bot.local_agent.aask.assert_not_called()
        update.effective_message.reply_text.assert_awaited_once_with(
            "Sorry, I couldn't transcribe that voice note. Please try again or type it.")

    def test_owner_flag_true_for_owner_id(self):
        bot, update, context, sent, o = self._make(owner_id="42", user_id=42)
        self._run(bot._handle_message, bot, update, context, o)
        self.assertIs(bot.local_agent.aask.call_args.kwargs["is_owner"], True)

    def test_owner_flag_false_when_unset_or_other_user(self):
        for owner_id, uid in (("", 42), ("99", 42)):
            bot, update, context, sent, o = self._make(owner_id=owner_id, user_id=uid)
            self._run(bot._handle_message, bot, update, context, o)
            self.assertIs(bot.local_agent.aask.call_args.kwargs["is_owner"], False)

    def test_owner_id_tolerates_quotes_spaces_usernames_and_lists(self):
        for owner_id, uid, username in (
            ('"42"', 42, None), (" 42 ", 42, None), ("7, 42", 42, None),
            ("@UzairK", 99, "uzairk"), ("uzairk", 99, "UzairK"),
        ):
            bot, update, context, sent, o = self._make(owner_id=owner_id, user_id=uid, username=username)
            self._run(bot._handle_message, bot, update, context, o)
            self.assertIs(bot.local_agent.aask.call_args.kwargs["is_owner"], True, owner_id)

    def test_owner_recognised_when_posting_as_channel(self):
        bot, update, context, sent, o = self._make(owner_id="-1001234")
        update.effective_user = None  # channel posts carry no user, only the sender chat
        update.effective_sender = MagicMock(id=-1001234, username=None)
        self._run(bot._handle_message, bot, update, context, o)
        self.assertIs(bot.local_agent.aask.call_args.kwargs["is_owner"], True)

    def test_whoami_shows_ids_and_owner_status(self):
        bot, update, context, sent, o = self._make(owner_id="42", user_id=42, username="UzairK")
        self._run(bot._handle_whoami, bot, update, context, o)
        text = update.effective_message.reply_text.call_args.args[0]
        self.assertIn("42, uzairk", text)
        self.assertIn("recognised as the owner", text)

    def test_ask_error_still_replies_sorry(self):
        bot, update, context, sent, o = self._make()
        bot.local_agent.aask = AsyncMock(side_effect=RuntimeError("x"))
        self._run(bot._handle_message, bot, update, context, o)
        sent.edit_text.assert_awaited_with("Sorry, something went wrong. Please try again.")

    def test_announces_picked_agent_with_remark_before_answer(self):
        from types import SimpleNamespace
        bot, update, context, sent, o = self._make()
        entry = SimpleNamespace(label="Expenditure Tracker", remarks=("Counting your coins…",))

        def ask(text, **kwargs):
            kwargs["on_route"](entry)  # the orchestrator picks an agent mid-request
            return "✅ Expense saved"

        bot.local_agent.ask = MagicMock(side_effect=ask)
        self._run(bot._handle_message, bot, update, context, o)
        update.message.reply_text.assert_awaited_once_with("🔀 Finding the right agent for your query…")
        self.assertEqual([c.args[0] for c in sent.edit_text.await_args_list], [
            "🤖 Expenditure Tracker is working on your query…\n\n😄 Counting your coins…",
            "✅ Expense saved",
        ])


if __name__ == "__main__":
    unittest.main()
