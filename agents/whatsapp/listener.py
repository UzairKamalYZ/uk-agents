"""
agents/whatsapp/listener.py — polls a WhatsApp group for new messages and
auto-replies when the owner (Uzair) is mentioned or asked for.

Flow
----
  1. Poll Green API for one pending notification.
  2. Ignore anything that is not an incoming text message from the target group.
  3. Ask the LLM: "Is this message directed at {user_name}?"
  4. If YES → send the configured away message to the group.
  5. Delete the notification regardless of outcome (so it doesn't come back).
  6. Repeat.

The poll_interval controls how long the loop sleeps when the notification
queue is empty.  A value of 2–5 seconds is a reasonable default.
"""

import logging
import time
from pathlib import Path

from core.src.config.config import Config
from core.src.model.llm_factory import create_llm
from core.src.utils import load_prompt
from .green_api_client import GreenAPIClient

logger = logging.getLogger(__name__)

_MENTION_PROMPT_FILE = Path(__file__).parent / "prompts" / "mention_check.txt"


class WhatsAppListener:
    """
    Polls a WhatsApp group and auto-replies when the owner is mentioned.

    Parameters
    ----------
    poll_interval : seconds to sleep when the notification queue is empty
    """

    def __init__(self, poll_interval: float = None) -> None:
        self._client = GreenAPIClient()
        self._group_id = Config.WHATSAPP_GROUP_ID
        self._user_name = Config.WHATSAPP_USER_NAME
        self._away_message = Config.WHATSAPP_AWAY_MESSAGE
        self._poll_interval = poll_interval or float(Config.WHATSAPP_POLL_INTERVAL)
        self._mention_prompt_template = load_prompt(_MENTION_PROMPT_FILE)
        self._llm = create_llm(Config.LLM_PROVIDER, Config.LLM_MODEL_NAME)
        self._running = False

        if not self._group_id:
            raise ValueError("WHATSAPP_GROUP_ID must be set in .env")

        logger.info(
            "[WhatsAppListener] ready — group=%s user=%s interval=%.1fs",
            self._group_id, self._user_name, self._poll_interval,
        )

    # ------------------------------------------------------------------
    # Mention detection
    # ------------------------------------------------------------------

    def _is_asking_for_user(self, message: str) -> bool:
        """
        Return True if the LLM decides this message requires the owner's attention.

        Uses a dedicated lightweight prompt (mention_check.txt) instead of the
        full orchestrator to keep latency low — this runs on every message.
        """
        prompt = self._mention_prompt_template.format(
            user_name=self._user_name,
            message=message,
        )
        response = self._llm.invoke(prompt)
        # LangChain chat models return an AIMessage; plain LLMs return a string
        text = response.text
        answer = text.strip().upper()
        is_mention = answer.startswith("YES")
        logger.debug(
            "[WhatsAppListener] mention check for %r → %s (raw=%r)",
            message[:60], "YES" if is_mention else "NO", text[:20],
        )
        return is_mention

    # ------------------------------------------------------------------
    # Notification processing
    # ------------------------------------------------------------------

    def _extract_group_text(self, notification: dict) -> tuple[str, str] | None:
        """
        Parse a raw Green API notification and extract (chat_id, message_text).

        Returns None if the notification is not an incoming text message from
        the configured target group.
        """
        body = notification.get("body", {})

        # Only handle incoming messages — ignore delivery receipts, state changes, etc.
        if body.get("typeWebhook") != "incomingMessageReceived":
            return None

        sender_data = body.get("senderData", {})
        chat_id = sender_data.get("chatId", "")

        # Filter to target group only
        if chat_id != self._group_id:
            logger.debug("[WhatsAppListener] ignoring message from chat %s", chat_id)
            return None

        message_data = body.get("messageData", {})

        # Green API nests the text under textMessageData for plain text messages
        text = message_data.get("textMessageData", {}).get("textMessage", "").strip()
        if not text:
            return None

        sender_name = sender_data.get("senderName", "unknown")
        logger.info(
            "[WhatsAppListener] group message from %s: %s",
            sender_name, text[:120],
        )
        return chat_id, text

    def _process_notification(self, notification: dict) -> None:
        """
        Handle one notification: check for mention, send reply if needed.

        This method is intentionally kept free of I/O side-effects beyond the
        API calls so it can be unit-tested with mock clients.
        """
        result = self._extract_group_text(notification)
        if result is None:
            return

        chat_id, text = result

        if self._is_asking_for_user(text):
            logger.info(
                "[WhatsAppListener] '%s' mentioned — sending away message to %s",
                self._user_name, chat_id,
            )
            self._client.send_message(chat_id, self._away_message)
        else:
            logger.debug("[WhatsAppListener] message not for %s — skipping", self._user_name)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Enter the polling loop.  Blocks until stop() is called.

        On each iteration:
          - Fetch one notification from Green API.
          - If the queue is empty, sleep for poll_interval seconds.
          - If a notification arrived, process it and delete it from the queue.
          - Any exception is logged and the loop sleeps for 5 seconds before retrying.
        """
        self._running = True
        logger.info("[WhatsAppListener] starting poll loop")

        while self._running:
            try:
                notification = self._client.receive_notification()

                if notification is None:
                    # Queue empty — wait before polling again
                    time.sleep(self._poll_interval)
                    continue

                receipt_id = notification.get("receiptId")
                try:
                    self._process_notification(notification)
                finally:
                    # Always delete the notification — even on processing errors —
                    # so the same event is never processed twice.
                    if receipt_id is not None:
                        self._client.delete_notification(receipt_id)

            except Exception:
                logger.exception(
                    "[WhatsAppListener] unexpected error — retrying in 5 s"
                )
                time.sleep(5.0)

        logger.info("[WhatsAppListener] stopped")

    def stop(self) -> None:
        """Signal the polling loop to exit after the current iteration."""
        self._running = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
