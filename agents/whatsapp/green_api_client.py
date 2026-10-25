"""
agents/whatsapp/green_api_client.py — thin HTTP wrapper over the Green API REST interface.

Green API (green-api.com) lets you connect a personal WhatsApp account via QR
code and then interact with it through a simple REST API.  No WhatsApp Business
account is required.

Setup (one-time):
  1. Register at https://console.green-api.com and create an instance.
  2. In the dashboard, click "Scan QR code" and scan with your phone's WhatsApp.
  3. Copy INSTANCE_ID and INSTANCE_TOKEN into .env.
  4. In the instance settings, enable the "Incoming webhooks" notification type.

Endpoint reference:
  receiveNotification  — returns one pending event (or null if queue is empty)
  deleteNotification   — acknowledges the event so it won't be returned again
  sendMessage          — sends a text message to a chat ID
  getContacts          — lists all contacts (useful for finding a group ID)
"""

import logging

import requests

from core.src.config.config import Config
from core.src.output_guard import WITHHELD_REPLY, scrub

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.green-api.com"
_TIMEOUT_LONG = 30   # seconds — used for receive (server holds connection briefly)
_TIMEOUT_SHORT = 10  # seconds — used for send / delete


class GreenAPIClient:
    """
    Minimal Green API client.

    Every public method maps 1-to-1 to a Green API endpoint.  All HTTP errors
    are raised as requests.HTTPError so callers can decide how to handle them.
    """

    def __init__(self) -> None:
        instance_id = Config.WHATSAPP_INSTANCE_ID
        token = Config.WHATSAPP_INSTANCE_TOKEN
        if not instance_id or not token:
            raise ValueError(
                "WHATSAPP_INSTANCE_ID and WHATSAPP_INSTANCE_TOKEN must be set in .env"
            )
        self._base = f"{_BASE_URL}/waInstance{instance_id}"
        self._token = token

    # ------------------------------------------------------------------
    # Receiving
    # ------------------------------------------------------------------

    def receive_notification(self) -> dict | None:
        """
        Fetch one pending notification from the queue.

        Returns the raw notification dict if a message is waiting, or None if
        the queue is empty.  The caller must call delete_notification() with the
        returned receiptId to remove the event from the queue.
        """
        url = f"{self._base}/receiveNotification/{self._token}"
        resp = requests.get(url, timeout=_TIMEOUT_LONG)
        resp.raise_for_status()
        data = resp.json()
        # Green API returns null (None in Python) when the queue is empty
        return data if data else None

    def delete_notification(self, receipt_id: int) -> bool:
        """
        Acknowledge a notification so it is removed from the queue.

        Must be called after every successful receive_notification() — even if
        the message was not relevant — otherwise the same event keeps coming back.
        """
        url = f"{self._base}/deleteNotification/{self._token}/{receipt_id}"
        resp = requests.delete(url, timeout=_TIMEOUT_SHORT)
        resp.raise_for_status()
        return bool(resp.json().get("result", False))

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    def send_message(self, chat_id: str, message: str) -> dict:
        """
        Send a plain-text message to a chat.

        chat_id format:
          - Individual: "79001234567@c.us"
          - Group:      "120363xxxxxxxxx@g.us"
        """
        result = scrub(message)
        if result.found:
            logger.warning("WhatsApp message to %s contained tool-call markup %s — removed", chat_id, list(result.found))
        url = f"{self._base}/sendMessage/{self._token}"
        payload = {"chatId": chat_id, "message": result.text or WITHHELD_REPLY}
        resp = requests.post(url, json=payload, timeout=_TIMEOUT_SHORT)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------

    def get_contacts(self) -> list[dict]:
        """
        Return all WhatsApp contacts, including group chats.

        Useful for discovering a group's chatId — look for entries where
        "type" == "group" and "name" matches the target group name.
        """
        url = f"{self._base}/getContacts/{self._token}"
        resp = requests.get(url, timeout=_TIMEOUT_SHORT)
        resp.raise_for_status()
        return resp.json()
