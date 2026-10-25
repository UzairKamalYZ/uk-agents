from unittest.mock import patch

from agents.whatsapp.green_api_client import GreenAPIClient
from core.src.output_guard import WITHHELD_REPLY


def _sent(message):
    client = GreenAPIClient.__new__(GreenAPIClient)  # no credentials needed to build the request
    client._base, client._token = "https://api.example", "token"
    with patch("agents.whatsapp.green_api_client.requests.post") as post:
        post.return_value.json.return_value = {"idMessage": "1"}
        client.send_message("123@g.us", message)
    return post.call_args.kwargs["json"]["message"]


def test_send_message_strips_tool_call_markup():
    assert _sent('Summary ready.<invoke name="Skill"><parameter name="a">x</parameter></invoke>') == "Summary ready."


def test_send_message_never_sends_bare_markup():
    assert _sent('<invoke name="Skill"></invoke>') == WITHHELD_REPLY


def test_send_message_leaves_plain_text_alone():
    assert _sent("3 new emails from the bank.") == "3 new emails from the bank."
