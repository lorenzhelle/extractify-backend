"""Callback handlers used in the app."""

from typing import Any, Callable
from app.models.server_sent_events import EventType, ServerSentMessage

from langchain.callbacks.base import AsyncCallbackHandler


class StreamingLLMCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def __init__(self, send_websocket_message: Callable[[Any], None]):
        self.send_message = send_websocket_message

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        resp = ServerSentMessage(
            event=EventType.SERVER_NEW_TOKEN,
            payload={
                "token": token,
            },
        )
        await self.send_message(resp)
