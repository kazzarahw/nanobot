"""Tests for Telegram channel message splitting."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# The telegram C-extension chain can fail in CI/test environments, so we
# stub the third-party ``telegram`` package before importing our module.
_telegram_mock = MagicMock()
sys.modules.setdefault("telegram", _telegram_mock)
sys.modules.setdefault("telegram.ext", _telegram_mock)
sys.modules.setdefault("telegram.request", _telegram_mock)

from nanobot.bus.events import OutboundMessage  # noqa: E402
from nanobot.channels.telegram import (  # noqa: E402
    _MARKDOWN_SPLIT_LIMIT,
    _TELEGRAM_MAX_LENGTH,
    _split_text,
)


# ---------------------------------------------------------------------------
# _split_text unit tests
# ---------------------------------------------------------------------------


class TestSplitText:
    def test_empty_string_returns_empty_list(self):
        assert _split_text("", 100) == []

    def test_short_text_returns_single_chunk(self):
        assert _split_text("hello", 100) == ["hello"]

    def test_exact_limit_returns_single_chunk(self):
        text = "a" * 100
        assert _split_text(text, 100) == [text]

    def test_splits_at_paragraph_boundary(self):
        para1 = "a" * 40
        para2 = "b" * 40
        text = f"{para1}\n\n{para2}"
        chunks = _split_text(text, 50)
        assert chunks == [para1, para2]

    def test_splits_at_line_boundary_when_no_paragraph_break(self):
        line1 = "a" * 40
        line2 = "b" * 40
        text = f"{line1}\n{line2}"
        chunks = _split_text(text, 50)
        assert chunks == [line1, line2]

    def test_splits_at_space_when_no_newline(self):
        word1 = "a" * 40
        word2 = "b" * 40
        text = f"{word1} {word2}"
        chunks = _split_text(text, 50)
        assert chunks == [word1, word2]

    def test_hard_split_when_no_boundary(self):
        text = "a" * 200
        chunks = _split_text(text, 100)
        assert len(chunks) == 2
        assert chunks[0] == "a" * 100
        assert chunks[1] == "a" * 100

    def test_multiple_chunks(self):
        paragraphs = [f"paragraph {i} " + "x" * 80 for i in range(5)]
        text = "\n\n".join(paragraphs)
        chunks = _split_text(text, 100)
        assert len(chunks) == 5
        for i, chunk in enumerate(chunks):
            assert f"paragraph {i}" in chunk

    def test_preserves_all_content(self):
        """All original content (modulo boundary whitespace) should appear in chunks."""
        text = "Hello world.\n\nThis is paragraph two.\n\nAnd paragraph three."
        chunks = _split_text(text, 30)
        joined = " ".join(chunks)
        assert "Hello world." in joined
        assert "paragraph two" in joined
        assert "paragraph three" in joined

    def test_each_chunk_within_limit(self):
        text = "word " * 1000  # ~5000 chars
        chunks = _split_text(text, 100)
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_realistic_telegram_limit(self):
        """Simulate a message that exceeds Telegram's 4096-char limit."""
        text = ("Some paragraph content. " * 20 + "\n\n") * 10  # ~5k+ chars
        chunks = _split_text(text, _TELEGRAM_MAX_LENGTH)
        for chunk in chunks:
            assert len(chunk) <= _TELEGRAM_MAX_LENGTH
        assert sum(len(c) for c in chunks) > 0


# ---------------------------------------------------------------------------
# TelegramChannel.send integration tests
# ---------------------------------------------------------------------------


def _make_channel():
    """Create a TelegramChannel with a mocked bot for testing send()."""
    from nanobot.bus.queue import MessageBus
    from nanobot.channels.telegram import TelegramChannel
    from nanobot.config.schema import TelegramConfig

    config = TelegramConfig(enabled=True, token="fake-token")
    channel = TelegramChannel(config, MessageBus())

    # Mock the Application and bot
    mock_app = MagicMock()
    mock_app.bot = MagicMock()
    mock_app.bot.send_message = AsyncMock()
    channel._app = mock_app
    return channel


@pytest.mark.asyncio
async def test_send_short_message_sends_once():
    channel = _make_channel()
    msg = OutboundMessage(channel="telegram", chat_id="12345", content="Hi there!")
    await channel.send(msg)

    channel._app.bot.send_message.assert_called_once()
    call_kwargs = channel._app.bot.send_message.call_args.kwargs
    assert call_kwargs["chat_id"] == 12345
    assert call_kwargs["parse_mode"] == "HTML"


@pytest.mark.asyncio
async def test_send_long_message_splits_into_multiple():
    channel = _make_channel()
    # Build a message well over the markdown split limit
    paragraphs = [f"Paragraph {i}: " + "x" * 500 for i in range(10)]
    content = "\n\n".join(paragraphs)
    assert len(content) > _MARKDOWN_SPLIT_LIMIT

    msg = OutboundMessage(channel="telegram", chat_id="12345", content=content)
    await channel.send(msg)

    # Should have been called multiple times
    call_count = channel._app.bot.send_message.call_count
    assert call_count > 1
    # All calls should use HTML parse mode
    for call in channel._app.bot.send_message.call_args_list:
        assert call.kwargs["chat_id"] == 12345
        assert call.kwargs["parse_mode"] == "HTML"


@pytest.mark.asyncio
async def test_send_falls_back_to_plain_text_on_html_error():
    channel = _make_channel()

    # First call raises (HTML parse failure), second succeeds (plain text fallback)
    channel._app.bot.send_message = AsyncMock(
        side_effect=[Exception("Bad HTML"), None]
    )

    msg = OutboundMessage(channel="telegram", chat_id="12345", content="Hello **world**")
    await channel.send(msg)

    assert channel._app.bot.send_message.call_count == 2
    # Second call should be plain text (no parse_mode)
    fallback_kwargs = channel._app.bot.send_message.call_args_list[1].kwargs
    assert fallback_kwargs.get("parse_mode") is None


@pytest.mark.asyncio
async def test_send_invalid_chat_id_logs_error():
    channel = _make_channel()
    msg = OutboundMessage(channel="telegram", chat_id="not-a-number", content="Hi")
    await channel.send(msg)
    # Should not attempt to send
    channel._app.bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_send_no_app_returns_early():
    channel = _make_channel()
    channel._app = None
    msg = OutboundMessage(channel="telegram", chat_id="12345", content="Hi")
    await channel.send(msg)
    # Nothing to assert beyond no crash
