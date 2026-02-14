"""LspManager: manages LSP server lifecycle and provides diagnostics API."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.lsp.client import AsyncLspClient

if TYPE_CHECKING:
    from nanobot.config.schema import LspConfig, LspServerConfig


class LspManager:
    """Manages LSP server lifecycle and provides a high-level API for diagnostics."""

    def __init__(self, workspace: str, config: LspConfig):
        self._workspace = workspace
        self._config = config
        self._clients: dict[str, AsyncLspClient] = {}  # language_id -> client
        self._ext_map: dict[str, str] = {}  # ".py" -> "python"
        self._server_map: dict[str, LspServerConfig] = {}  # language_id -> config
        self._lock = asyncio.Lock()
        self._build_maps()

    # -- Public API (used by file tools) ---------------------------------------

    async def notify_and_get_diagnostics(
        self, file_path: str, content: str, is_change: bool
    ) -> list[dict] | None:
        """Notify open/change and wait for diagnostics. Returns None if no LSP available."""
        language_id = self._language_for_file(file_path)
        if language_id is None:
            return None

        client = await self._get_or_start_client(language_id)
        if client is None:
            return None

        uri = Path(file_path).resolve().as_uri()

        try:
            if client.is_open(uri):
                if is_change:
                    await client.did_change(uri, content)
            else:
                await client.did_open(uri, language_id, content)

            # Wait for server to process and publish diagnostics
            delay = self._server_map[language_id].diagnostic_delay
            await asyncio.sleep(delay)

            diags = client.get_diagnostics(uri)
            return diags if diags else None
        except Exception as e:
            logger.debug(f"LSP diagnostics error for {file_path}: {e}")
            return None

    # -- Lifecycle -------------------------------------------------------------

    async def shutdown(self) -> None:
        """Shut down all running language servers."""
        for lang_id, client in self._clients.items():
            try:
                await client.shutdown()
            except Exception as e:
                logger.debug(f"Error shutting down LSP client for {lang_id}: {e}")
        self._clients.clear()

    # -- Internal --------------------------------------------------------------

    async def _get_or_start_client(self, language_id: str) -> AsyncLspClient | None:
        """Lazy-start a language server. Returns None if command not found."""
        if language_id in self._clients:
            return self._clients[language_id]

        async with self._lock:
            # Double-check after acquiring lock
            if language_id in self._clients:
                return self._clients[language_id]

            server_config = self._server_map.get(language_id)
            if server_config is None:
                return None

            # Verify the server command exists
            if not shutil.which(server_config.command):
                logger.debug(
                    f"LSP server command not found: {server_config.command} "
                    f"(language: {language_id})"
                )
                return None

            env = server_config.env if server_config.env else None
            client = AsyncLspClient(
                command=server_config.command,
                args=server_config.args,
                env=env,
            )

            try:
                await client.start(
                    self._workspace,
                    initialization_options=server_config.initialization_options or None,
                )
                self._clients[language_id] = client
                return client
            except Exception as e:
                logger.debug(f"Failed to start LSP server for {language_id}: {e}")
                return None

    def _language_for_file(self, file_path: str) -> str | None:
        """Map file extension to language ID."""
        ext = Path(file_path).suffix.lower()
        return self._ext_map.get(ext)

    def _build_maps(self) -> None:
        """Build ext_map and server_map from config."""
        for server in self._config.servers:
            self._server_map[server.language_id] = server
            for ext in server.extensions:
                self._ext_map[ext.lower()] = server.language_id
