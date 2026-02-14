"""Low-level JSON-RPC 2.0 client over stdio for LSP communication."""

import asyncio
import json
from typing import Any

from loguru import logger


class AsyncLspClient:
    """Manages a single language server process over stdio."""

    def __init__(self, command: str, args: list[str], env: dict[str, str] | None = None):
        self._command = command
        self._args = args
        self._env = env
        self._process: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._diagnostics: dict[str, list[dict]] = {}  # uri -> diagnostics
        self._reader_task: asyncio.Task | None = None
        self._version_counter: dict[str, int] = {}  # uri -> version
        self._open_docs: set[str] = set()  # tracked open URIs

    # -- Lifecycle -------------------------------------------------------------

    async def start(self, workspace_root: str, initialization_options: dict | None = None) -> None:
        """Spawn the language server process and perform the initialize handshake."""
        self._process = await asyncio.create_subprocess_exec(
            self._command,
            *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._env,
        )
        self._reader_task = asyncio.create_task(self._read_loop())

        uri = f"file://{workspace_root}"
        params: dict[str, Any] = {
            "processId": None,
            "capabilities": {
                "textDocument": {
                    "publishDiagnostics": {
                        "relatedInformation": False,
                    },
                    "synchronization": {
                        "didSave": True,
                        "dynamicRegistration": False,
                    },
                },
            },
            "rootUri": uri,
            "workspaceFolders": [{"uri": uri, "name": "workspace"}],
        }
        if initialization_options:
            params["initializationOptions"] = initialization_options

        await self._send_request("initialize", params)
        await self._send_notification("initialized", {})
        logger.debug(f"LSP server started: {self._command}")

    async def shutdown(self) -> None:
        """Send shutdown + exit, then terminate the process."""
        if self._process is None:
            return
        try:
            await asyncio.wait_for(self._send_request("shutdown", {}), timeout=5.0)
            await self._send_notification("exit", {})
        except Exception:
            pass
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._process and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                self._process.kill()
        self._process = None
        logger.debug(f"LSP server stopped: {self._command}")

    # -- Document sync ---------------------------------------------------------

    async def did_open(self, uri: str, language_id: str, text: str) -> None:
        """Notify the server that a document was opened."""
        self._version_counter[uri] = 1
        self._open_docs.add(uri)
        await self._send_notification(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": language_id,
                    "version": 1,
                    "text": text,
                }
            },
        )

    async def did_change(self, uri: str, text: str) -> None:
        """Notify the server that a document changed (full sync)."""
        version = self._version_counter.get(uri, 0) + 1
        self._version_counter[uri] = version
        await self._send_notification(
            "textDocument/didChange",
            {
                "textDocument": {"uri": uri, "version": version},
                "contentChanges": [{"text": text}],
            },
        )

    async def did_close(self, uri: str) -> None:
        """Notify the server that a document was closed."""
        self._open_docs.discard(uri)
        await self._send_notification(
            "textDocument/didClose",
            {
                "textDocument": {"uri": uri},
            },
        )

    # -- Diagnostics -----------------------------------------------------------

    def get_diagnostics(self, uri: str) -> list[dict]:
        """Return cached diagnostics for a URI."""
        return self._diagnostics.get(uri, [])

    def is_open(self, uri: str) -> bool:
        """Check whether a document URI is currently open."""
        return uri in self._open_docs

    # -- JSON-RPC internals ----------------------------------------------------

    async def _send_request(self, method: str, params: dict) -> Any:
        """Send a JSON-RPC request and await the response."""
        self._request_id += 1
        rid = self._request_id
        msg = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[rid] = future

        self._write_message(msg)
        return await future

    async def _send_notification(self, method: str, params: dict) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        self._write_message(msg)

    def _write_message(self, msg: dict) -> None:
        """Write one Content-Length framed JSON-RPC message to stdin."""
        if not self._process or not self._process.stdin:
            return
        body = json.dumps(msg).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        self._process.stdin.write(header + body)

    async def _read_loop(self) -> None:
        """Background task: read JSON-RPC messages, dispatch responses, cache diagnostics."""
        try:
            while self._process and self._process.returncode is None:
                try:
                    msg = await self._read_message()
                except (asyncio.IncompleteReadError, ConnectionError):
                    break

                if msg is None:
                    break

                # Response to a request
                if "id" in msg and ("result" in msg or "error" in msg):
                    rid = msg["id"]
                    future = self._pending.pop(rid, None)
                    if future and not future.done():
                        if "error" in msg:
                            future.set_exception(RuntimeError(f"LSP error: {msg['error']}"))
                        else:
                            future.set_result(msg.get("result"))

                # Notification from server
                elif "method" in msg:
                    if msg["method"] == "textDocument/publishDiagnostics":
                        params = msg.get("params", {})
                        uri = params.get("uri", "")
                        self._diagnostics[uri] = params.get("diagnostics", [])
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"LSP read loop error: {e}")

    async def _read_message(self) -> dict | None:
        """Read one Content-Length framed JSON-RPC message from stdout."""
        if not self._process or not self._process.stdout:
            return None

        headers: dict[str, str] = {}
        while True:
            line = await self._process.stdout.readline()
            if not line:
                return None
            decoded = line.decode("ascii", errors="replace").strip()
            if not decoded:
                break  # empty line separates headers from body
            if ":" in decoded:
                key, value = decoded.split(":", 1)
                headers[key.strip()] = value.strip()

        length = int(headers.get("Content-Length", 0))
        if length <= 0:
            return None

        body = await self._process.stdout.readexactly(length)
        return json.loads(body)
