"""Minimal keep-alive HTTP/1.1 client for the timed query path.

httpx adds about a millisecond per request and, for responses of a few
kilobytes on Linux loopback, a 40 ms delayed-ACK stall that would dominate the
client wall-clock metric. One persistent connection, one write per request,
and a Content-Length read give a client figure that reflects the server and
the loopback path only. Collection management and uploads keep using httpx.
"""

from __future__ import annotations

import asyncio
import os
import socket
from typing import Any
from urllib.parse import urlsplit

import orjson


class QueryConnection:
    def __init__(self, base_url: str, api_key: str | None, timeout: float = 120.0):
        parts = urlsplit(base_url)
        if parts.scheme != "http" or not parts.hostname:
            raise ValueError(
                "The timed query path needs a plain http:// endpoint on the VM; "
                f"got {base_url!r}"
            )
        self.host = parts.hostname
        self.port = parts.port or 80
        self.timeout = timeout
        self._headers = (
            f"Host: {self.host}:{self.port}\r\n"
            "Content-Type: application/json\r\n"
            "Connection: keep-alive\r\n"
            + (f"api-key: {api_key}\r\n" if api_key else "")
        ).encode()
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    async def _connect(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        if self._reader is None or self._writer is None or self._writer.is_closing():
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port), self.timeout
            )
            sock = writer.get_extra_info("socket")
            if sock is not None:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._reader, self._writer = reader, writer
        return self._reader, self._writer

    async def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        payload = orjson.dumps(body, option=orjson.OPT_SERIALIZE_NUMPY)
        request = (
            f"POST {path} HTTP/1.1\r\n".encode()
            + self._headers
            + f"Content-Length: {len(payload)}\r\n\r\n".encode()
            + payload
        )
        for attempt in range(2):
            reader, writer = await self._connect()
            try:
                writer.write(request)
                await writer.drain()
                head = await asyncio.wait_for(
                    reader.readuntil(b"\r\n\r\n"), self.timeout
                )
                break
            except (asyncio.IncompleteReadError, ConnectionError, OSError):
                await self.close()
                if attempt:
                    raise
        status_line, *header_lines = head.decode("latin-1").split("\r\n")
        status = int(status_line.split(" ", 2)[1])
        headers = {
            key.strip().lower(): value.strip()
            for key, _, value in (line.partition(":") for line in header_lines if line)
        }
        if "content-length" not in headers:
            await self.close()
            raise RuntimeError(
                "Query response without Content-Length; transport unsupported"
            )
        content = await asyncio.wait_for(
            reader.readexactly(int(headers["content-length"])), self.timeout
        )
        if headers.get("connection", "").lower() == "close":
            await self.close()
        if status >= 400:
            raise RuntimeError(f"Qdrant POST {path}: HTTP {status}: {content[:1000]!r}")
        return orjson.loads(content)

    async def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except OSError:
                pass
        self._reader = self._writer = None


def query_connection() -> QueryConnection:
    return QueryConnection(
        os.environ.get("QDRANT_URL", "http://127.0.0.1:6333"),
        os.environ.get("QDRANT_API_KEY"),
    )
