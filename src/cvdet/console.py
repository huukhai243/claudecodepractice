"""Buộc stdout/stderr dùng UTF-8.

Console Windows mặc định là codepage cp1252, không mã hoá được dấu tiếng Việt —
mọi `print` có dấu sẽ ném UnicodeEncodeError. Gọi `setup_console()` ở đầu mỗi
entrypoint để tránh việc đó.
"""

from __future__ import annotations

import sys


def setup_console() -> None:
    """Chuyển stdout/stderr sang UTF-8. An toàn khi gọi nhiều lần."""
    for stream in (sys.stdout, sys.stderr):
        # Stream bị redirect (pytest capture, pipe) có thể không có reconfigure.
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            # errors="replace": thà mất một ký tự hơn là làm sập cả script.
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass
