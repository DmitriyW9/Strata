"""serve/test_web_badges.py - the Recent requests table has a badge for every way a request can end (#3991).

A request that waits past the queue limit ends with finish "timeout"; the Status cell showed the raw word. Reads the page source.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parent / "web" / "app.js"
HTML = Path(__file__).resolve().parent / "web" / "index.html"
SERVER = Path(__file__).resolve().parent / "server.py"


class FinishBadges(unittest.TestCase):
    def test_a_timed_out_request_has_a_badge(self):
        src = APP.read_text(encoding="utf-8")
        table = src[src.index("const badge = {stop:"):]
        table = table[:table.index("};")]
        self.assertRegex(table, r'timeout: \["st-badge--queued", "Timeout"\]')

    def test_every_finish_the_server_stores_has_a_badge(self):
        src = APP.read_text(encoding="utf-8")
        table = src[src.index("const badge = {stop:"):]
        table = table[:table.index("};")]
        for finish in ("stop", "length", "cancel", "disconnect", "error", "timeout"):
            self.assertRegex(table, rf"\b{finish}:", finish)
        self.assertIn('"timeout"', SERVER.read_text(encoding="utf-8"))

    def test_recent_requests_shows_an_escaped_text_preview_as_the_third_column(self):
        html = HTML.read_text(encoding="utf-8")
        app = APP.read_text(encoding="utf-8")
        self.assertRegex(html, r"<th>Time</th><th>Status</th><th>Request</th>")
        self.assertIn("esc(promptPreview)", app)
        self.assertIn('r.prompt_preview_truncated ? "…" : ""', app)
        self.assertIn('colspan="10"', html)


if __name__ == "__main__":
    unittest.main()
