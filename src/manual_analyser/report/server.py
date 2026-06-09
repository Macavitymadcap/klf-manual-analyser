"""
report/server.py — Local HTTP server for the HTML report.

Serves:
    /           → latest report index.html from data/reports/
    /track_*    → track detail pages from same report directory
    /stems/*    → WAV files from data/stems/

Usage:
    from manual_analyser.report.server import serve
    serve(data_dir=Path("data"), port=8000)
"""

import logging
import mimetypes
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

logger = logging.getLogger(__name__)


def serve(data_dir: Path, port: int = 8000) -> None:
    """Start the local report server. Blocks until interrupted."""
    reports_dir = data_dir / "reports"
    stems_dir = data_dir / "stems"
    latest_dir = _find_latest_report(reports_dir)

    if latest_dir is None:
        logger.error("[server] No rendered reports found in %s — run 'report' first", reports_dir)
        return

    logger.info("[server] Serving report from %s", latest_dir)
    logger.info("[server] Stems from %s", stems_dir)
    logger.info("[server] Listening on http://localhost:%d", port)

    handler = _make_handler(latest_dir, stems_dir)
    httpd = HTTPServer(("localhost", port), handler)

    print(f"Report server running at http://localhost:{port}")
    print("Press Ctrl+C to stop.")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        httpd.server_close()


def _find_latest_report(reports_dir: Path) -> Path | None:
    """Return the most recently modified report directory, or None."""
    if not reports_dir.exists():
        return None
    dirs = [d for d in reports_dir.iterdir() if d.is_dir() and (d / "index.html").exists()]
    return max(dirs, key=lambda d: d.stat().st_mtime, default=None)


def _make_handler(report_dir: Path, stems_dir: Path):
    """Return a handler class bound to the given directories."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urllib.parse.urlparse(self.path).path
            if path == "/" or path == "/index.html":
                self._serve_file(report_dir / "index.html")
            elif path.startswith("/track_") and path.endswith(".html"):
                self._serve_file(report_dir / path.lstrip("/"))
            elif path.startswith("/stems/"):
                rel = path[len("/stems/") :]
                self._serve_file(stems_dir / rel)
            else:
                self._not_found()

        def _serve_file(self, file_path: Path) -> None:
            if not file_path.exists():
                self._not_found()
                return
            mime, _ = mimetypes.guess_type(str(file_path))
            mime = mime or "application/octet-stream"
            data = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _not_found(self) -> None:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")

        def log_message(self, fmt, *args):
            logger.debug("[server] " + fmt, *args)

    return Handler
