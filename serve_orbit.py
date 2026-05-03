"""Hot-reload dev server for orbit.html.

Watches orbit.html for mtime changes and pushes a reload event to any
connected browsers via Server-Sent Events. Zero third-party deps.

Usage:
    python serve_orbit.py            # http://localhost:5500
    python serve_orbit.py --port 8000 --file orbit.html
"""

from __future__ import annotations

import argparse
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RELOAD_SNIPPET = b"""
<script>
(function() {
  let backoff = 500;
  function connect() {
    const es = new EventSource("/__events");
    es.addEventListener("open",   () => { backoff = 500; });
    es.addEventListener("reload", () => { console.log("[hot-reload] reloading..."); location.reload(); });
    es.addEventListener("error",  () => {
      es.close();
      setTimeout(connect, backoff);
      backoff = Math.min(backoff * 2, 5000);
    });
  }
  connect();
})();
</script>
"""

_subscribers: "list[queue.Queue[str]]" = []
_subscribers_lock = threading.Lock()


def _broadcast(event: str) -> None:
    with _subscribers_lock:
        targets = list(_subscribers)
    for q in targets:
        try:
            q.put_nowait(event)
        except queue.Full:
            pass


def _watch_file(path: Path, poll_interval: float = 0.25) -> None:
    last_mtime: float | None = None
    last_size: int | None = None
    while True:
        try:
            st = path.stat()
            mtime, size = st.st_mtime, st.st_size
            if last_mtime is None:
                last_mtime, last_size = mtime, size
            elif (mtime, size) != (last_mtime, last_size):
                # Debounce: wait until file stops changing for ~150ms.
                stable_for = 0.0
                while stable_for < 0.15:
                    time.sleep(0.05)
                    try:
                        st2 = path.stat()
                    except FileNotFoundError:
                        st2 = None
                        break
                    if st2 and (st2.st_mtime, st2.st_size) == (mtime, size):
                        stable_for += 0.05
                    elif st2:
                        mtime, size = st2.st_mtime, st2.st_size
                        stable_for = 0.0
                last_mtime, last_size = mtime, size
                print(f"[watch] change detected -> reloading clients")
                _broadcast("reload")
        except FileNotFoundError:
            pass
        time.sleep(poll_interval)


def _make_handler(html_path: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quieter logs
            print(f"[http] {self.address_string()} - {fmt % args}")

        def _send_bytes(self, status: int, content_type: str, body: bytes, extra_headers: dict | None = None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra_headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html", "/orbit.html"):
                self._serve_html()
            elif path == "/__events":
                self._serve_events()
            elif path == "/__health":
                self._send_bytes(200, "text/plain; charset=utf-8", b"ok")
            else:
                self._send_bytes(404, "text/plain; charset=utf-8", b"not found")

        def _serve_html(self):
            try:
                data = html_path.read_bytes()
            except FileNotFoundError:
                msg = f"<h1>{html_path} not found yet</h1><p>Generate it from your notebook; this page will reload.</p>".encode()
                body = msg + RELOAD_SNIPPET
                self._send_bytes(200, "text/html; charset=utf-8", body)
                return
            # Inject reload snippet just before </body> (fallback: append).
            lower = data.lower()
            idx = lower.rfind(b"</body>")
            if idx == -1:
                body = data + RELOAD_SNIPPET
            else:
                body = data[:idx] + RELOAD_SNIPPET + data[idx:]
            self._send_bytes(200, "text/html; charset=utf-8", body)

        def _serve_events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            q: queue.Queue[str] = queue.Queue(maxsize=16)
            with _subscribers_lock:
                _subscribers.append(q)
            try:
                self.wfile.write(b": connected\n\n")
                self.wfile.flush()
                last_ping = time.time()
                while True:
                    try:
                        event = q.get(timeout=15)
                        payload = f"event: {event}\ndata: {int(time.time()*1000)}\n\n".encode()
                        self.wfile.write(payload)
                        self.wfile.flush()
                    except queue.Empty:
                        pass
                    if time.time() - last_ping > 15:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        last_ping = time.time()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with _subscribers_lock:
                    if q in _subscribers:
                        _subscribers.remove(q)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Hot-reload server for orbit.html")
    parser.add_argument("--file", default="orbit.html", help="HTML file to serve & watch")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5500)
    args = parser.parse_args()

    html_path = Path(args.file).resolve()
    print(f"[serve_orbit] watching {html_path}")
    print(f"[serve_orbit] open http://{args.host}:{args.port}/  (Ctrl+C to stop)")

    watcher = threading.Thread(target=_watch_file, args=(html_path,), daemon=True)
    watcher.start()

    server = ThreadingHTTPServer((args.host, args.port), _make_handler(html_path))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve_orbit] shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
