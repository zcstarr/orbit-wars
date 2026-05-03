# orbit_wars

Workspace for experimenting with the Kaggle [`orbit_wars`](https://www.kaggle.com/competitions) environment. `run_harness.py` runs games using agents from `main.py` and writes a self-contained replay viewer to `orbit.html`; `serve_orbit.py` serves it with browser hot-reload.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python --upgrade "kaggle-environments>=1.28.0"
```

## Hot-reloaded viewer

`serve_orbit.py` serves `orbit.html` and auto-refreshes any open browser tab whenever the file changes. Pure stdlib, no extra deps.

### Run

```bash
source .venv/bin/activate
python serve_orbit.py                       # http://127.0.0.1:5500
python serve_orbit.py --port 8000           # custom port
python serve_orbit.py --file other.html     # watch a different file
python serve_orbit.py --host 0.0.0.0        # expose on LAN
```

Then open `http://127.0.0.1:5500/` in a browser tab next to Cursor. In another terminal, run the harness whenever you want a fresh replay:

```bash
python run_harness.py                    # writes ./orbit.html, tab auto-reloads
python run_harness.py --out other.html   # write somewhere else
```

To change the matchup, edit the `AGENTS` list at the top of `run_harness.py` (e.g. swap `"random"` for `nearest_planet_sniper`, or pass 4 entries for a 4-player game).

### Endpoints

| Path           | Purpose                                                              |
| -------------- | -------------------------------------------------------------------- |
| `/`            | Serves `orbit.html` with a tiny SSE reload snippet injected.         |
| `/orbit.html`  | Same as `/`.                                                         |
| `/__events`    | Server-Sent Events stream emitting `reload` on file change.          |
| `/__health`    | Returns `ok`.                                                        |

### How it works

- A daemon thread polls `orbit.html`'s mtime+size every 250 ms; on change it debounces 150 ms (so a partial Jupyter write doesn't trigger a half-rendered reload), then broadcasts a `reload` event to every connected SSE subscriber.
- The injected `<script>` opens `EventSource("/__events")`, calls `location.reload()` on `reload`, and reconnects with exponential backoff (500 ms → 5 s) if the server restarts.

### Troubleshooting

- **Port in use** — `python serve_orbit.py --port 5501`.
- **Page doesn't reload** — open DevTools console; you should see `[hot-reload] reloading...` on each change. If you see SSE 404s, you're hitting a different server.
- **`orbit.html not found yet`** — placeholder shown when the file doesn't exist; it'll flip to the real renderer the moment the notebook cell finishes writing it.

## Files

- `getting-started.ipynb` — tutorial walking through observations, agent design, and submission.
- `main.py` — `nearest_planet_sniper` example bot (this is what gets submitted to Kaggle).
- `run_harness.py` — runs an `orbit_wars` match using agents from `main.py` and writes `orbit.html`.
- `orbit.html` — generated replay viewer (gitignore-worthy; large).
- `serve_orbit.py` — hot-reload dev server for `orbit.html`.
- `LLMLOG.MD` — log of agent-driven changes.
