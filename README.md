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

## Episode data cache

`download_episodes.py` reads `manifest.csv` and downloads daily Kaggle episode datasets into `episode_data/<YYYY-MM-DD>/`. Each day is a folder of replay JSON files (`{episode_id}.json`) plus that day's `manifest.csv`. Cached days are skipped unless you pass `--force`.

Requires the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) and API credentials (see `orbit-wars-data/agents.md`):

```bash
pip install kaggle
mkdir -p ~/.kaggle
# paste token from https://www.kaggle.com/settings/api
chmod 600 ~/.kaggle/access_token
```

### Run

```bash
source .venv/bin/activate
python download_episodes.py                         # all days in manifest.csv (~1 TB)
python download_episodes.py --date 2026-04-16         # one day
python download_episodes.py --from 2026-06-01 --to 2026-06-21
python download_episodes.py --dry-run                 # show planned downloads
python download_episodes.py --date 2026-04-16 --force   # redownload even if cached
python download_episodes.py --output /data/episodes   # custom cache root
```

## Winner-game dataset

`build_dataset.py` builds a compact training dataset of *winning* games out of the Parquet DB. It (1) quality-filters games, (2) computes per-player win-rate **within each player count** (2p baseline ~50%, 4p ~25%), (3) keeps players above a per-group win-rate / min-games bar, (4) keeps the games those players *won*, and (5) extracts the heavy tables filtered to just those episodes. Everything is config-driven by `dataset_config.yaml`, and the resolved config is written into the output dir so each dataset is self-describing/reproducible.

Prereq: a Parquet DB at `parquet_dir` (default `parquet_out/`), produced by `generate_parquet_db.py`. `fleet_state` additionally needs the raw replay JSONs in `data_root` (see *Episode data cache*) — missing JSONs are skipped/counted and backfilled on re-run.

### Run

```bash
source .venv/bin/activate
python build_dataset.py                         # uses dataset_config.yaml
python build_dataset.py --config other.yaml
python build_dataset.py --dry-run               # print selection stats only; write nothing
python build_dataset.py --group 2:0.6:50 --group 4:0.35:25   # replace per-count bars (N:winrate:min_games)
python build_dataset.py --max-silent 50         # override max_silent_ticks
python build_dataset.py --out-dir other_out     # override out_dir
python build_dataset.py --no-planet-state       # skip heavy planet_state (~670MB)
python build_dataset.py --no-fleet-state        # skip fleet_state JSON parse
python build_dataset.py --workers 8             # parallel workers for fleet_state parse
```

`--dry-run` is the fast way to tune thresholds: it prints per-group player/game counts and the top selected players without touching disk.

### Output (`out_dir`, default `dataset_out/`)

| File | Contents |
| ---- | -------- |
| `selected_games.parquet` | one row per won game — the spine; carries `winner_slot` + episode metadata (join key: `episode_id`). |
| `selected_players.parquet` | players that cleared the bar (`name, games, wins, win_rate, n_players`). |
| `player_stats.parquet` | win-rate stats for *all* players per group (pre-threshold). |
| `games/actions.parquet` | winner+opponent fleet commands (the action targets). |
| `games/tick_summary.parquet` | per-tick per-slot aggregates (cheap state context). |
| `games/episode_planets.parquet` | initial board layout per game. |
| `games/planet_state.parquet` | per-planet per-tick `(owner, ships)` trajectory (heavy; opt-in). |
| `games/fleet_state.parquet` | per-fleet per-tick `(x, y, angle, from_planet_id, ships)`, parsed from raw replays (opt-in). |
| `resolved_config.yaml` | the exact config used + selected player/game counts. |

Tune `dataset_config.yaml` for the filter (`end_reasons`, `max_silent_ticks`, `require_no_froze`), per-count selection (`groups`), `chunk_size_n` (recorded for the downstream encoder, not applied here), and which heavy tables to extract (`include_*`).

## Diffusion policy (lean v1)

Winner-conditioned action-chunk diffusion. Spec: [orbit_wars_diffusion_lean_spec.md](orbit_wars_diffusion_lean_spec.md) (full fleet design: [orbit_wars_diffusion_dataset_spec.md](orbit_wars_diffusion_dataset_spec.md)).

Prereq: `dataset_out/` from `build_dataset.py` with `planet_state` and `actions` enabled.

### Run

```bash
source .venv/bin/activate
python train_orbit.py build-cache                    # parquet -> cache/episodes/*.pt
python train_orbit.py train --config orbit_train.yaml
python train_orbit.py eval --checkpoint checkpoints/latest.pt
python train_orbit.py sample --checkpoint checkpoints/latest.pt
```

`EpisodeLRU` in `train_orbit.py` loads/unloads episode `.pt` caches; call `.clear()` to drop them from memory.

Checkpoints under `checkpoints/` store encoder + denoiser + DDPM schedule + configs in one file (`load_checkpoint`).

## Pack a submission

`pack.py` bundles `ddpm_act.py` + its import closure + a trained checkpoint into a Kaggle-ready `submission.tar.gz`. It generates a top-level `main.py` whose last `def` (`agent`) is the Kaggle entrypoint, forcing CPU since the eval image has no GPU. Torch/pandas/yaml are preinstalled on the Kaggle image, so no requirements/wheels are vendored.

Prereq: a trained checkpoint under `checkpoints/` (default `checkpoints/step_1800500.pt`).

### Run

```bash
source .venv/bin/activate
python pack.py                                              # -> submission.tar.gz
python pack.py --smoke                                      # build, then run one synthetic turn in a fresh subprocess
python pack.py --checkpoint checkpoints/latest.pt --out my_sub.tar.gz
python pack.py --keep-build                                 # keep build/submission/ staging dir
```

`--smoke` imports the packed bundle with cwd = bundle dir (mimicking Kaggle's flat layout) and runs one synthetic turn, catching missing-file / import-path / `torch.load` failures before a submission is spent. After building, the command prints the `kaggle competitions submit` line to copy.

## Files

- `getting-started.ipynb` — tutorial walking through observations, agent design, and submission.
- `main.py` — `nearest_planet_sniper` example bot (this is what gets submitted to Kaggle).
- `run_harness.py` — runs an `orbit_wars` match using agents from `main.py` and writes `orbit.html`.
- `orbit.html` — generated replay viewer (gitignore-worthy; large).
- `serve_orbit.py` — hot-reload dev server for `orbit.html`.
- `download_episodes.py` — download and cache daily episode datasets from `manifest.csv`.
- `manifest.csv` — index of daily Kaggle episode datasets.
- `generate_parquet_db.py` — parse cached replay JSONs into the Parquet DB (`parquet_out/`).
- `build_dataset.py` — select winning games + extract heavy tables into a training dataset (`dataset_out/`).
- `dataset_config.yaml` — config consumed by `build_dataset.py`.
- `dataset_schema.json` — JSON Schema for the winner-game dataset layout.
- `orbit_wars_diffusion_lean_spec.md` — lean v1 diffusion policy spec (planets-only obs).
- `orbit_wars_diffusion_dataset_spec.md` — full diffusion spec (fleet physics, reference).
- `train_orbit.py` — cache build, data load/unload, train/val/sample, checkpoints.
- `orbit_train.yaml` — optional training config for `train_orbit.py`.
- `ddpm_act.py` — Kaggle-side inference adapter; `make_ddpm_agent` loads a checkpoint into an `agent(obs)` callable.
- `pack.py` — bundle `ddpm_act.py` + closure + checkpoint into `submission.tar.gz` (`--smoke` to verify).
- `LLMLOG.MD` — log of agent-driven changes.
