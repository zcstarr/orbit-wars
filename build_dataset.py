#!/usr/bin/env python3
"""Build a winner-game dataset from the Orbit Wars Parquet DB.

Pipeline (driven by dataset_config.yaml):

  1. Quality-filter games (clean end_reason + no freeze / long-silence), exactly
     like the notebook's win-rate cell.
  2. Compute per-player win-rate on the filtered games (within a fixed
     n_players, since 4p games dilute win-rate toward 25%).
  3. Select players with win_rate >= threshold AND games >= min_games.
  4. For those players, keep the games they *won* -> the selected episodes.
  5. Extract everything associated with those episodes (actions, tick_summary,
     episode_planets, planet_state, and fleet_state) into a compact,
     fast-to-load dataset dir. Encoding into tensors / action-chunks is left to
     downstream code; `chunk_size_n` is recorded in resolved_config.yaml for it.

planet_state/actions/etc. are filtered out of the parquet DB; fleet_state has no
parquet table (only counts in tick_summary), so it's parsed from the raw replay
JSONs under `data_root` for just the selected games.

The heavy step (4) is a *selection*, not a tensorization: we emit Parquet so the
encoder can lazily read just the games it needs. `winner_slot` is stored per game
so the encoder can orient everything winner-centric.

Usage:
    python build_dataset.py                         # uses dataset_config.yaml
    python build_dataset.py --config other.yaml
    python build_dataset.py --group 2:0.6:50 --group 4:0.35:25  # per-count bars (N:WR:MG)
    python build_dataset.py --max-silent 50         # override max_silent_ticks
    python build_dataset.py --out-dir other_out     # override out_dir
    python build_dataset.py --no-planet-state       # skip planet_state (heavy)
    python build_dataset.py --no-fleet-state        # skip fleet_state JSON parse
    python build_dataset.py --workers 8             # workers for fleet_state parse
    python build_dataset.py --dry-run               # selection stats only, no writes
"""

import argparse
import dataclasses
import json
import os
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from generate_parquet_db import find_json_files, get_episode_id

REPO_ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "dataset_config.yaml"

# Heavy source tables -> whether they carry an episode_id column we filter on.
HEAVY_TABLES = ("actions", "tick_summary", "episode_planets", "planet_state")

# fleet_state is NOT in the parquet DB (only fleet COUNTS live in tick_summary);
# positions are parsed on demand from the raw replay JSONs for selected games.
# Fleet row layout (kaggle_environments orbit_wars Fleet):
#   [id, owner, x, y, angle, from_planet_id, ships]
FLEET_STATE_COLS = ("episode_id", "tick", "fleet_id", "owner",
                    "x", "y", "angle", "from_planet_id", "ships")
FLEET_STATE_SCHEMA = pa.schema([
    ("episode_id", pa.int32()), ("tick", pa.int16()), ("fleet_id", pa.int16()),
    ("owner", pa.int8()), ("x", pa.float32()), ("y", pa.float32()),
    ("angle", pa.float32()), ("from_planet_id", pa.int16()), ("ships", pa.int32()),
])


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class GroupSpec:
    """Per-player-count selection bar (win-rate baselines differ by count)."""
    win_rate_threshold: float
    min_games: int


@dataclasses.dataclass
class Config:
    parquet_dir: pathlib.Path
    out_dir: pathlib.Path
    data_root: pathlib.Path
    end_reasons: list
    max_silent_ticks: int
    require_no_froze: bool
    groups: dict  # {n_players(int): GroupSpec}
    chunk_size_n: int
    include_actions: bool
    include_tick_summary: bool
    include_episode_planets: bool
    include_planet_state: bool
    include_fleet_state: bool

    @classmethod
    def load(cls, path: pathlib.Path) -> "Config":
        raw = yaml.safe_load(path.read_text())
        raw["parquet_dir"] = (REPO_ROOT / raw["parquet_dir"]).resolve()
        raw["out_dir"] = (REPO_ROOT / raw["out_dir"]).resolve()
        raw["data_root"] = (REPO_ROOT / raw["data_root"]).resolve()
        raw["groups"] = {
            int(n): GroupSpec(**spec) for n, spec in raw["groups"].items()
        }
        return cls(**raw)

    def included_tables(self) -> list:
        flags = {
            "actions": self.include_actions,
            "tick_summary": self.include_tick_summary,
            "episode_planets": self.include_episode_planets,
            "planet_state": self.include_planet_state,
        }
        return [t for t in HEAVY_TABLES if flags[t]]

    def to_yaml_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["parquet_dir"] = str(self.parquet_dir)
        d["out_dir"] = str(self.out_dir)
        d["data_root"] = str(self.data_root)
        d["groups"] = {n: dataclasses.asdict(s) for n, s in self.groups.items()}
        return d


# --------------------------------------------------------------------------- #
# Selection (pure; no IO) -- this is the testable core.
# --------------------------------------------------------------------------- #
def filter_clean_games(players: pd.DataFrame, episodes: pd.DataFrame,
                       cfg: Config) -> pd.DataFrame:
    """Return the player rows belonging to cleanly-decided, freeze-free games.

    Mirrors the notebook: merge end_reason, keep wanted reasons, then drop any
    episode where an opponent went silent too long or any participant froze.
    Keeps all player counts (per-count selection happens in `select`).
    """
    pl = players.merge(
        episodes[["episode_id", "end_reason", "n_players"]],
        on="episode_id", how="left")
    pl = pl[pl["end_reason"].isin(cfg.end_reasons)]

    ep_max_silent = pl.groupby("episode_id")["opponent_silent_ticks"].transform("max")
    clean = ep_max_silent <= cfg.max_silent_ticks
    if cfg.require_no_froze:
        ep_any_froze = pl.groupby("episode_id")["opponent_froze"].transform("max")
        clean = clean & (ep_any_froze == 0)
    return pl[clean].copy()


def player_win_stats(clean_players: pd.DataFrame) -> pd.DataFrame:
    """games / wins / win_rate per player name over the given game set."""
    stats = clean_players.groupby("name").agg(
        games=("is_winner", "count"), wins=("is_winner", "sum")
    ).reset_index()
    stats["win_rate"] = stats["wins"] / stats["games"]
    return stats.sort_values("win_rate", ascending=False)


def select(players: pd.DataFrame, episodes: pd.DataFrame, cfg: Config):
    """Run the full per-group selection. Returns (stats, sel_players, sel_games).

    Win-rate is computed WITHIN each player count (group), each group applies
    its own threshold/min_games, and the results are unioned. `stats` and
    `selected_players` carry an `n_players` column so a name that qualifies in
    both 2p and 4p appears once per group. `selected_games` has one row per won
    game joined with episode metadata + the winner's per-player row.
    """
    clean = filter_clean_games(players, episodes, cfg)

    stats_parts, sel_parts, won_parts = [], [], []
    for n_players in sorted(cfg.groups):
        spec = cfg.groups[n_players]
        grp = clean[clean["n_players"] == n_players]
        if grp.empty:
            continue
        st = player_win_stats(grp)
        st["n_players"] = n_players
        sel = st[(st["win_rate"] >= spec.win_rate_threshold)
                 & (st["games"] >= spec.min_games)]
        names = set(sel["name"])
        won = grp[(grp["is_winner"] == 1) & (grp["name"].isin(names))]
        stats_parts.append(st)
        sel_parts.append(sel)
        won_parts.append(won)

    empty = pd.DataFrame(columns=["name", "games", "wins", "win_rate", "n_players"])
    stats = pd.concat(stats_parts, ignore_index=True) if stats_parts else empty
    selected_players = (pd.concat(sel_parts, ignore_index=True)
                        if sel_parts else empty.copy())
    won = (pd.concat(won_parts, ignore_index=True)
           if won_parts else clean.iloc[0:0])

    # Winner slot == this row's slot (it's the winning player's row). episodes
    # also carries winner_slot/n_players/end_reason -> drop to avoid _x/_y
    # collisions (the player-row values are authoritative here).
    won = won.rename(columns={"slot": "winner_slot"})
    ep_meta = episodes.drop(
        columns=["n_players", "end_reason", "winner_slot"], errors="ignore")
    selected_games = won.merge(ep_meta, on="episode_id", how="left")
    return stats, selected_players, selected_games


# --------------------------------------------------------------------------- #
# Streaming heavy-table extraction (bounded memory)
# --------------------------------------------------------------------------- #
def extract_table_by_episodes(src: pathlib.Path, dst: pathlib.Path,
                              episode_ids: set, batch_rows: int = 1_000_000):
    """Stream-filter a parquet by episode_id membership into a new parquet.

    Reads row-batches, masks by membership, appends to a single ParquetWriter so
    peak memory stays ~one batch regardless of source size (planet_state is GBs).
    """
    ids = pa.array(sorted(episode_ids), type=pa.int32())
    pf = pq.ParquetFile(src)
    writer = None
    kept = 0
    try:
        for batch in pf.iter_batches(batch_size=batch_rows):
            mask = pc.is_in(batch.column("episode_id"), value_set=ids)
            filtered = batch.filter(mask)
            if filtered.num_rows == 0:
                continue
            if writer is None:
                writer = pq.ParquetWriter(dst, filtered.schema, compression="snappy")
            writer.write_table(pa.Table.from_batches([filtered]))
            kept += filtered.num_rows
    finally:
        if writer is not None:
            writer.close()
    return kept


# --------------------------------------------------------------------------- #
# Fleet-state extraction (parsed from raw replays; not in the parquet DB)
# --------------------------------------------------------------------------- #
def _parse_fleets_one(item):
    """Worker: pull per-tick fleet positions out of one raw replay JSON.

    Returns (columnar dict, error). Columns mirror FLEET_STATE_COLS.
    """
    path, eid = item
    try:
        with open(path, "rb") as f:
            data = json.loads(f.read())
        steps = data.get("steps", [])
        cols = {k: [] for k in FLEET_STATE_COLS}
        for tick, step in enumerate(steps):
            if not isinstance(step, list) or not step:
                continue
            obs = step[0].get("observation", {})
            for fl in (obs.get("fleets", []) or []):
                if not isinstance(fl, (list, tuple)) or len(fl) < 7:
                    continue
                cols["episode_id"].append(eid)
                cols["tick"].append(tick)
                cols["fleet_id"].append(int(fl[0]))
                cols["owner"].append(int(fl[1]))
                cols["x"].append(float(fl[2]))
                cols["y"].append(float(fl[3]))
                cols["angle"].append(float(fl[4]))
                cols["from_planet_id"].append(int(fl[5]))
                cols["ships"].append(int(fl[6]))
        return cols, None
    except Exception as e:  # noqa: BLE001 - report and keep going
        return None, f"{path.name}: {e}"


def extract_fleet_state(data_root: pathlib.Path, dst: pathlib.Path,
                        episode_ids: set, workers: int,
                        chunksize: int = 8, flush_every: int = 1000):
    """Parse fleet positions for the selected episodes into a parquet.

    Builds episode_id -> raw JSON path from `data_root`, parses only the
    selected (and present) replays in parallel, and streams row-groups to a
    single ParquetWriter. Episodes whose JSON isn't downloaded yet are skipped
    and counted (the cache may still be filling).
    """
    id_to_path = {}
    for p, _day in find_json_files(data_root):
        try:
            id_to_path[get_episode_id(p)] = p
        except ValueError:
            continue
    items = [(id_to_path[e], e) for e in sorted(episode_ids) if e in id_to_path]
    missing = len(episode_ids) - len(items)

    writer = None
    buf = {k: [] for k in FLEET_STATE_COLS}
    buffered = kept = ok = err = 0

    def _flush():
        nonlocal writer, kept
        if not buf["episode_id"]:
            return
        table = pa.table(buf, schema=FLEET_STATE_SCHEMA)
        if writer is None:
            writer = pq.ParquetWriter(dst, FLEET_STATE_SCHEMA, compression="snappy")
        writer.write_table(table)
        kept += table.num_rows
        for k in buf:
            buf[k].clear()

    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for cols, error in pool.map(_parse_fleets_one, items, chunksize=chunksize):
                if error:
                    err += 1
                    continue
                ok += 1
                for k in FLEET_STATE_COLS:
                    buf[k].extend(cols[k])
                buffered += 1
                if buffered >= flush_every:
                    _flush()
                    buffered = 0
        _flush()
    finally:
        if writer is not None:
            writer.close()
    return kept, ok, err, missing


# --------------------------------------------------------------------------- #
# Build orchestration
# --------------------------------------------------------------------------- #
def build(cfg: Config, dry_run: bool = False, workers: int | None = None):
    t0 = time.time()
    episodes = pd.read_parquet(cfg.parquet_dir / "episodes.parquet")
    players = pd.read_parquet(cfg.parquet_dir / "player_episodes.parquet")
    print(f"Loaded {len(episodes):,} episodes / {len(players):,} player rows.")

    stats, selected_players, selected_games = select(players, episodes, cfg)
    n_games = selected_games["episode_id"].nunique()
    print(f"\nClean filter: end_reasons={cfg.end_reasons} "
          f"max_silent={cfg.max_silent_ticks} no_froze={cfg.require_no_froze}")
    for n_players in sorted(cfg.groups):
        spec = cfg.groups[n_players]
        gp = selected_players[selected_players["n_players"] == n_players]
        gg = selected_games[selected_games["n_players"] == n_players]
        print(f"  {n_players}p (wr>={spec.win_rate_threshold}, "
              f"games>={spec.min_games}): {len(gp)} players, "
              f"{gg['episode_id'].nunique():,} won games")
    print(f"Total: {len(selected_players)} player-groups -> {n_games:,} won games.")
    if len(selected_players):
        top = selected_players.sort_values("win_rate", ascending=False).head(10)
        print("\nTop selected players:")
        for _, r in top.iterrows():
            print(f"  {r['name']:<28} {int(r['n_players'])}p "
                  f"wr={r['win_rate']*100:5.1f}%  "
                  f"({int(r['wins'])}/{int(r['games'])})")

    if dry_run:
        print("\n[dry-run] no files written.")
        return

    if n_games == 0:
        print("\nNo games selected; nothing to write. Loosen thresholds.")
        return

    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "games").mkdir(exist_ok=True)

    stats.to_parquet(out / "player_stats.parquet", index=False)
    selected_players.to_parquet(out / "selected_players.parquet", index=False)
    selected_games.to_parquet(out / "selected_games.parquet", index=False)
    print(f"\nWrote selection: player_stats / selected_players / "
          f"selected_games ({len(selected_games):,} rows).")

    episode_ids = set(selected_games["episode_id"].tolist())
    for name in cfg.included_tables():
        src = cfg.parquet_dir / f"{name}.parquet"
        if not src.exists():
            print(f"  skip {name}: {src.name} missing")
            continue
        dst = out / "games" / f"{name}.parquet"
        kept = extract_table_by_episodes(src, dst, episode_ids)
        mb = dst.stat().st_size / 1e6 if dst.exists() else 0.0
        print(f"  {name}: {kept:,} rows ({mb:.1f} MB)")

    if cfg.include_fleet_state:
        w = workers or max(1, (os.cpu_count() or 4) - 2)
        dst = out / "games" / "fleet_state.parquet"
        rows, ok, err, missing = extract_fleet_state(
            cfg.data_root, dst, episode_ids, w)
        mb = dst.stat().st_size / 1e6 if dst.exists() else 0.0
        print(f"  fleet_state: {rows:,} rows ({mb:.1f} MB) from {ok:,} replays "
              f"[parse_err={err}, missing_json={missing:,}]")
        if missing:
            print(f"    NOTE: {missing:,} selected games have no raw JSON yet "
                  f"(download still filling {cfg.data_root}); re-run to backfill.")

    resolved = cfg.to_yaml_dict()
    resolved["_selected_players"] = len(selected_players)
    resolved["_selected_games"] = int(n_games)
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    print(f"\nDone in {time.time()-t0:.0f}s. Dataset at {out}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Build Orbit Wars winner-game dataset.")
    ap.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    ap.add_argument("--group", action="append", metavar="N:WR:MG",
                    help="replace config groups, e.g. --group 2:0.55:25 "
                         "--group 4:0.35:25 (repeatable)")
    ap.add_argument("--max-silent", type=int, help="override max_silent_ticks")
    ap.add_argument("--out-dir", type=pathlib.Path, help="override out_dir")
    ap.add_argument("--planet-state", dest="planet_state",
                    action="store_true", default=None,
                    help="force-extract planet_state (heavy, ~670MB)")
    ap.add_argument("--no-planet-state", dest="planet_state",
                    action="store_false",
                    help="skip planet_state extraction")
    ap.add_argument("--fleet-state", dest="fleet_state",
                    action="store_true", default=None,
                    help="force-parse fleet_state from raw replays")
    ap.add_argument("--no-fleet-state", dest="fleet_state",
                    action="store_false",
                    help="skip fleet_state extraction")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel workers for fleet_state parse "
                         "(default: cpu_count - 2)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print selection stats only; write nothing")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if args.group:
        cfg.groups = {}
        for g in args.group:
            n, wr, mg = g.split(":")
            cfg.groups[int(n)] = GroupSpec(
                win_rate_threshold=float(wr), min_games=int(mg))
    if args.max_silent is not None:
        cfg.max_silent_ticks = args.max_silent
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir.resolve()
    if args.planet_state is not None:
        cfg.include_planet_state = args.planet_state
    if args.fleet_state is not None:
        cfg.include_fleet_state = args.fleet_state

    build(cfg, dry_run=args.dry_run, workers=args.workers)


if __name__ == "__main__":
    main()
