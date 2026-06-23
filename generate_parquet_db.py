#!/usr/bin/env python3
"""Build the Orbit Wars Parquet DB from raw replay JSONs.

Parallel, streaming version of the notebook's Step 4. Parses replays across all
CPU cores and appends to Parquet via ParquetWriter (no quadratic re-read), then
the notebook only has to load the finished tables for analysis/viz.

Usage:
    python generate_parquet_db.py                  # all downloaded episodes
    python generate_parquet_db.py --max 6000       # cap for a quick run
    python generate_parquet_db.py --workers 16
    python generate_parquet_db.py --no-planet-state # skip the heaviest table
"""

import argparse
import json
import math
import os
import pathlib
import re
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# --------------------------------------------------------------------------- #
# Config (mirrors the notebook)
# --------------------------------------------------------------------------- #
REPO_ROOT = pathlib.Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "episode_data"
SUN_X, SUN_Y = 50.0, 50.0
DEFAULT_EPISODE_STEPS = 500
ERROR_STATUSES = frozenset({"ERROR", "TIMEOUT", "INVALID"})


def classify_end_reason(statuses, rewards, final_step, episode_steps):
    """Why the match terminated, mirroring orbit_wars.py's terminate logic.

    Precedence: a tainted game (any agent ended ERROR/TIMEOUT/INVALID) is
    flagged first so analysts can filter it out, then the two clean env-rule
    cases (step cap vs last-survivor), then the everyone-wiped-out edge case.
    """
    if any(s in ERROR_STATUSES for s in statuses if s):
        return "agent_error"
    # Env terminates when step >= episodeSteps - 2 (orbit_wars.py).
    if final_step >= episode_steps - 2:
        return "step_limit"
    if rewards and max(rewards) >= 1:
        return "domination"
    return "mutual_destruction"


def compute_opponent_flags(n_players, statuses, n_actions, last_action_tick,
                           last_alive_tick):
    """Per-slot view of "did the player I was up against fail to play?".

    For each slot the "primary opponent" is the *least active* other player
    (fewest actions, tie-break lowest slot) since that's the most likely one to
    have frozen/failed. Returns a list of dicts (one per slot) with the
    opponent_* columns.

    `opponent_froze` is intentionally *conservative*: it only flags the
    unambiguous "failed to play" cases -- the opponent never issued a single
    action while alive, or the engine marked them ERROR/TIMEOUT/INVALID. Note:
    every published Kaggle replay is status==DONE, so in practice this fires
    only on the never-acted case. The fuzzy "went quiet mid-game" case is
    deliberately NOT baked into the flag (agents routinely stop sending fleets
    once they've won the board but the game runs to step_limit). Instead use the
    raw `opponent_silent_ticks` (ticks alive without acting) to threshold a soft
    freeze yourself, e.g. `opponent_silent_ticks > 50`.
    """
    def status_of(o):
        return statuses[o] if o < len(statuses) else None

    flags = []
    for slot in range(n_players):
        opps = [o for o in range(n_players) if o != slot]
        if not opps:
            flags.append({
                "opponent_n_actions": 0, "opponent_last_action_tick": -1,
                "opponent_last_alive_tick": -1, "opponent_silent_ticks": 0,
                "opponent_status": None, "opponent_timed_out": 0,
                "opponent_froze": 0,
            })
            continue
        primary = min(opps, key=lambda o: (n_actions[o], o))
        opp_acts = int(n_actions[primary])
        opp_last = int(last_action_tick[primary])
        opp_alive = int(last_alive_tick[primary])
        # Span the opponent was alive but issued no action. If they never acted,
        # that's their whole lifetime (alive tick count); else the trailing gap.
        if opp_last < 0:
            silent = opp_alive + 1 if opp_alive >= 0 else 0
        else:
            silent = max(0, opp_alive - opp_last)
        any_error = any(status_of(o) in ERROR_STATUSES for o in opps)
        timed_out = 1 if any(status_of(o) == "TIMEOUT" for o in opps) else 0
        never_acted = opp_acts == 0 and opp_alive >= 0
        froze = 1 if (never_acted or any_error) else 0
        flags.append({
            "opponent_n_actions": opp_acts,
            "opponent_last_action_tick": opp_last,
            "opponent_last_alive_tick": opp_alive,
            "opponent_silent_ticks": int(silent),
            "opponent_status": status_of(primary),
            "opponent_timed_out": timed_out,
            "opponent_froze": froze,
        })
    return flags


SCHEMAS = {
    "episodes": pa.schema([
        ("episode_id", pa.int32()), ("dataset", pa.string()),
        ("n_players", pa.int8()), ("n_steps", pa.int16()),
        ("seed", pa.int32()), ("angular_velocity", pa.float32()),
        ("n_planets", pa.int8()), ("n_comets", pa.int8()), ("winner_slot", pa.int8()),
        ("final_step", pa.int16()), ("end_reason", pa.string()),
    ]),
    "player_episodes": pa.schema([
        ("episode_id", pa.int32()), ("slot", pa.int8()), ("name", pa.string()),
        ("reward", pa.float32()), ("is_winner", pa.int8()), ("status", pa.string()),
        ("n_actions", pa.int32()), ("last_action_tick", pa.int16()),
        ("last_alive_tick", pa.int16()),
        ("opponent_n_actions", pa.int32()),
        ("opponent_last_action_tick", pa.int16()),
        ("opponent_last_alive_tick", pa.int16()),
        ("opponent_silent_ticks", pa.int16()),
        ("opponent_status", pa.string()), ("opponent_timed_out", pa.int8()),
        ("opponent_froze", pa.int8()),
    ]),
    "tick_summary": pa.schema([
        ("episode_id", pa.int32()), ("tick", pa.int16()), ("slot", pa.int8()),
        ("ships_planets", pa.int32()), ("ships_fleets", pa.int32()),
        ("total_ships", pa.int32()), ("production", pa.int16()),
        ("n_planets", pa.int8()), ("n_fleets", pa.int16()),
    ]),
    "actions": pa.schema([
        ("episode_id", pa.int32()), ("tick", pa.int16()), ("slot", pa.int8()),
        ("src_planet_id", pa.int16()), ("angle", pa.float32()), ("n_ships", pa.int32()),
    ]),
    "episode_planets": pa.schema([
        ("episode_id", pa.int32()), ("planet_id", pa.int16()),
        ("initial_x", pa.float32()), ("initial_y", pa.float32()),
        ("radius", pa.float32()), ("production", pa.int8()),
        ("orbit_radius", pa.float32()), ("is_static", pa.bool_()),
        ("is_comet", pa.bool_()), ("initial_ships", pa.int16()), ("initial_owner", pa.int8()),
    ]),
    "planet_state": pa.schema([
        ("episode_id", pa.int32()), ("tick", pa.int16()),
        ("planet_id", pa.int16()), ("owner", pa.int8()), ("ships", pa.int32()),
    ]),
}

PLANET_STATE_COLS = ("episode_id", "tick", "planet_id", "owner", "ships")


# --------------------------------------------------------------------------- #
# Discovery (mirrors the notebook's find_json_files)
# --------------------------------------------------------------------------- #
def find_json_files(data_root):
    all_json = list(data_root.rglob("*.json"))

    by_day = {}
    for p in all_json:
        stem = p.stem
        if not (re.match(r"episode-(\d+)", stem) or re.match(r"^(\d+)$", stem)):
            continue
        day = None
        for ancestor in p.parents:
            if ancestor == data_root:
                break
            if ancestor.parent == data_root:
                day = ancestor.name
                break
        if day is None:
            day = p.parent.name
        by_day.setdefault(day, []).append(p)

    result = []
    for d in sorted(by_day.keys()):
        for p in sorted(by_day[d]):
            result.append((p, d))

    # Only keep days that actually exist as directories under data_root.
    days_downloaded = {p.name for p in data_root.iterdir() if p.is_dir()}
    return [(p, d) for p, d in result if d in days_downloaded]


# --------------------------------------------------------------------------- #
# Parser (verbatim logic from the notebook)
# --------------------------------------------------------------------------- #
def get_episode_id(path):
    stem = path.stem
    m = re.match(r"episode-(\d+)", stem)
    if m:
        return int(m.group(1))
    m = re.match(r"^(\d+)$", stem)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot extract episode ID from {path.name}")


def parse_episode(path, dataset_label):
    with open(path, "rb") as f:
        data = json.loads(f.read())

    rewards = data.get("rewards", [])
    steps = data.get("steps", [])
    info = data.get("info", {})
    agents = info.get("Agents", [])
    n_players = len(rewards)
    n_steps = len(steps)

    if n_steps < 2 or n_players < 2:
        return None

    episode_id = get_episode_id(path)

    max_r = max(rewards)
    winners = [i for i, r in enumerate(rewards) if r == max_r]
    winner = winners[0] if len(winners) == 1 else -1

    # Final per-agent status (DONE / ERROR / TIMEOUT / INVALID) lives at the
    # top level; fall back to the last step's per-agent status if absent.
    statuses = data.get("statuses")
    if not statuses:
        statuses = [a.get("status") for a in steps[-1]] if steps[-1] else []
    episode_steps = data.get("configuration", {}).get(
        "episodeSteps", DEFAULT_EPISODE_STEPS)
    final_step = n_steps - 1
    end_reason = classify_end_reason(statuses, rewards, final_step, episode_steps)

    obs0 = steps[0][0].get("observation", {})
    av = float(obs0.get("angular_velocity", 0.0))
    comet_ids = set(obs0.get("comet_planet_ids", []))
    init_planets = obs0.get("initial_planets", obs0.get("planets", []))

    episode_row = {
        "episode_id": episode_id, "dataset": dataset_label,
        "n_players": n_players, "n_steps": n_steps,
        "seed": info.get("seed", 0), "angular_velocity": av,
        "n_planets": len(init_planets), "n_comets": len(comet_ids),
        "winner_slot": winner,
        "final_step": final_step, "end_reason": end_reason,
    }

    player_rows = []
    for slot in range(n_players):
        name = agents[slot]["Name"] if slot < len(agents) else f"player_{slot}"
        player_rows.append({
            "episode_id": episode_id, "slot": slot, "name": name,
            "reward": rewards[slot], "is_winner": 1 if slot == winner else 0,
            "status": statuses[slot] if slot < len(statuses) else None,
        })

    ep_planet_rows = []
    for p in init_planets:
        pid, owner, x, y, r, ships, prod = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
        orbit_r = math.sqrt((x - SUN_X) ** 2 + (y - SUN_Y) ** 2)
        is_static = (orbit_r + r >= 50.0) or (av == 0)
        ep_planet_rows.append({
            "episode_id": episode_id, "planet_id": pid,
            "initial_x": x, "initial_y": y, "radius": r,
            "production": prod, "orbit_radius": orbit_r,
            "is_static": is_static, "is_comet": pid in comet_ids,
            "initial_ships": ships, "initial_owner": owner,
        })

    tick_rows, action_rows = [], []
    n_actions_slot = [0] * n_players
    last_action_tick_slot = [-1] * n_players
    last_alive_tick_slot = [-1] * n_players
    ps_eid, ps_tick, ps_pid, ps_owner, ps_ships = [], [], [], [], []

    for tick in range(n_steps):
        step = steps[tick]
        if not isinstance(step, list) or not step:
            continue
        obs = step[0].get("observation", {})
        planets = obs.get("planets", [])
        fleets = obs.get("fleets", [])

        for slot in range(n_players):
            sp = sum(p[5] for p in planets if p[1] == slot)
            sf = sum(f[6] for f in fleets if f[1] == slot)
            n_planets_owned = sum(1 for p in planets if p[1] == slot)
            if n_planets_owned > 0 or sf > 0:
                last_alive_tick_slot[slot] = tick
            tick_rows.append({
                "episode_id": episode_id, "tick": tick, "slot": slot,
                "ships_planets": sp, "ships_fleets": sf, "total_ships": sp + sf,
                "production": sum(p[6] for p in planets if p[1] == slot),
                "n_planets": sum(1 for p in planets if p[1] == slot),
                "n_fleets": sum(1 for f in fleets if f[1] == slot),
            })
            if tick >= 1 and slot < len(step):
                for a in (step[slot].get("action", []) or []):
                    if isinstance(a, (list, tuple)) and len(a) >= 3:
                        action_rows.append({
                            "episode_id": episode_id, "tick": tick, "slot": slot,
                            "src_planet_id": int(a[0]),
                            "angle": float(a[1]), "n_ships": int(a[2]),
                        })
                        n_actions_slot[slot] += 1
                        last_action_tick_slot[slot] = tick

        for p in planets:
            ps_eid.append(episode_id); ps_tick.append(tick)
            ps_pid.append(p[0]); ps_owner.append(p[1]); ps_ships.append(p[5])

    opp_flags = compute_opponent_flags(
        n_players, statuses, n_actions_slot, last_action_tick_slot,
        last_alive_tick_slot)
    for slot in range(n_players):
        player_rows[slot]["n_actions"] = n_actions_slot[slot]
        player_rows[slot]["last_action_tick"] = last_action_tick_slot[slot]
        player_rows[slot]["last_alive_tick"] = last_alive_tick_slot[slot]
        player_rows[slot].update(opp_flags[slot])

    return {
        "episode": episode_row,
        "players": player_rows,
        "ticks": tick_rows,
        "actions": action_rows,
        "episode_planets": ep_planet_rows,
        "planet_state": {"episode_id": ps_eid, "tick": ps_tick,
                         "planet_id": ps_pid, "owner": ps_owner, "ships": ps_ships},
    }


def _parse_one(item):
    """Worker entry point. Returns (result, error_message)."""
    path, day = item
    try:
        return parse_episode(path, day), None
    except Exception as e:  # noqa: BLE001 - report and keep going
        return None, f"{path.name}: {e}"


# --------------------------------------------------------------------------- #
# Lightweight end-reason scanner (for augmenting an already-built DB)
# --------------------------------------------------------------------------- #
def _extract_list(buf, key):
    """Pull a flat (non-nested) JSON list value out of raw bytes."""
    i = buf.find(key)
    if i == -1:
        return []
    lb = buf.find(b"[", i)
    rb = buf.find(b"]", lb)
    if lb == -1 or rb == -1:
        return []
    return json.loads(buf[lb:rb + 1])


def _scan_end_one(item):
    """Read only the top-level metadata (skips the giant `steps` array).

    `statuses`, `rewards`, and `configuration.episodeSteps` all live before
    `steps` in the JSON, so we slice the head and regex it out instead of
    json.loads-ing multi-MB replays.
    """
    path, _day = item
    try:
        with open(path, "rb") as f:
            raw = f.read()
        cut = raw.find(b'"steps"')
        head = raw[:cut] if cut != -1 else raw
        eid = get_episode_id(path)
        rewards = _extract_list(head, b'"rewards"')
        statuses = _extract_list(head, b'"statuses"')
        m = re.search(rb'"episodeSteps"\s*:\s*(\d+)', head)
        esteps = int(m.group(1)) if m else DEFAULT_EPISODE_STEPS
        return (eid, statuses, rewards, esteps), None
    except Exception as e:  # noqa: BLE001 - report and keep going
        return None, f"{path.name}: {e}"


def _scan_actions_one(item):
    """Full step parse to count actions per slot + last active tick.

    Unlike `_scan_end_one`, action data lives *inside* the giant `steps` array,
    so we must json.loads the whole replay (slower than the head-only scan).
    """
    path, _day = item
    try:
        with open(path, "rb") as f:
            data = json.loads(f.read())
        steps = data.get("steps", [])
        rewards = data.get("rewards", [])
        n_players = len(rewards)
        n_steps = len(steps)
        eid = get_episode_id(path)
        statuses = data.get("statuses")
        if not statuses:
            statuses = [a.get("status") for a in steps[-1]] if (
                steps and steps[-1]) else []
        n_actions = [0] * n_players
        last_action_tick = [-1] * n_players
        last_alive_tick = [-1] * n_players
        for tick in range(n_steps):
            step = steps[tick]
            if not isinstance(step, list) or not step:
                continue
            obs = step[0].get("observation", {})
            planets = obs.get("planets", [])
            fleets = obs.get("fleets", [])
            for slot in range(n_players):
                if any(p[1] == slot for p in planets) or any(
                        f[1] == slot for f in fleets):
                    last_alive_tick[slot] = tick
                if tick >= 1 and slot < len(step):
                    cnt = 0
                    for a in (step[slot].get("action", []) or []):
                        if isinstance(a, (list, tuple)) and len(a) >= 3:
                            cnt += 1
                    if cnt:
                        n_actions[slot] += cnt
                        last_action_tick[slot] = tick
        return (eid, n_players, statuses, n_actions, last_action_tick,
                last_alive_tick), None
    except Exception as e:  # noqa: BLE001 - report and keep going
        return None, f"{path.name}: {e}"


def augment_actions(out_dir, data_root, workers, chunksize, max_episodes=None):
    """Add per-player action counts + "did my opponent freeze/fail" flags.

    Rewrites only player_episodes.parquet. Requires a full step parse (actions
    live inside `steps`), so it's slower than `--augment`. New columns:
    n_actions, last_action_tick, opponent_n_actions, opponent_last_action_tick,
    opponent_status, opponent_timed_out, opponent_froze.
    """
    pl_path = out_dir / "player_episodes.parquet"
    if not pl_path.exists():
        raise SystemExit(f"Cannot augment: {pl_path} missing. Build first.")

    players = pd.read_parquet(pl_path)
    known_ids = set(players["episode_id"].tolist())
    print(f"Loaded {len(players)} player rows.")

    files = find_json_files(data_root)
    if max_episodes:
        files = files[:max_episodes]
    print(f"Parsing {len(files)} replays for action counts / freeze detection "
          f"with {workers} workers (full step parse, slower)...")

    flags_by_key = {}  # (eid, slot) -> dict of new columns
    t0 = time.time()
    ok = err = miss = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for res, error in pool.map(_scan_actions_one, files, chunksize=chunksize):
            if error:
                err += 1
                continue
            eid, n_players, statuses, n_actions, last_action_tick, last_alive_tick = res
            if eid not in known_ids:
                miss += 1
                continue
            ok += 1
            flags = compute_opponent_flags(
                n_players, statuses, n_actions, last_action_tick, last_alive_tick)
            for slot in range(n_players):
                row = {"n_actions": int(n_actions[slot]),
                       "last_action_tick": int(last_action_tick[slot]),
                       "last_alive_tick": int(last_alive_tick[slot])}
                row.update(flags[slot])
                flags_by_key[(eid, slot)] = row

    new_cols = ["n_actions", "last_action_tick", "last_alive_tick",
                "opponent_n_actions", "opponent_last_action_tick",
                "opponent_last_alive_tick", "opponent_silent_ticks",
                "opponent_status", "opponent_timed_out", "opponent_froze"]
    for col in new_cols:
        players[col] = [
            flags_by_key.get((eid, slot), {}).get(col)
            for eid, slot in zip(players["episode_id"], players["slot"])
        ]

    _write_table(pl_path, players, SCHEMAS["player_episodes"])

    elapsed = time.time() - t0
    print(f"\nAugmented {ok} episodes in {elapsed:.0f}s "
          f"(scan err={err}, unknown_id={miss}).")
    print("opponent_froze (never-acted / engine error) counts:")
    print(players["opponent_froze"].value_counts(dropna=False).to_string())
    print("opponent_timed_out counts:")
    print(players["opponent_timed_out"].value_counts(dropna=False).to_string())
    print("opponent_silent_ticks describe (soft-freeze: threshold yourself):")
    print(players["opponent_silent_ticks"].describe().to_string())


def augment_existing(out_dir, data_root, workers, chunksize, max_episodes=None):
    """Add end_reason/final_step + per-player status to an existing DB.

    Only rewrites episodes.parquet and player_episodes.parquet; the heavy
    tables (planet_state, actions, tick_summary, episode_planets) are left
    untouched. Far cheaper than a full rebuild.
    """
    ep_path = out_dir / "episodes.parquet"
    pl_path = out_dir / "player_episodes.parquet"
    if not ep_path.exists() or not pl_path.exists():
        raise SystemExit(
            f"Cannot augment: {ep_path} / {pl_path} missing. Build first.")

    episodes = pd.read_parquet(ep_path)
    players = pd.read_parquet(pl_path)
    known_ids = set(episodes["episode_id"].tolist())
    print(f"Loaded {len(episodes)} episodes / {len(players)} player rows.")

    files = find_json_files(data_root)
    if max_episodes:
        files = files[:max_episodes]
    print(f"Scanning {len(files)} replays for end metadata "
          f"with {workers} workers...")

    end_reason, final_step, status_by_key = {}, {}, {}
    t0 = time.time()
    ok = err = miss = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for res, error in pool.map(_scan_end_one, files, chunksize=chunksize):
            if error:
                err += 1
                continue
            eid, statuses, rewards, esteps = res
            if eid not in known_ids:
                miss += 1
                continue
            ok += 1
            for slot, st in enumerate(statuses):
                status_by_key[(eid, slot)] = st
            # final_step backfilled from the already-stored n_steps below.
            end_reason[eid] = (statuses, rewards, esteps)

    # Compute final_step from existing n_steps, then classify.
    fs = (episodes["n_steps"] - 1).astype("int64")
    episodes["final_step"] = fs
    reasons = []
    for eid, n in zip(episodes["episode_id"], fs):
        meta = end_reason.get(eid)
        if meta is None:
            reasons.append(None)
            continue
        statuses, rewards, esteps = meta
        reasons.append(classify_end_reason(statuses, rewards, int(n), esteps))
    episodes["end_reason"] = reasons

    players["status"] = [
        status_by_key.get((eid, slot))
        for eid, slot in zip(players["episode_id"], players["slot"])
    ]

    _write_table(ep_path, episodes, SCHEMAS["episodes"])
    _write_table(pl_path, players, SCHEMAS["player_episodes"])

    elapsed = time.time() - t0
    print(f"\nAugmented {ok} episodes in {elapsed:.0f}s "
          f"(scan err={err}, unknown_id={miss}).")
    print("end_reason counts:")
    print(episodes["end_reason"].value_counts(dropna=False).to_string())


def _write_table(path, df, schema):
    df = df.copy()
    # Missing columns must be object-None (not NaN floats) so pyarrow casts them
    # to proper nulls for int/string targets instead of failing the int cast.
    for name in schema.names:
        if name not in df.columns:
            df[name] = pd.Series([None] * len(df), dtype="object")
    df = df.reindex(columns=schema.names)
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, path, compression="snappy")
    print(f"  wrote {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")


# --------------------------------------------------------------------------- #
# Streaming Parquet writer (replaces the notebook's quadratic flush_all)
# --------------------------------------------------------------------------- #
class ParquetSink:
    """Opens one ParquetWriter per table and appends row groups on flush."""

    def __init__(self, out_dir, schemas, write_planet_state=True):
        self.out_dir = out_dir
        self.schemas = schemas
        self.write_planet_state = write_planet_state
        self.buffers = {k: [] for k in schemas}
        self.writers = {}

    def add(self, result):
        self.buffers["episodes"].append(result["episode"])
        self.buffers["player_episodes"].extend(result["players"])
        self.buffers["tick_summary"].extend(result["ticks"])
        self.buffers["actions"].extend(result["actions"])
        self.buffers["episode_planets"].extend(result["episode_planets"])
        if self.write_planet_state:
            self.buffers["planet_state"].append(result["planet_state"])

    def flush(self):
        for name, schema in self.schemas.items():
            if not self.buffers[name]:
                continue
            if name == "planet_state":
                combined = {k: [] for k in PLANET_STATE_COLS}
                for chunk in self.buffers[name]:
                    for k in combined:
                        combined[k].extend(chunk[k])
                df = pd.DataFrame(combined)
            else:
                df = pd.DataFrame(self.buffers[name])
            table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
            if name not in self.writers:
                self.writers[name] = pq.ParquetWriter(
                    self.out_dir / f"{name}.parquet", schema, compression="snappy"
                )
            self.writers[name].write_table(table)
            self.buffers[name].clear()

    def close(self):
        self.flush()
        for w in self.writers.values():
            w.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Build Orbit Wars Parquet DB.")
    ap.add_argument("--data-root", type=pathlib.Path, default=DATA_ROOT,
                    help="Directory of downloaded episode_data/<day>/ JSONs.")
    ap.add_argument("--out-dir", type=pathlib.Path, default=REPO_ROOT / "parquet_out")
    ap.add_argument("--max", type=int, default=None,
                    help="Cap on episodes to parse (default: all).")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--flush-every", type=int, default=200,
                    help="Episodes per Parquet row-group flush (bounds memory).")
    ap.add_argument("--chunksize", type=int, default=8,
                    help="Tasks dispatched per worker batch.")
    ap.add_argument("--no-planet-state", action="store_true",
                    help="Skip the heaviest table (per-planet, per-tick).")
    ap.add_argument("--augment", action="store_true",
                    help="Don't rebuild: just add end_reason/final_step/status "
                         "to existing episodes & player_episodes parquet.")
    ap.add_argument("--augment-actions", action="store_true",
                    help="Don't rebuild: add per-player n_actions + "
                         "opponent_froze/opponent_timed_out flags to existing "
                         "player_episodes parquet (full step parse, slower).")
    args = ap.parse_args()

    args.out_dir.mkdir(exist_ok=True)

    if args.augment:
        augment_existing(args.out_dir, args.data_root, args.workers,
                         args.chunksize, args.max)
        return

    if args.augment_actions:
        augment_actions(args.out_dir, args.data_root, args.workers,
                        args.chunksize, args.max)
        return

    # Fresh start: ParquetWriter creates new files, so clear stale outputs.
    for old in args.out_dir.glob("*.parquet"):
        old.unlink()

    print("Discovering replays...")
    json_files = find_json_files(args.data_root)
    files = json_files[: args.max] if args.max else json_files
    print(f"Found {len(json_files)} replays; processing {len(files)} "
          f"with {args.workers} workers.")
    if not files:
        print("Nothing to do.")
        return

    sink = ParquetSink(args.out_dir, SCHEMAS, write_planet_state=not args.no_planet_state)
    t0 = time.time()
    ok = skip = err = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (result, error) in enumerate(
            pool.map(_parse_one, files, chunksize=args.chunksize)
        ):
            if error:
                err += 1
                if err <= 3:
                    print(f"  ERROR {error}")
            elif result is None:
                skip += 1
            else:
                sink.add(result)
                ok += 1

            if (i + 1) % args.flush_every == 0:
                sink.flush()
                elapsed = time.time() - t0
                rate = ok / elapsed if elapsed > 0 else 0
                print(f"  [{i+1:>6}/{len(files)}] ok={ok} skip={skip} err={err}"
                      f"  ({elapsed:.0f}s, {rate:.1f} ep/s)")

    sink.close()
    elapsed = time.time() - t0
    rate = ok / elapsed if elapsed > 0 else 0
    print(f"\nDone: {ok} episodes in {elapsed:.0f}s ({rate:.1f} ep/s)")
    print(f"Skipped: {skip}  Errors: {err}")
    print(f"Output: {args.out_dir}")
    for name in SCHEMAS:
        f = args.out_dir / f"{name}.parquet"
        if f.exists():
            print(f"  {name}.parquet  {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
