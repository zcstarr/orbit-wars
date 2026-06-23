#!/usr/bin/env python3
"""Orbit Wars conditional DDPM training scaffold.

Fill-in-the-blanks entry point for cache build, data load/unload, train,
validate, sample, and checkpoint save/load. See orbit_wars_diffusion_lean_spec.md.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import pathlib
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.methods.ddpm import DDPM

try:
    import wandb
except ImportError:  # wandb is optional; training works without it.
    wandb = None

logger = logging.getLogger("train_orbit")

REPO_ROOT = pathlib.Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Constants (orbit_wars_diffusion_lean_spec.md §17)
# --------------------------------------------------------------------------- #
P_MAX = 60
ACTION_HORIZON = 4
EXECUTION_HORIZON = 1
ACTION_DIM = 4
PLANET_FEATURE_DIM = 12
GLOBAL_FEATURE_DIM = 9
MAX_GAME_TICK = 500
PLANET_SHIP_SCALE = 1518.0
COMET_SPAWN_TICKS = (50, 150, 250, 350, 450)
SUN_X = SUN_Y = 50.0
MAX_PLANET_RADIUS = 1.0 + math.log(5.0)


# --------------------------------------------------------------------------- #
# Small reusable blocks (TimestepEmbedding only — blocks.py pulls einops)
# --------------------------------------------------------------------------- #
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device)
            / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, time_embed_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or 4 * time_embed_dim
        self.sinusoidal = SinusoidalPositionalEmbedding(time_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, time_embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(t))


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class FeatureConfig:
    p_max: int = P_MAX
    action_horizon: int = ACTION_HORIZON
    planet_feature_dim: int = PLANET_FEATURE_DIM
    global_feature_dim: int = GLOBAL_FEATURE_DIM
    action_dim: int = ACTION_DIM
    max_game_tick: int = MAX_GAME_TICK

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclasses.dataclass
class ModelConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    num_timesteps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 0.20

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclasses.dataclass
class TrainConfig:
    dataset_dir: pathlib.Path = REPO_ROOT / "dataset_out"
    cache_dir: pathlib.Path = REPO_ROOT / "cache"
    checkpoint_dir: pathlib.Path = REPO_ROOT / "checkpoints"
    batch_size: int = 32
    num_workers: int = 4
    lr: float = 3e-4
    epochs: int = 10
    episode_cache_size: int = 4
    train_fraction: float = 0.80
    valid_fraction: float = 0.10
    log_every: int = 50
    save_every: int = 500
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_wandb: bool = False
    wandb_project: str = "orbit-wars"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["dataset_dir"] = str(self.dataset_dir)
        d["cache_dir"] = str(self.cache_dir)
        d["checkpoint_dir"] = str(self.checkpoint_dir)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        fields = {}
        for k in cls.__dataclass_fields__:
            if k not in d:
                continue
            v = d[k]
            if k in ("dataset_dir", "cache_dir", "checkpoint_dir"):
                v = pathlib.Path(v)
            fields[k] = v
        return cls(**fields)


# --------------------------------------------------------------------------- #
# Orbit / feature helpers
# --------------------------------------------------------------------------- #
def orbit_position(
    initial_x: float,
    initial_y: float,
    orbit_radius: float,
    angular_velocity: float,
    tick: int,
) -> Tuple[float, float]:
    phase0 = math.atan2(initial_y - SUN_Y, initial_x - SUN_X)
    phase = phase0 - angular_velocity * tick
    return (
        SUN_X + orbit_radius * math.cos(phase),
        SUN_Y + orbit_radius * math.sin(phase),
    )


def encode_owner(
    owner: int,
    controlled_player: int,
    opponent_map: Dict[int, int],
) -> Tuple[List[float], float]:
    if owner == -1:
        return [1.0, 0.0, 0.0], 0.0
    if owner == controlled_player:
        return [0.0, 1.0, 0.0], 0.0
    opp_id = opponent_map.get(owner, 1)
    return [0.0, 0.0, 1.0], opp_id / 3.0


def encode_global_features(
    tick: int,
    angular_velocity: float,
    n_players: int,
    active_mask: np.ndarray,
    is_comet: np.ndarray,
) -> List[float]:
    future_spawns = [s for s in COMET_SPAWN_TICKS if s >= tick]
    if future_spawns:
        turns_until_norm = min(future_spawns[0] - tick, 100) / 100.0
        no_future_spawns = 0.0
    else:
        turns_until_norm = 1.0
        no_future_spawns = 1.0

    active_regular = sum(
        bool(active_mask[i]) and not bool(is_comet[i]) for i in range(P_MAX)
    )
    active_comets = sum(
        bool(active_mask[i]) and bool(is_comet[i]) for i in range(P_MAX)
    )

    return [
        tick / float(MAX_GAME_TICK),
        (angular_velocity - 0.025) / 0.025,
        float(n_players == 2),
        float(n_players == 4),
        turns_until_norm,
        float(tick in COMET_SPAWN_TICKS),
        no_future_spawns,
        active_regular / 40.0,
        active_comets / 20.0,
    ]


def build_opponent_map(n_players: int, winner_slot: int) -> Dict[int, int]:
    opponents = [s for s in range(n_players) if s != winner_slot]
    return {slot: idx + 1 for idx, slot in enumerate(opponents)}


def build_dense_actions_for_tick(
    action_rows: pd.DataFrame,
    source_ships: np.ndarray,
    source_mask: np.ndarray,
    *,
    episode_id: Optional[int] = None,
    tick: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    dense = np.zeros((P_MAX, ACTION_DIM), dtype=np.float32)
    dense[source_mask, 0] = -1.0
    skipped = 0

    for row in action_rows.itertuples(index=False):
        planet_id = int(row.src_planet_id)
        ships_available = int(source_ships[planet_id])
        n_ships = int(row.n_ships)
        # A launch is only valid if the source planet is in source_mask at the
        # observation tick this action is aligned to (ships[t]): active,
        # owned by the winner, and ships>=1. The off-by-one alignment (action
        # row tick==t+1 -> obs tick t) can break any of these, e.g. ownership
        # flipped between obs t and the action at t+1, or the planet has no
        # planet_state entry at t (zero-init). Skip rather than raise so cache
        # building is robust to these data edge cases, and so the launch never
        # lands outside action_source_mask (validate_episode_cache invariant).
        if not bool(source_mask[planet_id]):
            owner_ok = ships_available >= 1
            logger.warning(
                "Skipping launch outside source_mask: episode=%s obs_tick=%s "
                "src_planet_id=%s ships_available=%s n_ships=%s "
                "reason=%s",
                episode_id,
                tick,
                planet_id,
                ships_available,
                n_ships,
                "no_ships" if not owner_ok else "not_winner_owned_or_inactive",
            )
            skipped += 1
            continue
        fraction = n_ships / ships_available
        dense[planet_id] = np.array(
            [
                1.0,
                math.sin(float(row.angle)),
                math.cos(float(row.angle)),
                2.0 * fraction - 1.0,
            ],
            dtype=np.float32,
        )
    return dense, skipped


def decode_action(
    prediction: np.ndarray,
    ships_available: int,
) -> Tuple[bool, float, int]:
    launch = float(prediction[0]) > 0.0
    if not launch:
        return False, 0.0, 0

    sin_v = float(prediction[1])
    cos_v = float(prediction[2])
    norm = max(math.sqrt(sin_v**2 + cos_v**2), 1e-6)
    angle = math.atan2(sin_v / norm, cos_v / norm)
    fraction = min(max((float(prediction[3]) + 1.0) / 2.0, 0.0), 1.0)
    n_ships = round(fraction * ships_available)
    n_ships = min(max(n_ships, 1), ships_available)
    return True, angle, n_ships


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
def split_bucket(
    group_key: str,
    train_fraction: float = 0.80,
    valid_fraction: float = 0.10,
) -> str:
    h = int(hashlib.md5(group_key.encode()).hexdigest(), 16) % 1000
    if h < int(train_fraction * 1000):
        return "train"
    if h < int((train_fraction + valid_fraction) * 1000):
        return "valid"
    return "test"


def make_split(games: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    games = games.copy()
    games["group_key"] = (
        games["dataset"].astype(str)
        + ":"
        + games["n_players"].astype(str)
        + ":"
        + games["seed"].astype(str)
    )
    games["split"] = games["group_key"].map(
        lambda g: split_bucket(g, cfg.train_fraction, cfg.valid_fraction)
    )
    return games


# --------------------------------------------------------------------------- #
# Stage A: episode cache builder
# --------------------------------------------------------------------------- #
def _planet_meta_from_episode(episode_planets: pd.DataFrame) -> dict:
    meta = {}
    for row in episode_planets.itertuples(index=False):
        pid = int(row.planet_id)
        meta[pid] = {
            "initial_x": float(row.initial_x),
            "initial_y": float(row.initial_y),
            "radius": float(row.radius),
            "production": int(row.production),
            "orbit_radius": float(row.orbit_radius),
            "is_static": bool(row.is_static),
            "is_comet": bool(row.is_comet),
        }
    return meta


def build_episode_tensors(
    game_row: pd.Series,
    episode_planets: pd.DataFrame,
    planet_state: pd.DataFrame,
    actions: pd.DataFrame,
    feature_cfg: FeatureConfig,
) -> Dict[str, torch.Tensor]:
    winner = int(game_row["winner_slot"])
    n_players = int(game_row["n_players"])
    angular_velocity = float(game_row["angular_velocity"])
    n_ticks = int(game_row["final_step"]) + 1
    opponent_map = build_opponent_map(n_players, winner)

    meta = _planet_meta_from_episode(episode_planets)
    is_comet = np.zeros(P_MAX, dtype=bool)
    for pid, m in meta.items():
        if 0 <= pid < P_MAX:
            is_comet[pid] = m["is_comet"]

    ps = planet_state.sort_values(["tick", "planet_id"])
    owners = np.full((n_ticks, P_MAX), -1, dtype=np.int8)
    ships = np.zeros((n_ticks, P_MAX), dtype=np.float32)
    active = np.zeros((n_ticks, P_MAX), dtype=bool)

    for row in ps.itertuples(index=False):
        t = int(row.tick)
        pid = int(row.planet_id)
        if 0 <= pid < P_MAX and 0 <= t < n_ticks:
            owners[t, pid] = int(row.owner)
            ships[t, pid] = float(row.ships)
            active[t, pid] = True

    planet_features = np.zeros((n_ticks, P_MAX, PLANET_FEATURE_DIM), dtype=np.float32)
    source_mask = np.zeros((n_ticks, P_MAX), dtype=bool)
    global_features = np.zeros((n_ticks, GLOBAL_FEATURE_DIM), dtype=np.float32)

    for t in range(n_ticks):
        for pid in range(P_MAX):
            if not active[t, pid] and pid not in meta:
                continue
            m = meta.get(pid)
            if m is None:
                continue

            if m["is_static"] or m["orbit_radius"] <= 0:
                x, y = m["initial_x"], m["initial_y"]
                is_orbiting = 0.0
            else:
                x, y = orbit_position(
                    m["initial_x"],
                    m["initial_y"],
                    m["orbit_radius"],
                    angular_velocity,
                    t,
                )
                is_orbiting = 1.0

            owner_enc, opp_norm = encode_owner(
                int(owners[t, pid]), winner, opponent_map
            )
            prod = m["production"]
            radius = m["radius"]
            ships_log = math.log1p(ships[t, pid]) / math.log1p(PLANET_SHIP_SCALE)

            planet_features[t, pid] = np.array(
                [
                    x / 100.0,
                    y / 100.0,
                    radius / MAX_PLANET_RADIUS,
                    is_orbiting,
                    float(m["is_comet"]),
                    m["orbit_radius"] / 50.0,
                    *owner_enc,
                    opp_norm,
                    ships_log,
                    (prod - 1.0) / 4.0,
                ],
                dtype=np.float32,
            )

        source_mask[t] = (
            active[t]
            & (owners[t] == winner)
            & (ships[t] >= 1.0)
        )
        global_features[t] = np.array(
            encode_global_features(
                t, angular_velocity, n_players, active[t], is_comet
            ),
            dtype=np.float32,
        )

    winner_actions = actions[actions["slot"] == winner]
    dense_actions = np.zeros((n_ticks, P_MAX, ACTION_DIM), dtype=np.float32)
    action_source_mask = source_mask.copy()

    # Off-by-one: the replay logs an action under the tick of the state it
    # *produced* (steps[t].action was decided from steps[t-1].observation, and
    # the parser skips tick 0). So the expert response to observation t is the
    # row at actions.tick == t+1, and its source ships come from ships[t].
    skipped_launches = 0
    for t in range(n_ticks):
        tick_rows = winner_actions[winner_actions["tick"] == t + 1]
        dense_actions[t], skipped = build_dense_actions_for_tick(
            tick_rows,
            ships[t],
            source_mask[t],
            episode_id=int(game_row["episode_id"]),
            tick=t,
        )
        skipped_launches += skipped

    done = np.zeros(n_ticks, dtype=np.float32)
    if n_ticks > 0:
        done[-1] = 1.0

    cache = {
        "episode_id": torch.tensor(int(game_row["episode_id"]), dtype=torch.int64),
        "global_features": torch.from_numpy(global_features),
        "planet_features": torch.from_numpy(planet_features),
        "planet_active_mask": torch.from_numpy(active),
        "planet_source_mask": torch.from_numpy(source_mask),
        "actions": torch.from_numpy(dense_actions),
        "action_source_mask": torch.from_numpy(action_source_mask),
        "done": torch.from_numpy(done),
        "ships_at_tick": torch.from_numpy(ships),
    }
    validate_episode_cache(cache)
    cache["_skipped_launches"] = int(skipped_launches)
    return cache


def validate_episode_cache(cache: Dict[str, torch.Tensor]) -> None:
    t = cache["actions"].shape[0]
    assert cache["global_features"].shape == (t, GLOBAL_FEATURE_DIM)
    assert cache["planet_features"].shape == (t, P_MAX, PLANET_FEATURE_DIM)
    assert cache["actions"].shape == (t, P_MAX, ACTION_DIM)
    launches = cache["actions"][..., 0] > 0
    assert torch.all(~launches | cache["action_source_mask"])
    for key, val in cache.items():
        if torch.is_floating_point(val):
            assert torch.isfinite(val).all(), key


# --------------------------------------------------------------------------- #
# Packed memmap store (replaces 75k per-episode .pt files)
#
# Every tick has a fixed shape, so we flatten *all* ticks across *all* episodes
# into a handful of contiguous .npy arrays (one per feature) and memory-map
# them. Build = sequential big writes (no 75k tiny zip archives + metadata
# ops). Load = O(1) slice into mmap with zero deserialization, which makes the
# per-tick random sampling in OrbitWarsDataset cheap and lets DataLoader scale
# across num_workers. A `samples.parquet` manifest carries the per-tick global
# index `gi` plus its episode's `[ep_start, ep_start+ep_len)` span so horizon
# windows never cross episode boundaries.
# --------------------------------------------------------------------------- #
# name -> (numpy dtype, per-tick trailing shape). The leading dim is the global
# tick axis T (sum of n_ticks over episodes).
PACKED_ARRAYS: Dict[str, Tuple[str, Tuple[int, ...]]] = {
    "global_features": ("float32", (GLOBAL_FEATURE_DIM,)),
    "planet_features": ("float32", (P_MAX, PLANET_FEATURE_DIM)),
    "planet_active_mask": ("bool", (P_MAX,)),
    "planet_source_mask": ("bool", (P_MAX,)),
    "actions": ("float32", (P_MAX, ACTION_DIM)),
    "action_source_mask": ("bool", (P_MAX,)),
    "done": ("float32", ()),
    "ships_at_tick": ("float32", (P_MAX,)),
    "has_any_launch": ("bool", ()),
}

# Populated in the parent *before* forking workers so children inherit them
# copy-on-write (Linux fork start method). Workers only read these.
_BUILD_STATE: Dict[str, Any] = {}
# Per-worker writable memmap handles, opened in the pool initializer.
_BUILD_MMAPS: Dict[str, np.ndarray] = {}


def _packed_path(cache_dir: pathlib.Path, name: str) -> pathlib.Path:
    return cache_dir / f"{name}.npy"


def _create_packed(cache_dir: pathlib.Path, total: int) -> None:
    """Preallocate the packed .npy files (sparse) and flush their headers."""
    for name, (dtype, shape) in PACKED_ARRAYS.items():
        mm = np.lib.format.open_memmap(
            _packed_path(cache_dir, name),
            mode="w+",
            dtype=np.dtype(dtype),
            shape=(total, *shape),
        )
        mm.flush()
        del mm


def _open_packed(cache_dir: pathlib.Path, mode: str = "r") -> Dict[str, np.ndarray]:
    return {
        name: np.load(_packed_path(cache_dir, name), mmap_mode=mode)
        for name in PACKED_ARRAYS
    }


def _write_episode(mm: Dict[str, np.ndarray], gi: int, cache: Dict[str, torch.Tensor]) -> int:
    """Write one episode's tensors into the global slice [gi, gi+n)."""
    n = int(cache["actions"].shape[0])
    sl = slice(gi, gi + n)
    mm["global_features"][sl] = cache["global_features"].numpy()
    mm["planet_features"][sl] = cache["planet_features"].numpy()
    mm["planet_active_mask"][sl] = cache["planet_active_mask"].numpy()
    mm["planet_source_mask"][sl] = cache["planet_source_mask"].numpy()
    mm["actions"][sl] = cache["actions"].numpy()
    mm["action_source_mask"][sl] = cache["action_source_mask"].numpy()
    mm["done"][sl] = cache["done"].numpy()
    mm["ships_at_tick"][sl] = cache["ships_at_tick"].numpy()
    mm["has_any_launch"][sl] = (cache["actions"][..., 0] > 0).any(dim=1).numpy()
    return n


def _build_one(eid: int, gi: int) -> dict:
    """Build + write a single episode. Uses inherited _BUILD_STATE / _BUILD_MMAPS."""
    st = _BUILD_STATE
    game = st["game_rows"][eid]
    ep_pl = st["pl_groups"].get(eid, st["empty_pl"])
    ep_ps = st["ps_groups"].get(eid, st["empty_ps"])
    ep_ac = st["ac_groups"].get(eid, st["empty_ac"])
    try:
        cache = build_episode_tensors(game, ep_pl, ep_ps, ep_ac, st["feature_cfg"])
    except AssertionError as exc:
        # One malformed episode shouldn't abort the whole build; its reserved
        # block stays zero and is excluded from the manifest.
        return {"eid": eid, "ok": False, "err": str(exc) or "assertion failed"}
    skipped = int(cache.pop("_skipped_launches", 0))
    n = _write_episode(_BUILD_MMAPS, gi, cache)
    return {"eid": eid, "ok": True, "gi": gi, "n": n, "skipped": skipped}


def _worker_init(cache_dir_str: str) -> None:
    global _BUILD_MMAPS
    _BUILD_MMAPS = _open_packed(pathlib.Path(cache_dir_str), mode="r+")


def _worker_task(task: Tuple[int, int]) -> dict:
    return _build_one(task[0], task[1])


def build_episode_cache(
    dataset_dir: pathlib.Path,
    cache_dir: pathlib.Path,
    feature_cfg: Optional[FeatureConfig] = None,
    train_cfg: Optional[TrainConfig] = None,
    workers: Optional[int] = None,
) -> pd.DataFrame:
    feature_cfg = feature_cfg or FeatureConfig()
    train_cfg = train_cfg or TrainConfig(dataset_dir=dataset_dir, cache_dir=cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    games = pd.read_parquet(dataset_dir / "selected_games.parquet")
    games = make_split(games, train_cfg).reset_index(drop=True)

    ep_planets = pd.read_parquet(dataset_dir / "games" / "episode_planets.parquet")
    planet_state = pd.read_parquet(dataset_dir / "games" / "planet_state.parquet")
    actions = pd.read_parquet(dataset_dir / "games" / "actions.parquet")

    # Group each frame by episode_id once (single O(N) pass) instead of
    # re-scanning the full frame per episode. The naive per-episode boolean
    # mask is O(episodes x rows) and is intractable at this dataset's scale
    # (~75k episodes x ~447M planet_state rows). Empty frames preserve column
    # schema for episodes with no matching rows.
    ps_groups = {eid: g for eid, g in planet_state.groupby("episode_id", sort=False)}
    ac_groups = {eid: g for eid, g in actions.groupby("episode_id", sort=False)}
    pl_groups = {eid: g for eid, g in ep_planets.groupby("episode_id", sort=False)}
    empty_ps = planet_state.iloc[0:0]
    empty_ac = actions.iloc[0:0]
    empty_pl = ep_planets.iloc[0:0]
    del planet_state, actions, ep_planets

    n_games = len(games)
    # n_ticks per episode is fixed (final_step + 1), so we can lay out the
    # global tick axis and assign every episode a disjoint write region up
    # front -> the build is embarrassingly parallel (workers write
    # non-overlapping slices of the same memmap, no locking needed).
    n_ticks_per = (games["final_step"].astype(np.int64) + 1).to_numpy()
    offsets = np.zeros(n_games + 1, dtype=np.int64)
    np.cumsum(n_ticks_per, out=offsets[1:])
    total = int(offsets[-1])

    eids = games["episode_id"].astype(np.int64).to_numpy()
    splits_per_ep = games["split"].astype(str).to_numpy()
    game_rows = {int(eids[i]): games.iloc[i] for i in range(n_games)}

    global _BUILD_STATE
    _BUILD_STATE = {
        "game_rows": game_rows,
        "ps_groups": ps_groups,
        "ac_groups": ac_groups,
        "pl_groups": pl_groups,
        "empty_ps": empty_ps,
        "empty_ac": empty_ac,
        "empty_pl": empty_pl,
        "feature_cfg": feature_cfg,
    }

    _create_packed(cache_dir, total)

    tasks = [(int(eids[i]), int(offsets[i])) for i in range(n_games)]
    workers = workers or (os.cpu_count() or 1)
    workers = max(1, min(workers, n_games))
    log_every = max(1, n_games // 100)
    logger.info(
        "build-cache: %d episodes, %d total ticks, %d workers -> %s",
        n_games, total, workers, cache_dir,
    )

    results: List[dict] = []

    def _maybe_log(done: int) -> None:
        if done % log_every == 0 or done == n_games:
            ok = sum(1 for r in results if r["ok"])
            skipped = sum(r.get("skipped", 0) for r in results)
            failed = done - ok
            logger.info(
                "build-cache progress: %d/%d episodes (%.1f%%), "
                "skipped_launches=%d, failed=%d",
                done, n_games, 100.0 * done / n_games, skipped, failed,
            )

    if workers == 1:
        # Serial path (debuggable; also the fallback when only one episode).
        global _BUILD_MMAPS
        _BUILD_MMAPS = _open_packed(cache_dir, mode="r+")
        for k, (eid, gi) in enumerate(tasks, start=1):
            results.append(_build_one(eid, gi))
            _maybe_log(k)
    else:
        # fork so children inherit _BUILD_STATE (the big group dicts) via
        # copy-on-write instead of pickling ~GBs across the pipe.
        ctx = mp.get_context("fork")
        with ctx.Pool(
            workers, initializer=_worker_init, initargs=(str(cache_dir),)
        ) as pool:
            for k, res in enumerate(
                pool.imap_unordered(_worker_task, tasks, chunksize=32), start=1
            ):
                results.append(res)
                _maybe_log(k)

    _BUILD_STATE = {}

    succ = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    succ.sort(key=lambda r: r["gi"])

    eid_by_idx = {int(eids[i]): i for i in range(n_games)}
    n_arr = np.fromiter((r["n"] for r in succ), dtype=np.int64, count=len(succ))
    gi0 = np.fromiter((r["gi"] for r in succ), dtype=np.int64, count=len(succ))
    succ_eids = np.fromiter((r["eid"] for r in succ), dtype=np.int64, count=len(succ))
    succ_splits = np.array([splits_per_ep[eid_by_idx[int(e)]] for e in succ_eids])

    # Vectorized per-tick manifest expansion (13.5M rows) — avoids 75k DataFrame
    # concats. base = start index of each episode in the concatenated stream.
    ep_start_all = np.repeat(gi0, n_arr)
    ep_len_all = np.repeat(n_arr, n_arr)
    eid_all = np.repeat(succ_eids, n_arr)
    split_all = np.repeat(succ_splits, n_arr)
    base = np.repeat(np.cumsum(n_arr) - n_arr, n_arr)
    tick_all = np.arange(ep_start_all.size, dtype=np.int64) - base
    gi_all = ep_start_all + tick_all

    has_mm = np.load(_packed_path(cache_dir, "has_any_launch"), mmap_mode="r")
    launch_all = np.asarray(has_mm[gi_all], dtype=bool)
    del has_mm

    samples = pd.DataFrame(
        {
            "episode_id": eid_all,
            "tick": tick_all,
            "split": pd.Categorical(split_all),
            "has_any_launch": launch_all,
            "gi": gi_all,
            "ep_start": ep_start_all,
            "ep_len": ep_len_all,
        }
    )
    splits = pd.DataFrame({"episode_id": succ_eids, "split": succ_splits})
    samples.to_parquet(cache_dir / "samples.parquet", index=False)
    splits.to_parquet(cache_dir / "splits.parquet", index=False)
    (cache_dir / "feature_config.json").write_text(
        json.dumps(feature_cfg.to_dict(), indent=2)
    )
    (cache_dir / "packed_meta.json").write_text(
        json.dumps(
            {
                "n_ticks_alloc": total,
                "n_ticks_used": int(n_arr.sum()),
                "n_episodes": len(succ),
                "arrays": {
                    name: {"dtype": dt, "shape": [total, *sh]}
                    for name, (dt, sh) in PACKED_ARRAYS.items()
                },
            },
            indent=2,
        )
    )
    total_skipped = sum(r.get("skipped", 0) for r in succ)
    episodes_with_skips = sum(1 for r in succ if r.get("skipped", 0))
    print(
        f"Cached {len(splits)} episodes, {len(samples)} samples -> {cache_dir} "
        f"(skipped {total_skipped} launches in {episodes_with_skips} episodes, "
        f"{len(failed)} episodes failed)"
    )
    if failed:
        logger.warning(
            "%d episodes failed cache build: %s%s",
            len(failed),
            ", ".join(str(r["eid"]) for r in failed[:20]),
            " ..." if len(failed) > 20 else "",
        )
    return samples


# --------------------------------------------------------------------------- #
# Dataset + load/unload
# --------------------------------------------------------------------------- #
class PackedEpisodeStore:
    """Lazily-opened, per-process handle to the packed memmap arrays.

    The memmaps are opened on first access in whatever process touches them, so
    a DataLoader worker (forked or spawned) gets its own read-only mappings and
    we never pickle an open memmap across the process boundary. mmap is
    inherently shareable read-only, so this scales cleanly across num_workers.
    """

    def __init__(self, cache_dir: pathlib.Path):
        self.cache_dir = pathlib.Path(cache_dir)
        self._arrs: Optional[Dict[str, np.ndarray]] = None
        self._pid: Optional[int] = None

    def __getstate__(self) -> dict:
        # Don't carry open memmaps into a worker process.
        return {"cache_dir": self.cache_dir, "_arrs": None, "_pid": None}

    def arr(self, name: str) -> np.ndarray:
        pid = os.getpid()
        if self._arrs is None or self._pid != pid:
            self._arrs = _open_packed(self.cache_dir, mode="r")
            self._pid = pid
        return self._arrs[name]


class OrbitWarsDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str,
        cache_dir: pathlib.Path,
        horizon: int = ACTION_HORIZON,
    ):
        self.rows = manifest.loc[manifest["split"] == split].reset_index(drop=True)
        self.horizon = horizon
        self.store = PackedEpisodeStore(cache_dir)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.rows.iloc[index]
        gi = int(row["gi"])
        ep_start = int(row["ep_start"])
        ep_len = int(row["ep_len"])
        tick = int(row["tick"])

        # Clamp the horizon window to this episode's span so we never read into
        # the next episode's ticks (the arrays are globally contiguous).
        stop = min(gi + self.horizon, ep_start + ep_len)
        valid_length = stop - gi

        action_target = torch.zeros(self.horizon, P_MAX, ACTION_DIM)
        action_source_mask = torch.zeros(self.horizon, P_MAX, dtype=torch.bool)
        done_target = torch.zeros(self.horizon)
        time_valid_mask = torch.zeros(self.horizon, dtype=torch.bool)

        # np.array(...) forces a writable, contiguous copy off the mmap so the
        # resulting tensors own their memory (safe for pin_memory / collate).
        action_target[:valid_length] = torch.from_numpy(
            np.array(self.store.arr("actions")[gi:stop])
        )
        action_source_mask[:valid_length] = torch.from_numpy(
            np.array(self.store.arr("action_source_mask")[gi:stop])
        )
        done_target[:valid_length] = torch.from_numpy(
            np.array(self.store.arr("done")[gi:stop])
        )
        time_valid_mask[:valid_length] = True

        return {
            "episode_id": torch.tensor(int(row["episode_id"]), dtype=torch.int64),
            "tick": torch.tensor(tick, dtype=torch.int64),
            "global_features": torch.from_numpy(
                np.array(self.store.arr("global_features")[gi])
            ),
            "planet_features": torch.from_numpy(
                np.array(self.store.arr("planet_features")[gi])
            ),
            "planet_active_mask": torch.from_numpy(
                np.array(self.store.arr("planet_active_mask")[gi])
            ),
            "planet_source_mask": torch.from_numpy(
                np.array(self.store.arr("planet_source_mask")[gi])
            ),
            "action_target": action_target,
            "action_source_mask": action_source_mask,
            "time_valid_mask": time_valid_mask,
            "done_target": done_target,
        }


def make_balanced_sampler(rows: pd.DataFrame) -> WeightedRandomSampler:
    has_launch = rows["has_any_launch"].to_numpy(dtype=bool)
    n_launch = max(int(has_launch.sum()), 1)
    n_noop = max(int((~has_launch).sum()), 1)
    weights = np.where(has_launch, 0.5 / n_launch, 0.5 / n_noop)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(rows),
        replacement=True,
    )


def make_loaders(
    manifest: pd.DataFrame,
    cfg: TrainConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = OrbitWarsDataset(manifest, "train", cfg.cache_dir)
    valid_ds = OrbitWarsDataset(manifest, "valid", cfg.cache_dir)
    test_ds = OrbitWarsDataset(manifest, "test", cfg.cache_dir)

    # persistent_workers keeps each worker's memmaps open across epochs instead
    # of reopening them every epoch.
    persistent = cfg.num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=make_balanced_sampler(train_ds.rows),
        num_workers=cfg.num_workers,
        pin_memory=cfg.device.startswith("cuda"),
        persistent_workers=persistent,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device.startswith("cuda"),
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device.startswith("cuda"),
        persistent_workers=persistent,
    )
    return train_loader, valid_loader, test_loader


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class ObsEncoder(nn.Module):
    def __init__(self, feature_cfg: FeatureConfig, model_cfg: ModelConfig):
        super().__init__()
        d = model_cfg.d_model
        self.planet_proj = nn.Linear(feature_cfg.planet_feature_dim, d)
        self.global_proj = nn.Linear(feature_cfg.global_feature_dim, d)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=model_cfg.n_heads,
            dim_feedforward=d * 2,
            dropout=model_cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.planet_encoder = nn.TransformerEncoder(
            enc_layer, num_layers=model_cfg.n_layers, enable_nested_tensor=False
        )
        self.board_proj = nn.Linear(d, d)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        planet_tokens = self.planet_proj(batch["planet_features"])
        mask = batch["planet_active_mask"]
        key_padding = ~mask

        planet_tokens = self.planet_encoder(
            planet_tokens, src_key_padding_mask=key_padding
        )

        active_f = mask.unsqueeze(-1).float()
        denom = active_f.sum(dim=1).clamp(min=1.0)
        board = (planet_tokens * active_f).sum(dim=1) / denom
        board_token = self.board_proj(board).unsqueeze(1)

        global_token = self.global_proj(batch["global_features"]).unsqueeze(1)

        return {
            "planet_tokens": planet_tokens,
            "planet_mask": mask,
            "source_mask": batch["planet_source_mask"],
            "global_token": global_token,
            "board_token": board_token,
        }


class ActionDenoiser(nn.Module):
    """Error predictor: eps_theta = f(x_t, diffusion_t, observation)."""

    def __init__(
        self,
        feature_cfg: FeatureConfig,
        model_cfg: ModelConfig,
    ):
        super().__init__()
        self.horizon = feature_cfg.action_horizon
        self.p_max = feature_cfg.p_max
        self.action_dim = feature_cfg.action_dim
        d = model_cfg.d_model

        flat_in = self.horizon * self.p_max * self.action_dim
        self.time_embed = TimestepEmbedding(d, hidden_dim=d * 2)
        self.noisy_proj = nn.Linear(flat_in, d)
        self.cond_proj = nn.Linear(d * 2, d)
        self.film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d, d * 2),
        )
        self.out = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d, flat_in),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        b = x_t.shape[0]
        flat = x_t.reshape(b, -1)
        h = self.noisy_proj(flat)

        cond_vec = torch.cat(
            [cond["board_token"], cond["global_token"]], dim=-1
        ).squeeze(1)
        h = h + self.cond_proj(cond_vec)

        t_emb = self.time_embed(t)
        scale, shift = self.film(t_emb).chunk(2, dim=-1)
        h = h * (1.0 + scale) + shift

        out = self.out(h).reshape(b, self.horizon, self.p_max, self.action_dim)
        return out


def build_model(
    feature_cfg: FeatureConfig,
    model_cfg: ModelConfig,
) -> Tuple[ObsEncoder, ActionDenoiser]:
    return ObsEncoder(feature_cfg, model_cfg), ActionDenoiser(feature_cfg, model_cfg)


# --------------------------------------------------------------------------- #
# Loss mask
# --------------------------------------------------------------------------- #
def build_loss_mask(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Mask over [B, H, P]: score noise only on eligible source planets within
    valid horizon steps. v1 uses this single gate for all 4 action features;
    splitting gate vs angle/ships supervision is a later refinement."""
    source = batch["action_source_mask"].float()
    time_valid = batch["time_valid_mask"].float().unsqueeze(-1)
    return source * time_valid


# --------------------------------------------------------------------------- #
# Train / validate / sample
# --------------------------------------------------------------------------- #
def _move_batch(batch: Dict[str, torch.Tensor], device: torch.device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def evaluate(
    ddpm: DDPM,
    encoder: ObsEncoder,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    ddpm.eval_mode()
    encoder.eval()

    tp = fp = fn = 0
    angle_errs: List[float] = []
    frac_errs: List[float] = []

    for batch in loader:
        batch = _move_batch(batch, device)
        x_0 = batch["action_target"]
        cond = encoder(batch)

        pred = ddpm.sample(
            batch_size=x_0.shape[0],
            image_shape=tuple(x_0.shape[1:]),
            cond=cond,
        )

        src = batch["action_source_mask"] & batch["time_valid_mask"].unsqueeze(-1)
        tgt_launch = batch["action_target"][..., 0] > 0
        pred_launch = pred[..., 0] > 0

        tp += int((pred_launch & tgt_launch & src).sum())
        fp += int((pred_launch & ~tgt_launch & src).sum())
        fn += int((~pred_launch & tgt_launch & src).sum())

        launch_mask = tgt_launch & src
        if launch_mask.any():
            tgt_a = batch["action_target"][..., 1:3]
            pred_a = pred[..., 1:3]
            dot = (tgt_a * pred_a).sum(-1)
            cross = tgt_a[..., 0] * pred_a[..., 1] - tgt_a[..., 1] * pred_a[..., 0]
            ang = torch.abs(torch.atan2(cross, dot.clamp(-1, 1)))
            angle_errs.extend(ang[launch_mask].cpu().tolist())

            tgt_frac = (batch["action_target"][..., 3] + 1) / 2
            pred_frac = (pred[..., 3] + 1) / 2
            frac_errs.extend(
                torch.abs(tgt_frac - pred_frac)[launch_mask].cpu().tolist()
            )

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "launch_precision": precision,
        "launch_recall": recall,
        "launch_f1": f1,
        "angle_mae_rad": float(np.mean(angle_errs)) if angle_errs else 0.0,
        "frac_mae": float(np.mean(frac_errs)) if frac_errs else 0.0,
    }


def sample_actions(
    ddpm: DDPM,
    encoder: ObsEncoder,
    obs_batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Generate action chunk; execute only timestep 0 at deployment."""
    ddpm.eval_mode()
    encoder.eval()
    with torch.no_grad():
        cond = encoder(obs_batch)
        b = obs_batch["global_features"].shape[0]
        shape = (ACTION_HORIZON, P_MAX, ACTION_DIM)
        return ddpm.sample(batch_size=b, image_shape=shape, cond=cond)


def _init_wandb(
    train_cfg: TrainConfig,
    feature_cfg: FeatureConfig,
    model_cfg: ModelConfig,
    n_train: int,
    n_valid: int,
    n_params: int,
):
    """Return an active wandb run or None. Disabled cleanly if the flag is off
    or the package isn't installed (training still runs + logs to stdout)."""
    if not train_cfg.use_wandb:
        return None
    if wandb is None:
        logger.warning("use_wandb=True but wandb is not installed; skipping.")
        return None
    run = wandb.init(
        project=train_cfg.wandb_project,
        entity=train_cfg.wandb_entity,
        name=train_cfg.wandb_run_name,
        config={
            "train": train_cfg.to_dict(),
            "feature": feature_cfg.to_dict(),
            "model": model_cfg.to_dict(),
            "n_train_samples": n_train,
            "n_valid_samples": n_valid,
            "n_params": n_params,
        },
    )
    logger.info("wandb run: %s", getattr(run, "url", run.id))
    return run


def train(
    train_cfg: TrainConfig,
    feature_cfg: FeatureConfig,
    model_cfg: ModelConfig,
) -> None:
    device = torch.device(train_cfg.device)
    manifest = pd.read_parquet(train_cfg.cache_dir / "samples.parquet")
    train_loader, valid_loader, _ = make_loaders(manifest, train_cfg)

    encoder, denoiser = build_model(feature_cfg, model_cfg)
    encoder = encoder.to(device)
    denoiser = denoiser.to(device)

    # The denoiser IS the noise predictor (the U-Net replacement): its forward
    # signature is (x_t, t, cond), exactly what the conditional DDPM expects.
    ddpm = DDPM(
        denoiser,
        device,
        num_timesteps=model_cfg.num_timesteps,
        beta_start=model_cfg.beta_start,
        beta_end=model_cfg.beta_end,
    )
    ddpm.model_cfg = model_cfg.to_dict()
    ddpm.feature_cfg = feature_cfg.to_dict()

    params = list(encoder.parameters()) + list(denoiser.parameters())
    n_params = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=train_cfg.lr)
    global_step = 0
    train_cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    n_train = len(train_loader.dataset)
    n_valid = len(valid_loader.dataset)
    steps_per_epoch = len(train_loader)
    logger.info(
        "train: device=%s params=%d train_samples=%d valid_samples=%d "
        "batch_size=%d steps/epoch=%d epochs=%d -> %s",
        device, n_params, n_train, n_valid, train_cfg.batch_size,
        steps_per_epoch, train_cfg.epochs, train_cfg.checkpoint_dir,
    )

    run = _init_wandb(train_cfg, feature_cfg, model_cfg, n_train, n_valid, n_params)

    def _save(path: pathlib.Path) -> None:
        t0 = time.perf_counter()
        save_checkpoint(
            path, ddpm, encoder, denoiser, opt, global_step,
            model_cfg, feature_cfg,
        )
        size_mb = path.stat().st_size / 1e6
        logger.info(
            "checkpoint saved: %s (step=%d, %.1f MB, %.2fs)",
            path, global_step, size_mb, time.perf_counter() - t0,
        )
        if run is not None:
            run.log({"checkpoint/size_mb": size_mb}, step=global_step)

    try:
        for epoch in range(train_cfg.epochs):
            encoder.train()
            denoiser.train()
            # Throughput is measured over each log window (reset at every log).
            window_t0 = time.perf_counter()
            window_start_step = global_step
            for batch in train_loader:
                batch = _move_batch(batch, device)
                x_0 = batch["action_target"]
                cond = encoder(batch)
                loss_mask = build_loss_mask(batch)
                loss, metrics = ddpm.compute_loss(
                    x_0, cond=cond, loss_mask=loss_mask
                )

                opt.zero_grad()
                loss.backward()
                opt.step()
                global_step += 1

                if global_step % train_cfg.log_every == 0:
                    elapsed = max(time.perf_counter() - window_t0, 1e-9)
                    n_steps = global_step - window_start_step
                    steps_per_sec = n_steps / elapsed
                    samples_per_sec = steps_per_sec * train_cfg.batch_size
                    # Fraction of action slots actually scored this batch — near
                    # zero means the loss is supervised on almost nothing.
                    mask_cov = float(loss_mask.mean())
                    logger.info(
                        "epoch=%d step=%d mse=%.4f mask_cov=%.4f "
                        "%.1f steps/s %.0f samples/s",
                        epoch, global_step, metrics["mse"], mask_cov,
                        steps_per_sec, samples_per_sec,
                    )
                    if run is not None:
                        run.log(
                            {
                                "train/mse": metrics["mse"],
                                "train/loss": float(loss.item()),
                                "train/mask_coverage": mask_cov,
                                "train/lr": opt.param_groups[0]["lr"],
                                "perf/steps_per_sec": steps_per_sec,
                                "perf/samples_per_sec": samples_per_sec,
                                "epoch": epoch,
                            },
                            step=global_step,
                        )
                    window_t0 = time.perf_counter()
                    window_start_step = global_step

                if global_step % train_cfg.save_every == 0:
                    _save(train_cfg.checkpoint_dir / f"step_{global_step}.pt")

            val_metrics = evaluate(ddpm, encoder, valid_loader, device)
            logger.info("epoch=%d valid %s", epoch, val_metrics)
            if run is not None:
                run.log(
                    {f"valid/{k}": v for k, v in val_metrics.items()}
                    | {"epoch": epoch},
                    step=global_step,
                )

        _save(train_cfg.checkpoint_dir / "latest.pt")
    finally:
        if run is not None:
            run.finish()


# --------------------------------------------------------------------------- #
# Checkpoint save / load
# --------------------------------------------------------------------------- #
def save_checkpoint(
    path: pathlib.Path,
    ddpm: DDPM,
    encoder: ObsEncoder,
    denoiser: ActionDenoiser,
    optimizer: torch.optim.Optimizer,
    step: int,
    model_cfg: ModelConfig,
    feature_cfg: FeatureConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "denoiser": denoiser.state_dict(),
            "method_cfg": ddpm.method_cfg,
            "model_cfg": model_cfg.to_dict(),
            "feature_cfg": feature_cfg.to_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        path,
    )


def load_checkpoint(
    path: pathlib.Path,
    device: Optional[torch.device] = None,
) -> Tuple[DDPM, ObsEncoder, ActionDenoiser, dict]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    feature_cfg = FeatureConfig.from_dict(ckpt["feature_cfg"])
    model_cfg = ModelConfig.from_dict(ckpt["model_cfg"])
    encoder, denoiser = build_model(feature_cfg, model_cfg)
    encoder.load_state_dict(ckpt["encoder"])
    denoiser.load_state_dict(ckpt["denoiser"])
    encoder.to(device)
    denoiser.to(device)

    ddpm = DDPM.from_config(denoiser, ckpt["method_cfg"], device)
    ddpm.model_cfg = ckpt["model_cfg"]
    ddpm.feature_cfg = ckpt["feature_cfg"]
    return ddpm, encoder, denoiser, ckpt


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Orbit Wars conditional DDPM")
    ap.add_argument(
        "command",
        choices=["build-cache", "train", "eval", "sample"],
        help="pipeline stage",
    )
    ap.add_argument("--dataset-dir", type=pathlib.Path, default=REPO_ROOT / "dataset_out")
    ap.add_argument("--cache-dir", type=pathlib.Path, default=REPO_ROOT / "cache")
    ap.add_argument("--checkpoint", type=pathlib.Path, default=REPO_ROOT / "checkpoints" / "latest.pt")
    ap.add_argument("--config", type=pathlib.Path, help="optional orbit_train.yaml")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="parallel processes for build-cache (default: all CPUs)",
    )
    ap.add_argument("--wandb", action="store_true", help="log training to wandb")
    ap.add_argument("--wandb-project", type=str, default=None)
    ap.add_argument("--wandb-entity", type=str, default=None)
    ap.add_argument("--wandb-run-name", type=str, default=None)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    feature_cfg = FeatureConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig(
        dataset_dir=args.dataset_dir,
        cache_dir=args.cache_dir,
    )

    if args.config and args.config.exists():
        raw = yaml.safe_load(args.config.read_text())
        if "feature" in raw:
            feature_cfg = FeatureConfig.from_dict(raw["feature"])
        if "model" in raw:
            model_cfg = ModelConfig.from_dict(raw["model"])
        if "train" in raw:
            train_cfg = TrainConfig.from_dict({**train_cfg.to_dict(), **raw["train"]})

    if args.epochs is not None:
        train_cfg.epochs = args.epochs
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.device:
        train_cfg.device = args.device
    if args.wandb:
        train_cfg.use_wandb = True
    if args.wandb_project:
        train_cfg.wandb_project = args.wandb_project
    if args.wandb_entity:
        train_cfg.wandb_entity = args.wandb_entity
    if args.wandb_run_name:
        train_cfg.wandb_run_name = args.wandb_run_name

    if args.command == "build-cache":
        build_episode_cache(
            args.dataset_dir, args.cache_dir, feature_cfg, train_cfg,
            workers=args.workers,
        )
    elif args.command == "train":
        train(train_cfg, feature_cfg, model_cfg)
    elif args.command == "eval":
        device = torch.device(train_cfg.device)
        ddpm, encoder, _, _ = load_checkpoint(args.checkpoint, device)
        manifest = pd.read_parquet(args.cache_dir / "samples.parquet")
        _, valid_loader, _ = make_loaders(manifest, train_cfg)
        print(evaluate(ddpm, encoder, valid_loader, device))
    elif args.command == "sample":
        device = torch.device(train_cfg.device)
        ddpm, encoder, _, _ = load_checkpoint(args.checkpoint, device)
        manifest = pd.read_parquet(args.cache_dir / "samples.parquet")
        ds = OrbitWarsDataset(manifest, "valid", args.cache_dir)
        batch = ds[0]
        batch = {k: v.unsqueeze(0).to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        actions = sample_actions(ddpm, encoder, batch)
        print("sampled action shape:", tuple(actions.shape))
        # Only eligible source planets (owned, >=1 ship) can actually launch at
        # deployment, so gate the readout by the turn-0 source mask.
        turn0_sources = batch["planet_source_mask"][0]
        launches = (actions[0, 0, :, 0] > 0) & turn0_sources
        print("turn-0 launches (planet_id -> gate):",
              launches.nonzero(as_tuple=False).squeeze(-1).tolist())


if __name__ == "__main__":
    main()
