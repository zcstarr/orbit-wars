"""Deploy-time DDPM action adapter for the Orbit Wars harness."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from train_orbit import (
    GLOBAL_FEATURE_DIM,
    MAX_PLANET_RADIUS,
    P_MAX,
    PLANET_FEATURE_DIM,
    PLANET_SHIP_SCALE,
    build_opponent_map,
    decode_action,
    encode_global_features,
    encode_owner,
    load_checkpoint,
    sample_actions,
)

SUN_X = SUN_Y = 50.0
ROTATION_RADIUS_LIMIT = 50.0
BOARD_MIN = 0.0
BOARD_MAX = 100.0
SUN_RADIUS = 10.0


@dataclass(frozen=True)
class DDPMActConfig:
    checkpoint_path: Path
    device: str | None = None
    launch_threshold: float = 0.0
    max_moves: int | None = None
    resample_attempts: int = 4


@dataclass
class ObsBatch:
    batch: dict[str, torch.Tensor]
    source_mask: np.ndarray
    ships_by_pid: np.ndarray
    active_mask: np.ndarray
    x_by_pid: np.ndarray
    y_by_pid: np.ndarray
    radius_by_pid: np.ndarray


def _get(obs: Any, key: str, default: Any = None) -> Any:
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _raw_planets(obs: Any, key: str) -> list[Any]:
    return list(_get(obs, key, []) or [])


def _row_value(row: Any, index: int, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    if hasattr(row, name):
        return getattr(row, name)
    try:
        return row[index]
    except (IndexError, TypeError):
        return default


def _planet_id(row: Any) -> int:
    return int(_row_value(row, 0, "id", -1))


def _owner(row: Any) -> int:
    return int(_row_value(row, 1, "owner", -1))


def _infer_n_players(obs: Any) -> int:
    owners: set[int] = {
        _owner(p) for p in _raw_planets(obs, "initial_planets") if _owner(p) >= 0
    }
    if not owners:
        owners.update(_owner(p) for p in _raw_planets(obs, "planets") if _owner(p) >= 0)
    if not owners:
        owners.update(_owner(f) for f in _raw_planets(obs, "fleets") if _owner(f) >= 0)
    return 4 if any(owner >= 2 for owner in owners) else 2


def _ray_circle_distance(
    ox: float,
    oy: float,
    dx: float,
    dy: float,
    cx: float,
    cy: float,
    radius: float,
) -> float | None:
    fx = ox - cx
    fy = oy - cy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - radius * radius
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return None
    root = math.sqrt(disc)
    t1 = (-b - root) / 2.0
    t2 = (-b + root) / 2.0
    hits = [t for t in (t1, t2) if t >= 0.0]
    return min(hits) if hits else None


def _ray_board_exit_distance(ox: float, oy: float, dx: float, dy: float) -> float:
    distances: list[float] = []
    if dx > 0.0:
        distances.append((BOARD_MAX - ox) / dx)
    elif dx < 0.0:
        distances.append((BOARD_MIN - ox) / dx)
    if dy > 0.0:
        distances.append((BOARD_MAX - oy) / dy)
    elif dy < 0.0:
        distances.append((BOARD_MIN - oy) / dy)
    positive = [d for d in distances if d >= 0.0]
    return min(positive) if positive else float("inf")


def _launch_hits_planet_before_hazard(
    obs_batch: ObsBatch,
    source_pid: int,
    angle: float,
) -> bool:
    dx = math.cos(angle)
    dy = math.sin(angle)
    sx = float(obs_batch.x_by_pid[source_pid])
    sy = float(obs_batch.y_by_pid[source_pid])
    sr = float(obs_batch.radius_by_pid[source_pid])

    # Match the engine's launch offset so the ray does not immediately hit its
    # origin planet.
    ox = sx + dx * (sr + 0.1)
    oy = sy + dy * (sr + 0.1)
    board_t = _ray_board_exit_distance(ox, oy, dx, dy)
    sun_t = _ray_circle_distance(ox, oy, dx, dy, SUN_X, SUN_Y, SUN_RADIUS)
    hazard_t = min(board_t, sun_t if sun_t is not None else float("inf"))

    planet_t = float("inf")
    for pid in np.flatnonzero(obs_batch.active_mask):
        if int(pid) == source_pid:
            continue
        hit_t = _ray_circle_distance(
            ox,
            oy,
            dx,
            dy,
            float(obs_batch.x_by_pid[pid]),
            float(obs_batch.y_by_pid[pid]),
            float(obs_batch.radius_by_pid[pid]),
        )
        if hit_t is not None and hit_t < planet_t:
            planet_t = hit_t

    # The engine checks planet hits before sun/bounds for a tick; allow ties in
    # favor of planets to match that precedence.
    return planet_t <= hazard_t


def _obs_to_batch(obs: Any, device: torch.device) -> ObsBatch:
    player = int(_get(obs, "player", 0))
    step = int(_get(obs, "step", 0))
    angular_velocity = float(_get(obs, "angular_velocity", 0.0))
    n_players = _infer_n_players(obs)
    opponent_map = build_opponent_map(n_players, player)
    comet_ids = {int(pid) for pid in (_get(obs, "comet_planet_ids", []) or [])}

    owners = np.full(P_MAX, -1, dtype=np.int64)
    ships = np.zeros(P_MAX, dtype=np.float32)
    ships_by_pid = np.zeros(P_MAX, dtype=np.int64)
    active_mask = np.zeros(P_MAX, dtype=bool)
    is_comet = np.zeros(P_MAX, dtype=bool)
    initial_x = np.zeros(P_MAX, dtype=np.float32)
    initial_y = np.zeros(P_MAX, dtype=np.float32)
    current_x = np.zeros(P_MAX, dtype=np.float32)
    current_y = np.zeros(P_MAX, dtype=np.float32)
    radius = np.zeros(P_MAX, dtype=np.float32)
    production = np.ones(P_MAX, dtype=np.float32)
    orbit_radius = np.zeros(P_MAX, dtype=np.float32)
    has_meta = np.zeros(P_MAX, dtype=bool)

    for raw in _raw_planets(obs, "initial_planets"):
        pid = _planet_id(raw)
        if not 0 <= pid < P_MAX:
            continue
        ix = float(_row_value(raw, 2, "x", 0.0))
        iy = float(_row_value(raw, 3, "y", 0.0))
        r = float(_row_value(raw, 4, "radius", 0.0))
        prod = float(_row_value(raw, 6, "production", 1.0))
        initial_x[pid] = ix
        initial_y[pid] = iy
        current_x[pid] = ix
        current_y[pid] = iy
        radius[pid] = r
        production[pid] = prod
        orbit_radius[pid] = math.hypot(ix - SUN_X, iy - SUN_Y)
        is_comet[pid] = pid in comet_ids
        has_meta[pid] = True

    for raw in _raw_planets(obs, "planets"):
        pid = _planet_id(raw)
        if not 0 <= pid < P_MAX:
            continue
        active_mask[pid] = True
        owners[pid] = _owner(raw)
        sx = float(_row_value(raw, 2, "x", 0.0))
        sy = float(_row_value(raw, 3, "y", 0.0))
        r = float(_row_value(raw, 4, "radius", radius[pid]))
        ship_count = float(_row_value(raw, 5, "ships", 0.0))
        prod = float(_row_value(raw, 6, "production", production[pid]))
        current_x[pid] = sx
        current_y[pid] = sy
        radius[pid] = r
        ships[pid] = ship_count
        ships_by_pid[pid] = max(0, int(ship_count))
        production[pid] = prod
        if not has_meta[pid]:
            initial_x[pid] = sx
            initial_y[pid] = sy
            orbit_radius[pid] = math.hypot(sx - SUN_X, sy - SUN_Y)
            has_meta[pid] = True
        if pid in comet_ids:
            is_comet[pid] = True

    source_mask = active_mask & (owners == player) & (ships >= 1.0)
    planet_features = np.zeros((P_MAX, PLANET_FEATURE_DIM), dtype=np.float32)

    for pid in np.flatnonzero(has_meta):
        owner_enc, opponent_id_norm = encode_owner(int(owners[pid]), player, opponent_map)
        orbiting = bool(orbit_radius[pid] + radius[pid] < ROTATION_RADIUS_LIMIT)
        if is_comet[pid]:
            orbiting = False
        planet_features[pid] = np.array(
            [
                current_x[pid] / 100.0,
                current_y[pid] / 100.0,
                radius[pid] / MAX_PLANET_RADIUS,
                float(orbiting),
                float(is_comet[pid]),
                orbit_radius[pid] / 50.0,
                *owner_enc,
                opponent_id_norm,
                math.log1p(float(ships[pid])) / math.log1p(PLANET_SHIP_SCALE),
                (production[pid] - 1.0) / 4.0,
            ],
            dtype=np.float32,
        )

    global_features = np.array(
        encode_global_features(step, angular_velocity, n_players, active_mask, is_comet),
        dtype=np.float32,
    )
    batch = {
        "global_features": torch.from_numpy(global_features)
        .reshape(1, GLOBAL_FEATURE_DIM)
        .to(device),
        "planet_features": torch.from_numpy(planet_features).unsqueeze(0).to(device),
        "planet_active_mask": torch.from_numpy(active_mask).unsqueeze(0).to(device),
        "planet_source_mask": torch.from_numpy(source_mask).unsqueeze(0).to(device),
    }
    return ObsBatch(
        batch=batch,
        source_mask=source_mask,
        ships_by_pid=ships_by_pid,
        active_mask=active_mask,
        x_by_pid=current_x,
        y_by_pid=current_y,
        radius_by_pid=radius,
    )


def _coerce_config(
    config_or_path: DDPMActConfig | str | Path,
    *,
    device: str | torch.device | None = None,
    launch_threshold: float | None = None,
    max_moves: int | None = None,
    resample_attempts: int | None = None,
) -> DDPMActConfig:
    if isinstance(config_or_path, DDPMActConfig):
        cfg = config_or_path
    else:
        cfg = DDPMActConfig(checkpoint_path=Path(config_or_path))

    return DDPMActConfig(
        checkpoint_path=Path(cfg.checkpoint_path).resolve(),
        device=str(device) if device is not None else cfg.device,
        launch_threshold=cfg.launch_threshold
        if launch_threshold is None
        else float(launch_threshold),
        max_moves=cfg.max_moves if max_moves is None else max_moves,
        resample_attempts=cfg.resample_attempts
        if resample_attempts is None
        else int(resample_attempts),
    )


def make_ddpm_agent(
    config_or_path: DDPMActConfig | str | Path,
    *,
    device: str | torch.device | None = None,
    launch_threshold: float | None = None,
    max_moves: int | None = None,
    resample_attempts: int | None = None,
) -> Callable[[Any], list[list[float | int]]]:
    config = _coerce_config(
        config_or_path,
        device=device,
        launch_threshold=launch_threshold,
        max_moves=max_moves,
        resample_attempts=resample_attempts,
    )
    resolved_device = torch.device(
        config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ddpm, encoder, _, _ = load_checkpoint(config.checkpoint_path, resolved_device)
    ddpm.eval_mode()
    encoder.eval()

    def ddpm_act(obs: Any) -> list[list[float | int]]:
        obs_batch = _obs_to_batch(obs, resolved_device)
        moves: list[list[float | int]] = []
        pending = set(int(pid) for pid in np.flatnonzero(obs_batch.source_mask))
        attempts = max(1, config.resample_attempts)

        for _ in range(attempts):
            if not pending:
                break
            actions = sample_actions(ddpm, encoder, obs_batch.batch)
            turn0 = actions[0, 0].detach().cpu().numpy()
            resample: set[int] = set()

            for pid in sorted(pending):
                launch, angle, n_ships = decode_action(
                    turn0[pid], int(obs_batch.ships_by_pid[pid])
                )
                if not launch:
                    continue
                if float(turn0[pid, 0]) <= config.launch_threshold:
                    continue
                if not _launch_hits_planet_before_hazard(obs_batch, pid, angle):
                    resample.add(pid)
                    continue
                moves.append([int(pid), float(angle), int(n_ships)])
                if config.max_moves is not None and len(moves) >= config.max_moves:
                    return moves

            pending = resample
        return moves

    ddpm_act.__name__ = f"ddpm_act_{config.checkpoint_path.stem}"
    return ddpm_act
