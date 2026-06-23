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
# Engine constant: the asymptotic fleet speed cap a >=1000-ship fleet approaches.
# Baked into the engine and dataset normalization (speed_norm divides by 5.0 == 6.0-1.0),
# so this is NOT tunable without retraining; keep it a constant, not a config field.
MAX_FLEET_SPEED = 6.0


@dataclass(frozen=True)
class DDPMActConfig:
    checkpoint_path: Path
    device: str | None = None
    launch_threshold: float = 0.0
    max_moves: int | None = None
    resample_attempts: int = 4
    correct_misses: bool = True
    max_correction_deg: float = 20.0


@dataclass
class ObsBatch:
    batch: dict[str, torch.Tensor]
    source_mask: np.ndarray
    ships_by_pid: np.ndarray
    active_mask: np.ndarray
    x_by_pid: np.ndarray
    y_by_pid: np.ndarray
    radius_by_pid: np.ndarray
    is_comet: np.ndarray
    angular_velocity: float


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


def _ray_target_planet(
    obs_batch: ObsBatch,
    source_pid: int,
    angle: float,
) -> int | None:
    """PID of the first planet the launch ray strikes before any hazard, else None.

    Static raycast against planets' CURRENT positions: identifies which planet the
    model is aiming at. NOTE this is motion-blind — for an orbiting target the
    planet moves out of the straight path during flight, so a hit here does NOT
    mean the fleet actually arrives (or that it avoids the sun). Callers must
    confirm delivery with the swept, sun-aware `_simulate_target_hit`.
    """
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

    target_pid: int | None = None
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
            target_pid = int(pid)

    # The engine checks planet hits before sun/bounds for a tick; allow ties in
    # favor of planets to match that precedence.
    if target_pid is not None and planet_t <= hazard_t:
        return target_pid
    return None


# --------------------------------------------------------------------------- #
# Post-model aim correction
#
# When a decoded launch misses everything (raycast above == False), snap its
# angle to a lead/intercept solution for the most-likely-intended planet. All
# angles are RADIANS (matching the engine, decode_action, and angular_velocity);
# only the human-facing threshold is given in degrees and converted once at the
# call site. Self-contained (math/numpy only) so pack.py's closure stays
# Kaggle-safe.
# --------------------------------------------------------------------------- #
def _fleet_speed(n_ships: int) -> float:
    """Mirror the engine's size-dependent fleet speed (constant per launched fleet)."""
    n = max(1, int(n_ships))
    speed = 1.0 + (MAX_FLEET_SPEED - 1.0) * (math.log(n) / math.log(1000.0)) ** 1.5
    return min(speed, MAX_FLEET_SPEED)


def _orbit_state(obs_batch: ObsBatch, pid: int) -> tuple[float, float, bool]:
    """Return (orbit_radius, current_orbit_angle, is_orbiting) for a planet."""
    cx = float(obs_batch.x_by_pid[pid])
    cy = float(obs_batch.y_by_pid[pid])
    r = float(obs_batch.radius_by_pid[pid])
    orbit_r = math.hypot(cx - SUN_X, cy - SUN_Y)
    is_orbiting = (not bool(obs_batch.is_comet[pid])) and (
        orbit_r + r < ROTATION_RADIUS_LIMIT
    )
    phi0 = math.atan2(cy - SUN_Y, cx - SUN_X)
    return orbit_r, phi0, is_orbiting


def _planet_pos_at(obs_batch: ObsBatch, pid: int, k: int) -> tuple[float, float]:
    """Predicted planet position k ticks ahead (linearised orbital motion)."""
    orbit_r, phi0, is_orbiting = _orbit_state(obs_batch, pid)
    if not is_orbiting:
        return float(obs_batch.x_by_pid[pid]), float(obs_batch.y_by_pid[pid])
    phi = phi0 + obs_batch.angular_velocity * k
    return SUN_X + orbit_r * math.cos(phi), SUN_Y + orbit_r * math.sin(phi)


def _solve_intercept(
    sx: float,
    sy: float,
    obs_batch: ObsBatch,
    pid: int,
    speed: float,
) -> float:
    """Lead angle to intercept planet `pid`.

    Static target -> direct bearing. Orbiting target -> fixed-point lead that
    converges fast for circular motion: guess time-to-impact, advance the orbit,
    re-aim, repeat. Start is the source planet center; the sub-unit launch offset
    (sr + 0.1) is ignored here and re-applied during swept validation.
    """
    px0 = float(obs_batch.x_by_pid[pid])
    py0 = float(obs_batch.y_by_pid[pid])
    orbit_r, phi0, is_orbiting = _orbit_state(obs_batch, pid)
    if not is_orbiting or speed <= 0.0:
        return math.atan2(py0 - sy, px0 - sx)

    omega = obs_batch.angular_velocity
    t = math.hypot(px0 - sx, py0 - sy) / speed
    px, py = px0, py0
    for _ in range(16):
        phi = phi0 + omega * t
        px = SUN_X + orbit_r * math.cos(phi)
        py = SUN_Y + orbit_r * math.sin(phi)
        t_new = math.hypot(px - sx, py - sy) / speed
        if abs(t_new - t) < 1e-3:
            t = t_new
            break
        t = t_new
    return math.atan2(py - sy, px - sx)


def _swept_pair_hit(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    p0x: float,
    p0y: float,
    p1x: float,
    p1y: float,
    r: float,
) -> bool:
    """Local mirror of the engine's swept_pair_hit (fleet A->B vs planet P0->P1)."""
    d0x, d0y = ax - p0x, ay - p0y
    dvx = (bx - ax) - (p1x - p0x)
    dvy = (by - ay) - (p1y - p0y)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    if a < 1e-12:
        return c <= 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return t2 >= 0.0 and t1 <= 1.0


def _point_to_segment_distance(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    """Minimum distance from point (px,py) to segment (ax,ay)-(bx,by)."""
    l2 = (ax - bx) ** 2 + (ay - by) ** 2
    if l2 == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / l2))
    projx = ax + t * (bx - ax)
    projy = ay + t * (by - ay)
    return math.hypot(px - projx, py - projy)


def _simulate_target_hit(
    sx: float,
    sy: float,
    source_r: float,
    angle: float,
    speed: float,
    obs_batch: ObsBatch,
    pid: int,
    max_ticks: int | None = None,
) -> bool:
    """True iff a fleet fired at `angle` reaches planet `pid` before a hazard.

    Mirrors the engine tick loop ordering: per tick, continuous swept hit on the
    target planet is checked first, then board exit, then sun crossing. Only the
    target planet is tested (other planets en route are ignored, matching the
    correction's intent of validating delivery to the chosen target).

    `max_ticks` defaults to enough ticks for the fleet to traverse the whole board
    at its (size-dependent) speed, so slow fleets aren't falsely rejected before
    they could reach a far target.
    """
    if max_ticks is None:
        diagonal = math.hypot(BOARD_MAX - BOARD_MIN, BOARD_MAX - BOARD_MIN)
        max_ticks = int(diagonal / max(speed, 1e-6)) + 2
    dx = math.cos(angle)
    dy = math.sin(angle)
    fx = sx + dx * (source_r + 0.1)
    fy = sy + dy * (source_r + 0.1)
    r = float(obs_batch.radius_by_pid[pid])
    for k in range(1, max_ticks + 1):
        old_x, old_y = fx, fy
        fx += dx * speed
        fy += dy * speed
        p_old = _planet_pos_at(obs_batch, pid, k - 1)
        p_new = _planet_pos_at(obs_batch, pid, k)
        if _swept_pair_hit(
            old_x, old_y, fx, fy, p_old[0], p_old[1], p_new[0], p_new[1], r
        ):
            return True
        if not (BOARD_MIN <= fx <= BOARD_MAX and BOARD_MIN <= fy <= BOARD_MAX):
            return False
        if (
            _point_to_segment_distance(SUN_X, SUN_Y, old_x, old_y, fx, fy)
            < SUN_RADIUS
        ):
            return False
    return False


def _angdiff(a: float, b: float) -> float:
    """Absolute wrapped difference between two angles (radians), in [0, pi]."""
    d = a - b
    return abs(math.atan2(math.sin(d), math.cos(d)))


def _correct_angle(
    obs_batch: ObsBatch,
    source_pid: int,
    model_angle: float,
    n_ships: int,
    max_dev_rad: float,
) -> float | None:
    """Corrected launch angle for a missed shot, or None if no valid snap.

    Considers every active non-comet planet except the source (any owner: enemy
    and neutral captures plus own-planet reinforcement). For each, solve the
    lead/intercept angle, keep those within `max_dev_rad` of the model's angle,
    validate with the swept simulation, and return the min-deviation valid one.
    """
    sx = float(obs_batch.x_by_pid[source_pid])
    sy = float(obs_batch.y_by_pid[source_pid])
    source_r = float(obs_batch.radius_by_pid[source_pid])
    speed = _fleet_speed(n_ships)

    best_angle: float | None = None
    best_dev: float | None = None
    for raw_pid in np.flatnonzero(obs_batch.active_mask):
        pid = int(raw_pid)
        if pid == source_pid or bool(obs_batch.is_comet[pid]):
            continue
        angle = _solve_intercept(sx, sy, obs_batch, pid, speed)
        dev = _angdiff(angle, model_angle)
        if dev > max_dev_rad:
            continue
        if best_dev is not None and dev >= best_dev:
            continue
        if not _simulate_target_hit(sx, sy, source_r, angle, speed, obs_batch, pid):
            continue
        best_angle = angle
        best_dev = dev
    return best_angle


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
        is_comet=is_comet,
        angular_velocity=angular_velocity,
    )


def _coerce_config(
    config_or_path: DDPMActConfig | str | Path,
    *,
    device: str | torch.device | None = None,
    launch_threshold: float | None = None,
    max_moves: int | None = None,
    resample_attempts: int | None = None,
    correct_misses: bool | None = None,
    max_correction_deg: float | None = None,
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
        correct_misses=cfg.correct_misses
        if correct_misses is None
        else bool(correct_misses),
        max_correction_deg=cfg.max_correction_deg
        if max_correction_deg is None
        else float(max_correction_deg),
    )


def make_ddpm_agent(
    config_or_path: DDPMActConfig | str | Path,
    *,
    device: str | torch.device | None = None,
    launch_threshold: float | None = None,
    max_moves: int | None = None,
    resample_attempts: int | None = None,
    correct_misses: bool | None = None,
    max_correction_deg: float | None = None,
) -> Callable[[Any], list[list[float | int]]]:
    config = _coerce_config(
        config_or_path,
        device=device,
        launch_threshold=launch_threshold,
        max_moves=max_moves,
        resample_attempts=resample_attempts,
        correct_misses=correct_misses,
        max_correction_deg=max_correction_deg,
    )
    resolved_device = torch.device(
        config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ddpm, encoder, _, _ = load_checkpoint(config.checkpoint_path, resolved_device)
    ddpm.eval_mode()
    encoder.eval()
    max_dev_rad = math.radians(config.max_correction_deg)

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
                sx = float(obs_batch.x_by_pid[pid])
                sy = float(obs_batch.y_by_pid[pid])
                sr = float(obs_batch.radius_by_pid[pid])
                speed = _fleet_speed(n_ships)
                # Identify the intended target (static ray) then confirm DELIVERY
                # with the swept, sun-aware sim. A static hit alone is not enough:
                # an orbiting planet moves out of the straight path during flight,
                # leaving the fleet to sail straight into the sun. Validating here
                # means every emitted launch (raw or corrected) is sun-safe.
                target = _ray_target_planet(obs_batch, pid, angle)
                hits = target is not None and _simulate_target_hit(
                    sx, sy, sr, angle, speed, obs_batch, target
                )
                if not hits:
                    if config.correct_misses:
                        corrected = _correct_angle(
                            obs_batch,
                            pid,
                            angle,
                            n_ships,
                            max_dev_rad,
                        )
                        if corrected is not None:
                            moves.append([int(pid), float(corrected), int(n_ships)])
                            if (
                                config.max_moves is not None
                                and len(moves) >= config.max_moves
                            ):
                                return moves
                            continue
                    resample.add(pid)
                    continue
                moves.append([int(pid), float(angle), int(n_ships)])
                if config.max_moves is not None and len(moves) >= config.max_moves:
                    return moves

            pending = resample
        return moves

    ddpm_act.__name__ = f"ddpm_act_{config.checkpoint_path.stem}"
    return ddpm_act
