#!/usr/bin/env python3
"""Pure-function tests for the post-model aim corrector in ddpm_act.py.

No torch / checkpoint / env needed: the correction helpers operate purely on the
numpy side-channel of ObsBatch, so we build minimal ObsBatch fixtures directly.
All angles are radians (matching the engine and decode_action); the only degree
value is the human-facing threshold, converted at the boundary.

Run: `.venv/bin/python test_correction.py` (exits non-zero on failure).
"""

from __future__ import annotations

import math

import numpy as np

from ddpm_act import (
    ObsBatch,
    _angdiff,
    _correct_angle,
    _fleet_speed,
    _ray_target_planet,
    _simulate_target_hit,
    _solve_intercept,
)


def make_ob(planets: list[dict], angular_velocity: float = 0.0) -> ObsBatch:
    """Build a minimal ObsBatch from a list of planet dicts.

    Each dict: {id, x, y, radius?, ships?, comet?, source?}. The torch `batch`
    is irrelevant to the correction path, so it's left empty.
    """
    n = max(p["id"] for p in planets) + 1
    x = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    r = np.zeros(n, dtype=np.float64)
    ships = np.zeros(n, dtype=np.int64)
    active = np.zeros(n, dtype=bool)
    comet = np.zeros(n, dtype=bool)
    source = np.zeros(n, dtype=bool)
    for p in planets:
        i = p["id"]
        x[i] = p["x"]
        y[i] = p["y"]
        r[i] = p.get("radius", 1.0)
        ships[i] = p.get("ships", 10)
        active[i] = True
        comet[i] = p.get("comet", False)
        source[i] = p.get("source", False)
    return ObsBatch(
        batch={},
        source_mask=source,
        ships_by_pid=ships,
        active_mask=active,
        x_by_pid=x,
        y_by_pid=y,
        radius_by_pid=r,
        is_comet=comet,
        angular_velocity=angular_velocity,
    )


def test_static_intercept_is_direct_bearing() -> None:
    # Target at (90,20): orbit_radius == 50 (+r) >= 50 -> static, so the lead
    # solve must collapse to the plain atan2 bearing. (Kept off the y=x diagonal
    # so the path doesn't graze the sun.)
    ob = make_ob(
        [
            {"id": 0, "x": 10.0, "y": 10.0, "radius": 2.0, "source": True},
            {"id": 1, "x": 90.0, "y": 20.0, "radius": 2.0},
        ]
    )
    speed = _fleet_speed(15)
    angle = _solve_intercept(10.0, 10.0, ob, 1, speed)
    expected = math.atan2(20.0 - 10.0, 90.0 - 10.0)
    assert abs(_angdiff(angle, expected)) < 1e-9, (angle, expected)


def test_orbiting_lead_hits_when_direct_bearing_misses() -> None:
    # Orbiting target (orbit_radius 20 < 50). A naive shot at its CURRENT
    # position misses after the planet rotates away; the lead solution hits.
    ob = make_ob(
        [
            {"id": 0, "x": 10.0, "y": 50.0, "radius": 2.0, "source": True},
            {"id": 1, "x": 50.0, "y": 70.0, "radius": 1.0},
        ],
        angular_velocity=0.05,
    )
    speed = _fleet_speed(20)
    led = _solve_intercept(10.0, 50.0, ob, 1, speed)
    direct = math.atan2(70.0 - 50.0, 50.0 - 10.0)

    assert _angdiff(led, direct) > 1e-2, "lead should differ from direct bearing"
    assert _simulate_target_hit(10.0, 50.0, 2.0, led, speed, ob, 1), "lead must hit"
    assert not _simulate_target_hit(
        10.0, 50.0, 2.0, direct, speed, ob, 1
    ), "direct bearing should miss a mover"


def test_correct_angle_snaps_to_own_planet() -> None:
    # Owner isn't tracked by the corrector, so any planet (incl. one we own) is a
    # valid snap target. Model angle is slightly off the only candidate.
    ob = make_ob(
        [
            {"id": 0, "x": 10.0, "y": 10.0, "radius": 2.0, "source": True},
            {"id": 1, "x": 90.0, "y": 20.0, "radius": 2.0},
        ]
    )
    true_bearing = math.atan2(20.0 - 10.0, 90.0 - 10.0)
    model_angle = true_bearing + math.radians(8.0)  # slight miss
    out = _correct_angle(
        ob,
        source_pid=0,
        model_angle=model_angle,
        n_ships=15,
        max_dev_rad=math.radians(20.0),
    )
    assert out is not None, "should snap to the planet"
    assert abs(_angdiff(out, true_bearing)) < 1e-6, out


def test_deviation_threshold_rejects_far_target() -> None:
    # Same geometry, but the model angle is ~40deg off the only target while the
    # threshold is 20deg -> no correction.
    ob = make_ob(
        [
            {"id": 0, "x": 10.0, "y": 10.0, "radius": 2.0, "source": True},
            {"id": 1, "x": 90.0, "y": 20.0, "radius": 2.0},
        ]
    )
    model_angle = math.atan2(20.0 - 10.0, 90.0 - 10.0) + math.radians(40.0)
    out = _correct_angle(
        ob,
        source_pid=0,
        model_angle=model_angle,
        n_ships=15,
        max_dev_rad=math.radians(20.0),
    )
    assert out is None, f"beyond-threshold target should be rejected, got {out}"


def test_sun_hazard_rejects_shot_through_sun() -> None:
    # Source and target straddle the sun on the y=50 line; the only intercept
    # path crosses the sun (r=10), so the shot is invalid and no correction is
    # returned.
    ob = make_ob(
        [
            {"id": 0, "x": 10.0, "y": 50.0, "radius": 2.0, "source": True},
            {"id": 1, "x": 90.0, "y": 50.0, "radius": 2.0},
        ]
    )
    model_angle = 0.0  # straight at the target, but through the sun
    out = _correct_angle(
        ob,
        source_pid=0,
        model_angle=model_angle,
        n_ships=15,
        max_dev_rad=math.radians(20.0),
    )
    assert out is None, f"sun-crossing shot must be rejected, got {out}"


def test_moving_planet_static_hit_still_crosses_sun() -> None:
    # The bug: source (10,50), orbiting planet currently at (35,50), sun (50,50).
    # A shot straight along y=50 hits the planet's CURRENT disc before the sun, so
    # the motion-blind static gate ACCEPTS it. But the planet orbits off the line
    # during flight; the fleet sails through the vacated spot and into the sun. The
    # swept, motion+sun-aware sim must reject delivery so the launch isn't emitted.
    ob = make_ob(
        [
            {"id": 0, "x": 10.0, "y": 50.0, "radius": 2.0, "source": True},
            {"id": 1, "x": 35.0, "y": 50.0, "radius": 2.0},
        ],
        angular_velocity=0.15,
    )
    angle = 0.0  # straight +x at the planet's current position
    # Static gate alone would accept (planet hit before the sun)...
    assert _ray_target_planet(ob, 0, angle) == 1, "static gate should identify target"
    speed = _fleet_speed(15)
    # ...but the swept sim rejects it because the fleet reaches the sun.
    assert not _simulate_target_hit(10.0, 50.0, 2.0, angle, speed, ob, 1), (
        "moving-planet shot that crosses the sun must be rejected"
    )


def test_board_exit_rejects_offboard_shot() -> None:
    # Source near the +x edge (90,50) firing straight off the board (away from the
    # sun at (50,50), so only the boundary can reject it). The only planet sits off
    # the path. Both the static gate and the swept sim must reject: the fleet exits
    # the board before reaching anything.
    ob = make_ob(
        [
            {"id": 0, "x": 90.0, "y": 50.0, "radius": 2.0, "source": True},
            {"id": 1, "x": 50.0, "y": 90.0, "radius": 2.0},
        ]
    )
    angle = 0.0  # straight +x, toward x=100 boundary
    assert _ray_target_planet(ob, 0, angle) is None, "off-board shot hits no planet"
    speed = _fleet_speed(15)
    assert not _simulate_target_hit(90.0, 50.0, 2.0, angle, speed, ob, 1), (
        "fleet that exits the board must be rejected"
    )


def main() -> int:
    tests = [
        test_static_intercept_is_direct_bearing,
        test_orbiting_lead_hits_when_direct_bearing_misses,
        test_correct_angle_snaps_to_own_planet,
        test_deviation_threshold_rejects_far_target,
        test_sun_hazard_rejects_shot_through_sun,
        test_moving_planet_static_hit_still_crosses_sun,
        test_board_exit_rejects_offboard_shot,
    ]
    for t in tests:
        t()
        print(f"[ok] {t.__name__}")
    print("all correction tests PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
