import math
import numpy as np
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
import utils


def holding_player_unit_test(obs):
    moves = []
    return moves


def never_miss(obs):
    moves = []

    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(
        obs, dict) else obs.planets
    planets = [Planet(*p) for p in raw_planets]

    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]

    if not targets:
        return moves

    for mine in my_planets:
        # Find the nearest planet we don't own
        nearest = None
        min_dist = float('inf')
        for t in targets:
            dist = math.sqrt((mine.x - t.x)**2 + (mine.y - t.y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest = t
    return moves


def nearest_planet_sniper(obs):
    classifier = utils.make_classifier(obs)

    num_ships = len(obs.get('fleets', []))
    speed = 1.0 + (6.0 - 1.0) * (np.log(num_ships) / np.log(1000)) ** 1.5
    print(f"fleet speed: {speed}")
    moves = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(
        obs, dict) else obs.planets
    planets = [Planet(*p) for p in raw_planets]
    planet_production = {p.id: p.production for p in planets}
    planets_classified = [classifier(p) for p in planets]
    print(f"planets_classified: {planets_classified}")

    # Separate our planets from targets
    my_planets = [p for p in planets if p.owner == player]
    targets: list[Planet] = [p for p in planets if p.owner != player]

    if not targets:
        return moves

    for mine in my_planets:
        # Find the nearest planet we don't own
        nearest: Planet | None = None
        min_dist = float('inf')
        for t in targets:
            if planets_classified[t.id] == "orbiting" or planets_classified[t.id] == "comet":
                continue
            dist = math.sqrt((mine.x - t.x)**2 + (mine.y - t.y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest = t

        if nearest is None:
            continue

        # How many ships do we need? Target's garrison + 1
        ships_needed = max(nearest.ships + 1, 10)

        # Only send if we have enough
        if mine.ships >= ships_needed:
            # Calculate angle from our planet to the target
            angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
            if (utils.segment_intersects_circle(mine.x, mine.y, nearest.x, nearest.y, 50, 50, 10) == False):
                moves.append([mine.id, angle, ships_needed])

    return moves
