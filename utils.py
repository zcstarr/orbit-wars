import math
from kaggle_environments.envs.orbit_wars.orbit_wars import (
    CENTER,
    ROTATION_RADIUS_LIMIT,
)

# we want to know if the path of a ship will intersect with the sun

# planet source and destination coordinates, circle center and radius


def segment_intersects_circle(x1, y1, x2, y2, h, k, r):
    dx = x2 - x1
    dy = y2 - y1

    fx = x1 - h
    fy = y1 - k

    a = dx*dx + dy*dy
    b = 2 * (fx*dx + fy*dy)
    c = fx*fx + fy*fy - r*r

    discriminant = b*b - 4*a*c

    if discriminant < 0:
        return False

    discriminant = math.sqrt(discriminant)

    t1 = (-b - discriminant) / (2*a)
    t2 = (-b + discriminant) / (2*a)

    # negative means the segment doesn't intersect the circle
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)


def get_initial_planet_data(obs):
    raw_planets = obs.planets if not isinstance(obs, dict) else obs["planets"]
    raw_initial = obs.initial_planets if not isinstance(
        obs, dict) else obs["initial_planets"]
    comet_ids = set(obs.comet_planet_ids if not isinstance(
        obs, dict) else obs["comet_planet_ids"])

    initial_by_id = {p[0]: p for p in raw_initial}
    return initial_by_id, comet_ids


def make_classifier(obs):
    CENTER = 50.0
    ROT_LIMIT = 50.0
    initial_by_id, comet_ids = get_initial_planet_data(obs)
    print(f"initial_by_id: {initial_by_id}")

    def classify(p):
        if p.id in comet_ids:
            return "comet"
        ip = initial_by_id[p.id]
        orb_r = math.hypot(ip[2] - CENTER, ip[3] - CENTER)
        return "static" if orb_r + p.radius >= ROT_LIMIT else "orbiting"
    return classify
