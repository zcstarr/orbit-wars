"""Run an orbit_wars match and write orbit.html for serve_orbit.py to pick up.

Usage:
    python run_harness.py                # writes ./orbit.html
    python run_harness.py --out foo.html

Edit AGENTS below to change the matchup. Keep nearest_planet_sniper's
signature stable -- main.py is the Kaggle submission file.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from kaggle_environments import make

from agents import holding_player_unit_test, nearest_planet_sniper, never_miss
from ddpm_act import make_ddpm_agent

# Edit this list to change who plays. 2 or 4 entries.
# Each entry is either a callable (agent fn) or a string ("random").
CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "step_1800500.pt"
ddpm_agent = make_ddpm_agent(CHECKPOINT)
AGENTS = [ddpm_agent, nearest_planet_sniper, nearest_planet_sniper, nearest_planet_sniper]
AGENTS = [ddpm_agent, holding_player_unit_test, holding_player_unit_test, holding_player_unit_test]
# AGENTS = [never_miss, holding_player_unit_test]
# AGENTS = [nearest_planet_sniper, holding_player_unit_test]

WIDTH, HEIGHT = 800, 600


def _agent_label(agent) -> str:
    return getattr(agent, "__name__", str(agent))


def _timed_agent(agent, durations: list[float]):
    """Wrap a callable agent so each act() call's wall time is recorded.

    String agents (e.g. "random") are returned untouched -- they run inside the
    env and we can't time them here.
    """
    if not callable(agent):
        return agent

    def wrapped(obs):
        t0 = time.perf_counter()
        try:
            return agent(obs)
        finally:
            durations.append(time.perf_counter() - t0)

    wrapped.__name__ = _agent_label(agent)
    return wrapped


def _pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def _print_timing_report(agents, per_slot, act_timeout: float, overage: float) -> None:
    print("\n=== per-turn timing (vs Kaggle limits) ===")
    print(
        f"actTimeout={act_timeout:.0f}s/turn (hard), "
        f"remainingOverageTime={overage:.0f}s total bank/episode"
    )
    for i, durations in enumerate(per_slot):
        label = _agent_label(agents[i])
        if not durations:
            print(f"Player {i} ({label}): no timed turns (non-callable agent)")
            continue
        s = sorted(durations)
        n = len(s)
        mean = sum(s) / n
        over = [d for d in s if d > act_timeout]
        consumed = sum(d - act_timeout for d in over)
        print(
            f"Player {i} ({label}): turns={n} "
            f"mean={mean * 1000:.0f}ms p50={_pct(s, 0.50) * 1000:.0f}ms "
            f"p95={_pct(s, 0.95) * 1000:.0f}ms max={s[-1] * 1000:.0f}ms"
        )
        flag = " <-- WOULD TIME OUT" if consumed > overage else ""
        print(
            f"           over {act_timeout:.0f}s: {len(over)}/{n} turns, "
            f"overage consumed={consumed:.1f}s / {overage:.0f}s bank{flag}"
        )


GRID_OVERLAY = """
<style>
  .ow-grid-wrap { position: relative; display: inline-block; }
  .ow-grid-wrap canvas { display: block; }
  .ow-grid {
    position: absolute; inset: 0; pointer-events: none;
    background-image:
      linear-gradient(to right,  rgba(255,255,255,0.25) 1px, transparent 1px),
      linear-gradient(to bottom, rgba(255,255,255,0.25) 1px, transparent 1px),
      linear-gradient(to right,  rgba(255,255,255,0.06) 1px, transparent 1px),
      linear-gradient(to bottom, rgba(255,255,255,0.06) 1px, transparent 1px);
    background-size: 10% 10%, 10% 10%, 1% 1%, 1% 1%;
    background-position: 0 0;
  }
</style>
<script>
  // Wrap the first canvas the renderer creates with a grid overlay.
  (function attachGrid() {
    const tryAttach = () => {
      const cv = document.querySelector('canvas');
      if (!cv || cv.dataset.gridded) return false;
      cv.dataset.gridded = '1';
      const wrap = document.createElement('div');
      wrap.className = 'ow-grid-wrap';
      cv.parentNode.insertBefore(wrap, cv);
      wrap.appendChild(cv);
      const grid = document.createElement('div');
      grid.className = 'ow-grid';
      wrap.appendChild(grid);
      return true;
    };
    if (!tryAttach()) {
      const obs = new MutationObserver(() => { if (tryAttach()) obs.disconnect(); });
      obs.observe(document.body, { childList: true, subtree: true });
    }
  })();
</script>
"""


def run(agents, out: Path) -> None:
    env = make("orbit_wars", debug=True)

    per_slot: list[list[float]] = [[] for _ in agents]
    timed = [_timed_agent(a, per_slot[i]) for i, a in enumerate(agents)]
    env.run(timed)

    for i, s in enumerate(env.steps[-1]):
        print(
            f"Player {i} ({_agent_label(agents[i])}): reward={s.reward}, status={s.status}")

    cfg = env.configuration
    act_timeout = float(cfg.get("actTimeout", 1))
    overage = float(cfg.get("remainingOverageTime", 60))
    _print_timing_report(agents, per_slot, act_timeout, overage)

    html = env.render(mode="html", width=WIDTH, height=HEIGHT)
    # Inject overlay just before </body>; fall back to append if absent.
    if "</body>" in html:
        html = html.replace("</body>", GRID_OVERLAY + "</body>", 1)
    else:
        html += GRID_OVERLAY
    out.write_text(html)
    print(f"[harness] wrote {out} ({out.stat().st_size:,} bytes)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="orbit.html", type=Path,
                    help="Output HTML path (default: orbit.html)")
    args = ap.parse_args()
    run(AGENTS, args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
