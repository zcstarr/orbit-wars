"""Run an orbit_wars match and write orbit.html for serve_orbit.py to pick up.

Usage:
    python run_harness.py                # writes ./orbit.html
    python run_harness.py --out foo.html

Edit AGENTS below to change the matchup. Keep nearest_planet_sniper's
signature stable -- main.py is the Kaggle submission file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kaggle_environments import make

from agents import holding_player_unit_test, nearest_planet_sniper, never_miss

# Edit this list to change who plays. 2 or 4 entries.
# Each entry is either a callable (agent fn) or a string ("random").
AGENTS = [nearest_planet_sniper, holding_player_unit_test]
# AGENTS = [never_miss, holding_player_unit_test]

WIDTH, HEIGHT = 800, 600


def _agent_label(agent) -> str:
    return getattr(agent, "__name__", str(agent))


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
    env.run(agents)

    for i, s in enumerate(env.steps[-1]):
        print(
            f"Player {i} ({_agent_label(agents[i])}): reward={s.reward}, status={s.status}")

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
