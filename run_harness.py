"""Run an orbit_wars match and write orbit.html for serve_orbit.py to pick up.

Usage:
    python run_harness.py                # writes ./orbit.html
    python run_harness.py --out foo.html

Edit AGENTS below to change the matchup. Keep nearest_planet_sniper's
signature stable -- main.py is the Kaggle submission file.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from kaggle_environments import make

from agents import holding_player_unit_test, nearest_planet_sniper, never_miss
from ddpm_act import make_ddpm_agent

# Edit this list to change who plays. 2 or 4 entries.
# Each entry is either a callable (agent fn) or a string ("random").
CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "step_1800500.pt"
CHECKPOINT = Path(__file__).resolve().parent / "checkpoints_v2" / "latest.pt"
CHECKPOINT = Path(__file__).resolve().parent / "checkpoints_v3" / "step_120000.pt"
# correct_misses snaps a missed sampled launch to a lead/intercept on the
# nearest intended planet (any owner). Flip to False to A/B the raw policy.
ddpm_agent = make_ddpm_agent(CHECKPOINT, correct_misses=True)
# AGENTS = [ddpm_agent, nearest_planet_sniper, nearest_planet_sniper, nearest_planet_sniper]
AGENTS = [ddpm_agent, nearest_planet_sniper]
# AGENTS = [ddpm_agent, holding_player_unit_test, holding_player_unit_test, holding_player_unit_test]
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


def _scoreboard_overlay(agents) -> str:
    """Live panel: maps each player color->agent name and shows a running
    score (planets owned, total ships on planets + in flight) plus per-player
    reward/status. Stays in sync with playback by wrapping the env renderer,
    which is invoked every frame with the current step.

    NOTE: window.kaggle.step is NOT live -- the player Object.assigns
    window.kaggle into its own React ref and never writes the step back, so
    polling window.kaggle.step freezes at step 0. The renderer ctx.step is the
    only reliable live playhead.

    Score is recomputed client-side from environment.steps[step][0].observation
    (full visibility): planets owner=p[1] ships=p[5], fleets owner=f[1] ships=f[6].
    Colors mirror the renderer's Wong palette.
    """
    labels = [_agent_label(a) for a in agents]
    return """
<style>
  #ow-scoreboard {
    position: fixed; top: 8px; right: 8px; z-index: 99999;
    font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px;
    color: #fff; background: rgba(0,0,0,0.66); padding: 8px 10px;
    border-radius: 6px; line-height: 1.45; box-shadow: 0 2px 8px rgba(0,0,0,0.5);
  }
  #ow-scoreboard table { border-collapse: collapse; }
  #ow-scoreboard td, #ow-scoreboard th { padding: 1px 6px; text-align: right; }
  #ow-scoreboard th { color: #bbb; font-weight: normal; }
  #ow-scoreboard td.ow-name { text-align: left; }
  #ow-scoreboard .ow-sw {
    display: inline-block; width: 10px; height: 10px; border-radius: 2px;
    margin-right: 4px; vertical-align: middle;
  }
  #ow-scoreboard .ow-step { color: #9cf; margin-bottom: 3px; }
  #ow-scoreboard .ow-dead { opacity: 0.45; }
</style>
<script>
  (function () {
    const LABELS = __LABELS__;
    const COLORS = ['#0072B2', '#D55E00', '#009E73', '#F0E442', '#888888'];

    function panel() {
      let el = document.getElementById('ow-scoreboard');
      if (!el) {
        el = document.createElement('div');
        el.id = 'ow-scoreboard';
        document.body.appendChild(el);
      }
      return el;
    }

    function compute(steps, step) {
      const s = steps && steps[step];
      if (!s) return null;
      const obs = (s[0] && s[0].observation) || {};
      const n = LABELS.length;
      const planets = new Array(n).fill(0);
      const ships = new Array(n).fill(0);
      (obs.planets || []).forEach((p) => {
        const o = p[1];
        if (o >= 0 && o < n) { planets[o] += 1; ships[o] += Math.floor(p[5]); }
      });
      (obs.fleets || []).forEach((f) => {
        const o = f[1];
        if (o >= 0 && o < n) ships[o] += Math.floor(f[6]);
      });
      const reward = s.map((x) => (x && x.reward != null ? x.reward : null));
      const status = s.map((x) => (x && x.status ? x.status : ''));
      return { step, planets, ships, reward, status };
    }

    function render(steps, step) {
      const d = compute(steps, step);
      if (!d) return;
      let rows =
        '<tr><th></th><th class="ow-name">agent</th>' +
        '<th>plnts</th><th>ships</th><th>rew</th></tr>';
      for (let i = 0; i < LABELS.length; i++) {
        const dead = d.status[i] && d.status[i] !== 'ACTIVE' && d.status[i] !== 'DONE';
        rows +=
          '<tr class="' + (dead ? 'ow-dead' : '') + '">' +
          '<td><span class="ow-sw" style="background:' + COLORS[i] + '"></span></td>' +
          '<td class="ow-name">P' + i + ' ' + LABELS[i] + '</td>' +
          '<td>' + d.planets[i] + '</td>' +
          '<td>' + d.ships[i] + '</td>' +
          '<td>' + (d.reward[i] == null ? '-' : d.reward[i]) + '</td>' +
          '</tr>';
      }
      panel().innerHTML =
        '<div class="ow-step">step ' + d.step + '</div><table>' + rows + '</table>';
    }

    // Wrap the env renderer so we repaint with the exact step it draws each
    // frame. Runs synchronously at end-of-body, before the player's mount
    // effect copies window.kaggle.renderer into its React ref.
    function hook() {
      const k = window.kaggle;
      if (!k || typeof k.renderer !== 'function') return false;
      if (k.renderer.__owWrapped) return true;
      const orig = k.renderer;
      const wrapped = async function (ctx) {
        const r = await orig(ctx);
        try {
          const steps = (ctx.environment || k.environment || {}).steps;
          render(steps, ctx.step || 0);
        } catch (e) {}
        return r;
      };
      wrapped.__owWrapped = true;
      k.renderer = wrapped;
      try { render(k.environment && k.environment.steps, k.step || 0); } catch (e) {}
      return true;
    }
    if (!hook()) {
      const t = setInterval(() => { if (hook()) clearInterval(t); }, 50);
    }
  })();
</script>
""".replace("__LABELS__", json.dumps(labels))


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
    # Inject overlays just before </body>; fall back to append if absent.
    overlay = GRID_OVERLAY + _scoreboard_overlay(agents)
    if "</body>" in html:
        html = html.replace("</body>", overlay + "</body>", 1)
    else:
        html += overlay
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
