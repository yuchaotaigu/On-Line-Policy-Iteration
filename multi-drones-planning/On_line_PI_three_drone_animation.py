"""

Animation of the algorithm from:
  "On-Line Policy Iteration for Finite-Horizon Deterministic Optimal Control
   with A Fixed Initial State" (Li, Chen, Li, Fan, Bertsekas, 2026)

Copyright (c) 2026 Fei Chen, Yuchao Li

Policy Iteration via Rollout — Multi-Agent (3 Drones), Env 2 obstacles.

Joint state: [p1(3), v1(3), p2(3), v2(3), p3(3), v3(3)] = 18D
Joint control: [a1(3), a2(3), a3(3)] = 9D
Same 16 static obstacles as env2, plus pairwise inter-agent repulsion.

Drone 1: (-2.5, 1.5, 1.8) → goal (2.5, -1.5, 0.3)
Drone 2: (2.5, 1.5, 0.3)  → goal (-2.5, -1.5, 1.5)
Drone 3: (0.0, -2.0, 0.2) → goal (0.0, 2.0, 1.2)
Trajectories designed to cross near the center.
"""

import os
os.environ["FORCE_HEADLESS"] = "1"

import io, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

from On_line_PI_env2_d3 import (
    Config, OBSTACLES, BoxObstacle, SphereObstacle,
    N_AGENTS, STATE_DIM_PER, CTRL_DIM_PER, JOINT_STATE_DIM, JOINT_CTRL_DIM,
    train_ppo, rollout_policy_fn, trajectory_cost,
    PolicyWithTrajectory, run_rollout_iteration,
    generate_perturbation_data, train_residual_nn,
    check_consistency,
    draw_box_3d, draw_sphere_3d,
)

"""# Problem Definition & Algorithm"""

# ══════════════════════════════════════════════════════════════════════════════
# Run full PI and collect trajectories
# ══════════════════════════════════════════════════════════════════════════════

def run_pi():
    cfg = Config()
    np.random.seed(123)
    torch.manual_seed(123)

    print("=" * 60)
    print("Multi-agent PI (3 drones)")
    print("=" * 60)

    ppo_policy = train_ppo(cfg)
    traj_0, ctrl_0 = rollout_policy_fn(
        lambda x, k: ppo_policy.act(x, cfg, deterministic=True)[0], cfg)
    cost_0 = trajectory_cost(traj_0, ctrl_0, cfg)
    print(f"\n[Iter 0] PPO: cost={cost_0:.2f}")

    all_trajs = [traj_0]
    all_costs = [cost_0]
    current_policy = PolicyWithTrajectory(traj_0, ctrl_0, None, cfg)
    no_improve = 0

    for it in range(1, cfg.n_policy_iters + 1):
        print(f"\n[Iter {it}]", flush=True)

        new_traj, new_ctrl, n_imp, step_times, rt = \
            run_rollout_iteration(current_policy, cfg)
        new_cost = trajectory_cost(new_traj, new_ctrl, cfg)
        print(f"  Rollout: cost={new_cost:.2f} "
              f"(Δ={new_cost - all_costs[-1]:+.2f}), "
              f"improved={n_imp}/{cfg.N}, time={rt:.1f}s")

        perturb = generate_perturbation_data(new_traj, new_ctrl, cfg)
        resnn = train_residual_nn(new_traj, new_ctrl, perturb, cfg)
        current_policy = PolicyWithTrajectory(new_traj, new_ctrl, resnn, cfg)

        all_trajs.append(new_traj)
        all_costs.append(new_cost)

        if n_imp == 0:
            no_improve += 1
        else:
            no_improve = 0
        if no_improve >= 2:
            print("  Converged.")
            break

    return all_trajs, all_costs, cfg

"""# Drawing Nodes & Animation"""

# ══════════════════════════════════════════════════════════════════════════════
# Drone drawing (Crazyflie quad-rotor)
# ══════════════════════════════════════════════════════════════════════════════

def draw_crazyflie(ax, pos, size=0.18):
    """Draw the Crazyflie at position pos."""
    cx, cy, cz = pos
    arm_len = size
    prop_r = size * 0.7
    arms = [
        np.array([1, 1, 0]) / np.sqrt(2),
        np.array([1, -1, 0]) / np.sqrt(2),
        np.array([-1, 1, 0]) / np.sqrt(2),
        np.array([-1, -1, 0]) / np.sqrt(2),
    ]
    for arm_dir in arms:
        tip = np.array([cx, cy, cz]) + arm_len * arm_dir
        ax.plot([cx, tip[0]], [cy, tip[1]], [cz, tip[2]],
                color="#222222", linewidth=2.5, solid_capstyle="round",
                zorder=15)
        theta = np.linspace(0, 2 * np.pi, 24)
        ax.plot(tip[0] + prop_r * np.cos(theta),
                tip[1] + prop_r * np.sin(theta),
                np.full(24, tip[2]),
                color="#1565c0", linewidth=1.4, alpha=0.85, zorder=15)
    # Body
    ax.scatter([cx], [cy], [cz], color="#1565c0", s=45,
               edgecolor="#0d47a1", linewidth=0.6, zorder=16)


# ══════════════════════════════════════════════════════════════════════════════
# Scene setup
# ══════════════════════════════════════════════════════════════════════════════

def setup_scene(ax, cfg):
    for obs in OBSTACLES:
        if isinstance(obs, BoxObstacle):
            draw_box_3d(ax, obs.center, obs.half_size,
                        color="#6b7280", alpha=0.28, edgecolor="#374151")
        elif isinstance(obs, SphereObstacle):
            draw_sphere_3d(ax, obs.center, obs.radius,
                           color="#e11d48", alpha=0.22)
    # Goal
    draw_box_3d(ax, cfg.goal_center, cfg.goal_half_size,
                color="#22c55e", alpha=0.28, edgecolor="#15803d")
    # Start marker
    ax.scatter(*cfg.x0[:3], s=45, color="#0b5394",
               edgecolor="white", linewidth=0.8, zorder=12,
               label="Start")

    ax.set_xlim(-2.8, 3.2)
    ax.set_ylim(-2.8, 3.2)
    ax.set_zlim(-0.8, 2.0)
    ax.set_box_aspect((1.0, 1.0, 0.5))
    ax.view_init(elev=26, azim=-55)
    ax.set_xlabel("X", fontsize=10, labelpad=4)
    ax.set_ylabel("Y", fontsize=10, labelpad=4)
    ax.set_zlabel("Z", fontsize=10, labelpad=4)
    ax.tick_params(axis="both", labelsize=7, pad=1)
    ax.grid(True, alpha=0.15)
    try:
        ax.xaxis.pane.set_facecolor((0.97, 0.97, 0.97, 0.3))
        ax.yaxis.pane.set_facecolor((0.97, 0.97, 0.97, 0.3))
        ax.zaxis.pane.set_facecolor((0.97, 0.97, 0.97, 0.3))
    except Exception:
        pass


def fig_to_pil(fig, dpi=110):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    buf.seek(0)
    return Image.open(buf).convert("RGB")

# ══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ══════════════════════════════════════════════════════════════════════════════

AGENT_COLORS = ["#0b5394", "#0b8a8a", "#7b2d8e"]
AGENT_NAMES = ["Drone 1", "Drone 2", "Drone 3"]
TRAIL_COLORS_FADED = ["#90caf9", "#80cbc4", "#ce93d8"]


def draw_crazyflie(ax, pos, size=0.18, color="#1565c0"):
    """Draw a Crazyflie quad-rotor at the given position."""
    cx, cy, cz = pos
    prop_r = size * 0.7
    arms = [
        np.array([1, 1, 0]) / np.sqrt(2),
        np.array([1, -1, 0]) / np.sqrt(2),
        np.array([-1, 1, 0]) / np.sqrt(2),
        np.array([-1, -1, 0]) / np.sqrt(2),
    ]
    for arm_dir in arms:
        tip = np.array([cx, cy, cz]) + size * arm_dir
        ax.plot([cx, tip[0]], [cy, tip[1]], [cz, tip[2]],
                color="#222222", linewidth=2.2, solid_capstyle="round",
                zorder=15)
        theta = np.linspace(0, 2 * np.pi, 24)
        ax.plot(tip[0] + prop_r * np.cos(theta),
                tip[1] + prop_r * np.sin(theta),
                np.full(24, tip[2]),
                color=color, linewidth=1.3, alpha=0.85, zorder=15)
    ax.scatter([cx], [cy], [cz], color=color, s=35,
               edgecolor="white", linewidth=0.4, zorder=16)


def setup_scene(ax, cfg):
    """Draw obstacles, goal regions, start markers, and configure axes."""
    # Obstacles
    for obs in OBSTACLES:
        if isinstance(obs, BoxObstacle):
            draw_box_3d(ax, obs.center, obs.half_size,
                        color="#6b7280", alpha=0.28, edgecolor="#374151")
        elif isinstance(obs, SphereObstacle):
            draw_sphere_3d(ax, obs.center, obs.radius,
                           color="#e11d48", alpha=0.22)

    # Goal regions (colored per agent)
    for a in range(N_AGENTS):
        draw_box_3d(ax, cfg.goal_centers[a], cfg.goal_half_size,
                    color=AGENT_COLORS[a], alpha=0.20,
                    edgecolor=AGENT_COLORS[a])

    # Start markers
    for a in range(N_AGENTS):
        ps = a * STATE_DIM_PER
        ax.scatter(*cfg.x0[ps:ps + 3], s=40, color=AGENT_COLORS[a],
                   edgecolor="white", linewidth=0.6, zorder=12)

    # Axes
    ax.set_xlim(-4.0, 4.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_zlim(-1.0, 3.0)
    ax.set_box_aspect((1.0, 0.85, 0.50))
    ax.view_init(elev=28, azim=-52)
    ax.set_xlabel("X", fontsize=10, labelpad=4)
    ax.set_ylabel("Y", fontsize=10, labelpad=4)
    ax.set_zlabel("Z", fontsize=10, labelpad=4)
    ax.tick_params(axis="both", labelsize=7, pad=1)
    ax.grid(True, alpha=0.15)
    try:
        ax.xaxis.pane.set_facecolor((0.97, 0.97, 0.97, 0.3))
        ax.yaxis.pane.set_facecolor((0.97, 0.97, 0.97, 0.3))
        ax.zaxis.pane.set_facecolor((0.97, 0.97, 0.97, 0.3))
    except Exception:
        pass


def fig_to_pil(fig, dpi=110):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ══════════════════════════════════════════════════════════════════════════════
# Animation: all 3 drones fly simultaneously
# ══════════════════════════════════════════════════════════════════════════════

def generate_flight_animation(
    all_trajs, all_costs, cfg,
    show_iters=None,
    output="multiagent_flight_animation.gif",
    dpi=110,
    step_skip=2,
    ms_per_step=120,
    ms_hold_start=1800,
    ms_hold_end=2500,
):
    """
    Generate a GIF of 3 drones flying for selected PI iterations.

    Parameters
    ----------
    all_trajs : list of trajectories from run_pi()
    all_costs : list of costs from run_pi()
    cfg : Config object
    show_iters : list of iteration indices to animate (e.g. [0, 4, 22])
    output : output GIF filename
    step_skip : show every Nth timestep (2 = every other step)
    ms_per_step : frame duration in ms for normal flight frames
    ms_hold_start : frame duration in ms for the first frame of each iteration
    ms_hold_end : frame duration in ms for the last frame of each iteration
    """
    n_total = len(all_trajs)
    if show_iters is None:
        show_iters = [0, min(4, n_total - 1), n_total - 1]
    show_iters = sorted(set(i for i in show_iters if i < n_total))

    print(f"\nAnimating iterations: {show_iters}")

    fig = plt.figure(figsize=(10, 7))
    fig.patch.set_facecolor("white")

    pil_frames = []
    durations = []

    for iter_idx in show_iters:
        traj = all_trajs[iter_idx]
        arr = np.array(traj)  # shape (N+1, JOINT_STATE_DIM)
        cost = all_costs[iter_idx]
        n_pts = len(arr)

        # Which timesteps to render
        steps = list(range(0, n_pts, step_skip))
        if steps[-1] != n_pts - 1:
            steps.append(n_pts - 1)

        # Label
        if iter_idx == 0:
            iter_label = "Iteration 0 (PPO)"
        elif iter_idx == n_total - 1:
            iter_label = f"Iteration {iter_idx} (final)"
        else:
            iter_label = f"Iteration {iter_idx}"

        for frame_i, t in enumerate(steps):
            fig.clf()
            ax = fig.add_subplot(111, projection="3d")
            fig.subplots_adjust(left=0.0, right=1.0, bottom=0.02, top=0.91)

            setup_scene(ax, cfg)

            # ── Previous iterations: faded dashed trails ────────────────
            for prev_it in show_iters:
                if prev_it >= iter_idx:
                    break
                prev_arr = np.array(all_trajs[prev_it])
                for a in range(N_AGENTS):
                    ps = a * STATE_DIM_PER
                    ax.plot(prev_arr[:, ps], prev_arr[:, ps + 1],
                            prev_arr[:, ps + 2],
                            color=TRAIL_COLORS_FADED[a], alpha=0.3,
                            linewidth=0.9, linestyle="--", zorder=2)

            # ── Current iteration: trail + future + drone ───────────────
            for a in range(N_AGENTS):
                ps = a * STATE_DIM_PER
                col = AGENT_COLORS[a]

                # Solid trail up to current time
                if t > 0:
                    trail = arr[:t + 1]
                    ax.plot(trail[:, ps], trail[:, ps + 1], trail[:, ps + 2],
                            color=col, linewidth=2.0, alpha=0.75, zorder=4)

                # Dotted future path
                if t < n_pts - 1:
                    future = arr[t:]
                    ax.plot(future[:, ps], future[:, ps + 1],
                            future[:, ps + 2],
                            color=col, linewidth=0.8, alpha=0.25,
                            linestyle=":", zorder=3)

                # Drone at current position
                pos = arr[t, ps:ps + 3]
                draw_crazyflie(ax, pos, size=0.17, color=col)

            # ── Title & info ────────────────────────────────────────────
            fig.suptitle(iter_label, fontsize=13, fontweight="bold",
                         y=0.95, color="#1A1A1A")

            time_val = t * cfg.dt
            info = f"J = {cost:.2f}    t = {time_val:.1f}s  (step {t}/{n_pts - 1})"
            ax.text2D(0.5, 0.02, info, transform=ax.transAxes,
                      ha="center", fontsize=10, fontweight="bold",
                      color="#2255AA")

            # ── Render frame ────────────────────────────────────────────
            pil_frames.append(fig_to_pil(fig, dpi=dpi))

            if frame_i == 0:
                durations.append(ms_hold_start)
            elif t == n_pts - 1:
                durations.append(ms_hold_end)
            else:
                durations.append(ms_per_step)

        print(f"  {iter_label}: {len(steps)} frames rendered")

    plt.close(fig)

    # ── Assemble GIF ────────────────────────────────────────────────────
    pil_frames[0].save(
        output,
        save_all=True,
        append_images=pil_frames[1:],
        duration=durations,
        loop=0,
    )
    print(f"\nSaved: {output}  ({len(pil_frames)} frames, "
          f"{sum(durations) / 1000:.1f}s total)")

"""# GIF Generation"""

if __name__ == "__main__":
    # Step 1: Run full PI pipeline
    all_trajs, all_costs, cfg = run_pi()

    # Step 2: Animate iterations 0, 4, and the last one
    n = len(all_trajs)
    show = [0, min(4, n - 1), n - 1]

    generate_flight_animation(
        all_trajs, all_costs, cfg,
        show_iters=show,
        output="multiagent_flight_animation.gif",
        step_skip=2,       # show every 2nd timestep
        ms_per_step=120,   # 120ms per flight frame
        ms_hold_start=1800,
        ms_hold_end=2500,
    )