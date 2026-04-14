"""
Policy Iteration via Rollout — Multi-Agent (2 Drones), Env 1 obstacles.

Copyright (c) 2026 Fei Chen

Joint state: [p1(3), v1(3), p2(3), v2(3)] = 12D
Joint control: [a1(3), a2(3)] = 6D
Same 12 static obstacles as env1, plus inter-agent repulsion.

Drone 1: (-2, -2, 0) → goal (2, 2, 1)
Drone 2: (2, -2, 0.5) → goal (-2, 2, 0.5)
Trajectories designed to cross near the center.

Requirements: numpy, torch, matplotlib
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import matplotlib
# # Detect Colab / Jupyter: they handle display via the inline backend, so don't force Agg.  ---- uncomment for creating jupyter notebook
# def _in_notebook():
#     try:
#         from IPython import get_ipython
#         shell = get_ipython().__class__.__name__
#         return shell in ("ZMQInteractiveShell", "Shell", "Colab")
#     except Exception:
#         return False

# IN_NOTEBOOK = _in_notebook()

# Prefer interactive backend on local macOS; use Agg only when explicitly headless.
if os.environ.get("FORCE_HEADLESS", "0") == "1":
    matplotlib.use("Agg")
# elif IN_NOTEBOOK:
#     # Let Colab/Jupyter pick its inline backend automatically; do not override.
#     pass
elif sys.platform == "darwin":
    try:
        matplotlib.use("MacOSX")
    except Exception:
        pass
elif os.environ.get("DISPLAY") is None and os.environ.get("WAYLAND_DISPLAY") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ======================================================================
# Configuration
# ======================================================================
@dataclass
class Config:
    dt: float = 0.25
    N: int = 40
    V_MAX: float = 4.0
    A_MAX_XY: float = 3.0
    A_MAX_Z: float = 3.5

    x0_agent1: np.ndarray = None
    x0_agent2: np.ndarray = None
    goal1_center: np.ndarray = None
    goal2_center: np.ndarray = None
    goal_half_size: np.ndarray = None
    cost_goal_half_size: np.ndarray = None

    lambda_u: float = 0.1
    c_obs: float = 0.3 
    eps_obs: float = 0.05
    alpha_goal: float = 0.5
    C_p: float = 40.0
    C_v: float = 10.0
    # Half-width of the centered control grid used in the rollout coordinate
    # descent (in acceleration units). Linearly decays from _max at k=0 to
    # _min at k=N-1: wider trust region early (faster correction of bad base
    # actions), finer grid late (smooth, wobble-free terminal control).
    grid_half_width_max: float = 0.6
    grid_half_width_min: float = 0.1

    c_inter: float = 0.8
    eps_inter: float = 0.02
    safe_radius: float = 0.4

    ppo_total_steps: int = 100_000
    ppo_lr: float = 3e-4
    ppo_hidden: int = 128
    ppo_gamma: float = 1.0
    ppo_lam: float = 0.95
    ppo_clip_eps: float = 0.2
    ppo_epochs: int = 10
    ppo_batch_size: int = 64
    ppo_entropy_coef: float = 0.01

    grid_points: int = 7
    n_policy_iters: int = 16

    nn_hidden: int = 128
    nn_epochs: int = 300
    nn_lr: float = 1e-3
    nn_batch: int = 128
    nn_perturb_samples: int = 600

    def __post_init__(self):
        if self.x0_agent1 is None:
            self.x0_agent1 = np.array([-2.0, -2.0, 0.0, 0, 0, 0], float)
        if self.x0_agent2 is None:
            self.x0_agent2 = np.array([2.0, -2.0, 0.5, 0, 0, 0], float)
        if self.goal1_center is None:
            self.goal1_center = np.array([2.0, 2.0, 1.0], float)
        if self.goal2_center is None:
            self.goal2_center = np.array([-2.0, 2.0, 0.5], float)
        if self.goal_half_size is None:
            self.goal_half_size = np.array([0.55, 0.55, 0.55], float)
        if self.cost_goal_half_size is None:
            self.cost_goal_half_size = np.array([0.1, 0.1, 0.1], float)

    @property
    def x0(self):
        return np.concatenate([self.x0_agent1, self.x0_agent2])


# ======================================================================
# Obstacles (same 12 as env1)
# ======================================================================
@dataclass
class BoxObstacle:
    center: np.ndarray
    half_size: np.ndarray

    def surface_dist_sq(self, p):
        d = np.abs(p - self.center) - self.half_size
        if np.all(d < 0):
            return 0.0
        return np.sum(np.maximum(d, 0.0)**2)

    def barrier(self, p, c, eps):
        return c / (self.surface_dist_sq(p) + eps)


@dataclass
class SphereObstacle:
    center: np.ndarray
    radius: float

    def surface_dist_sq(self, p):
        d = np.linalg.norm(p - self.center) - self.radius
        if d <= 0:
            return 0.0
        return d**2

    def barrier(self, p, c, eps):
        return c / (self.surface_dist_sq(p) + eps)


def build_obstacles():
    return [
        BoxObstacle(center=np.array([-1.2, -0.8, 0.2]), half_size=np.array([0.25, 0.3, 0.3])),
        SphereObstacle(center=np.array([-0.5, -1.5, 0.5]), radius=0.3),
        SphereObstacle(center=np.array([0.0, 0.3, 0.0]), radius=0.35),
        BoxObstacle(center=np.array([0.5, -0.5, 0.8]), half_size=np.array([0.3, 0.25, 0.25])),
        SphereObstacle(center=np.array([-0.8, 1.0, 0.4]), radius=0.28),
        BoxObstacle(center=np.array([1.2, 0.5, 0.3]), half_size=np.array([0.2, 0.3, 0.35])),
        SphereObstacle(center=np.array([0.3, 1.5, 1.0]), radius=0.25),
        SphereObstacle(center=np.array([1.5, 1.8, 0.6]), radius=0.3),
        BoxObstacle(center=np.array([-1.5, 0.5, 0.8]), half_size=np.array([0.2, 0.2, 0.3])),
        SphereObstacle(center=np.array([1.0, -1.0, 0.2]), radius=0.22),
        BoxObstacle(center=np.array([0.8, 0.8, 1.2]), half_size=np.array([0.2, 0.2, 0.2])),
        SphereObstacle(center=np.array([-0.3, -0.3, 1.0]), radius=0.2),
    ]


OBSTACLES = build_obstacles()

# ======================================================================
# Joint dynamics: x = [p1(3), v1(3), p2(3), v2(3)], u = [a1(3), a2(3)]
# ======================================================================
def step_single(p, v, a, cfg):
    a = a.copy()
    a[0:2] = np.clip(a[0:2], -cfg.A_MAX_XY, cfg.A_MAX_XY)
    a[2] = np.clip(a[2], -cfg.A_MAX_Z, cfg.A_MAX_Z)
    v = np.clip(v, -cfg.V_MAX, cfg.V_MAX)
    pn = p + cfg.dt * v + 0.5 * cfg.dt**2 * a
    vn = np.clip(v + cfg.dt * a, -cfg.V_MAX, cfg.V_MAX)
    return pn, vn


def step_dynamics(x, u, cfg):
    p1, v1, p2, v2 = x[0:3], x[3:6], x[6:9], x[9:12]
    a1, a2 = np.array(u[:3], float), np.array(u[3:6], float)
    p1n, v1n = step_single(p1, v1, a1, cfg)
    p2n, v2n = step_single(p2, v2, a2, cfg)
    return np.concatenate([p1n, v1n, p2n, v2n])


def clamp_control(u, cfg):
    u = np.array(u, float).ravel()[:6]
    u[0:2] = np.clip(u[0:2], -cfg.A_MAX_XY, cfg.A_MAX_XY)
    u[2] = np.clip(u[2], -cfg.A_MAX_Z, cfg.A_MAX_Z)
    u[3:5] = np.clip(u[3:5], -cfg.A_MAX_XY, cfg.A_MAX_XY)
    u[5] = np.clip(u[5], -cfg.A_MAX_Z, cfg.A_MAX_Z)
    return u


# ======================================================================
# Cost
# ======================================================================
def obstacle_barrier_cost_single(p, cfg):
    total = 0.0
    for obs in OBSTACLES:
        total += obs.barrier(p, cfg.c_obs, cfg.eps_obs)
    return total


def inter_agent_barrier(p1, p2, cfg):
    d = np.linalg.norm(p1 - p2) - cfg.safe_radius
    if d <= 0:
        dist_sq = 0.0
    else:
        dist_sq = d**2
    return cfg.c_inter / (dist_sq + cfg.eps_inter)


def goal_region_dist_sq(p, goal_center, goal_half_size):
    d = np.abs(p - goal_center) - goal_half_size
    return np.sum(np.maximum(d, 0.0)**2)


def stage_cost(x, u, cfg):
    p1, p2 = x[0:3], x[6:9]
    a1, a2 = u[:3], u[3:6]
    cost = cfg.lambda_u * (np.sum(a1**2) + np.sum(a2**2))
    cost += obstacle_barrier_cost_single(p1, cfg) + obstacle_barrier_cost_single(p2, cfg)
    cost += cfg.alpha_goal * np.sum((p1 - cfg.goal1_center)**2)
    cost += cfg.alpha_goal * np.sum((p2 - cfg.goal2_center)**2)
    cost += inter_agent_barrier(p1, p2, cfg)
    return cost


"""
def terminal_cost(x, cfg):
    p1, v1, p2, v2 = x[0:3], x[3:6], x[6:9], x[9:12]
    cost = 0.0
    cost += cfg.C_p * goal_region_dist_sq(p1, cfg.goal1_center, cfg.cost_goal_half_size)
    cost += 2.0 * np.sum((p1 - cfg.goal1_center)**2)
    cost += cfg.C_v * np.sum(v1**2)
    cost += obstacle_barrier_cost_single(p1, cfg)
    cost += cfg.C_p * goal_region_dist_sq(p2, cfg.goal2_center, cfg.cost_goal_half_size)
    cost += 2.0 * np.sum((p2 - cfg.goal2_center)**2)
    cost += cfg.C_v * np.sum(v2**2)
    cost += obstacle_barrier_cost_single(p2, cfg)
    cost += inter_agent_barrier(p1, p2, cfg)
    return cost
"""
def terminal_cost(x, cfg):
    p1, v1, p2, v2 = x[0:3], x[3:6], x[6:9], x[9:12]
    cost = 0.0
    cost += cfg.C_p * np.sum((p1 - cfg.goal1_center)**2)
    cost += cfg.C_v * np.sum(v1**2)
    cost += obstacle_barrier_cost_single(p1, cfg)
    cost += cfg.C_p * np.sum((p2 - cfg.goal2_center)**2)
    cost += cfg.C_v * np.sum(v2**2)
    cost += obstacle_barrier_cost_single(p2, cfg)
    cost += inter_agent_barrier(p1, p2, cfg)
    return cost


def trajectory_cost(traj, controls, cfg):
    J = terminal_cost(traj[-1], cfg)
    for k in range(len(controls)):
        J += stage_cost(traj[k], controls[k], cfg)
    return J


def grid_half_width_at(k, cfg):
    """Linearly decaying trust region for the rollout coordinate descent.

    delta_k = delta_max + (delta_min - delta_max) * k / (N - 1)

    Wider grid early (k=0 -> grid_half_width_max) speeds correction of bad
    base actions; finer grid late (k=N-1 -> grid_half_width_min) gives
    high-resolution terminal control and prevents tail wobble.
    """
    if cfg.N <= 1:
        return cfg.grid_half_width_min
    frac = k / (cfg.N - 1)
    return cfg.grid_half_width_max + (cfg.grid_half_width_min - cfg.grid_half_width_max) * frac



# ======================================================================
# PPO (joint policy for both agents)
# ======================================================================
class PPOPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden):
        super().__init__()
        self.mu = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )
        self.log_std = nn.Parameter(-0.5 * torch.ones(action_dim))
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.mu(x)

    def act(self, x_np, cfg, deterministic=False):
        with torch.no_grad():
            x_t = torch.tensor(x_np[:12], dtype=torch.float32).unsqueeze(0)
            mu = self.mu(x_t).squeeze(0)
            std = torch.exp(self.log_std)
            if deterministic:
                a = mu
            else:
                dist = torch.distributions.Normal(mu, std)
                a = dist.sample()
            v = self.value(x_t).squeeze()
            log_p = torch.distributions.Normal(mu, std).log_prob(a).sum()
            return clamp_control(a.numpy(), cfg), log_p.item(), v.item()


def rollout_policy_fn(action_fn, cfg):
    x0 = cfg.x0.copy()
    traj = [x0.copy()]
    controls = []
    x = x0.copy()
    for k in range(cfg.N):
        u = clamp_control(action_fn(x, k), cfg)
        controls.append(u.copy())
        x = step_dynamics(x, u, cfg)
        traj.append(x.copy())
    return traj, controls


def train_ppo(cfg):
    print("[PPO] Training joint initial policy (clipped surrogate)...")
    state_dim = 12
    action_dim = 6
    policy = PPOPolicy(state_dim, action_dim, cfg.ppo_hidden)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.ppo_lr)
    best_cost = float('inf')
    best_weights = None
    total_episodes = 0
    steps_collected = 0

    t0 = time.perf_counter()
    while steps_collected < cfg.ppo_total_steps:
        all_states, all_actions, all_logprobs, all_rewards, all_values, all_dones = \
            [], [], [], [], [], []
        for _ in range(5):
            x = cfg.x0.copy()
            ep_s, ep_a, ep_lp, ep_r, ep_v = [], [], [], [], []
            for k in range(cfg.N):
                s = x[:12].copy()
                a_np, logp, v = policy.act(x, cfg, deterministic=False)
                r = -stage_cost(x, a_np, cfg)
                ep_s.append(s); ep_a.append(a_np.copy())
                ep_lp.append(logp); ep_r.append(r); ep_v.append(v)
                x = step_dynamics(x, a_np, cfg)
            ep_r.append(-terminal_cost(x, cfg))
            _, _, v_last = policy.act(x, cfg, deterministic=False)
            ep_v.append(v_last)
            advantages = np.zeros(cfg.N, dtype=np.float32)
            gae = 0.0
            for t in reversed(range(cfg.N)):
                r_t = ep_r[t] + (ep_r[cfg.N] if t == cfg.N - 1 else 0)
                delta = r_t + cfg.ppo_gamma * ep_v[t + 1] - ep_v[t]
                gae = delta + cfg.ppo_gamma * cfg.ppo_lam * gae
                advantages[t] = gae
            returns = advantages + np.array(ep_v[:cfg.N], dtype=np.float32)
            all_states.extend(ep_s); all_actions.extend(ep_a)
            all_logprobs.extend(ep_lp); all_rewards.append(returns)
            all_values.extend(ep_v[:cfg.N]); all_dones.extend(advantages)
            total_episodes += 1; steps_collected += cfg.N

        states_t = torch.tensor(np.array(all_states), dtype=torch.float32)
        actions_t = torch.tensor(np.array(all_actions), dtype=torch.float32)
        old_lp_t = torch.tensor(np.array(all_logprobs), dtype=torch.float32)
        returns_t = torch.tensor(np.concatenate(all_rewards), dtype=torch.float32)
        adv_t = torch.tensor(np.array(all_dones), dtype=torch.float32)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        n_samples = len(states_t)
        for _ in range(cfg.ppo_epochs):
            idx = np.random.permutation(n_samples)
            for start in range(0, n_samples, cfg.ppo_batch_size):
                end = min(start + cfg.ppo_batch_size, n_samples)
                b = idx[start:end]
                mu = policy.mu(states_t[b])
                std = torch.exp(policy.log_std)
                dist = torch.distributions.Normal(mu, std)
                new_lp = dist.log_prob(actions_t[b]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                ratio = torch.exp(new_lp - old_lp_t[b])
                clipped = torch.clamp(ratio, 1 - cfg.ppo_clip_eps, 1 + cfg.ppo_clip_eps)
                p_loss = -torch.min(ratio * adv_t[b], clipped * adv_t[b]).mean()
                v_loss = nn.functional.mse_loss(policy.value(states_t[b]).squeeze(-1), returns_t[b])
                loss = p_loss + 0.5 * v_loss - cfg.ppo_entropy_coef * entropy
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5); optimizer.step()

        if total_episodes % 20 == 0:
            traj, ctrls = rollout_policy_fn(
                lambda x_np, k: policy.act(x_np, cfg, deterministic=True)[0], cfg)
            cost = trajectory_cost(traj, ctrls, cfg)
            if cost < best_cost:
                best_cost = cost
                best_weights = {k: v.clone() for k, v in policy.state_dict().items()}
            print(f"  steps={steps_collected}, ep={total_episodes}: "
                  f"cost={cost:.2f} (best={best_cost:.2f})")

    if best_weights is not None:
        policy.load_state_dict(best_weights)
    elapsed = time.perf_counter() - t0
    print(f"[PPO] Done in {elapsed:.1f}s, best cost={best_cost:.2f}")
    return policy


# ======================================================================
# Residual NN: 12D state deviation + k/N → 6D control correction
# ======================================================================
class ResidualNN(nn.Module):
    def __init__(self, state_dim=12, action_dim=6, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )
        self.state_dim = state_dim

    def forward(self, dx, k_norm):
        inp = torch.cat([dx, k_norm], dim=-1)
        out = self.net(inp)
        inp_zero = torch.cat([torch.zeros_like(dx), k_norm], dim=-1)
        out_zero = self.net(inp_zero)
        return out - out_zero


class PolicyWithTrajectory:
    def __init__(self, ref_traj, ref_controls, residual_nn, cfg):
        self.ref_traj = [x.copy() for x in ref_traj]
        self.ref_controls = [u.copy() for u in ref_controls]
        self.residual_nn = residual_nn
        self.cfg = cfg
        self.N = len(ref_controls)

    def get_action(self, x, k):
        if k < 0 or k >= self.N:
            return np.zeros(6)
        u_ff = self.ref_controls[k]
        if self.residual_nn is None:
            return clamp_control(u_ff, self.cfg)
        dx = x[:12] - self.ref_traj[k][:12]
        with torch.no_grad():
            dx_t = torch.tensor(dx, dtype=torch.float32).unsqueeze(0)
            k_norm = torch.tensor([[k / self.N]], dtype=torch.float32)
            du = self.residual_nn(dx_t, k_norm).squeeze(0).numpy()
        return clamp_control(u_ff + du, self.cfg)


def generate_perturbation_data(ref_traj, ref_controls, cfg):
    data = []
    N = len(ref_controls)
    u_min = np.array([-cfg.A_MAX_XY, -cfg.A_MAX_XY, -cfg.A_MAX_Z,
                       -cfg.A_MAX_XY, -cfg.A_MAX_XY, -cfg.A_MAX_Z])
    u_max = -u_min
    for _ in range(cfg.nn_perturb_samples):
        k = np.random.randint(0, N)
        x_pert = ref_traj[k].copy()
        x_pert[0:3] += np.random.randn(3) * 0.3
        x_pert[3:6] += np.random.randn(3) * 0.2
        x_pert[6:9] += np.random.randn(3) * 0.3
        x_pert[9:12] += np.random.randn(3) * 0.2
        best_u = ref_controls[k].copy()
        best_J = _eval_cost_to_go_stored(x_pert, best_u, k, ref_controls, cfg)
        for _ in range(15):
            u_cand = np.clip(ref_controls[k] + np.random.randn(6) * 0.5, u_min, u_max)
            J_cand = _eval_cost_to_go_stored(x_pert, u_cand, k, ref_controls, cfg)
            if J_cand < best_J:
                best_J = J_cand; best_u = u_cand.copy()
        data.append((x_pert.copy(), best_u.copy(), k))
    return data


def _eval_cost_to_go_stored(x, u_first, k, ref_controls, cfg):
    J = stage_cost(x, u_first, cfg)
    xc = step_dynamics(x, u_first, cfg)
    for t in range(k + 1, len(ref_controls)):
        J += stage_cost(xc, ref_controls[t], cfg)
        xc = step_dynamics(xc, ref_controls[t], cfg)
    J += terminal_cost(xc, cfg)
    return J


def train_residual_nn(ref_traj, ref_controls, extra_data, cfg):
    nn_model = ResidualNN(state_dim=12, action_dim=6, hidden=cfg.nn_hidden)
    optimizer = optim.Adam(nn_model.parameters(), lr=cfg.nn_lr)
    N = len(ref_controls)
    dx_list, du_list = [], []
    for x_actual, u_target, k in extra_data:
        if k >= N:
            continue
        dx = x_actual[:12] - ref_traj[k][:12]
        du = u_target - ref_controls[k]
        dx_list.append(np.append(dx, k / N))
        du_list.append(du)
    if len(dx_list) < 10:
        return nn_model
    dx_arr = torch.tensor(np.array(dx_list), dtype=torch.float32)
    du_arr = torch.tensor(np.array(du_list), dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(dx_arr, du_arr)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.nn_batch, shuffle=True)
    nn_model.train()
    for _ in range(cfg.nn_epochs):
        for batch_dx, batch_du in loader:
            pred = nn_model(batch_dx[:, :12], batch_dx[:, 12:13])
            loss = nn.functional.mse_loss(pred, batch_du)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    nn_model.eval()
    return nn_model


# ======================================================================
# Rollout: 6D coordinate descent with parallel grid evaluation
# ======================================================================
_worker_policy = None
_worker_cfg = None


def _init_worker(ref_traj_np, ref_controls_np, nn_state_dict, nn_hidden, cfg_dict):
    """Initialize per-worker policy object (called once per process)."""
    global _worker_policy, _worker_cfg
    _worker_cfg = Config(**cfg_dict)
    ref_traj = [ref_traj_np[i] for i in range(len(ref_traj_np))]
    ref_controls = [ref_controls_np[i] for i in range(len(ref_controls_np))]
    if nn_state_dict is not None:
        rnn = ResidualNN(state_dim=12, action_dim=6, hidden=nn_hidden)
        rnn.load_state_dict(nn_state_dict)
        rnn.eval()
    else:
        rnn = None
    _worker_policy = PolicyWithTrajectory(ref_traj, ref_controls, rnn, _worker_cfg)


def _eval_candidate(args):
    """Worker: evaluate one candidate control, return cost."""
    x, u_cand, k = args
    return _eval_tail_cost_nn(x, u_cand, k, _worker_policy, _worker_cfg)


def _eval_candidate_batch(args):
    """Worker: evaluate a batch of candidate controls, return list of costs."""
    x, u_list, k = args
    return [_eval_tail_cost_nn(x, u, k, _worker_policy, _worker_cfg) for u in u_list]


def _pack_policy_for_workers(policy, cfg):
    """Serialize policy + cfg into picklable objects for worker init."""
    ref_traj_np = np.array(policy.ref_traj)
    ref_controls_np = np.array(policy.ref_controls)
    nn_sd = None
    nn_hidden = cfg.nn_hidden
    if policy.residual_nn is not None:
        nn_sd = {key: val.cpu() for key, val in policy.residual_nn.state_dict().items()}
    cfg_dict = {
        'dt': cfg.dt, 'N': cfg.N, 'V_MAX': cfg.V_MAX,
        'A_MAX_XY': cfg.A_MAX_XY, 'A_MAX_Z': cfg.A_MAX_Z,
        'x0_agent1': cfg.x0_agent1.copy(), 'x0_agent2': cfg.x0_agent2.copy(),
        'goal1_center': cfg.goal1_center.copy(), 'goal2_center': cfg.goal2_center.copy(),
        'goal_half_size': cfg.goal_half_size.copy(),
        'cost_goal_half_size': cfg.cost_goal_half_size.copy(),
        'lambda_u': cfg.lambda_u, 'c_obs': cfg.c_obs, 'eps_obs': cfg.eps_obs,
        'alpha_goal': cfg.alpha_goal, 'C_p': cfg.C_p, 'C_v': cfg.C_v,
        'c_inter': cfg.c_inter, 'eps_inter': cfg.eps_inter, 'safe_radius': cfg.safe_radius,
        'grid_points': cfg.grid_points,
        'grid_half_width_max': cfg.grid_half_width_max,
        'grid_half_width_min': cfg.grid_half_width_min,
        'n_policy_iters': cfg.n_policy_iters,
        'nn_hidden': cfg.nn_hidden, 'nn_epochs': cfg.nn_epochs, 'nn_lr': cfg.nn_lr,
        'nn_batch': cfg.nn_batch, 'nn_perturb_samples': cfg.nn_perturb_samples,
        'ppo_total_steps': cfg.ppo_total_steps, 'ppo_lr': cfg.ppo_lr,
        'ppo_hidden': cfg.ppo_hidden, 'ppo_gamma': cfg.ppo_gamma, 'ppo_lam': cfg.ppo_lam,
        'ppo_clip_eps': cfg.ppo_clip_eps, 'ppo_epochs': cfg.ppo_epochs,
        'ppo_batch_size': cfg.ppo_batch_size, 'ppo_entropy_coef': cfg.ppo_entropy_coef,
    }
    return ref_traj_np, ref_controls_np, nn_sd, nn_hidden, cfg_dict


N_WORKERS = 7


def run_rollout_iteration(policy, cfg):
    u_min = np.array([-cfg.A_MAX_XY, -cfg.A_MAX_XY, -cfg.A_MAX_Z,
                       -cfg.A_MAX_XY, -cfg.A_MAX_XY, -cfg.A_MAX_Z])
    u_max = -u_min
    # Note: the grid is now built per-step, centered on u_base, inside the k-loop.
    # This implements the centered-grid construction from Eq. 31 of the paper
    # and prevents tail wobble caused by a coarse full-range grid.

    init_args = _pack_policy_for_workers(policy, cfg)
    pool = mp.Pool(processes=N_WORKERS, initializer=_init_worker, initargs=init_args)

    # Warmup: ensure all worker processes are spawned before timing starts
    pool.map(_eval_candidate, [(cfg.x0, np.zeros(6), 0)] * N_WORKERS)

    traj = [cfg.x0.copy()]
    controls = []
    n_improved = 0
    per_step_times = []
    t_total = time.perf_counter()
    x = cfg.x0.copy()

    for k in range(cfg.N):
        t_step = time.perf_counter()
        u_base = policy.get_action(x, k)
        best_u = u_base.copy()
        best_J = _eval_tail_cost_nn(x, u_base, k, policy, cfg)

        # Build a centered grid around u_base for this step, clipped to
        # the actuator box. Half-width decays linearly with k from
        # grid_half_width_max (at k=0) to grid_half_width_min (at k=N-1).
        hw = grid_half_width_at(k, cfg)
        grid_vals = []
        for d in range(6):
            lo = max(u_min[d], u_base[d] - hw)
            hi = min(u_max[d], u_base[d] + hw)
            if hi <= lo:
                lo, hi = u_min[d], u_max[d]
            grid_vals.append(np.linspace(lo, hi, cfg.grid_points))

        cur = u_base.copy()

        for dim in range(6):
            cands = []
            for val in grid_vals[dim]:
                cand = cur.copy(); cand[dim] = float(val)
                cands.append(clamp_control(cand, cfg))

            # 7 candidates → 7 workers in parallel, one pool.map call
            tasks = [(x.copy(), c, k) for c in cands]
            costs = pool.map(_eval_candidate, tasks)

            for idx, J_cand in enumerate(costs):
                if J_cand < best_J - 1e-12:
                    best_J = J_cand
                    best_u = cands[idx].copy()
                    cur = cands[idx].copy()

        if not np.allclose(best_u, u_base, atol=1e-8):
            n_improved += 1
        per_step_times.append(time.perf_counter() - t_step)
        controls.append(best_u.copy())
        x = step_dynamics(x, best_u, cfg)
        traj.append(x.copy())

    pool.close()
    pool.join()
    elapsed = time.perf_counter() - t_total
    return traj, controls, n_improved, per_step_times, elapsed


def _eval_tail_cost_nn(x, u_first, k, policy, cfg):
    J = stage_cost(x, u_first, cfg)
    xc = step_dynamics(x, u_first, cfg)
    for t in range(k + 1, cfg.N):
        u_t = policy.get_action(xc, t)
        J += stage_cost(xc, u_t, cfg)
        xc = step_dynamics(xc, u_t, cfg)
    J += terminal_cost(xc, cfg)
    return J


# ======================================================================
# Consistency check
# ======================================================================
def check_consistency(policy, cfg):
    x = cfg.x0.copy()
    max_dev = 0.0
    for k in range(policy.N):
        u = policy.get_action(x, k)
        dev = np.max(np.abs(u - clamp_control(policy.ref_controls[k], cfg)))
        max_dev = max(max_dev, dev)
        x = step_dynamics(x, u, cfg)
        pos_dev = np.max(np.abs(x[:12] - policy.ref_traj[k + 1][:12]))
        max_dev = max(max_dev, pos_dev)
    return max_dev


# ======================================================================
# Visualization
# ======================================================================
def draw_crazyflie(ax, pos, size=0.15, color_prop='#1a73e8'):
    cx, cy, cz = pos
    prop_r = size * 0.8
    arms = [np.array([1, 1, 0]) / np.sqrt(2), np.array([1, -1, 0]) / np.sqrt(2),
            np.array([-1, 1, 0]) / np.sqrt(2), np.array([-1, -1, 0]) / np.sqrt(2)]
    for arm_dir in arms:
        tip = np.array([cx, cy, cz]) + size * arm_dir
        ax.plot([cx, tip[0]], [cy, tip[1]], [cz, tip[2]],
                color='#333333', linewidth=2.5, zorder=10)
        theta = np.linspace(0, 2 * np.pi, 20)
        ax.plot(tip[0] + prop_r * np.cos(theta), tip[1] + prop_r * np.sin(theta),
                np.full(20, tip[2]), color=color_prop, linewidth=1.5, alpha=0.8, zorder=10)
    ax.scatter([cx], [cy], [cz], color=color_prop, s=40, zorder=11)


def draw_box_3d(ax, center, half_size, color='gray', alpha=0.3, edgecolor='black'):
    c, h = center, half_size
    lo, hi = c - h, c + h
    v = np.array([[lo[0],lo[1],lo[2]], [hi[0],lo[1],lo[2]], [hi[0],hi[1],lo[2]],
                   [lo[0],hi[1],lo[2]], [lo[0],lo[1],hi[2]], [hi[0],lo[1],hi[2]],
                   [hi[0],hi[1],hi[2]], [lo[0],hi[1],hi[2]]])
    faces = [[v[0],v[1],v[5],v[4]], [v[2],v[3],v[7],v[6]], [v[0],v[3],v[7],v[4]],
             [v[1],v[2],v[6],v[5]], [v[0],v[1],v[2],v[3]], [v[4],v[5],v[6],v[7]]]
    ax.add_collection3d(Poly3DCollection(faces, alpha=alpha, facecolor=color,
                                          edgecolor=edgecolor, linewidth=0.5))


def draw_sphere_3d(ax, center, radius, color='red', alpha=0.2):
    u = np.linspace(0, 2*np.pi, 32)
    v = np.linspace(0, np.pi, 20)
    ax.plot_surface(center[0] + radius * np.outer(np.cos(u), np.sin(v)),
                    center[1] + radius * np.outer(np.sin(u), np.sin(v)),
                    center[2] + radius * np.outer(np.ones_like(u), np.cos(v)),
                    alpha=alpha, color=color, linewidth=0)


# ======================================================================
# Main
# ======================================================================
def main():
    cfg = Config()
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 70)
    print("Policy Iteration via Rollout — Multi-Agent (2 Drones)")
    print("=" * 70)
    print(f"N={cfg.N}, dt={cfg.dt}")
    print(f"Drone 1: {cfg.x0_agent1[:3]} → goal {cfg.goal1_center}")
    print(f"Drone 2: {cfg.x0_agent2[:3]} → goal {cfg.goal2_center}")
    print(f"Obstacles: {len(OBSTACLES)} static + inter-agent repulsion")
    print()

    ppo_t0 = time.perf_counter()
    ppo_policy = train_ppo(cfg)
    ppo_time = time.perf_counter() - ppo_t0

    traj_0, ctrl_0 = rollout_policy_fn(
        lambda x, k: ppo_policy.act(x, cfg, deterministic=True)[0], cfg)
    cost_0 = trajectory_cost(traj_0, ctrl_0, cfg)
    print(f"\n[Iter 0] PPO policy: cost={cost_0:.2f}, time={ppo_time:.1f}s")

    all_trajs = [traj_0]
    all_ctrls = [ctrl_0]
    all_costs = [cost_0]
    all_rollout_times = []
    all_rollout_step_means = []
    all_rollout_step_stds = []
    all_nn_times = []
    all_n_improved = []
    all_consistency = []

    current_policy = PolicyWithTrajectory(traj_0, ctrl_0, None, cfg)
    no_improve_count = 0

    for iteration in range(1, cfg.n_policy_iters + 1):
        print(f"\n{'='*60}")
        print(f"[Iter {iteration}]")

        cons_dev = check_consistency(current_policy, cfg)
        all_consistency.append(cons_dev)
        print(f"  Consistency check: max deviation = {cons_dev:.2e}")

        new_traj, new_ctrl, n_improved, step_times, rollout_time = \
            run_rollout_iteration(current_policy, cfg)
        new_cost = trajectory_cost(new_traj, new_ctrl, cfg)
        step_mean = np.mean(step_times)
        step_std = np.std(step_times)
        all_rollout_times.append(rollout_time)
        all_rollout_step_means.append(step_mean)
        all_rollout_step_stds.append(step_std)
        all_n_improved.append(n_improved)

        prev_cost = all_costs[-1]
        print(f"  Rollout: cost={new_cost:.2f} (prev={prev_cost:.2f}, "
              f"Δ={new_cost - prev_cost:+.2f}), "
              f"improved={n_improved}/{cfg.N}, time={rollout_time:.2f}s "
              f"(per-step: {step_mean*1000:.1f}±{step_std*1000:.1f}ms)")

        nn_t0 = time.perf_counter()
        perturb_data = generate_perturbation_data(new_traj, new_ctrl, cfg)
        residual_nn = train_residual_nn(new_traj, new_ctrl, perturb_data, cfg)
        nn_time = time.perf_counter() - nn_t0
        all_nn_times.append(nn_time)
        print(f"  NN training: {len(perturb_data)} samples, time={nn_time:.1f}s")

        current_policy = PolicyWithTrajectory(new_traj, new_ctrl, residual_nn, cfg)

        all_trajs.append(new_traj)
        all_ctrls.append(new_ctrl)
        all_costs.append(new_cost)

        if n_improved == 0:
            no_improve_count += 1
        else:
            no_improve_count = 0
        if no_improve_count >= 2:
            print("  Converged (no improvement for 2 consecutive iterations).")
            break

    cons_final = check_consistency(current_policy, cfg)
    all_consistency.append(cons_final)
    print(f"\n  Final consistency check: max deviation = {cons_final:.2e}")

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Iter':>4} | {'Cost':>8} | {'ΔCost':>8} | {'Impr':>4} | "
          f"{'Rollout(s)':>10} | {'Step(ms)':>14} | {'NN(s)':>6} | {'Consist':>9}")
    print("-" * 90)
    print(f"   0 | {all_costs[0]:>8.2f} | {'—':>8} | {'—':>4} | "
          f"{'PPO':>10} | {'—':>14} | {'—':>6} | {'—':>9}")
    for i in range(len(all_rollout_times)):
        dc = all_costs[i+1] - all_costs[i]
        sm = all_rollout_step_means[i]
        ss = all_rollout_step_stds[i]
        cons = all_consistency[i]
        print(f"{i+1:>4} | {all_costs[i+1]:>8.2f} | {dc:>+8.2f} | {all_n_improved[i]:>4} | "
              f"{all_rollout_times[i]:>10.2f} | {sm*1000:>6.1f}±{ss*1000:<5.1f} | "
              f"{all_nn_times[i]:>6.1f} | {cons:>9.2e}")

    final_x = all_trajs[-1][-1]
    p1f, p2f = final_x[0:3], final_x[6:9]
    in_g1 = np.all(np.abs(p1f - cfg.goal1_center) <= cfg.goal_half_size)
    in_g2 = np.all(np.abs(p2f - cfg.goal2_center) <= cfg.goal_half_size)
    print(f"\nDrone 1 final: {np.round(p1f, 3)}, in_goal={in_g1}")
    print(f"Drone 2 final: {np.round(p2f, 3)}, in_goal={in_g2}")

    min_dist = float('inf')
    for st in all_trajs[-1]:
        d = np.linalg.norm(st[0:3] - st[6:9])
        min_dist = min(min_dist, d)
    print(f"Min inter-agent distance (final traj): {min_dist:.3f}")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8.5, 6))
    iters = np.arange(len(all_costs))
    ax1.plot(iters, all_costs, color='#1f4bd8', linewidth=2.4, alpha=0.95, zorder=2)
    ax1.scatter(iters, all_costs, color='white', edgecolor='#1f4bd8',
                s=60, linewidth=1.8, zorder=3)
    ax1.set_xlabel(r"Iteration $\ell$", fontsize=24, fontname="Times New Roman")
    ax1.set_ylabel(r"Total Cost $J_{\pi^{\ell}}$", fontsize=24, fontname="Times New Roman")
    ax1.tick_params(axis='both', labelsize=16)
    ax1.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    if len(all_costs) > 1:
        ax1.fill_between(iters, all_costs, np.max(all_costs), color='#1f4bd8', alpha=0.06)
    for i, c in enumerate(all_costs):
        ax1.annotate(f"{c:.1f}", (i, c), textcoords="offset points",
                     xytext=(0, 8), ha='center', fontsize=18, color='#1a1a1a')

    fig2 = plt.figure(figsize=(11, 8))
    ax2 = fig2.add_subplot(111, projection='3d')

    n_trajs = len(all_trajs)
    highlight_iter4 = min(4, n_trajs - 1)
    show_iters = sorted({0, highlight_iter4, n_trajs - 1})

    # Drone identity: color, marker, goal box color (extensible to 3-4 drones)
    agent_colors = ['#0b5394', '#0b8a8a', '#7b2d8e', '#8c564b']
    agent_markers = ['^', 's', 'D', 'P']
    agent_names = ['Drone 1', 'Drone 2']
    goal_centers = [cfg.goal1_center, cfg.goal2_center]
    n_agents = 2
    agent_state_slices = [(0, 3, 6), (6, 9, 8)]  # (pos_start, vel_start_unused, col_start_for_arr)

    # Iteration styles: Iter 0 = dashed, mid = solid thinner, final = solid thick
    iter_palette = ['#ff7f0e', '#d62728', '#2ca02c', '#9467bd',
                    '#8c564b', '#e377c2', '#17becf', '#bcbd22']
    iter_colors = {}
    iter_styles = {}
    rank = 0
    for i in show_iters:
        if i == 0:
            iter_styles[i] = ('--', 2.0)
        elif i == n_trajs - 1:
            iter_colors[i] = iter_palette[rank % len(iter_palette)]
            iter_styles[i] = ('-', 3.0)
            rank += 1
        else:
            iter_colors[i] = iter_palette[rank % len(iter_palette)]
            iter_styles[i] = ('-', 2.4)
            rank += 1

    for i in show_iters:
        arr = np.array(all_trajs[i])
        ls, lw = iter_styles[i]
        n_pts = len(arr)
        mk_idx = [n_pts // 3, 2 * n_pts // 3]
        for a in range(n_agents):
            px = a * 6  # position index in joint state
            col = agent_colors[a] if i == 0 else iter_colors[i]
            ax2.plot(arr[:, px], arr[:, px+1], arr[:, px+2],
                     color=col, linewidth=lw, linestyle=ls, alpha=0.95)
            ax2.plot(arr[mk_idx, px], arr[mk_idx, px+1], arr[mk_idx, px+2],
                     color=agent_colors[a], marker=agent_markers[a], markersize=8,
                     linestyle='none', alpha=0.9,
                     markeredgecolor='white', markeredgewidth=0.5)

    final_arr = np.array(all_trajs[-1])
    for a in range(n_agents):
        px = a * 6
        ax2.scatter(*final_arr[0, px:px+3], s=55, color=agent_colors[a],
                    edgecolor='white', linewidth=0.8, zorder=12)
        ax2.scatter(*final_arr[-1, px:px+3], s=85, marker='*',
                    color=agent_colors[a], edgecolor='white', linewidth=0.8, zorder=13)

    for obs in OBSTACLES:
        if isinstance(obs, BoxObstacle):
            draw_box_3d(ax2, obs.center, obs.half_size, color='#6b7280',
                        alpha=0.30, edgecolor='#374151')
        elif isinstance(obs, SphereObstacle):
            draw_sphere_3d(ax2, obs.center, obs.radius, color='#e11d48', alpha=0.24)

    for a in range(n_agents):
        draw_box_3d(ax2, goal_centers[a], cfg.goal_half_size,
                    color=agent_colors[a], alpha=0.22,
                    edgecolor=agent_colors[a])

    for a in range(n_agents):
        px = a * 6
        draw_crazyflie(ax2, cfg.x0[px:px+3], size=0.18, color_prop=agent_colors[a])

    # Legend: agent entries (marker + color) then iteration entries (line style)
    from matplotlib.lines import Line2D
    legend_handles = []
    for a in range(n_agents):
        legend_handles.append(Line2D([], [], color=agent_colors[a],
                                     marker=agent_markers[a], markersize=9,
                                     linestyle='-', linewidth=2.0,
                                     markeredgecolor='white', markeredgewidth=0.5,
                                     label=agent_names[a]))
    legend_handles.append(Line2D([], [], color='#333333', linewidth=2.0,
                                 linestyle='--', label='Iter 0 (PPO)'))
    for i in show_iters:
        if i == 0:
            continue
        ls, lw = iter_styles[i]
        legend_handles.append(Line2D([], [], color=iter_colors[i], linewidth=lw,
                                     linestyle=ls, label=f'Iter {i}'))
    legend2 = ax2.legend(handles=legend_handles, fontsize=14, loc='upper right',
                         framealpha=0.92)
    legend2.set_draggable(True)

    ax2.set_xlim(-3.0, 3.5)
    ax2.set_ylim(-3.0, 3.5)
    ax2.set_zlim(-1.0, 2.0)
    ax2.set_box_aspect((1.0, 1.0, 0.55))
    ax2.view_init(elev=26, azim=-55)
    ax2.set_xlabel("$X$", fontsize=24, labelpad=10)
    ax2.set_ylabel("$Y$", fontsize=24, labelpad=10)
    ax2.set_zlabel("$Z$", fontsize=24, labelpad=10)
    ax2.tick_params(axis='x', labelsize=14, pad=4)
    ax2.tick_params(axis='y', labelsize=14, pad=4)
    ax2.tick_params(axis='z', labelsize=14, pad=4)
    ax2.grid(True, alpha=0.18)
    try:
        ax2.xaxis.pane.set_facecolor((0.96, 0.96, 0.96, 0.35))
        ax2.yaxis.pane.set_facecolor((0.96, 0.96, 0.96, 0.35))
        ax2.zaxis.pane.set_facecolor((0.96, 0.96, 0.96, 0.35))
    except Exception:
        pass

    fig1.tight_layout()
    fig2.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)

    # # Always save figures to disk so they are accessible on Colab / headless runs. ---- uncomment for saving figures
    # out_dir = os.environ.get("FIG_OUT_DIR", ".")
    # os.makedirs(out_dir, exist_ok=True)
    # fig1_path = os.path.join(out_dir, "two_drone_env1_cost.pdf")
    # fig2_path = os.path.join(out_dir, "two_drone_env1_trajectory.pdf")
    # fig1.savefig(fig1_path, dpi=150, bbox_inches="tight")
    # fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
    # print(f"[figures] saved:\n  {fig1_path}\n  {fig2_path}")

    plt.show()

if __name__ == "__main__":
    main()