"""
Policy Iteration via Rollout for Nonlinear Optimal Control of a 3D Drone.

Copyright (c) 2026 Fei Chen

Pipeline:
  1. Train initial policy π⁰ via PPO
  2. For each iteration ℓ:
     a. Roll out using NN policy as base with coordinate-descent improvement
        → improved trajectory τ^{ℓ+1}
        Tail uses NN policy; on-trajectory consistency guarantees monotonicity.
     b. Train residual NN from perturbation data.
     c. Assemble π^{ℓ+1}(x,k) = û_k + [nn(x-x̂_k, k/N) - nn(0, k/N)]
        The centering ensures π^{ℓ+1}(x̂_k, k) = û_k exactly.
  3. Report cost, timing, trajectory plots.

Cost (stage-wise additive):
  J = g_N(x_N) + Σ_{k=0}^{N-1} g_k(x_k, u_k)

Dynamics: 3D double integrator
  p_{k+1} = p_k + dt v_k + 0.5 dt² a_k
  v_{k+1} = v_k + dt a_k

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
import matplotlib

# Prefer interactive backend on local macOS; use Agg only when explicitly headless.
if os.environ.get("FORCE_HEADLESS", "0") == "1":
    matplotlib.use("Agg")
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

    x0: np.ndarray = None
    goal_center: np.ndarray = None
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

    ppo_total_steps: int = 20_000
    ppo_lr: float = 3e-4
    ppo_hidden: int = 64
    ppo_gamma: float = 1.0
    ppo_lam: float = 0.95
    ppo_clip_eps: float = 0.2
    ppo_epochs: int = 10
    ppo_batch_size: int = 64
    ppo_entropy_coef: float = 0.01

    grid_points: int = 7
    n_policy_iters: int = 12

    nn_hidden: int = 64
    nn_epochs: int = 300
    nn_lr: float = 1e-3
    nn_batch: int = 128
    nn_perturb_samples: int = 500

    def __post_init__(self):
        if self.x0 is None:
            self.x0 = np.array([-2.0, -2.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0], float)
        if self.goal_center is None:
            self.goal_center = np.array([2.0, 2.0, 1.0], float)
        if self.goal_half_size is None:
            self.goal_half_size = np.array([0.5, 0.5, 0.5], float)
        if self.cost_goal_half_size is None:
            self.cost_goal_half_size = np.array([0.1, 0.1, 0.1], float)


# ======================================================================
# Obstacles (dispersed across the workspace)
# ======================================================================
@dataclass
class BoxObstacle:
    center: np.ndarray
    half_size: np.ndarray

    def surface_dist_sq(self, p: np.ndarray) -> float:
        d = np.abs(p - self.center) - self.half_size
        if np.all(d < 0):
            return 0.0
        return np.sum(np.maximum(d, 0.0)**2)

    def barrier(self, p: np.ndarray, c: float, eps: float) -> float:
        return c / (self.surface_dist_sq(p) + eps)


@dataclass
class SphereObstacle:
    center: np.ndarray
    radius: float

    def surface_dist_sq(self, p: np.ndarray) -> float:
        d = np.linalg.norm(p - self.center) - self.radius
        if d <= 0:
            return 0.0
        return d**2

    def barrier(self, p: np.ndarray, c: float, eps: float) -> float:
        return c / (self.surface_dist_sq(p) + eps)


def build_obstacles():
    """Obstacles dispersed across the workspace [-2.5, 2.5]^2 x [-0.5, 2]."""
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
# Dynamics
# ======================================================================
def step_dynamics(x: np.ndarray, u: np.ndarray, cfg: Config) -> np.ndarray:
    x = x.copy()
    x[3:6] = np.clip(x[3:6], -cfg.V_MAX, cfg.V_MAX)
    p, v = x[0:3], x[3:6]
    a = np.array(u, float).ravel()[:3]
    a[0:2] = np.clip(a[0:2], -cfg.A_MAX_XY, cfg.A_MAX_XY)
    a[2] = np.clip(a[2], -cfg.A_MAX_Z, cfg.A_MAX_Z)
    xn = x.copy()
    xn[0:3] = p + cfg.dt * v + 0.5 * cfg.dt**2 * a
    xn[3:6] = np.clip(v + cfg.dt * a, -cfg.V_MAX, cfg.V_MAX)
    xn[6:12] = 0.0
    return xn


def clamp_control(u: np.ndarray, cfg: Config) -> np.ndarray:
    u = np.array(u, float).ravel()[:3]
    u[0:2] = np.clip(u[0:2], -cfg.A_MAX_XY, cfg.A_MAX_XY)
    u[2] = np.clip(u[2], -cfg.A_MAX_Z, cfg.A_MAX_Z)
    return u


# ======================================================================
# Cost (scaled to ~100-200 range for a reasonable PPO trajectory)
# ======================================================================
def obstacle_barrier_cost(p: np.ndarray, cfg: Config) -> float:
    total = 0.0
    for obs in OBSTACLES:
        total += obs.barrier(p, cfg.c_obs, cfg.eps_obs)
    return total


def goal_region_dist_sq(p: np.ndarray, cfg: Config) -> float:
    d = np.abs(p - cfg.goal_center) - cfg.goal_half_size
    return np.sum(np.maximum(d, 0.0)**2)


def goal_cost_dist_sq(p: np.ndarray, cfg: Config) -> float:
    d = np.abs(p - cfg.goal_center) - cfg.cost_goal_half_size
    return np.sum(np.maximum(d, 0.0)**2)


def goal_attraction_cost(p: np.ndarray, cfg: Config) -> float:
    return cfg.alpha_goal * goal_cost_dist_sq(p, cfg)


def stage_cost(x: np.ndarray, u: np.ndarray, cfg: Config) -> float:
    p = x[0:3]
    return (cfg.lambda_u * np.sum(u**2)
            + obstacle_barrier_cost(p, cfg)
            + cfg.alpha_goal * np.sum((p - cfg.goal_center)**2))

"""
def terminal_cost(x: np.ndarray, cfg: Config) -> float:
    p, v = x[0:3], x[3:6]
    dist_sq_cost = goal_cost_dist_sq(p, cfg)
    dist_sq_center = np.sum((p - cfg.goal_center)**2)
    return (cfg.C_p * dist_sq_cost
            + 2.0 * dist_sq_center
            + cfg.C_v * np.sum(v**2)
            + obstacle_barrier_cost(p, cfg))
"""
def terminal_cost(x: np.ndarray, cfg: Config) -> float:
    p, v = x[0:3], x[3:6]
    dist_sq_center = np.sum((p - cfg.goal_center)**2)
    return (cfg.C_p * dist_sq_center
            + cfg.C_v * np.sum(v**2)
            + obstacle_barrier_cost(p, cfg))

def trajectory_cost(traj: List[np.ndarray], controls: List[np.ndarray], cfg: Config) -> float:
    J = terminal_cost(traj[-1], cfg)
    for k in range(len(controls)):
        J += stage_cost(traj[k], controls[k], cfg)
    return J


def grid_half_width_at(k: int, cfg: Config) -> float:
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
# PPO (proper clipped surrogate)
# ======================================================================
class PPOPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int):
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

    def forward(self, x: torch.Tensor):
        return self.mu(x)

    def act(self, x_np: np.ndarray, cfg: Config, deterministic: bool = False):
        with torch.no_grad():
            x_t = torch.tensor(x_np[:6], dtype=torch.float32).unsqueeze(0)
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


def rollout_policy_fn(action_fn, cfg: Config) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    traj = [cfg.x0.copy()]
    controls = []
    x = cfg.x0.copy()
    for k in range(cfg.N):
        u = clamp_control(action_fn(x, k), cfg)
        controls.append(u.copy())
        x = step_dynamics(x, u, cfg)
        traj.append(x.copy())
    return traj, controls


def train_ppo(cfg: Config) -> PPOPolicy:
    """Train initial policy with proper PPO (clipped surrogate, GAE, multiple epochs)."""
    print("[PPO] Training initial policy (clipped surrogate)...")
    state_dim = 6
    action_dim = 3
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
            ep_states, ep_actions, ep_logprobs, ep_rewards, ep_values = \
                [], [], [], [], []
            for k in range(cfg.N):
                s = x[:6].copy()
                a_np, logp, v = policy.act(x, cfg, deterministic=False)
                r = -stage_cost(x, a_np, cfg)
                ep_states.append(s)
                ep_actions.append(a_np.copy())
                ep_logprobs.append(logp)
                ep_rewards.append(r)
                ep_values.append(v)
                x = step_dynamics(x, a_np, cfg)

            ep_rewards.append(-terminal_cost(x, cfg))
            _, _, v_last = policy.act(x, cfg, deterministic=False)
            ep_values.append(v_last)

            advantages = np.zeros(cfg.N, dtype=np.float32)
            gae = 0.0
            for t in reversed(range(cfg.N)):
                r_t = ep_rewards[t]
                if t == cfg.N - 1:
                    r_t += ep_rewards[cfg.N]
                delta = r_t + cfg.ppo_gamma * ep_values[t + 1] - ep_values[t]
                gae = delta + cfg.ppo_gamma * cfg.ppo_lam * gae
                advantages[t] = gae

            returns = advantages + np.array(ep_values[:cfg.N], dtype=np.float32)

            all_states.extend(ep_states)
            all_actions.extend(ep_actions)
            all_logprobs.extend(ep_logprobs)
            all_rewards.append(returns)
            all_values.extend(ep_values[:cfg.N])
            all_dones.extend(advantages)

            total_episodes += 1
            steps_collected += cfg.N

        states_t = torch.tensor(np.array(all_states), dtype=torch.float32)
        actions_t = torch.tensor(np.array(all_actions), dtype=torch.float32)
        old_logprobs_t = torch.tensor(np.array(all_logprobs), dtype=torch.float32)
        returns_t = torch.tensor(np.concatenate(all_rewards), dtype=torch.float32)
        advantages_t = torch.tensor(np.array(all_dones), dtype=torch.float32)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        n_samples = len(states_t)
        for _ in range(cfg.ppo_epochs):
            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, cfg.ppo_batch_size):
                end = min(start + cfg.ppo_batch_size, n_samples)
                idx = indices[start:end]

                s_b = states_t[idx]
                a_b = actions_t[idx]
                old_lp_b = old_logprobs_t[idx]
                ret_b = returns_t[idx]
                adv_b = advantages_t[idx]

                mu = policy.mu(s_b)
                std = torch.exp(policy.log_std)
                dist = torch.distributions.Normal(mu, std)
                new_lp = dist.log_prob(a_b).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                ratio = torch.exp(new_lp - old_lp_b)
                clipped = torch.clamp(ratio, 1 - cfg.ppo_clip_eps, 1 + cfg.ppo_clip_eps)
                policy_loss = -torch.min(ratio * adv_b, clipped * adv_b).mean()

                values = policy.value(s_b).squeeze(-1)
                value_loss = nn.functional.mse_loss(values, ret_b)

                loss = policy_loss + 0.5 * value_loss - cfg.ppo_entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

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
# Residual NN with centering for on-trajectory consistency
# ======================================================================
class ResidualNN(nn.Module):
    """Maps (Δx, k/N) → Δu with centering: output = f(Δx, k/N) - f(0, k/N).
    Guarantees output = 0 when Δx = 0 for any k."""
    def __init__(self, state_dim: int = 6, action_dim: int = 3, hidden: int = 64):
        super().__init__()
        input_dim = state_dim + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )
        self.state_dim = state_dim

    def forward(self, dx: torch.Tensor, k_norm: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([dx, k_norm], dim=-1)
        out = self.net(inp)
        zero_dx = torch.zeros_like(dx)
        inp_zero = torch.cat([zero_dx, k_norm], dim=-1)
        out_zero = self.net(inp_zero)
        return out - out_zero


class PolicyWithTrajectory:
    """π(x,k) = û_k + ResidualNN(x - x̂_k, k/N)
    On-trajectory (x = x̂_k): ResidualNN outputs 0 by centering → π = û_k exactly."""
    def __init__(self, ref_traj: List[np.ndarray], ref_controls: List[np.ndarray],
                 residual_nn: Optional[ResidualNN], cfg: Config):
        self.ref_traj = [x.copy() for x in ref_traj]
        self.ref_controls = [u.copy() for u in ref_controls]
        self.residual_nn = residual_nn
        self.cfg = cfg
        self.N = len(ref_controls)

    def get_action(self, x: np.ndarray, k: int) -> np.ndarray:
        if k < 0 or k >= self.N:
            return np.zeros(3)
        u_ff = self.ref_controls[k]
        if self.residual_nn is None:
            return clamp_control(u_ff, self.cfg)
        ref_x = self.ref_traj[k]
        dx = x[:6] - ref_x[:6]
        with torch.no_grad():
            dx_t = torch.tensor(dx, dtype=torch.float32).unsqueeze(0)
            k_norm = torch.tensor([[k / self.N]], dtype=torch.float32)
            du = self.residual_nn(dx_t, k_norm).squeeze(0).numpy()
        return clamp_control(u_ff + du, self.cfg)


def generate_perturbation_data(ref_traj: List[np.ndarray], ref_controls: List[np.ndarray],
                               cfg: Config) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    data = []
    N = len(ref_controls)
    u_min = np.array([-cfg.A_MAX_XY, -cfg.A_MAX_XY, -cfg.A_MAX_Z])
    u_max = np.array([cfg.A_MAX_XY, cfg.A_MAX_XY, cfg.A_MAX_Z])

    for _ in range(cfg.nn_perturb_samples):
        k = np.random.randint(0, N)
        ref_x = ref_traj[k].copy()
        noise_p = np.random.randn(3) * 0.3
        noise_v = np.random.randn(3) * 0.2
        x_pert = ref_x.copy()
        x_pert[0:3] += noise_p
        x_pert[3:6] += noise_v

        best_u = ref_controls[k].copy()
        best_J = _eval_cost_to_go_stored(x_pert, best_u, k, ref_controls, cfg)

        for _ in range(10):
            u_cand = ref_controls[k] + np.random.randn(3) * 0.5
            u_cand = np.clip(u_cand, u_min, u_max)
            J_cand = _eval_cost_to_go_stored(x_pert, u_cand, k, ref_controls, cfg)
            if J_cand < best_J:
                best_J = J_cand
                best_u = u_cand.copy()

        data.append((x_pert.copy(), best_u.copy(), k))

    return data


def _eval_cost_to_go_stored(x, u_first, k, ref_controls, cfg):
    """Cost-to-go using stored controls for the tail."""
    J = stage_cost(x, u_first, cfg)
    xc = step_dynamics(x, u_first, cfg)
    for t in range(k + 1, len(ref_controls)):
        J += stage_cost(xc, ref_controls[t], cfg)
        xc = step_dynamics(xc, ref_controls[t], cfg)
    J += terminal_cost(xc, cfg)
    return J


def train_residual_nn(ref_traj, ref_controls, extra_data, cfg):
    nn_model = ResidualNN(state_dim=6, action_dim=3, hidden=cfg.nn_hidden)
    optimizer = optim.Adam(nn_model.parameters(), lr=cfg.nn_lr)
    N = len(ref_controls)

    dx_list, du_list = [], []
    for x_actual, u_target, k in extra_data:
        if k >= N:
            continue
        dx = x_actual[:6] - ref_traj[k][:6]
        du = u_target - ref_controls[k]
        k_feat = k / N
        dx_list.append(np.append(dx, k_feat))
        du_list.append(du)

    if len(dx_list) < 10:
        return nn_model

    dx_arr = torch.tensor(np.array(dx_list), dtype=torch.float32)
    du_arr = torch.tensor(np.array(du_list), dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(dx_arr, du_arr)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.nn_batch, shuffle=True)

    nn_model.train()
    for epoch in range(cfg.nn_epochs):
        total_loss = 0.0
        for batch_dx, batch_du in loader:
            dx_inp = batch_dx[:, :6]
            k_inp = batch_dx[:, 6:7]
            pred = nn_model(dx_inp, k_inp)
            loss = nn.functional.mse_loss(pred, batch_du)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_dx)

    nn_model.eval()
    return nn_model


# ======================================================================
# Rollout with NN policy tail (consistency guarantees monotonic improvement)
# ======================================================================
def _init_worker(ref_traj_np, ref_controls_np, nn_state_dict, nn_hidden, cfg_dict):
    global _worker_policy, _worker_cfg
    _worker_cfg = Config(**cfg_dict)
    ref_traj = [ref_traj_np[i] for i in range(len(ref_traj_np))]
    ref_controls = [ref_controls_np[i] for i in range(len(ref_controls_np))]
    if nn_state_dict is not None:
        rnn = ResidualNN(state_dim=6, action_dim=3, hidden=nn_hidden)
        rnn.load_state_dict(nn_state_dict)
        rnn.eval()
    else:
        rnn = None
    _worker_policy = PolicyWithTrajectory(ref_traj, ref_controls, rnn, _worker_cfg)


def _eval_candidate(args):
    x, u_cand, k = args
    return _eval_tail_cost_nn(x, u_cand, k, _worker_policy, _worker_cfg)


def _pack_policy_for_workers(policy, cfg):
    ref_traj_np = np.array(policy.ref_traj)
    ref_controls_np = np.array(policy.ref_controls)
    nn_sd = None
    nn_hidden = cfg.nn_hidden
    if policy.residual_nn is not None:
        nn_sd = {key: val.cpu() for key, val in policy.residual_nn.state_dict().items()}
    cfg_dict = {
        'dt': cfg.dt, 'N': cfg.N, 'V_MAX': cfg.V_MAX,
        'A_MAX_XY': cfg.A_MAX_XY, 'A_MAX_Z': cfg.A_MAX_Z,
        'x0': cfg.x0.copy(), 'goal_center': cfg.goal_center.copy(),
        'goal_half_size': cfg.goal_half_size.copy(),
        'cost_goal_half_size': cfg.cost_goal_half_size.copy(),
        'lambda_u': cfg.lambda_u, 'c_obs': cfg.c_obs, 'eps_obs': cfg.eps_obs,
        'alpha_goal': cfg.alpha_goal, 'C_p': cfg.C_p, 'C_v': cfg.C_v,
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
USE_PARALLEL = 1          # 0 = serial (single-process), 1 = parallel (multiprocess)


def run_rollout_iteration(policy: PolicyWithTrajectory, cfg: Config
                          ) -> Tuple[List[np.ndarray], List[np.ndarray], int,
                                     List[float], float]:
    u_min = np.array([-cfg.A_MAX_XY, -cfg.A_MAX_XY, -cfg.A_MAX_Z])
    u_max = np.array([cfg.A_MAX_XY, cfg.A_MAX_XY, cfg.A_MAX_Z])
    # Note: the grid is now built per-step, centered on u_base, inside the k-loop.
    # This implements the centered-grid construction from Eq. 31 of the paper
    # and prevents tail wobble caused by a coarse full-range grid.

    pool = None
    if USE_PARALLEL:
        init_args = _pack_policy_for_workers(policy, cfg)
        pool = mp.Pool(processes=N_WORKERS, initializer=_init_worker, initargs=init_args)
        pool.map(_eval_candidate, [(cfg.x0, np.zeros(3), 0)] * N_WORKERS)

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
        for d in range(3):
            lo = max(u_min[d], u_base[d] - hw)
            hi = min(u_max[d], u_base[d] + hw)
            if hi <= lo:
                lo, hi = u_min[d], u_max[d]
            grid_vals.append(np.linspace(lo, hi, cfg.grid_points))

        cur = u_base.copy()
        for dim in range(3):
            cands = []
            for val in grid_vals[dim]:
                cand = cur.copy(); cand[dim] = float(val)
                cands.append(clamp_control(cand, cfg))
            if USE_PARALLEL:
                tasks = [(x.copy(), c, k) for c in cands]
                costs = pool.map(_eval_candidate, tasks)
            else:
                costs = [_eval_tail_cost_nn(x, c, k, policy, cfg) for c in cands]
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

    if pool is not None:
        pool.close()
        pool.join()
    elapsed = time.perf_counter() - t_total
    return traj, controls, n_improved, per_step_times, elapsed


def _eval_tail_cost_nn(x: np.ndarray, u_first: np.ndarray, k: int,
                       policy: PolicyWithTrajectory, cfg: Config) -> float:
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
def check_consistency(policy: PolicyWithTrajectory, cfg: Config) -> float:
    """Verify that rolling out the policy from x0 on the stored trajectory
    reproduces the stored trajectory exactly (since NN output = 0 on-trajectory).
    Returns the max deviation across all steps."""
    x = cfg.x0.copy()
    max_dev = 0.0
    for k in range(policy.N):
        u = policy.get_action(x, k)
        u_ref = policy.ref_controls[k]
        dev = np.max(np.abs(u - clamp_control(u_ref, cfg)))
        max_dev = max(max_dev, dev)
        x = step_dynamics(x, u, cfg)
        pos_dev = np.max(np.abs(x[:3] - policy.ref_traj[k + 1][:3]))
        max_dev = max(max_dev, pos_dev)
    return max_dev


# ======================================================================
# Visualization
# ======================================================================
def draw_crazyflie(ax, pos, size=0.15):
    cx, cy, cz = pos
    arm_len = size
    prop_r = size * 0.8
    arms = [np.array([1, 1, 0]) / np.sqrt(2), np.array([1, -1, 0]) / np.sqrt(2),
            np.array([-1, 1, 0]) / np.sqrt(2), np.array([-1, -1, 0]) / np.sqrt(2)]
    for arm_dir in arms:
        tip = np.array([cx, cy, cz]) + arm_len * arm_dir
        ax.plot([cx, tip[0]], [cy, tip[1]], [cz, tip[2]],
                color='#333333', linewidth=2.5, zorder=10)
        theta = np.linspace(0, 2 * np.pi, 20)
        ax.plot(tip[0] + prop_r * np.cos(theta), tip[1] + prop_r * np.sin(theta),
                np.full(20, tip[2]), color='#1a73e8', linewidth=1.5, alpha=0.8, zorder=10)
    ax.scatter([cx], [cy], [cz], color='#1a73e8', s=40, zorder=11)


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
    print("Policy Iteration via Rollout — 3D Drone Optimal Control")
    print("=" * 70)
    print(f"N={cfg.N}, dt={cfg.dt}")
    print(f"Start: {cfg.x0[:3]}")
    print(f"Goal region: center={cfg.goal_center}, half_size={cfg.goal_half_size}")
    print(f"Obstacles: {len(OBSTACLES)} (box + sphere)")
    print()

    # ------------------------------------------------------------------
    # Step 0: PPO initial policy
    # ------------------------------------------------------------------
    ppo_t0 = time.perf_counter()
    ppo_policy = train_ppo(cfg)
    ppo_time = time.perf_counter() - ppo_t0

    traj_0, ctrl_0 = rollout_policy_fn(
        lambda x, k: ppo_policy.act(x, cfg, deterministic=True)[0], cfg)
    cost_0 = trajectory_cost(traj_0, ctrl_0, cfg)
    print(f"\n[Iter 0] PPO policy: cost={cost_0:.2f}, time={ppo_time:.1f}s")

    # ------------------------------------------------------------------
    # Policy iteration
    # ------------------------------------------------------------------
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

        # Consistency check: base policy reproduces stored trajectory
        cons_dev = check_consistency(current_policy, cfg)
        all_consistency.append(cons_dev)
        print(f"  Consistency check: max deviation = {cons_dev:.2e}")

        # Rollout
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

        # Train NN
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

    # Final consistency check
    cons_final = check_consistency(current_policy, cfg)
    all_consistency.append(cons_final)
    print(f"\n  Final consistency check: max deviation = {cons_final:.2e}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
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

    final_pos = all_trajs[-1][-1][:3]
    in_goal = np.all(np.abs(final_pos - cfg.goal_center) <= cfg.goal_half_size)
    goal_dist = np.sqrt(goal_region_dist_sq(final_pos, cfg))
    print(f"\nFinal: pos={np.round(final_pos, 3)}, in_goal={in_goal}, "
          f"dist_to_boundary={goal_dist:.4f}")

    # ------------------------------------------------------------------
    # Plotting: 2 separate windows (cost + 3D scenario), polished style
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

    fig2 = plt.figure(figsize=(10, 7.5))
    ax2 = fig2.add_subplot(111, projection='3d')

    n_trajs = len(all_trajs)
    highlight_iter4 = min(4, n_trajs - 1)
    show_iters = sorted({0, highlight_iter4, n_trajs - 1})

    for i, traj in enumerate(all_trajs):
        if i not in show_iters:
            continue
        arr = np.array(traj)
        if i == 0:
            label = "Iter 0 (PPO)"
            color, alpha, lw, ls = '#1f77b4', 0.95, 2.2, '--'
        elif i == n_trajs - 1:
            label = f"Iter {i} (final)"
            color, alpha, lw, ls = '#d62728', 1.0, 3.0, '-'
        elif i == highlight_iter4:
            label = f"Iter {i}"
            color, alpha, lw, ls = '#ff7f0e', 0.95, 2.4, '-'
        ax2.plot(arr[:, 0], arr[:, 1], arr[:, 2], color=color, alpha=alpha,
                 linewidth=lw, linestyle=ls, label=label)

    final_arr = np.array(all_trajs[-1])
    ax2.scatter(final_arr[0, 0], final_arr[0, 1], final_arr[0, 2], s=55,
                color='#0b5394', edgecolor='white', linewidth=0.8, zorder=12)
    ax2.scatter(final_arr[-1, 0], final_arr[-1, 1], final_arr[-1, 2], s=85, marker='*',
                color='#c00000', edgecolor='white', linewidth=0.8, zorder=13)

    for obs in OBSTACLES:
        if isinstance(obs, BoxObstacle):
            draw_box_3d(ax2, obs.center, obs.half_size, color='#6b7280',
                        alpha=0.30, edgecolor='#374151')
        elif isinstance(obs, SphereObstacle):
            draw_sphere_3d(ax2, obs.center, obs.radius, color='#e11d48', alpha=0.24)

    draw_box_3d(ax2, cfg.goal_center, cfg.goal_half_size,
                color='#22c55e', alpha=0.30, edgecolor='#15803d')
    draw_crazyflie(ax2, cfg.x0[:3], size=0.18)

    ax2.set_xlim(-2.8, 3.2)
    ax2.set_ylim(-2.8, 3.2)
    ax2.set_zlim(-1.5, 2.0)
    ax2.set_box_aspect((1.0, 1.0, 0.70))
    ax2.view_init(elev=24, azim=-58)
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
    legend2 = ax2.legend(fontsize=20, loc='upper right', framealpha=0.92)
    legend2.set_draggable(True)

    fig1.tight_layout()
    fig2.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)

    # # Always save figures to disk so they are accessible on Colab / headless runs. --- uncomment for saving figures
    # out_dir = os.environ.get("FIG_OUT_DIR", ".")
    # os.makedirs(out_dir, exist_ok=True)
    # fig1_path = os.path.join(out_dir, "one_drone_cost.pdf")
    # fig2_path = os.path.join(out_dir, "one_drone_trajectory.pdf")
    # fig1.savefig(fig1_path, dpi=150, bbox_inches="tight")
    # fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
    # print(f"[figures] saved:\n  {fig1_path}\n  {fig2_path}")

    plt.show()

if __name__ == "__main__":
    main()