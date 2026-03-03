#!/usr/bin/env python3
"""Collect one large quad2d dataset for static predictor fitting.

This is intentionally simple:
- one file to read and edit
- one output dataset with focused + robust rollouts
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import safe_control_gym  # noqa: F401  (ensures registrations are loaded)
from safe_control_gym.controllers.lqr.lqr_utils import compute_lqr_gain, get_cost_weight_matrix
from safe_control_gym.utils.registration import make

# -----------------------------------------------------------------------------
# Global settings (edit these directly)
# -----------------------------------------------------------------------------
DATASET_NAME = "theta_focused_plus_robust_fresh"
SEED = 20260226

# Target size matches your current combined dataset: 12000 + 4000 = 16000.
N_FOCUSED = 12000
N_ROBUST = 4000

# Rollout and env settings.
ROLLOUT_STEPS = 180
CTRL_FREQ = 50
PYB_FREQ = 1000
GUI = False

# LQR baseline.
Q_LQR = [1, 1, 1, 1, 1, 1]
R_LQR = [0.1, 0.1]

# If a regime is hard to satisfy, these caps prevent infinite loops.
MAX_ATTEMPTS_FOCUSED = 12 * N_FOCUSED
MAX_ATTEMPTS_ROBUST = 15 * N_ROBUST

# Focused regime: small initial condition and gentle excitation.
FOCUSED = {
    "name": "focused",
    "init": {
        "x": 0.06,
        "x_dot": 0.20,
        "z": 0.06,
        "z_dot": 0.20,
        "theta": 0.035,
        "theta_dot": 0.35,
    },
    "excitation_amp": np.array([0.015, 0.015], dtype=np.float64),
    "excitation_freqs": np.array([0.4, 0.8, 1.2, 1.8, 2.6], dtype=np.float64),
    "delta_u_clip": np.array([0.05, 0.05], dtype=np.float64),
    "accept_input_dev_max": np.array([0.05, 0.05], dtype=np.float64),
    "accept_x_max": 0.25,
    "accept_z_max": 0.25,
    "accept_theta_max": 0.10,
    "accept_xdot_max": 0.8,
    "accept_zdot_max": 0.8,
    "accept_thetadot_max": 1.2,
    "coupled_weight": 0.6,
    "coupled_max": 1.1,
    "sat_frac_max": 0.25,
    "mean_clip_max": 0.11,
    "mean_explicit_clip_max": 0.11,
    "action_noise_std": 0.0,
    "pulse_prob": 0.0,
    "pulse_min_steps": 1,
    "pulse_max_steps": 1,
    "pulse_diff_amplitude": 0.0,
    "pulse_collective_amplitude": 0.0,
    "require_violent": False,
}

# Robust regime: broader initial condition and richer excitation.
ROBUST = {
    "name": "robust",
    "init": {
        "x": 0.30,
        "x_dot": 1.00,
        "z": 0.35,
        "z_dot": 1.00,
        "theta": 0.12,
        "theta_dot": 2.00,
    },
    "excitation_amp": np.array([0.03, 0.03], dtype=np.float64),
    "excitation_freqs": np.array([0.4, 0.8, 1.2, 1.8, 2.6], dtype=np.float64),
    "delta_u_clip": np.array([0.12, 0.12], dtype=np.float64),
    "accept_input_dev_max": np.array([0.16, 0.16], dtype=np.float64),
    "accept_x_max": 1.6,
    "accept_z_max": 1.2,
    "accept_theta_max": 1.2,
    "accept_xdot_max": 8.0,
    "accept_zdot_max": 8.0,
    "accept_thetadot_max": 25.0,
    "coupled_weight": 0.8,
    "coupled_max": 2.4,
    "sat_frac_max": 0.50,
    "mean_clip_max": 0.20,
    "mean_explicit_clip_max": 0.20,
    "action_noise_std": 0.004,
    "pulse_prob": 0.06,
    "pulse_min_steps": 6,
    "pulse_max_steps": 22,
    "pulse_diff_amplitude": 0.08,
    "pulse_collective_amplitude": 0.04,
    # Keep robust data truly robust by requiring "violent" behavior.
    "require_violent": True,
}


def sample_initial_state(rng: np.random.Generator, x_eq: np.ndarray, bounds: dict) -> np.ndarray:
    """Sample [x, x_dot, z, z_dot, theta, theta_dot] around equilibrium."""
    out = np.zeros(6, dtype=np.float64)
    out[0] = x_eq[0] + rng.uniform(-bounds["x"], bounds["x"])
    out[1] = x_eq[1] + rng.uniform(-bounds["x_dot"], bounds["x_dot"])
    out[2] = x_eq[2] + rng.uniform(-bounds["z"], bounds["z"])
    out[3] = x_eq[3] + rng.uniform(-bounds["z_dot"], bounds["z_dot"])
    out[4] = x_eq[4] + rng.uniform(-bounds["theta"], bounds["theta"])
    out[5] = x_eq[5] + rng.uniform(-bounds["theta_dot"], bounds["theta_dot"])
    return out


def multisine(t_idx: int, dt: float, amplitudes: np.ndarray, frequencies: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """Two-channel multisine excitation."""
    t = float(t_idx) * dt
    s = np.sin(2.0 * np.pi * frequencies[None, :] * t + phases)
    return amplitudes * np.mean(s, axis=1)


def run_one_rollout(
    env,
    rng: np.random.Generator,
    regime: dict,
    x_eq: np.ndarray,
    u_eq: np.ndarray,
    lqr_gain: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> dict | None:
    """Run one rollout and return all arrays/metrics, or None if invalid."""
    init_state = sample_initial_state(rng, x_eq, regime["init"])
    env.INIT_X = float(init_state[0])
    env.INIT_X_DOT = float(init_state[1])
    env.INIT_Z = float(init_state[2])
    env.INIT_Z_DOT = float(init_state[3])
    env.INIT_THETA = float(init_state[4])
    env.INIT_THETA_DOT = float(init_state[5])

    obs, info_reset = env.reset(seed=int(rng.integers(0, np.iinfo(np.int32).max)))
    obs = np.asarray(obs, dtype=np.float64)

    raw_obs_seq = np.zeros((ROLLOUT_STEPS + 1, obs.shape[0]), dtype=np.float64)
    state_seq = np.zeros((ROLLOUT_STEPS + 1, 6), dtype=np.float64)
    y_seq = np.zeros((ROLLOUT_STEPS + 1, 3), dtype=np.float64)

    u_applied_seq = np.zeros((ROLLOUT_STEPS, 2), dtype=np.float64)
    u_cmd_seq = np.zeros((ROLLOUT_STEPS, 2), dtype=np.float64)
    u_unclipped_seq = np.zeros((ROLLOUT_STEPS, 2), dtype=np.float64)
    exploration_seq = np.zeros((ROLLOUT_STEPS, 2), dtype=np.float64)
    pulse_seq = np.zeros((ROLLOUT_STEPS, 2), dtype=np.float64)
    explicit_clip_mag = np.zeros((ROLLOUT_STEPS, 2), dtype=np.float64)
    actuator_clip_mag = np.zeros((ROLLOUT_STEPS, 2), dtype=np.float64)
    saturation_mask = np.zeros((ROLLOUT_STEPS, 2), dtype=np.int8)

    raw_obs_seq[0] = obs
    state_seq[0] = np.asarray(env.state, dtype=np.float64)
    y_seq[0] = obs[[0, 2, 4]]

    phases = rng.uniform(-np.pi, np.pi, size=(2, regime["excitation_freqs"].shape[0]))
    headroom = np.minimum(action_high - u_eq, u_eq - action_low)
    delta_u_clip = np.minimum(np.asarray(regime["delta_u_clip"], dtype=np.float64), 0.98 * headroom)

    pulse_remaining = 0
    pulse_vec = np.zeros(2, dtype=np.float64)

    for t in range(ROLLOUT_STEPS):
        x_hat = obs[:6]

        u_fb = u_eq - lqr_gain @ (x_hat - x_eq)
        explore = multisine(
            t_idx=t,
            dt=float(env.CTRL_TIMESTEP),
            amplitudes=regime["excitation_amp"],
            frequencies=regime["excitation_freqs"],
            phases=phases,
        )

        if pulse_remaining <= 0 and rng.uniform() < regime["pulse_prob"]:
            pulse_remaining = int(rng.integers(regime["pulse_min_steps"], regime["pulse_max_steps"] + 1))
            diff = float(rng.uniform(-regime["pulse_diff_amplitude"], regime["pulse_diff_amplitude"]))
            collective = float(
                rng.uniform(-regime["pulse_collective_amplitude"], regime["pulse_collective_amplitude"])
            )
            pulse_vec = np.array([collective - diff, collective + diff], dtype=np.float64)
        elif pulse_remaining <= 0:
            pulse_vec = np.zeros(2, dtype=np.float64)

        pulse_remaining = max(0, pulse_remaining - 1)
        noise = rng.normal(0.0, regime["action_noise_std"], size=2).astype(np.float64)

        u_unclipped = u_fb + explore + pulse_vec + noise
        delta_unclipped = u_unclipped - u_eq
        delta_clipped = np.clip(delta_unclipped, -delta_u_clip, delta_u_clip)
        u_cmd = np.clip(u_eq + delta_clipped, action_low, action_high)

        obs_next, _, done, _ = env.step(u_cmd)

        applied = np.asarray(env.current_clipped_action, dtype=np.float64)
        obs = np.asarray(obs_next, dtype=np.float64)
        state_now = np.asarray(env.state, dtype=np.float64)

        # Keep only full-length rollouts.
        if done and t < ROLLOUT_STEPS - 1:
            return None

        u_cmd_seq[t] = u_cmd
        u_unclipped_seq[t] = u_unclipped
        u_applied_seq[t] = applied
        exploration_seq[t] = explore
        pulse_seq[t] = pulse_vec

        explicit_clip_mag[t] = np.abs(delta_unclipped - delta_clipped)
        actuator_clip_mag[t] = np.abs(applied - u_cmd)
        sat_low = np.isclose(applied, action_low, atol=1e-6)
        sat_high = np.isclose(applied, action_high, atol=1e-6)
        saturation_mask[t] = np.logical_or(sat_low, sat_high).astype(np.int8)

        raw_obs_seq[t + 1] = obs
        state_seq[t + 1] = state_now
        y_seq[t + 1] = obs[[0, 2, 4]]

    state_dev = state_seq - x_eq[None, :]
    u_dev = u_applied_seq - u_eq[None, :]

    state_ratio = np.maximum.reduce(
        [
            np.abs(state_dev[:, 0]) / regime["accept_x_max"],
            np.abs(state_dev[:, 2]) / regime["accept_z_max"],
            np.abs(state_dev[:, 4]) / regime["accept_theta_max"],
            np.abs(state_dev[:, 1]) / regime["accept_xdot_max"],
            np.abs(state_dev[:, 3]) / regime["accept_zdot_max"],
            np.abs(state_dev[:, 5]) / regime["accept_thetadot_max"],
        ]
    )
    input_ratio = np.max(np.abs(u_dev) / regime["accept_input_dev_max"][None, :], axis=1)
    coupled_score = state_ratio[:-1] + regime["coupled_weight"] * input_ratio

    u_diff = u_applied_seq[:, 1] - u_applied_seq[:, 0]

    metrics = {
        "max_abs_theta": float(np.max(np.abs(state_dev[:, 4]))),
        "max_abs_thetadot": float(np.max(np.abs(state_dev[:, 5]))),
        "max_abs_xdot": float(np.max(np.abs(state_dev[:, 1]))),
        "max_abs_u_diff": float(np.max(np.abs(u_diff))),
        "state_ratio_max": float(np.max(state_ratio)),
        "input_ratio_max": float(np.max(input_ratio)),
        "coupled_score_max": float(np.max(coupled_score)),
        "saturation_fraction": float(np.mean(np.any(saturation_mask.astype(bool), axis=1))),
        "mean_clip_magnitude": float(np.mean(np.mean(np.abs(u_applied_seq - u_unclipped_seq), axis=1))),
        "mean_explicit_clip_magnitude": float(np.mean(np.mean(explicit_clip_mag, axis=1))),
    }

    # "Violent" flag from your robust collector logic.
    is_violent = bool(
        (metrics["max_abs_theta"] >= 0.25)
        or (metrics["max_abs_thetadot"] >= 5.0)
        or (metrics["max_abs_xdot"] >= 0.8)
        or (metrics["max_abs_u_diff"] >= 0.05)
    )

    return {
        "regime": regime["name"],
        "init_state": init_state,
        "raw_obs": raw_obs_seq,
        "state": state_seq,
        "y": y_seq,
        "u_applied": u_applied_seq,
        "u_command": u_cmd_seq,
        "u_unclipped": u_unclipped_seq,
        "exploration": exploration_seq,
        "pulse": pulse_seq,
        "explicit_clip_mag": explicit_clip_mag,
        "actuator_clip_mag": actuator_clip_mag,
        "saturation_mask": saturation_mask,
        "metrics": metrics,
        "is_violent": is_violent,
        "physical_parameters": info_reset.get("physical_parameters", {}),
    }


def accept_rollout(rollout: dict, regime: dict) -> bool:
    """Simple acceptance gate for data quality."""
    m = rollout["metrics"]
    if m["state_ratio_max"] > 1.0:
        return False
    if m["input_ratio_max"] > 1.0:
        return False
    if m["coupled_score_max"] > regime["coupled_max"]:
        return False
    if m["saturation_fraction"] > regime["sat_frac_max"]:
        return False
    if m["mean_clip_magnitude"] > regime["mean_clip_max"]:
        return False
    if m["mean_explicit_clip_magnitude"] > regime["mean_explicit_clip_max"]:
        return False
    if regime["require_violent"] and (not rollout["is_violent"]):
        return False
    return True


def save_rollout(path: Path, rollout: dict, env, x_eq: np.ndarray, u_eq: np.ndarray, action_low: np.ndarray, action_high: np.ndarray) -> None:
    """Save one rollout in the same style used by your other collectors."""
    np.savez(
        path,
        y=rollout["y"].astype(np.float32),
        u=rollout["u_applied"].astype(np.float32),
        raw_obs=rollout["raw_obs"].astype(np.float32),
        state=rollout["state"].astype(np.float32),
        u_command=rollout["u_command"].astype(np.float32),
        u_unclipped=rollout["u_unclipped"].astype(np.float32),
        exploration=rollout["exploration"].astype(np.float32),
        pulse=rollout["pulse"].astype(np.float32),
        explicit_clip_mag=rollout["explicit_clip_mag"].astype(np.float32),
        actuator_clip_mag=rollout["actuator_clip_mag"].astype(np.float32),
        saturation_mask=rollout["saturation_mask"].astype(np.int8),
        dt=np.asarray([env.CTRL_TIMESTEP], dtype=np.float64),
        x_eq=x_eq.astype(np.float64),
        u_eq=u_eq.astype(np.float64),
        action_low=action_low.astype(np.float64),
        action_high=action_high.astype(np.float64),
        init_state=rollout["init_state"].astype(np.float64),
        max_abs_theta=np.asarray([rollout["metrics"]["max_abs_theta"]], dtype=np.float64),
        max_abs_thetadot=np.asarray([rollout["metrics"]["max_abs_thetadot"]], dtype=np.float64),
        max_abs_xdot=np.asarray([rollout["metrics"]["max_abs_xdot"]], dtype=np.float64),
        max_abs_u_diff=np.asarray([rollout["metrics"]["max_abs_u_diff"]], dtype=np.float64),
        state_ratio_max=np.asarray([rollout["metrics"]["state_ratio_max"]], dtype=np.float64),
        input_ratio_max=np.asarray([rollout["metrics"]["input_ratio_max"]], dtype=np.float64),
        coupled_score_max=np.asarray([rollout["metrics"]["coupled_score_max"]], dtype=np.float64),
        saturation_fraction=np.asarray([rollout["metrics"]["saturation_fraction"]], dtype=np.float64),
        mean_clip_magnitude=np.asarray([rollout["metrics"]["mean_clip_magnitude"]], dtype=np.float64),
        mean_explicit_clip_magnitude=np.asarray([rollout["metrics"]["mean_explicit_clip_magnitude"]], dtype=np.float64),
        is_violent=np.asarray([int(rollout["is_violent"])], dtype=np.int8),
        regime=np.asarray([rollout["regime"]]),
        physical_parameters_json=np.asarray([json.dumps(rollout["physical_parameters"])]),
    )


def collect_regime(
    env,
    rng: np.random.Generator,
    regime: dict,
    n_target: int,
    max_attempts: int,
    start_index: int,
    rollout_dir: Path,
    x_eq: np.ndarray,
    u_eq: np.ndarray,
    lqr_gain: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> tuple[int, int, int]:
    """Collect one regime and return (accepted, attempted, next_index)."""
    accepted = 0
    attempted = 0
    rollout_index = start_index

    print(f"[collect_data] start regime={regime['name']} target={n_target} max_attempts={max_attempts}")

    while accepted < n_target and attempted < max_attempts:
        attempted += 1
        rollout = run_one_rollout(
            env=env,
            rng=rng,
            regime=regime,
            x_eq=x_eq,
            u_eq=u_eq,
            lqr_gain=lqr_gain,
            action_low=action_low,
            action_high=action_high,
        )

        if rollout is None:
            continue
        if not accept_rollout(rollout, regime):
            continue

        save_rollout(
            path=rollout_dir / f"rollout_{rollout_index:07d}.npz",
            rollout=rollout,
            env=env,
            x_eq=x_eq,
            u_eq=u_eq,
            action_low=action_low,
            action_high=action_high,
        )

        accepted += 1
        rollout_index += 1

        if accepted % 20 == 0 or accepted == n_target:
            print(f"[collect_data] regime={regime['name']} accepted={accepted}/{n_target} attempts={attempted}")

    return accepted, attempted, rollout_index


def main() -> None:
    rng = np.random.default_rng(SEED)

    script_dir = Path(__file__).resolve().parent
    rollout_dir = script_dir / "rollouts" / DATASET_NAME
    rollout_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = {
        "gui": GUI,
        "quad_type": 2,
        "task": "stabilization",
        "cost": "quadratic",
        "ctrl_freq": CTRL_FREQ,
        "pyb_freq": PYB_FREQ,
        "episode_len_sec": int(max(1, math.ceil(float(ROLLOUT_STEPS) / float(CTRL_FREQ))) + 1),
        "normalized_rl_action_space": False,
        "done_on_out_of_bound": False,
        "randomized_init": False,
        "task_info": {
            "stabilization_goal": [0.0, 1.0],
            "stabilization_goal_tolerance": 1e-12,
        },
    }

    env = make("quadrotor", **env_kwargs)

    x_eq = np.asarray(env.X_GOAL, dtype=np.float64)
    u_eq = np.asarray(env.U_GOAL, dtype=np.float64)
    action_low = np.asarray(env.physical_action_bounds[0], dtype=np.float64)
    action_high = np.asarray(env.physical_action_bounds[1], dtype=np.float64)

    q_mat = get_cost_weight_matrix(Q_LQR, env.symbolic.nx)
    r_mat = get_cost_weight_matrix(R_LQR, env.symbolic.nu)
    lqr_gain = compute_lqr_gain(
        env.symbolic,
        env.symbolic.X_EQ,
        env.symbolic.U_EQ,
        q_mat,
        r_mat,
        discrete_dynamics=True,
    )

    focused_acc, focused_att, next_idx = collect_regime(
        env=env,
        rng=rng,
        regime=FOCUSED,
        n_target=N_FOCUSED,
        max_attempts=MAX_ATTEMPTS_FOCUSED,
        start_index=0,
        rollout_dir=rollout_dir,
        x_eq=x_eq,
        u_eq=u_eq,
        lqr_gain=lqr_gain,
        action_low=action_low,
        action_high=action_high,
    )

    robust_acc, robust_att, next_idx = collect_regime(
        env=env,
        rng=rng,
        regime=ROBUST,
        n_target=N_ROBUST,
        max_attempts=MAX_ATTEMPTS_ROBUST,
        start_index=next_idx,
        rollout_dir=rollout_dir,
        x_eq=x_eq,
        u_eq=u_eq,
        lqr_gain=lqr_gain,
        action_low=action_low,
        action_high=action_high,
    )

    env.close()

    summary = {
        "dataset_name": DATASET_NAME,
        "seed": SEED,
        "rollout_steps": ROLLOUT_STEPS,
        "ctrl_freq": CTRL_FREQ,
        "pyb_freq": PYB_FREQ,
        "targets": {"focused": N_FOCUSED, "robust": N_ROBUST, "total": N_FOCUSED + N_ROBUST},
        "accepted": {
            "focused": int(focused_acc),
            "robust": int(robust_acc),
            "total": int(focused_acc + robust_acc),
        },
        "attempted": {
            "focused": int(focused_att),
            "robust": int(robust_att),
            "total": int(focused_att + robust_att),
        },
        "rollout_dir": str(rollout_dir),
        "focused_config": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in FOCUSED.items()
        },
        "robust_config": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in ROBUST.items()
        },
    }

    summary_path = rollout_dir / "collect_data_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("[collect_data] done")
    print(f"[collect_data] accepted total: {focused_acc + robust_acc}/{N_FOCUSED + N_ROBUST}")
    print(f"[collect_data] saved to: {rollout_dir}")
    print(f"[collect_data] summary: {summary_path}")


if __name__ == "__main__":
    main()
