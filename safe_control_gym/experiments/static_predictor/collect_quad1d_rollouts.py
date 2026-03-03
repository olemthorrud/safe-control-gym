#!/usr/bin/env python3
"""Collect quad1d rollout data for fitting a static linear multi-step predictor.

Structured like collect_quad2d_rollouts.py but simplified for quad1d:
  - state = [z, z_dot] (2D)
  - y     = [z]        (1D, position only — analogous to quad2d's [x, z, theta])
  - u     = [T]        (1D, total thrust)

The dataset can be used with Linear_Predictor (fit_static_phi_ddpc.py) to verify
that the predictor error goes to zero for an exactly linear system.
"""

import json
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import safe_control_gym  # noqa: F401
from safe_control_gym.controllers.lqr.lqr_utils import compute_lqr_gain, get_cost_weight_matrix
from safe_control_gym.utils.registration import make

# ----------------------------
# Collection settings
# ----------------------------
DATASET_NAME        = 'quad1d_linear_verify'
NUM_ROLLOUTS        = 200
MAX_ATTEMPTS        = 2000      # max collection attempts before giving up
ROLLOUT_STEPS       = 100       # timesteps per rollout (must be > T_INI + N_HORIZON)

SEED                = 0
CTRL_FREQ           = 50        # Hz
PYB_FREQ            = 1000      # Hz

Q_LQR               = [1.0, 1.0]
R_LQR               = [0.1]

EXCITATION_AMP      = 0.04
EXCITATION_FREQS    = [0.4, 0.8, 1.2, 1.8, 2.6]  # Hz
DELTA_U_CLIP        = 0.18      # max |delta_u| before action-bound clipping

INIT_Z_RANGE        = 0.25      # m   (initial z ~ U[z_eq ± range])
INIT_ZDOT_RANGE     = 0.7       # m/s

ACCEPT_Z_MAX        = 2.00      # max |z - z_eq| to keep rollout
ACCEPT_ZDOT_MAX     = 4.0       # max |z_dot - z_dot_eq| to keep rollout
SATURATION_FRAC_MAX = 0.50      # max fraction of saturating timesteps
MEAN_CLIP_MAX       = 0.50      # max mean clip magnitude
SAT_ATOL            = 1e-6

ZDEV_EDGES          = [0.05, 0.15, 0.30, 0.60, 1.00]


# ----------------------------
# Helpers
# ----------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def init_vector_stats(dim: int) -> Dict[str, Any]:
    return {'dim': int(dim), 'count': 0, 'sum': [0.0] * dim,
            'sum_sq': [0.0] * dim, 'max_abs': [0.0] * dim}


def ensure_vector_stats_shape(stats: Dict[str, Any], dim: int) -> None:
    if stats.get('dim') == dim:
        return
    if int(stats.get('count', 0)) > 0:
        raise ValueError(
            f'Incompatible stats dimension: existing {stats.get("dim")} vs current {dim}.')
    stats.clear()
    stats.update(init_vector_stats(dim))


def update_vector_stats(stats: Dict[str, Any], values: np.ndarray) -> None:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    dim = arr.shape[1]
    ensure_vector_stats_shape(stats, dim)
    arr = arr[np.isfinite(arr).all(axis=1)]
    if arr.size == 0:
        return
    sv = np.asarray(stats['sum'], dtype=np.float64)
    ss = np.asarray(stats['sum_sq'], dtype=np.float64)
    mx = np.asarray(stats['max_abs'], dtype=np.float64)
    sv += np.sum(arr, axis=0)
    ss += np.sum(np.square(arr), axis=0)
    mx = np.maximum(mx, np.max(np.abs(arr), axis=0))
    stats['count'] = int(stats['count']) + int(arr.shape[0])
    stats['sum'] = sv.tolist()
    stats['sum_sq'] = ss.tolist()
    stats['max_abs'] = mx.tolist()


def summarize_vector_stats(stats: Dict[str, Any], labels: Sequence[str]) -> Dict[str, Dict[str, float]]:
    count = int(stats.get('count', 0))
    dim = int(stats.get('dim', len(labels)))
    sv = np.asarray(stats.get('sum', [0.0] * dim), dtype=np.float64)
    ss = np.asarray(stats.get('sum_sq', [0.0] * dim), dtype=np.float64)
    mx = np.asarray(stats.get('max_abs', [0.0] * dim), dtype=np.float64)
    if count > 0:
        mean = sv / count
        std = np.sqrt(np.maximum(ss / count - mean ** 2, 0.0))
    else:
        mean = std = np.zeros(dim)
    return {lbl: {'mean': float(mean[i]), 'std': float(std[i]),
                  'max_abs': float(mx[i]), 'count': count}
            for i, lbl in enumerate(labels)}


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(path)


def bucket_index(value: float, edges: Sequence[float]) -> int:
    return int(np.digitize([float(value)], np.asarray(edges, dtype=np.float64), right=True)[0])


def bucket_labels(edges: Sequence[float]) -> List[str]:
    if not edges:
        return ['all']
    lbls = [f'<= {edges[0]:.6g}']
    for lo, hi in zip(edges[:-1], edges[1:]):
        lbls.append(f'({lo:.6g}, {hi:.6g}]')
    lbls.append(f'> {edges[-1]:.6g}')
    return lbls


def ensure_bucket_counts_shape(counts: List[int], n: int) -> List[int]:
    if len(counts) == n:
        return counts
    if len(counts) < n:
        return list(counts) + [0] * (n - len(counts))
    return list(counts[:n])


def load_or_initialize_metadata(
    path: Path, dataset_name: str, dt: float, raw_obs_dim: int,
    zdev_edges: Sequence[float], obs_mapping: Dict[str, Any],
    env_settings: Dict[str, Any],
) -> Dict[str, Any]:
    default = {
        'schema_version': 1, 'dataset_name': dataset_name,
        'created_utc': utc_now(), 'last_updated_utc': utc_now(),
        'num_runs': 0,
        'counts': {'attempted_rollouts': 0, 'accepted_rollouts': 0, 'discarded_rollouts': 0},
        'discard_reasons': {}, 'sampling_time_sec': float(dt),
        'observation_mapping': obs_mapping, 'environment_settings': env_settings,
        'bucket_edges': {'max_z_deviation': list(zdev_edges)},
        'bucket_labels': {'max_z_deviation': bucket_labels(zdev_edges)},
        'bucket_counts': {
            'accepted': {'max_z_deviation': [0] * (len(zdev_edges) + 1)},
            'discarded': {'max_z_deviation': [0] * (len(zdev_edges) + 1)},
        },
        'stats': {'accepted': {
            'raw_obs': init_vector_stats(raw_obs_dim), 'state': init_vector_stats(2),
            'y': init_vector_stats(1), 'u': init_vector_stats(1),
            'y_deviation': init_vector_stats(1), 'u_deviation': init_vector_stats(1),
        }},
        'saturation': {
            'timesteps_total': 0, 'timesteps_saturating': 0,
            'sum_clip_magnitude': 0.0, 'max_clip_magnitude': 0.0,
            'sum_saturation_fraction_per_rollout': 0.0, 'rollouts_accounted': 0,
        },
        'last_run_knobs': {},
    }
    if not path.exists():
        return default

    with path.open('r', encoding='utf-8') as f:
        loaded = json.load(f)
    for k, v in default.items():
        if k not in loaded:
            loaded[k] = v
    loaded['counts'] = {**default['counts'], **loaded.get('counts', {})}
    loaded['discard_reasons'] = dict(loaded.get('discard_reasons', {}))
    loaded['stats'] = loaded.get('stats', {})
    loaded['stats']['accepted'] = loaded['stats'].get('accepted', {})
    for k, v in default['stats']['accepted'].items():
        loaded['stats']['accepted'].setdefault(k, v)
    loaded['bucket_edges'] = loaded.get('bucket_edges', default['bucket_edges'])
    loaded['bucket_labels'] = loaded.get('bucket_labels', default['bucket_labels'])
    loaded['bucket_counts'] = loaded.get('bucket_counts', default['bucket_counts'])
    n = len(zdev_edges) + 1
    for split in ['accepted', 'discarded']:
        d = loaded['bucket_counts'].setdefault(split, {})
        d['max_z_deviation'] = ensure_bucket_counts_shape(d.get('max_z_deviation', []), n)
    loaded['saturation'] = {**default['saturation'], **loaded.get('saturation', {})}
    if abs(float(loaded.get('sampling_time_sec', dt)) - dt) > 1e-12:
        raise ValueError(f'Dataset sampling time mismatch.')
    return loaded


def next_rollout_index(rollout_dir: Path) -> int:
    rollout_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r'^rollout_(\d+)\.npz$')
    max_idx = -1
    for p in rollout_dir.glob('rollout_*.npz'):
        m = pattern.match(p.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def sample_initial_state(rng: np.random.Generator, x_eq: np.ndarray) -> np.ndarray:
    # state = [z, z_dot] for quad1d
    init = np.zeros(2, dtype=np.float64)
    init[0] = x_eq[0] + rng.uniform(-INIT_Z_RANGE, INIT_Z_RANGE)
    init[1] = x_eq[1] + rng.uniform(-INIT_ZDOT_RANGE, INIT_ZDOT_RANGE)
    return init


def build_multisine_excitation(t_index: int, dt: float, freqs: np.ndarray, phase: np.ndarray) -> float:
    t = float(t_index) * dt
    return float(EXCITATION_AMP * np.mean(np.sin(2.0 * np.pi * freqs * t + phase)))


def collect_rollout(
    env,
    rng: np.random.Generator,
    episode_seed: int,
    x_eq: np.ndarray,
    u_eq: np.ndarray,
    lqr_gain: np.ndarray,
    delta_u_clip: float,
    excitation_freqs: np.ndarray,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    init_state = sample_initial_state(rng, x_eq)
    # For quad1d INIT_STATE_LABELS = ['init_x', 'init_x_dot'] mapping to z and z_dot.
    env.INIT_X = float(init_state[0])
    env.INIT_X_DOT = float(init_state[1])

    try:
        obs, info_reset = env.reset(seed=int(episode_seed))
    except Exception:
        return None, ['numerical_failure_reset']

    obs_arr = np.asarray(obs, dtype=np.float64)
    if obs_arr.shape[0] < 2:
        return None, ['observation_too_small']

    raw_obs_dim = obs_arr.shape[0]
    action_low = np.asarray(env.physical_action_bounds[0], dtype=np.float64)
    action_high = np.asarray(env.physical_action_bounds[1], dtype=np.float64)

    state_seq    = np.zeros((ROLLOUT_STEPS + 1, 2), dtype=np.float64)
    raw_obs_seq  = np.zeros((ROLLOUT_STEPS + 1, raw_obs_dim), dtype=np.float64)
    y_seq        = np.zeros((ROLLOUT_STEPS + 1, 1), dtype=np.float64)
    u_applied_seq = np.zeros((ROLLOUT_STEPS, 1), dtype=np.float64)
    u_cmd_seq    = np.zeros((ROLLOUT_STEPS, 1), dtype=np.float64)
    u_unclipped_seq = np.zeros((ROLLOUT_STEPS, 1), dtype=np.float64)
    saturation_mask = np.zeros((ROLLOUT_STEPS, 1), dtype=np.int8)

    state_seq[0]   = np.asarray(env.state, dtype=np.float64)
    raw_obs_seq[0] = obs_arr
    y_seq[0]       = obs_arr[[0]]

    phase = rng.uniform(-np.pi, np.pi, size=excitation_freqs.shape[0])
    discard_reasons: List[str] = []

    for t in range(ROLLOUT_STEPS):
        x_hat = np.asarray(obs_arr[:2], dtype=np.float64)
        if not np.isfinite(x_hat).all():
            discard_reasons.append('numerical_failure_state')
            break

        excitation = build_multisine_excitation(t, float(env.CTRL_TIMESTEP), excitation_freqs, phase)
        u_fb = u_eq - lqr_gain @ (x_hat - x_eq)
        u_unclipped = u_fb + excitation
        delta_clipped = np.clip(u_unclipped - u_eq, -delta_u_clip, delta_u_clip)
        u_cmd = np.clip(u_eq + delta_clipped, action_low, action_high)

        u_cmd_arr = np.atleast_1d(u_cmd)
        u_cmd_seq[t]       = u_cmd_arr
        u_unclipped_seq[t] = np.atleast_1d(u_unclipped)

        try:
            obs_next, _, done, _ = env.step(u_cmd_arr)
        except Exception:
            discard_reasons.append('numerical_failure_step')
            break

        applied_u = np.asarray(env.current_clipped_action, dtype=np.float64).reshape(1)
        u_applied_seq[t] = applied_u
        sat_low  = np.isclose(applied_u, action_low,  atol=SAT_ATOL)
        sat_high = np.isclose(applied_u, action_high, atol=SAT_ATOL)
        saturation_mask[t] = np.logical_or(sat_low, sat_high).astype(np.int8)

        obs_arr = np.asarray(obs_next, dtype=np.float64)
        if not np.isfinite(obs_arr).all() or obs_arr.shape[0] < 2:
            discard_reasons.append('numerical_failure_observation')
            break

        state_now = np.asarray(env.state, dtype=np.float64)
        if not np.isfinite(state_now).all():
            discard_reasons.append('numerical_failure_state')
            break

        state_seq[t + 1]   = state_now
        raw_obs_seq[t + 1] = obs_arr
        y_seq[t + 1]       = obs_arr[[0]]

        if done and t < ROLLOUT_STEPS - 1:
            discard_reasons.append('early_termination')
            break

    if discard_reasons:
        non_zero = np.nonzero(np.linalg.norm(u_applied_seq, axis=1) > 0.0)[0]
        effective = int(non_zero[-1] + 1) if non_zero.size > 0 else 0
        if effective != ROLLOUT_STEPS:
            return None, list(dict.fromkeys(discard_reasons))

    state_dev = state_seq - x_eq[None, :]
    saturation_fraction = float(np.mean(saturation_mask.astype(bool)))
    mean_clip = float(np.mean(np.abs(u_applied_seq - u_unclipped_seq)))

    rollout = {
        'episode_seed': int(episode_seed), 'init_state': init_state,
        'raw_obs': raw_obs_seq, 'state': state_seq, 'y': y_seq,
        'u_applied': u_applied_seq, 'u_command': u_cmd_seq, 'u_unclipped': u_unclipped_seq,
        'saturation_mask': saturation_mask,
        'max_z_dev':    float(np.max(np.abs(state_dev[:, 0]))),
        'max_zdot_dev': float(np.max(np.abs(state_dev[:, 1]))),
        'saturation_fraction': saturation_fraction,
        'mean_clip_magnitude': mean_clip,
        'physical_parameters': info_reset.get('physical_parameters', {}),
    }
    return rollout, []


def main() -> None:
    excitation_freqs = np.asarray(EXCITATION_FREQS, dtype=np.float64)
    rng = np.random.default_rng(SEED)

    script_dir   = Path(__file__).resolve().parent
    rollout_dir  = script_dir / 'rollouts'  / DATASET_NAME
    metadata_dir = script_dir / 'metadata' / DATASET_NAME
    metadata_path = metadata_dir / 'dataset_stats.json'
    runs_path    = metadata_dir / 'runs.jsonl'
    rollout_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = {
        'gui': False, 'quad_type': 1, 'task': 'stabilization', 'cost': 'quadratic',
        'ctrl_freq': CTRL_FREQ, 'pyb_freq': PYB_FREQ,
        'episode_len_sec': int(max(1, ROLLOUT_STEPS // CTRL_FREQ) + 2),
        'normalized_rl_action_space': False, 'done_on_out_of_bound': False,
        'randomized_init': False,
        # stabilization_goal[1] is used as z_ref in quad1d code.
        'task_info': {'stabilization_goal': [0.0, 1.0], 'stabilization_goal_tolerance': 1e-12},
    }

    env = make('quadrotor', **env_kwargs)

    x_eq = np.asarray(env.X_GOAL, dtype=np.float64)   # [z_eq=1.0, z_dot_eq=0.0]
    u_eq = np.asarray(env.U_GOAL, dtype=np.float64)   # [T_eq = m*g]
    action_low  = np.asarray(env.physical_action_bounds[0], dtype=np.float64)
    action_high = np.asarray(env.physical_action_bounds[1], dtype=np.float64)

    headroom = float(np.min(np.minimum(action_high - u_eq, u_eq - action_low)))
    delta_u_clip = float(min(DELTA_U_CLIP, 0.95 * headroom))

    q_mat    = get_cost_weight_matrix(Q_LQR, env.symbolic.nx)
    r_mat    = get_cost_weight_matrix(R_LQR, env.symbolic.nu)
    lqr_gain = compute_lqr_gain(env.symbolic, env.symbolic.X_EQ, env.symbolic.U_EQ,
                                 q_mat, r_mat, discrete_dynamics=True)

    obs_mapping = {
        'quad1d_state_labels': ['z', 'z_dot'], 'y_labels': ['z'],
        'y_from_raw_observation_indices': [0], 'y_formula': 'y = [raw_obs[0]]',
    }

    metadata = load_or_initialize_metadata(
        path=metadata_path, dataset_name=DATASET_NAME, dt=float(env.CTRL_TIMESTEP),
        raw_obs_dim=int(env.obs_dim), zdev_edges=ZDEV_EDGES,
        obs_mapping=obs_mapping, env_settings=to_jsonable(env_kwargs),
    )

    start_rollout_idx = next_rollout_index(rollout_dir)
    run_id    = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    run_start = utc_now()
    run_discard_reasons = Counter()
    accepted_count = attempt_count = 0

    print(f"[collect] dataset={DATASET_NAME} target={NUM_ROLLOUTS} dt={env.CTRL_TIMESTEP:.4f}")
    print(f"[collect] x_eq={x_eq.tolist()} u_eq={u_eq.tolist()} delta_u_clip={delta_u_clip:.4f}")
    print(f"[collect] lqr_gain={lqr_gain.tolist()}")

    while accepted_count < NUM_ROLLOUTS and attempt_count < MAX_ATTEMPTS:
        episode_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        attempt_count += 1
        metadata['counts']['attempted_rollouts'] = int(metadata['counts']['attempted_rollouts']) + 1

        rollout_data, fail_reasons = collect_rollout(
            env=env, rng=rng, episode_seed=episode_seed,
            x_eq=x_eq, u_eq=u_eq, lqr_gain=lqr_gain,
            delta_u_clip=delta_u_clip, excitation_freqs=excitation_freqs,
        )

        if rollout_data is None:
            reason = fail_reasons[0] if fail_reasons else 'numerical_failure_unknown'
            run_discard_reasons[reason] += 1
            metadata['discard_reasons'][reason] = int(metadata['discard_reasons'].get(reason, 0)) + 1
            metadata['counts']['discarded_rollouts'] = int(metadata['counts']['discarded_rollouts']) + 1
            continue

        zdev_idx = bucket_index(rollout_data['max_z_dev'], ZDEV_EDGES)
        discard_reasons: List[str] = []
        if rollout_data['max_z_dev']    > ACCEPT_Z_MAX:       discard_reasons.append('z_deviation_exceeded')
        if rollout_data['max_zdot_dev'] > ACCEPT_ZDOT_MAX:    discard_reasons.append('zdot_deviation_exceeded')
        if rollout_data['saturation_fraction'] > SATURATION_FRAC_MAX: discard_reasons.append('saturation_fraction_exceeded')
        if rollout_data['mean_clip_magnitude'] > MEAN_CLIP_MAX: discard_reasons.append('clip_mean_exceeded')

        if discard_reasons:
            metadata['counts']['discarded_rollouts'] = int(metadata['counts']['discarded_rollouts']) + 1
            metadata['bucket_counts']['discarded']['max_z_deviation'][zdev_idx] += 1
            for r in discard_reasons:
                run_discard_reasons[r] += 1
                metadata['discard_reasons'][r] = int(metadata['discard_reasons'].get(r, 0)) + 1
            continue

        accepted_count += 1
        metadata['counts']['accepted_rollouts'] = int(metadata['counts']['accepted_rollouts']) + 1
        metadata['bucket_counts']['accepted']['max_z_deviation'][zdev_idx] += 1

        as_ = metadata['stats']['accepted']
        update_vector_stats(as_['raw_obs'], rollout_data['raw_obs'])
        update_vector_stats(as_['state'],   rollout_data['state'])
        update_vector_stats(as_['y'],       rollout_data['y'])
        update_vector_stats(as_['u'],       rollout_data['u_applied'])
        update_vector_stats(as_['y_deviation'], rollout_data['y'] - x_eq[[0]][None, :])
        update_vector_stats(as_['u_deviation'], rollout_data['u_applied'] - u_eq[None, :])

        sat = metadata['saturation']
        sat_steps = int(np.any(rollout_data['saturation_mask'].astype(bool), axis=1).sum())
        sat['timesteps_total']        = int(sat['timesteps_total']) + ROLLOUT_STEPS
        sat['timesteps_saturating']   = int(sat['timesteps_saturating']) + sat_steps
        sat['sum_clip_magnitude']     = float(sat['sum_clip_magnitude']) + float(np.sum(np.abs(rollout_data['u_applied'] - rollout_data['u_unclipped'])))
        sat['max_clip_magnitude']     = max(float(sat['max_clip_magnitude']), float(np.max(np.abs(rollout_data['u_applied'] - rollout_data['u_unclipped']))))
        sat['sum_saturation_fraction_per_rollout'] = float(sat['sum_saturation_fraction_per_rollout']) + rollout_data['saturation_fraction']
        sat['rollouts_accounted']     = int(sat['rollouts_accounted']) + 1

        rollout_name = f'rollout_{start_rollout_idx + accepted_count - 1:07d}.npz'
        np.savez(
            rollout_dir / rollout_name,
            y=rollout_data['y'].astype(np.float32),
            u=rollout_data['u_applied'].astype(np.float32),
            raw_obs=rollout_data['raw_obs'].astype(np.float32),
            state=rollout_data['state'].astype(np.float32),
            u_command=rollout_data['u_command'].astype(np.float32),
            u_unclipped=rollout_data['u_unclipped'].astype(np.float32),
            saturation_mask=rollout_data['saturation_mask'].astype(np.int8),
            dt=np.asarray([env.CTRL_TIMESTEP], dtype=np.float64),
            x_eq=x_eq.astype(np.float64), u_eq=u_eq.astype(np.float64),
            action_low=action_low.astype(np.float64), action_high=action_high.astype(np.float64),
            init_state=rollout_data['init_state'].astype(np.float64),
            episode_seed=np.asarray([rollout_data['episode_seed']], dtype=np.int64),
            physical_parameters_json=np.asarray([json.dumps(rollout_data['physical_parameters'])]),
        )

        if accepted_count % 10 == 0 or accepted_count == NUM_ROLLOUTS:
            print(f"[collect] accepted={accepted_count}/{NUM_ROLLOUTS} attempts={attempt_count} last={rollout_name}")

    env.close()

    metadata['num_runs'] = int(metadata.get('num_runs', 0)) + 1
    metadata['last_updated_utc'] = utc_now()
    metadata['last_run_knobs'] = {
        'effective_delta_u_clip': delta_u_clip, 'x_eq': x_eq.tolist(), 'u_eq': u_eq.tolist(),
        'lqr_gain': lqr_gain.tolist(), 'dataset_name': DATASET_NAME,
        'num_rollouts': NUM_ROLLOUTS, 'rollout_steps': ROLLOUT_STEPS,
    }

    # Write summary stats.
    as_ = metadata['stats']['accepted']
    metadata['summary'] = {
        'accepted_distributions': {
            'state': summarize_vector_stats(as_['state'], ['z', 'z_dot']),
            'y': summarize_vector_stats(as_['y'], ['z']),
            'u': summarize_vector_stats(as_['u'], ['T']),
        },
    }
    write_json_atomic(metadata_path, to_jsonable(metadata))

    run_summary = {'run_id': run_id, 'start_utc': run_start, 'end_utc': utc_now(),
                   'dataset_name': DATASET_NAME, 'accepted': accepted_count,
                   'attempted': attempt_count, 'discard_reasons': dict(run_discard_reasons)}
    with runs_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(run_summary, sort_keys=True) + '\n')

    print(f"[collect] Done. accepted={accepted_count}/{attempt_count}. Discards: {dict(run_discard_reasons)}")
    print(f"[collect] Rollouts → {rollout_dir}")


if __name__ == '__main__':
    main()
