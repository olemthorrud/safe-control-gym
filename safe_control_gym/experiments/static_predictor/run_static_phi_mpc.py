#!/usr/bin/env python3
"""End-to-end runner: fit predictor and run StaticPhiMPC.

Usage:
  python safe_control_gym/experiments/static_predictor/run_static_phi_mpc.py --ls
  python safe_control_gym/experiments/static_predictor/run_static_phi_mpc.py --sgd
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

try:
    from .fit_static_phi_ddpc import Linear_Predictor
    print("it was fine")
except ImportError:
    from fit_static_phi_ddpc import Linear_Predictor

# ---------------------------------------------------------------------------
# Process noise injected into the quad2d dynamics during the MPC experiment.
# std is in Newtons; hovering thrust is ~0.53 N so 0.01 ≈ 2% noise.
# Set to 0.0 to disable.
# ---------------------------------------------------------------------------
PROCESS_NOISE_STD = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit predictor and run StaticPhiMPC end-to-end.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ls", action="store_true", help="Use analytical least-squares fit.")
    mode.add_argument("--sgd", action="store_true", help="Use SGD/DDPC-style fit.")

    parser.add_argument("--sys", type=str, default="quadrotor_2D", help="System override name.")
    parser.add_argument("--task", type=str, default="stab", choices=["stab", "track"], help="Task override name.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="safe_control_gym/experiments/static_phi_mpc_experiment/results",
        help="Output directory for experiment results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fit_mode = "ls" if args.ls else "sgd"

    # 1) Fit predictor.
    trainer = Linear_Predictor(fit_mode=fit_mode)
    run_cfg = trainer.run()
    phi_path = str(run_cfg["saved_phi"])
    print(f"[runner] predictor mode={fit_mode}")
    print(f"[runner] predictor phi={phi_path}")

    # 2) Build generated algo override with fitted phi path.
    repo_root = Path(__file__).resolve().parents[3]
    exp_dir = repo_root / "safe_control_gym" / "experiments" / "static_phi_mpc_experiment"
    cfg_dir = exp_dir / "config_overrides"

    task_override = cfg_dir / f"{args.sys}_{args.task}.yaml"
    base_algo_override = cfg_dir / f"static_phi_mpc_{args.sys}_{args.task}.yaml"

    if not task_override.exists():
        raise FileNotFoundError(f"Task override not found: {task_override}")
    if not base_algo_override.exists():
        raise FileNotFoundError(f"Algo override not found: {base_algo_override}")

    with base_algo_override.open("r", encoding="utf-8") as f:
        algo_cfg = yaml.safe_load(f)

    algo_cfg["algo_config"]["phi_path"] = phi_path
    algo_cfg["algo_config"]["phi_bias_path"] = None
    algo_cfg["algo_config"]["t_ini"] = int(run_cfg["history"])
    algo_cfg["algo_config"]["horizon"] = int(run_cfg["horizon"])

    generated_dir = cfg_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    generated_override = generated_dir / f"static_phi_mpc_{args.sys}_{args.task}_{fit_mode}_{ts}.yaml"
    with generated_override.open("w", encoding="utf-8") as f:
        yaml.safe_dump(algo_cfg, f, sort_keys=False)

    # Optionally inject process noise into the task config.
    if PROCESS_NOISE_STD > 0.0:
        with task_override.open("r", encoding="utf-8") as f:
            task_cfg = yaml.safe_load(f)
        task_cfg["task_config"]["disturbances"] = {
            "dynamics": [{"disturbance_func": "white_noise", "std": PROCESS_NOISE_STD}]
        }
        generated_task_override = generated_dir / f"{args.sys}_{args.task}_noise_{ts}.yaml"
        with generated_task_override.open("w", encoding="utf-8") as f:
            yaml.safe_dump(task_cfg, f, sort_keys=False)
        task_path = generated_task_override
        print(f"[runner] process noise std={PROCESS_NOISE_STD} N (dynamics)")
    else:
        task_path = task_override

    # 3) Run StaticPhiMPC experiment.
    if args.sys == "cartpole":
        task_name = "cartpole"
    else:
        task_name = "quadrotor"

    run_cmd = [
        sys.executable,
        str(exp_dir / "static_phi_mpc_experiment.py"),
        "--task",
        task_name,
        "--algo",
        "static_phi_mpc",
        "--output_dir",
        str((repo_root / args.output_dir).resolve()),
        "--overrides",
        str(task_path),
        str(generated_override),
    ]

    env = os.environ.copy()
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root}:{py_path}" if py_path else str(repo_root)

    print(f"[runner] task override: {task_override}")
    print(f"[runner] algo override: {generated_override}")
    print(f"[runner] running: {' '.join(run_cmd)}")
    subprocess.run(run_cmd, check=True, env=env)


if __name__ == "__main__":
    main()
