#!/usr/bin/env python3
"""Evaluate predictor accuracy as a function of pitch angle θ.

For each prediction window, record θ at the start of the horizon, compute
the per-channel prediction error, then bin and plot.  Reveals that the
linear predictor is most accurate near θ≈0 and degrades at larger angles,
as expected from the sin(θ)/cos(θ) nonlinearities in the quad2d dynamics.

Usage:
    python analyze_predictor_vs_theta.py

Edit the constants below to point at a different predictor or dataset.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Settings — edit these
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

# Path to a specific predictor run directory.  None → auto-select the most
# recently modified subdirectory of analytic_predictors/.
PREDICTOR_DIR: Path | None = None

# Bin edges (radians) for θ.  Signed so asymmetry is visible.
# Fine bins near hover (θ≈0), coarser out to the data limit (~1.2 rad / 68 deg).
THETA_BIN_EDGES = [
    -1.20, -0.80, -0.50, -0.35, -0.25, -0.15, -0.08, -0.03,
     0.03,  0.08,  0.15,  0.25,  0.35,  0.50,  0.80,  1.20,
]

# Output directory for the figure (created if needed).
OUTPUT_DIR = SCRIPT_DIR / "analysis_output"

# y channel names for quad2d
Y_LABELS = ["x", "z", "θ"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_predictor_dir() -> Path:
    if PREDICTOR_DIR is not None:
        return Path(PREDICTOR_DIR)
    candidates = sorted(
        (SCRIPT_DIR / "analytic_predictors").iterdir(),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError("No predictor runs found in analytic_predictors/")
    return candidates[-1]


def load_rollouts(rollout_dir: Path) -> list[dict]:
    files = sorted(rollout_dir.glob("rollout_*.npz"))
    if not files:
        raise FileNotFoundError(f"No rollout files found in {rollout_dir}")
    rollouts = []
    for path in files:
        data = np.load(path)
        rollouts.append({
            "u": np.asarray(data["u"], dtype=np.float64),
            "y": np.asarray(data["y"], dtype=np.float64),
        })
    return rollouts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Load predictor
    pred_dir = resolve_predictor_dir()
    print(f"[analyze] predictor: {pred_dir.name}")

    phi = np.load(pred_dir / "phi.npy")             # (n_y*N, n_reg)

    import json
    with (pred_dir / "run_config.json").open() as f:
        cfg = json.load(f)

    H   = int(cfg["history"])
    N   = int(cfg["horizon"])
    S   = int(cfg["stride"])
    n_y = int(cfg["n_y"])
    n_u = int(cfg["n_u"])
    rollout_dir = Path(cfg["rollout_dir"])

    print(f"[analyze] H={H}  N={N}  stride={S}  n_y={n_y}  n_u={n_u}")
    print(f"[analyze] rollout_dir: {rollout_dir}")

    # 2. Load rollouts
    rollouts = load_rollouts(rollout_dir)
    print(f"[analyze] {len(rollouts)} rollouts loaded")

    # 3. Sweep all windows, collect (theta_t, squared_error per channel)
    theta_list: list[float] = []
    sq_err_list: list[np.ndarray] = []   # each entry shape (n_y,)

    for rollout in rollouts:
        u = rollout["u"]   # (T, n_u)
        y = rollout["y"]   # (T+1, n_y)
        T = u.shape[0]

        for t in range(H, T - N + 1, S):
            # Regressor
            u_hist   = u[t - H : t, :].reshape(-1)
            y_hist   = y[t - H : t + 1, :].reshape(-1)
            u_future = u[t : t + N, :].reshape(-1)
            z = np.concatenate([u_hist, y_hist, u_future])

            # Ground truth and prediction
            y_true = y[t + 1 : t + N + 1, :]           # (N, n_y)
            y_pred = (z @ phi.T).reshape(N, n_y)        # (N, n_y)
            residual = y_true - y_pred                  # (N, n_y)

            # θ at the start of this window (last entry of y_hist, channel 2)
            theta_t = float(y[t, 2])

            # Per-channel mean squared error over the horizon
            sq_err = np.mean(residual ** 2, axis=0)     # (n_y,)

            theta_list.append(theta_t)
            sq_err_list.append(sq_err)

    theta_arr  = np.array(theta_list)                   # (W,)
    sq_err_arr = np.array(sq_err_list)                  # (W, n_y)
    print(f"[analyze] {len(theta_arr)} windows total")

    # 4. Bin by theta
    edges = np.array(THETA_BIN_EDGES, dtype=np.float64)
    bin_idx = np.digitize(theta_arr, edges) - 1          # 0-indexed bin
    n_bins  = len(edges) - 1
    bin_centres = 0.5 * (edges[:-1] + edges[1:])

    bin_counts  = np.zeros(n_bins, dtype=int)
    bin_mse     = np.zeros((n_bins, n_y), dtype=np.float64)

    for b in range(n_bins):
        mask = (bin_idx == b)
        bin_counts[b] = int(mask.sum())
        if bin_counts[b] > 0:
            bin_mse[b] = sq_err_arr[mask].mean(axis=0)

    # 5. Print table
    col_w = 10
    header = f"{'bin centre':>12}  {'count':>6}" + "".join(f"  {'MSE_' + l:>{col_w}}" for l in Y_LABELS) + f"  {'MSE_overall':>{col_w}}"
    print("\n" + header)
    print("-" * len(header))
    for b in range(n_bins):
        overall = bin_mse[b].mean() if bin_counts[b] > 0 else float("nan")
        row = f"{bin_centres[b]:>12.3f}  {bin_counts[b]:>6}"
        for ch in range(n_y):
            v = bin_mse[b, ch] if bin_counts[b] > 0 else float("nan")
            row += f"  {v:>{col_w}.4e}"
        row += f"  {overall:>{col_w}.4e}"
        print(row)

    # 6. Plot
    bin_widths = np.diff(edges)
    bar_width = bin_widths / n_y * 0.85        # each bar occupies 1/n_y of its bin
    offsets = np.arange(n_y) - (n_y - 1) / 2  # [-1, 0, 1] for n_y=3

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["steelblue", "darkorange", "seagreen"]
    for ch in range(n_y):
        heights = np.where(bin_counts > 0, bin_mse[:, ch], np.nan)
        ax.bar(
            bin_centres + offsets[ch] * bar_width,
            heights,
            width=bar_width,
            label=f"MSE {Y_LABELS[ch]}",
            color=colors[ch],
            alpha=0.85,
        )

    ax.set_yscale("log")
    ax.set_xlabel("θ at window start (rad)")
    ax.set_ylabel("Mean squared prediction error  [log scale]")
    ax.set_title("Predictor accuracy vs pitch angle θ")
    ax.set_xticks(bin_centres)
    ax.set_xticklabels([f"{c:.2f}" for c in bin_centres], rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "predictor_mse_vs_theta.png"
    fig.savefig(out_path, dpi=150)
    print(f"\n[analyze] figure saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
