from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

try:
    from .replay_buffer import ReplayBuffer
except ImportError:
    from replay_buffer import ReplayBuffer


# ----------------------------
# Global settings
# ----------------------------
DATASET_NAME = "theta_focused_plus_robust_fresh"
ROLLOUT_FILE_PATTERN = "rollout_*.npz"

FIT_MODE = "ls"  # "ls" or "sgd"

HISTORY = 20
HORIZON = 40
STRIDE = 5

TRAIN_FRACTION = 0.9
SPLIT_SEED = 123

NORMALIZATION_MODE = "scale"
NORM_EPS = 1e-8
RIDGE = 1e-3

BATCH_SIZE = 64
TRAIN_EPOCHS = 5
LEARNING_RATE = 3e-3
L1_LAMBDA = 0.0
L2_LAMBDA = 0.0
SEED = 123
PRINT_EVERY = 200


class Linear_Predictor:
    """Owns data loading, fitting, evaluation, and artifact saving."""

    def __init__(self, fit_mode: str | None = None):
        self.dataset_name = DATASET_NAME
        self.rollout_file_pattern = ROLLOUT_FILE_PATTERN

        self.fit_mode = FIT_MODE if fit_mode is None else str(fit_mode)
        if self.fit_mode not in ("ls", "sgd"):
            raise ValueError(f"Unsupported fit mode: {self.fit_mode}")

        self.history = HISTORY
        self.horizon = HORIZON
        self.stride = STRIDE

        self.train_fraction = TRAIN_FRACTION
        self.split_seed = SPLIT_SEED

        self.normalization_mode = NORMALIZATION_MODE
        self.norm_eps = NORM_EPS
        self.ridge = RIDGE

        self.batch_size = BATCH_SIZE
        self.train_epochs = TRAIN_EPOCHS
        self.learning_rate = LEARNING_RATE
        self.l1_lambda = L1_LAMBDA
        self.l2_lambda = L2_LAMBDA
        self.seed = SEED
        self.print_every = PRINT_EVERY

        self.script_dir = Path(__file__).resolve().parent
        self.rollout_dir = self.script_dir / "rollouts" / self.dataset_name
        if self.fit_mode == "ls":
            self.output_dir = self.script_dir / "analytic_predictors"
        else:
            self.output_dir = self.script_dir / "sgd_predictors"
        self.replay_buffer_path = (
            self.script_dir / "static_phi_ddpc_stats"
            / f"replay_buffer_{self.dataset_name}_H-{self.history}_N-{self.horizon}.pkl"
        )

    # ----------------------------
    # Data helpers
    # ----------------------------
    def load_rollouts_from_npz(self) -> list[dict]:
        """Load rollout files and keep only u and y."""
        files = sorted(self.rollout_dir.glob(self.rollout_file_pattern))
        rollouts: list[dict] = []
        for path in files:
            data = np.load(path)
            rollouts.append(
                {
                    "u": np.asarray(data["u"], dtype=np.float64),
                    "y": np.asarray(data["y"], dtype=np.float64),
                }
            )
        return rollouts

    def split_rollouts(self, rollouts: list[dict]) -> tuple[list[dict], list[dict], np.ndarray, np.ndarray]:
        """Split rollouts by index into train/test."""
        n = len(rollouts)
        rng = np.random.default_rng(self.split_seed)
        perm = rng.permutation(n)
        n_train = int(np.floor(self.train_fraction * n))
        n_train = max(1, min(n - 1, n_train))

        train_idx = perm[:n_train]
        test_idx = perm[n_train:]

        train_rollouts = [rollouts[int(i)] for i in train_idx.tolist()]
        test_rollouts = [rollouts[int(i)] for i in test_idx.tolist()]
        return train_rollouts, test_rollouts, train_idx, test_idx

    def make_examples_from_rollout(self, rollout: dict) -> tuple[np.ndarray, np.ndarray]:
        """Build (z, Y) windows from one rollout using current history/horizon/stride."""
        u = rollout["u"]
        y = rollout["y"]
        t_total = u.shape[0]

        z_list = []
        y_list = []

        for t in range(self.history, t_total - self.horizon + 1, self.stride):
            u_hist = u[t - self.history : t, :].reshape(-1)
            y_hist = y[t - self.history : t + 1, :].reshape(-1)
            u_future = u[t : t + self.horizon, :].reshape(-1)
            y_target = y[t + 1 : t + self.horizon + 1, :].reshape(-1)

            z_list.append(np.concatenate((u_hist, y_hist, u_future), axis=0))
            y_list.append(y_target)

        if len(z_list) == 0:
            n_y = y.shape[1]
            n_u = u.shape[1]
            n_reg = n_u * self.history + n_y * (self.history + 1) + n_u * self.horizon
            n_tar = n_y * self.horizon
            return (
                np.zeros((0, n_reg), dtype=np.float64),
                np.zeros((0, n_tar), dtype=np.float64),
            )

        return (
            np.asarray(z_list, dtype=np.float64),
            np.asarray(y_list, dtype=np.float64),
        )

    def convert_rollouts_to_replay_format(self, rollouts: list[dict]) -> list[list[tuple]]:
        """Convert rollout arrays to ReplayBuffer rollout format."""
        replay_rollouts: list[list[tuple]] = []

        for rollout in rollouts:
            u = rollout["u"]
            y = rollout["y"]
            t_total = u.shape[0]

            one_rollout: list[tuple] = []
            for t in range(self.history, t_total):
                u_hist = u[t - self.history : t, :].reshape(-1)
                y_hist = y[t - self.history : t + 1, :].reshape(-1)
                hist = np.concatenate((u_hist, y_hist), axis=0).astype(np.float64)

                obs = np.asarray(y[t + 1], dtype=np.float64).copy()
                action = np.asarray(u[t], dtype=np.float64).copy()
                next_obs = np.asarray(y[min(t + 2, t_total)], dtype=np.float64).copy()

                state = obs.copy()
                next_state = next_obs.copy()
                reward = 0.0
                done = bool(t == (t_total - 1))
                info = {"hist": hist}

                one_rollout.append(
                    (state, obs, action, reward, next_state, next_obs, done, info)
                )

            replay_rollouts.append(one_rollout)

        return replay_rollouts

    def load_or_create_replay_rollouts(self, rollouts: list[dict]) -> list[list[tuple]]:
        """Build replay pickle once and reuse it."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.replay_buffer_path.parent.mkdir(parents=True, exist_ok=True)

        if self.replay_buffer_path.exists():
            with self.replay_buffer_path.open("rb") as f:
                return pickle.load(f)

        replay_rollouts = self.convert_rollouts_to_replay_format(rollouts)
        with self.replay_buffer_path.open("wb") as f:
            pickle.dump(replay_rollouts, f)
        return replay_rollouts

    # ----------------------------
    # Phi parameterization
    # ----------------------------
    def count_phi_params(self, n_y: int, n_u: int) -> int:
        n_target = n_y * self.horizon
        n_hist = n_u * self.history + n_y * (self.history + 1)
        n_past = n_target * n_hist
        n_future = n_y * n_u * self.horizon * (self.horizon + 1) // 2
        return int(n_past + n_future)

    def build_phi_from_p_dyn_torch(self, p_dyn: torch.Tensor, n_y: int, n_u: int) -> torch.Tensor:
        """Build structured Phi from p_dyn (torch)."""
        n_target = n_y * self.horizon
        n_hist = n_u * self.history + n_y * (self.history + 1)
        n_future_u = n_u * self.horizon

        n_past = n_target * n_hist
        phi_past = p_dyn[:n_past].reshape(n_target, n_hist)

        phi_future = torch.zeros((n_target, n_future_u), dtype=p_dyn.dtype)
        cursor = n_past

        for step in range(self.horizon):
            row_start = step * n_y
            row_end = row_start + n_y
            col_end = (step + 1) * n_u

            block_len = n_y * col_end
            block = p_dyn[cursor : cursor + block_len].reshape(n_y, col_end)
            phi_future[row_start:row_end, :col_end] = block
            cursor += block_len

        return torch.cat((phi_past, phi_future), dim=1)

    def pack_p_dyn_from_phi(self, phi: np.ndarray, n_y: int, n_u: int) -> np.ndarray:
        """Pack raw-space Phi into raw-space p_dyn in the DDPC ordering."""
        n_target = n_y * self.horizon
        n_hist = n_u * self.history + n_y * (self.history + 1)

        parts = [phi[:, :n_hist].reshape(-1)]

        phi_future = phi[:, n_hist:]
        for step in range(self.horizon):
            row_start = step * n_y
            row_end = row_start + n_y
            col_end = (step + 1) * n_u
            parts.append(phi_future[row_start:row_end, :col_end].reshape(-1))

        return np.concatenate(parts, axis=0).astype(np.float64)

    def build_allowed_feature_indices(self, n_y: int, n_u: int) -> list[np.ndarray]:
        """Allowed columns for each target row in masked Phi."""
        n_hist = n_u * self.history + n_y * (self.history + 1)
        n_target = n_y * self.horizon
        past_idx = np.arange(n_hist, dtype=np.int64)

        allowed: list[np.ndarray] = []
        for out_row in range(n_target):
            step = out_row // n_y
            fut_len = (step + 1) * n_u
            fut_idx = np.arange(n_hist, n_hist + fut_len, dtype=np.int64)
            allowed.append(np.concatenate((past_idx, fut_idx), axis=0))
        return allowed

    # ----------------------------
    # Normalization
    # ----------------------------
    def compute_scale_normalization(
        self,
        rollouts: list[dict],
        n_y: int,
        n_u: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute scale-only normalization from selected rollouts."""
        n_reg = n_u * self.history + n_y * (self.history + 1) + n_u * self.horizon
        n_tar = n_y * self.horizon

        z_sum = np.zeros(n_reg, dtype=np.float64)
        z_sum_sq = np.zeros(n_reg, dtype=np.float64)
        y_sum = np.zeros(n_tar, dtype=np.float64)
        y_sum_sq = np.zeros(n_tar, dtype=np.float64)
        n_total = 0

        for rollout in rollouts:
            z_batch, y_batch = self.make_examples_from_rollout(rollout)
            if z_batch.shape[0] == 0:
                continue
            z_sum += np.sum(z_batch, axis=0)
            z_sum_sq += np.sum(np.square(z_batch), axis=0)
            y_sum += np.sum(y_batch, axis=0)
            y_sum_sq += np.sum(np.square(y_batch), axis=0)
            n_total += int(z_batch.shape[0])

        z_mean = z_sum / float(n_total)
        y_mean = y_sum / float(n_total)
        z_var = np.maximum(z_sum_sq / float(n_total) - np.square(z_mean), 0.0)
        y_var = np.maximum(y_sum_sq / float(n_total) - np.square(y_mean), 0.0)

        z_scale = np.maximum(np.sqrt(z_var), self.norm_eps)
        y_scale = np.maximum(np.sqrt(y_var), self.norm_eps)
        return z_scale.astype(np.float64), y_scale.astype(np.float64)

    # ----------------------------
    # LS fit
    # ----------------------------
    def fit_phi_ls(
        self,
        train_rollouts: list[dict],
        n_y: int,
        n_u: int,
        z_scale: np.ndarray,
        y_scale: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Fit masked Phi with analytical least squares."""
        n_reg = n_u * self.history + n_y * (self.history + 1) + n_u * self.horizon
        n_tar = n_y * self.horizon

        gram = np.zeros((n_reg, n_reg), dtype=np.float64)
        cross = np.zeros((n_reg, n_tar), dtype=np.float64)

        for rollout in train_rollouts:
            z_batch, y_batch = self.make_examples_from_rollout(rollout)
            if z_batch.shape[0] == 0:
                continue
            z_norm = z_batch / z_scale[None, :]
            y_norm = y_batch / y_scale[None, :]
            gram += z_norm.T @ z_norm
            cross += z_norm.T @ y_norm

        allowed = self.build_allowed_feature_indices(n_y=n_y, n_u=n_u)
        phi_norm = np.zeros((n_tar, n_reg), dtype=np.float64)

        for out_row in range(n_tar):
            idx = allowed[out_row]
            g_sub = gram[np.ix_(idx, idx)].copy()
            c_sub = cross[idx, out_row]
            if self.ridge > 0.0:
                g_sub += self.ridge * np.eye(g_sub.shape[0], dtype=np.float64)
            try:
                w = np.linalg.solve(g_sub, c_sub)
            except np.linalg.LinAlgError:
                w = np.linalg.pinv(g_sub) @ c_sub
            phi_norm[out_row, idx] = w

        phi_raw = (y_scale[:, None] * phi_norm) / z_scale[None, :]
        p_dyn_raw = self.pack_p_dyn_from_phi(phi=phi_raw, n_y=n_y, n_u=n_u)
        return phi_raw, p_dyn_raw, phi_norm, 0

    # ----------------------------
    # SGD fit
    # ----------------------------
    def build_window_indices(
        self,
        replay: ReplayBuffer,
        selected_rollout_indices: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Build all train windows from selected rollouts."""
        windows: list[tuple[int, int]] = []
        for rollout_idx in selected_rollout_indices.tolist():
            rollout = replay.buffer[int(rollout_idx)]
            max_start = len(rollout) - self.horizon
            if max_start < 0:
                continue
            for start in range(0, max_start + 1, self.stride):
                windows.append((int(rollout_idx), start))
        return windows

    def fit_phi_sgd(
        self,
        replay: ReplayBuffer,
        train_windows: list[tuple[int, int]],
        n_y: int,
        n_u: int,
        z_scale: np.ndarray,
        y_scale: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Fit masked Phi with Adam in DDPC-style parameterization."""
        torch.manual_seed(self.seed)
        train_rng = np.random.default_rng(self.seed + 77)

        n_params = self.count_phi_params(n_y=n_y, n_u=n_u)
        p_dyn_train = 0.25 * torch.rand(n_params, dtype=torch.float64)
        p_dyn_train.requires_grad_()
        optimizer = optim.Adam([p_dyn_train], lr=self.learning_rate, betas=(0.9, 0.999))

        z_scale_t = torch.tensor(z_scale, dtype=torch.float64)
        y_scale_t = torch.tensor(y_scale, dtype=torch.float64)

        updates_per_epoch = int(np.ceil(len(train_windows) / self.batch_size))
        expected_updates = int(self.train_epochs * updates_per_epoch)
        print(
            f"[fit_phi_sgd] windows={len(train_windows)} "
            f"updates_per_epoch={updates_per_epoch} expected_updates={expected_updates}"
        )

        total_updates = 0

        for epoch in range(self.train_epochs):
            perm = train_rng.permutation(len(train_windows))

            for batch_start in range(0, len(train_windows), self.batch_size):
                batch_ids = perm[batch_start : batch_start + self.batch_size]

                hist_batch = []
                u_batch = []
                y_batch = []

                for bi in batch_ids.tolist():
                    rollout_idx, start = train_windows[bi]
                    rollout = replay.buffer[rollout_idx]
                    seq = rollout[start : start + self.horizon]

                    obss = [tr[1] for tr in seq]
                    acts = [tr[2] for tr in seq]
                    infos = [tr[7] for tr in seq]

                    y_batch.append(np.asarray(obss, dtype=np.float64).reshape(-1))
                    u_batch.append(np.asarray(acts, dtype=np.float64).reshape(-1))
                    hist_batch.append(np.asarray(infos[0]["hist"], dtype=np.float64).reshape(-1))

                hist_t = torch.tensor(np.asarray(hist_batch, dtype=np.float64), dtype=torch.float64)
                u_t = torch.tensor(np.asarray(u_batch, dtype=np.float64), dtype=torch.float64)
                y_t = torch.tensor(np.asarray(y_batch, dtype=np.float64), dtype=torch.float64)

                phi_norm_t = self.build_phi_from_p_dyn_torch(p_dyn_train, n_y=n_y, n_u=n_u)
                z_t = torch.cat((hist_t, u_t), dim=1)

                z_norm_t = z_t / z_scale_t[None, :]
                y_norm_t = y_t / y_scale_t[None, :]
                y_hat_norm = torch.matmul(z_norm_t, phi_norm_t.T)

                mse_loss = F.mse_loss(y_hat_norm, y_norm_t)
                l1_loss = self.l1_lambda * torch.norm(p_dyn_train, 1)
                l2_loss = self.l2_lambda * torch.sum(p_dyn_train * p_dyn_train)
                loss = mse_loss + l1_loss + l2_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_updates += 1

                if total_updates == 1 or (total_updates % self.print_every) == 0:
                    print(
                        f"[fit_phi_sgd] epoch={epoch + 1}/{self.train_epochs} "
                        f"update={total_updates} loss={float(loss.detach().cpu().item()):.8f} "
                        f"mse={float(mse_loss.detach().cpu().item()):.8f}"
                    )

        phi_norm = self.build_phi_from_p_dyn_torch(p_dyn_train, n_y=n_y, n_u=n_u)
        phi_norm_np = phi_norm.detach().cpu().numpy()
        phi_raw = (y_scale[:, None] * phi_norm_np) / z_scale[None, :]

        # Save raw-space p_dyn so downstream parameterization is consistent with MPC-space Phi.
        p_dyn_raw = self.pack_p_dyn_from_phi(phi=phi_raw, n_y=n_y, n_u=n_u)
        return phi_raw, p_dyn_raw, phi_norm_np, total_updates

    # ----------------------------
    # Evaluation and run
    # ----------------------------
    def evaluate_phi_mse(self, rollouts: list[dict], phi: np.ndarray) -> tuple[int, float, list[float]]:
        """Compute overall and per-step MSE on provided rollouts."""
        n_y = int(rollouts[0]["y"].shape[1])
        n_windows = 0
        overall_sq = 0.0
        overall_count = 0

        per_step_sq = np.zeros(self.horizon, dtype=np.float64)
        per_step_count = np.zeros(self.horizon, dtype=np.int64)

        for rollout in rollouts:
            z_batch, y_batch = self.make_examples_from_rollout(rollout)
            if z_batch.shape[0] == 0:
                continue
            y_pred = z_batch @ phi.T
            residual = y_batch - y_pred

            n_windows += int(z_batch.shape[0])
            overall_sq += float(np.sum(np.square(residual)))
            overall_count += int(residual.size)

            residual_step = residual.reshape(-1, self.horizon, n_y)
            per_step_sq += np.sum(np.square(residual_step), axis=(0, 2))
            per_step_count += residual_step.shape[0] * n_y

        overall_mse = overall_sq / float(overall_count)
        per_step_mse = (per_step_sq / per_step_count.astype(np.float64)).tolist()
        return n_windows, float(overall_mse), per_step_mse

    def make_run_tag(self, n_y: int, n_u: int) -> str:
        lr_tag = f"{self.learning_rate:.0e}".replace("-", "m")
        return (
            f"dataset-{self.dataset_name}_"
            f"y-{n_y}_u-{n_u}_"
            f"H-{self.history}_N-{self.horizon}_s-{self.stride}_"
            f"bs-{self.batch_size}_ep-{self.train_epochs}_"
            f"lr-{lr_tag}_l1-{self.l1_lambda:.0e}_l2-{self.l2_lambda:.0e}_"
            f"ridge-{self.ridge:.0e}_seed-{self.seed}"
        )

    def run(self) -> dict:
        rollouts = self.load_rollouts_from_npz()
        train_rollouts, test_rollouts, train_idx, test_idx = self.split_rollouts(rollouts)

        n_y = int(rollouts[0]["y"].shape[1])
        n_u = int(rollouts[0]["u"].shape[1])

        z_scale, y_scale = self.compute_scale_normalization(
            rollouts=train_rollouts,
            n_y=n_y,
            n_u=n_u,
        )

        replay_rollouts = self.load_or_create_replay_rollouts(rollouts)
        replay = ReplayBuffer(max_size=max(1, len(replay_rollouts)), seed=self.seed)
        replay.update_buffer(replay_rollouts)

        if self.fit_mode == "ls":
            phi, p_dyn, phi_norm, total_updates = self.fit_phi_ls(
                train_rollouts=train_rollouts,
                n_y=n_y,
                n_u=n_u,
                z_scale=z_scale,
                y_scale=y_scale,
            )
        else:
            train_windows = self.build_window_indices(
                replay=replay,
                selected_rollout_indices=train_idx,
            )
            phi, p_dyn, phi_norm, total_updates = self.fit_phi_sgd(
                replay=replay,
                train_windows=train_windows,
                n_y=n_y,
                n_u=n_u,
                z_scale=z_scale,
                y_scale=y_scale,
            )

        eval_rollouts = test_rollouts if len(test_rollouts) > 0 else train_rollouts
        n_eval_windows, overall_mse, per_step_mse = self.evaluate_phi_mse(
            rollouts=eval_rollouts,
            phi=phi,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        tag = self.make_run_tag(n_y=n_y, n_u=n_u)
        run_dir = self.output_dir / tag
        run_dir.mkdir(parents=True, exist_ok=True)

        phi_path = run_dir / "phi.npy"
        phi_norm_path = run_dir / "phi_normalized.npy"
        p_dyn_path = run_dir / "p_dyn.npy"
        config_path = run_dir / "run_config.json"

        np.save(phi_path, phi)
        np.save(phi_norm_path, phi_norm)
        np.save(p_dyn_path, p_dyn)

        run_config = {
            "fit_mode": self.fit_mode,
            "dataset_name": self.dataset_name,
            "rollout_dir": str(self.rollout_dir.resolve()),
            "replay_buffer_path": str(self.replay_buffer_path.resolve()),
            "history": self.history,
            "horizon": self.horizon,
            "stride": self.stride,
            "train_fraction": self.train_fraction,
            "split_seed": self.split_seed,
            "ridge": self.ridge,
            "normalization_mode": self.normalization_mode,
            "norm_eps": self.norm_eps,
            "batch_size": self.batch_size,
            "train_epochs": self.train_epochs,
            "learning_rate": self.learning_rate,
            "l1_lambda": self.l1_lambda,
            "l2_lambda": self.l2_lambda,
            "seed": self.seed,
            "total_optimizer_updates": int(total_updates),
            "n_rollouts_total": int(len(rollouts)),
            "n_rollouts_train": int(len(train_rollouts)),
            "n_rollouts_test": int(len(test_rollouts)),
            "train_rollout_indices": train_idx.astype(int).tolist(),
            "test_rollout_indices": test_idx.astype(int).tolist(),
            "n_y": n_y,
            "n_u": n_u,
            "phi_shape": [int(x) for x in phi.shape],
            "predictor_form": "Y = Phi z (bias-free)",
            "num_eval_windows": int(n_eval_windows),
            "overall_mse": float(overall_mse),
            "per_step_mse": per_step_mse,
            "stacking": {
                "z": "[u_{t-H}...u_{t-1}, y_{t-H}...y_t, u_t...u_{t+N-1}]",
                "Y": "[y_{t+1}...y_{t+N}]",
            },
            "saved_phi": str(phi_path.resolve()),
            "saved_phi_normalized": str(phi_norm_path.resolve()),
            "saved_p_dyn": str(p_dyn_path.resolve()),
            "saved_run_dir": str(run_dir.resolve()),
            "normalization_scales": {
                "z_scale": z_scale.tolist(),
                "y_scale": y_scale.tolist(),
            },
        }

        with config_path.open("w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2, sort_keys=True)

        print(f"[Linear_Predictor] mode={self.fit_mode}")
        print(f"[Linear_Predictor] eval_windows={n_eval_windows} overall_mse={overall_mse:.10f}")
        print(f"[Linear_Predictor] per_step_mse={per_step_mse}")
        print(f"[Linear_Predictor] saved run folder: {run_dir}")
        print(f"[Linear_Predictor] saved phi: {phi_path}")
        print(f"[Linear_Predictor] saved p_dyn: {p_dyn_path}")
        print(f"[Linear_Predictor] saved config: {config_path}")
        return run_config


def main() -> None:
    trainer = Linear_Predictor()
    trainer.run()


if __name__ == "__main__":
    main()
