"""Static-Phi MPC controller solved as a single QP with cvxpy."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Sequence

import numpy as np
from termcolor import colored

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.envs.benchmark_env import Task

try:
    import cvxpy as cp
except ImportError:
    cp = None


class StaticPhiController(BaseController):
    """MPC using a static multi-step predictor Y = Phi * z + b.

    One QP is built at init and solved at every time step with updated parameters.
    """

    def __init__(
        self,
        env_func,
        phi_path: str,
        phi_bias_path: str | None = None,
        t_ini: int = 20,
        horizon: int = 40,
        y_indices: Sequence[int] = (0, 2, 4),
        Qy: Sequence[float] = (5.0, 5.0, 0.1),
        R: Sequence[float] = (0.1, 0.1),
        terminal_weight_scale: float = 10.0,
        theta_max: float = 0.3,
        theta_index_in_y: int = 2,
        solver_max_iter: int = 20000,
        **kwargs,
    ):
        super().__init__(env_func=env_func, **kwargs)
        if cp is None:
            raise ImportError("cvxpy is required but not installed.")

        self.env = env_func()
        self.t_ini = int(t_ini)
        self.horizon = int(horizon)
        self.y_indices = [int(i) for i in y_indices]
        self.theta_max = float(theta_max)
        self.theta_index_in_y = int(theta_index_in_y)
        self.terminal_weight_scale = float(terminal_weight_scale)
        self.solver_max_iter = int(solver_max_iter)

        self.n_u = int(np.asarray(self.env.action_space.shape).prod())
        self.n_y = len(self.y_indices)
        self.n_past = self.t_ini * self.n_u + (self.t_ini + 1) * self.n_y
        self.n_future_u = self.horizon * self.n_u
        self.n_target = self.horizon * self.n_y

        # Load predictor
        phi = np.load(phi_path).astype(np.float64)
        assert phi.shape == (self.n_target, self.n_past + self.n_future_u), (
            f"Phi shape mismatch: expected ({self.n_target}, {self.n_past + self.n_future_u}), got {phi.shape}"
        )
        self.phi_past = phi[:, : self.n_past]
        self.phi_future = phi[:, self.n_past :]
        self.b = (
            np.load(phi_bias_path).astype(np.float64).reshape(-1)
            if phi_bias_path is not None
            else np.zeros(self.n_target)
        )

        # Input bounds
        self.u_lo = np.asarray(self.env.physical_action_bounds[0], dtype=np.float64).reshape(-1)
        self.u_hi = np.asarray(self.env.physical_action_bounds[1], dtype=np.float64).reshape(-1)

        # Equilibrium
        if hasattr(self.env, "symbolic") and hasattr(self.env.symbolic, "U_EQ"):
            self.u_eq = np.asarray(self.env.symbolic.U_EQ, dtype=np.float64).reshape(-1)
        elif hasattr(self.env, "U_GOAL"):
            self.u_eq = np.asarray(self.env.U_GOAL, dtype=np.float64).reshape(-1)
        else:
            self.u_eq = 0.5 * (self.u_lo + self.u_hi)

        if hasattr(self.env, "X_GOAL"):
            x_goal = np.asarray(self.env.X_GOAL, dtype=np.float64)
            target = x_goal[0] if x_goal.ndim > 1 else x_goal
            self.y_eq = target[self.y_indices]
        else:
            self.y_eq = np.zeros(self.n_y, dtype=np.float64)

        # Cost weights – terminal step scaled up
        Qy_arr = np.asarray(Qy, dtype=np.float64)
        if Qy_arr.size == 1:
            Qy_arr = np.repeat(Qy_arr, self.n_y)
        R_arr = np.asarray(R, dtype=np.float64)
        if R_arr.size == 1:
            R_arr = np.repeat(R_arr, self.n_u)

        Qy_stack = np.tile(Qy_arr, self.horizon)
        Qy_stack[-self.n_y :] *= self.terminal_weight_scale
        self.sqrt_Qy = np.sqrt(np.maximum(Qy_stack, 0.0))
        self.sqrt_R = np.sqrt(np.maximum(np.tile(R_arr, self.horizon), 0.0))

        self._build_qp()
        self.reset()

    def _build_qp(self):
        self.z_past_p = cp.Parameter(self.n_past)
        self.y_ref_p = cp.Parameter(self.n_target)
        self.u_ref_p = cp.Parameter(self.n_future_u)
        self.u_var = cp.Variable(self.n_future_u)

        y_pred = self.phi_past @ self.z_past_p + self.phi_future @ self.u_var + self.b

        cost = cp.sum_squares(cp.multiply(self.sqrt_Qy, y_pred - self.y_ref_p)) + cp.sum_squares(
            cp.multiply(self.sqrt_R, self.u_var - self.u_ref_p)
        )

        theta_idx = [k * self.n_y + self.theta_index_in_y for k in range(self.horizon)]
        constraints = [
            self.u_var >= np.tile(self.u_lo, self.horizon),
            self.u_var <= np.tile(self.u_hi, self.horizon),
            y_pred[theta_idx] >= -self.theta_max,
            y_pred[theta_idx] <= self.theta_max,
        ]

        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    def reset(self):
        self.u_hist = None
        self.y_hist = None
        self.setup_results_dict()

    def reset_before_run(self, obs=None, info=None, env=None):
        if env is not None:
            self.env = env
        y0 = np.asarray(obs, dtype=np.float64).reshape(-1)[self.y_indices] if obs is not None else self.y_eq.copy()
        self.u_hist = deque([self.u_eq.copy()] * self.t_ini, maxlen=self.t_ini)
        self.y_hist = deque([y0.copy()] * (self.t_ini + 1), maxlen=self.t_ini + 1)
        super().reset_before_run(obs=obs, info=info, env=env)

    def select_action(self, obs, info=None):
        step = int(self.extract_step(info))
        y_now = np.asarray(obs, dtype=np.float64).reshape(-1)[self.y_indices]

        if self.y_hist is None:
            self.u_hist = deque([self.u_eq.copy()] * self.t_ini, maxlen=self.t_ini)
            self.y_hist = deque([y_now.copy()] * (self.t_ini + 1), maxlen=self.t_ini + 1)
        else:
            self.y_hist.append(y_now.copy())

        z_past = np.concatenate([
            np.stack(self.u_hist).reshape(-1),
            np.stack(self.y_hist).reshape(-1),
        ])
        y_ref = self._get_y_ref(step)
        u_ref = np.tile(self.u_eq, self.horizon)

        self.z_past_p.value = z_past
        self.y_ref_p.value = y_ref
        self.u_ref_p.value = u_ref

        try:
            self.problem.solve(solver=cp.OSQP, warm_start=True, max_iter=self.solver_max_iter)
            status = str(self.problem.status)
        except Exception as exc:
            print(colored(f"QP solve failed: {exc}", "red"))
            status = "exception"

        if status in ("optimal", "optimal_inaccurate") and self.u_var.value is not None:
            action = np.clip(self.u_var.value[: self.n_u], self.u_lo, self.u_hi)
        else:
            action = self.u_eq.copy()

        self.u_hist.append(action.copy())

        self.results_dict["action"].append(deepcopy(action))
        self.results_dict["y"].append(deepcopy(y_now))
        self.results_dict["y_ref"].append(deepcopy(y_ref[: self.n_y]))
        self.results_dict["qp_status"].append(status)

        return action

    def _get_y_ref(self, step: int) -> np.ndarray:
        if getattr(self.env, "TASK", None) == Task.TRAJ_TRACKING and hasattr(self.env, "X_GOAL"):
            x_goal = np.asarray(self.env.X_GOAL, dtype=np.float64)
            if x_goal.ndim == 2:
                idx = [min(step + 1 + k, x_goal.shape[0] - 1) for k in range(self.horizon)]
                return np.vstack([x_goal[i][self.y_indices] for i in idx]).reshape(-1)
        return np.tile(self.y_eq, self.horizon)

    def setup_results_dict(self):
        self.results_dict = {"action": [], "y": [], "y_ref": [], "qp_status": []}

    def close(self):
        self.env.close()
