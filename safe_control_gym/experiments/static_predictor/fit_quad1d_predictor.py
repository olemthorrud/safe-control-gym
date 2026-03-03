from pathlib import Path

try:
    from .fit_static_phi_ddpc import Linear_Predictor
except ImportError:
    from fit_static_phi_ddpc import Linear_Predictor

# ----------------------------
# Quad1d-specific settings
# ----------------------------
DATASET_NAME = "quad1d_linear_verify"
HISTORY  = 3    # T_ini (theoretical min = 2 for a 2-state/1-output system)
HORIZON  = 40   # N (same as quad2d)
FIT_MODE = "sgd" # "ls" (analytical least-squares) or "sgd" (Adam)


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    predictor = Linear_Predictor(fit_mode=FIT_MODE)

    # Patch dataset and dimension settings onto the existing instance.
    predictor.dataset_name = DATASET_NAME
    predictor.rollout_dir  = script_dir / "rollouts" / DATASET_NAME
    predictor.output_dir   = script_dir / "analytic_predictors_quad1d"
    predictor.replay_buffer_path = (
        script_dir / "static_phi_ddpc_stats"
        / f"replay_buffer_{DATASET_NAME}_H-{HISTORY}_N-{HORIZON}.pkl"
    )
    predictor.history = HISTORY
    predictor.horizon = HORIZON

    print(f"[fit_quad1d] dataset  : {predictor.rollout_dir}")
    print(f"[fit_quad1d] T_ini    : {HISTORY}")
    print(f"[fit_quad1d] N        : {HORIZON}")
    print(f"[fit_quad1d] ridge    : {predictor.ridge}")
    print(f"[fit_quad1d] fit_mode : {FIT_MODE}")
    print()

    result = predictor.run()

    mse = result["overall_mse"]
    print()
    print(f"[fit_quad1d] overall_mse = {mse:.2e}")
    if mse < 1e-6:
        print("[fit_quad1d] PASS — MSE is effectively zero (model class is correct).")
    else:
        print(f"[fit_quad1d] WARNING — MSE={mse:.2e} > 1e-6, "
              "check rollout collection or y-extraction.")


if __name__ == "__main__":
    main()
