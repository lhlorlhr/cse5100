#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="${PYTHON_BIN}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
    PYTHON_BIN="python"
fi

ENV_NAME="MPE"
SCENARIO_NAME="simple_tag"
ALGORITHM_NAME="mappo"
USER_NAME="kristin"

NUM_AGENTS=4
NUM_GOOD_AGENTS=1
NUM_ADVERSARIES=3
NUM_LANDMARKS=2

SEEDS=(1 2 3)
COMM_DIMS=(2 4 8 16 32 64)

check_tensorboard_writer() {
    "${PYTHON_BIN}" - <<'PY'
import sys

print(f"python_executable: {sys.executable}")
print(f"python_version: {sys.version.split()[0]}")

try:
    import torch
    print(f"torch: OK ({torch.__version__})")
except Exception as exc:
    raise SystemExit(
        "Preflight failed: cannot import torch with the selected Python interpreter.\n"
        f"Reason: {exc!r}"
    )

tensorboard_backend = None

try:
    import tensorboardX
    tensorboard_backend = f"tensorboardX ({tensorboardX.__version__})"
except Exception as tensorboardx_exc:
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        tensorboard_backend = "torch.utils.tensorboard"
    except Exception as torch_tb_exc:
        raise SystemExit(
            "Preflight failed: TensorBoard logging backend is unavailable.\n"
            "Training is configured to require TensorBoard logging.\n"
            f"tensorboardX import error: {tensorboardx_exc!r}\n"
            f"torch.utils.tensorboard import error: {torch_tb_exc!r}\n"
            "Fix by installing `tensorboard` or `tensorboardX` in this same Python environment,\n"
            "or run with PYTHON_BIN pointing to the environment that already has them."
        )

print(f"tensorboard_backend: {tensorboard_backend}")
PY
}

echo "root_dir: ${ROOT_DIR}"
echo "python_bin: ${PYTHON_BIN}"
echo "env: ${ENV_NAME}, scenario: ${SCENARIO_NAME}, algo: ${ALGORITHM_NAME}"
echo "seeds: ${SEEDS[*]}"
echo "comm_dims: ${COMM_DIMS[*]}"

check_tensorboard_writer

for comm_dim in "${COMM_DIMS[@]}"; do
    if [[ "${comm_dim}" -eq 0 ]]; then
        exp_name="base_reward_nocomm"
    else
        exp_name="base_reward_comm${comm_dim}"
    fi

    for seed in "${SEEDS[@]}"; do
        echo "running exp=${exp_name}, seed=${seed}, comm_dim=${comm_dim}"

        cmd=(
          "${PYTHON_BIN}" -m onpolicy.scripts.train.train_mpe
          --env_name "${ENV_NAME}"
          --scenario_name "${SCENARIO_NAME}"
          --algorithm_name "${ALGORITHM_NAME}"
          --experiment_name "${exp_name}"
          --user_name "${USER_NAME}"
          --num_agents "${NUM_AGENTS}"
          --num_good_agents "${NUM_GOOD_AGENTS}"
          --num_adversaries "${NUM_ADVERSARIES}"
          --num_landmarks "${NUM_LANDMARKS}"
          --share_policy
          --seed "${seed}"
          --n_training_threads 1
          --n_rollout_threads 32
          --episode_length 100
          --num_env_steps 10000000
          --ppo_epoch 10
          --num_mini_batch 1
          --lr 7e-4
          --critic_lr 7e-4
          --gain 0.01
          --use_eval
          --n_eval_rollout_threads 10
          --eval_episodes 32
          --eval_interval 50
        )

        if [[ "${comm_dim}" -gt 0 ]]; then
            cmd+=(
              --use_simple_comm
              --comm_dim "${comm_dim}"
              --comm_target adversaries
            )
        fi

        CUDA_VISIBLE_DEVICES=0 "${cmd[@]}"
    done
done
