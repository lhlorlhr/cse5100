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
USER_NAME="${USER_NAME:-kristin}"

NUM_AGENTS=4
NUM_GOOD_AGENTS=1
NUM_ADVERSARIES=3
NUM_LANDMARKS=2

COMM_DIM="${COMM_DIM:-8}"
SEEDS=(${SEEDS:-1})
LAMBDA_VALUES=(${LAMBDA_VALUES:-0 10})

NUM_ENV_STEPS="${NUM_ENV_STEPS:-500000}"
EPISODE_LENGTH="${EPISODE_LENGTH:-100}"
N_ROLLOUT_THREADS="${N_ROLLOUT_THREADS:-32}"
N_EVAL_ROLLOUT_THREADS="${N_EVAL_ROLLOUT_THREADS:-10}"
EVAL_EPISODES="${EVAL_EPISODES:-16}"
EVAL_INTERVAL="${EVAL_INTERVAL:-25}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

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
echo "comm_dim: ${COMM_DIM}"
echo "lambda_values: ${LAMBDA_VALUES[*]}"
echo "seeds: ${SEEDS[*]}"
echo "num_env_steps: ${NUM_ENV_STEPS}"

check_tensorboard_writer

for lambda_value in "${LAMBDA_VALUES[@]}"; do
    exp_name="step0_comm${COMM_DIM}_lambda${lambda_value}"

    for seed in "${SEEDS[@]}"; do
        echo "running exp=${exp_name}, seed=${seed}, lambda=${lambda_value}, comm_dim=${COMM_DIM}"

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
          --n_rollout_threads "${N_ROLLOUT_THREADS}"
          --episode_length "${EPISODE_LENGTH}"
          --num_env_steps "${NUM_ENV_STEPS}"
          --ppo_epoch 10
          --num_mini_batch 1
          --lr 7e-4
          --critic_lr 7e-4
          --gain 0.01
          --use_eval
          --n_eval_rollout_threads "${N_EVAL_ROLLOUT_THREADS}"
          --eval_episodes "${EVAL_EPISODES}"
          --eval_interval "${EVAL_INTERVAL}"
          --use_simple_comm
          --comm_dim "${COMM_DIM}"
          --comm_target adversaries
          --use_comm_l1_penalty
          --comm_l1_coef "${lambda_value}"
        )

        CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${cmd[@]}"
    done
done
