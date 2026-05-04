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

PARTIAL_OBS_RADIUS="${PARTIAL_OBS_RADIUS:-1.0}"
SEEDS=(${SEEDS:-1})
COMM_DIMS=(${COMM_DIMS:-0 2 8 32})

NUM_ENV_STEPS="${NUM_ENV_STEPS:-10000000}"
EPISODE_LENGTH="${EPISODE_LENGTH:-100}"
N_ROLLOUT_THREADS="${N_ROLLOUT_THREADS:-32}"
N_EVAL_ROLLOUT_THREADS="${N_EVAL_ROLLOUT_THREADS:-10}"
EVAL_EPISODES="${EVAL_EPISODES:-32}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
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

sanitize_float() {
    local value="$1"
    value="${value//./p}"
    value="${value//-/m}"
    echo "${value}"
}

echo "root_dir: ${ROOT_DIR}"
echo "python_bin: ${PYTHON_BIN}"
echo "env: ${ENV_NAME}, scenario: ${SCENARIO_NAME}, algo: ${ALGORITHM_NAME}"
echo "partial_obs_radius: ${PARTIAL_OBS_RADIUS}"
echo "comm_dims: ${COMM_DIMS[*]}"
echo "seeds: ${SEEDS[*]}"
echo "num_env_steps: ${NUM_ENV_STEPS}"

check_tensorboard_writer

radius_tag="$(sanitize_float "${PARTIAL_OBS_RADIUS}")"

for comm_dim in "${COMM_DIMS[@]}"; do
    if [[ "${comm_dim}" -eq 0 ]]; then
        exp_name="partial_obs_r${radius_tag}_nocomm"
    else
        exp_name="partial_obs_r${radius_tag}_comm${comm_dim}"
    fi

    for seed in "${SEEDS[@]}"; do
        echo "running exp=${exp_name}, seed=${seed}, comm_dim=${comm_dim}, partial_obs_radius=${PARTIAL_OBS_RADIUS}"

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
          --use_partial_obs
          --partial_obs_radius "${PARTIAL_OBS_RADIUS}"
        )

        if [[ "${comm_dim}" -gt 0 ]]; then
            cmd+=(
              --use_simple_comm
              --comm_dim "${comm_dim}"
              --comm_target adversaries
            )
        fi

        CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${cmd[@]}"
    done
done
