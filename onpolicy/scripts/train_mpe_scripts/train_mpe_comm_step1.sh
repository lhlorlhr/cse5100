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
SEEDS=(${SEEDS:-1 2 3})
LAMBDA_VALUES=(${LAMBDA_VALUES:-0 0.5 0.71 1.0 1.41 2.0 10.0})

NUM_ENV_STEPS="${NUM_ENV_STEPS:-10000000}"
EPISODE_LENGTH="${EPISODE_LENGTH:-100}"
N_ROLLOUT_THREADS="${N_ROLLOUT_THREADS:-32}"
N_EVAL_ROLLOUT_THREADS="${N_EVAL_ROLLOUT_THREADS:-10}"
EVAL_EPISODES="${EVAL_EPISODES:-32}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
RUN_FIXED_OPP_EVAL="${RUN_FIXED_OPP_EVAL:-1}"
RUN_CUTOFF_EVAL="${RUN_CUTOFF_EVAL:-1}"
FINAL_EVAL_EPISODES="${FINAL_EVAL_EPISODES:-100}"
RANDOM_BUFFER_STEPS="${RANDOM_BUFFER_STEPS:-2000}"
NOISE_SCALE="${NOISE_SCALE:-0.5}"

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

find_latest_run_dir() {
    local exp_root="$1"
    python - "$exp_root" <<'PY'
from pathlib import Path
import sys

exp_root = Path(sys.argv[1])
run_dirs = [p for p in exp_root.iterdir() if p.is_dir() and p.name.startswith("run")]
if not run_dirs:
    raise SystemExit("")

def run_num(path: Path) -> int:
    try:
        return int(path.name.replace("run", "", 1))
    except ValueError:
        return -1

print(sorted(run_dirs, key=run_num)[-1])
PY
}

run_final_evals() {
    local exp_name="$1"
    local seed="$2"
    local latest_run_dir="$3"
    local model_dir="${latest_run_dir}/models"

    if [[ ! -d "${model_dir}" ]]; then
        echo "skip final eval: missing model_dir=${model_dir}"
        return
    fi

    if [[ "${RUN_FIXED_OPP_EVAL}" == "1" ]]; then
        echo "running fixed-opponent eval for exp=${exp_name}, seed=${seed}"
        CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" -m onpolicy.scripts.eval.eval_fixed_opponent_mpe \
          --env_name "${ENV_NAME}" \
          --scenario_name "${SCENARIO_NAME}" \
          --algorithm_name "${ALGORITHM_NAME}" \
          --experiment_name "${exp_name}" \
          --user_name "${USER_NAME}" \
          --model_dir "${model_dir}" \
          --num_good_agents "${NUM_GOOD_AGENTS}" \
          --num_adversaries "${NUM_ADVERSARIES}" \
          --num_landmarks "${NUM_LANDMARKS}" \
          --episode_length "${EPISODE_LENGTH}" \
          --eval_episodes "${FINAL_EVAL_EPISODES}" \
          --seed "${seed}" \
          --use_simple_comm \
          --comm_dim "${COMM_DIM}" \
          --comm_target adversaries \
          --use_comm_l1_penalty \
          --comm_l1_coef 0.0 \
          --fixed_opponent_policy all \
          --share_policy
    fi

    if [[ "${RUN_CUTOFF_EVAL}" == "1" ]]; then
        echo "running cutoff-intervention eval for exp=${exp_name}, seed=${seed}"
        CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" -m onpolicy.scripts.eval.eval_cutoff_intervention \
          --env_name "${ENV_NAME}" \
          --scenario_name "${SCENARIO_NAME}" \
          --algorithm_name "${ALGORITHM_NAME}" \
          --experiment_name "${exp_name}" \
          --user_name "${USER_NAME}" \
          --model_dir "${model_dir}" \
          --num_good_agents "${NUM_GOOD_AGENTS}" \
          --num_adversaries "${NUM_ADVERSARIES}" \
          --num_landmarks "${NUM_LANDMARKS}" \
          --episode_length "${EPISODE_LENGTH}" \
          --eval_episodes "${FINAL_EVAL_EPISODES}" \
          --seed "${seed}" \
          --use_simple_comm \
          --comm_dim "${COMM_DIM}" \
          --comm_target adversaries \
          --use_comm_l1_penalty \
          --comm_l1_coef 0.0 \
          --fixed_opponent_policy all \
          --random_buffer_steps "${RANDOM_BUFFER_STEPS}" \
          --noise_scale "${NOISE_SCALE}" \
          --share_policy
    fi
}

echo "root_dir: ${ROOT_DIR}"
echo "python_bin: ${PYTHON_BIN}"
echo "env: ${ENV_NAME}, scenario: ${SCENARIO_NAME}, algo: ${ALGORITHM_NAME}"
echo "comm_dim: ${COMM_DIM}"
echo "lambda_values: ${LAMBDA_VALUES[*]}"
echo "seeds: ${SEEDS[*]}"
echo "num_env_steps: ${NUM_ENV_STEPS}"
echo "run_fixed_opp_eval: ${RUN_FIXED_OPP_EVAL}"
echo "run_cutoff_eval: ${RUN_CUTOFF_EVAL}"
echo "final_eval_episodes: ${FINAL_EVAL_EPISODES}"

check_tensorboard_writer

for lambda_value in "${LAMBDA_VALUES[@]}"; do
    sanitized_lambda="${lambda_value//./p}"
    exp_name="step1_full_comm${COMM_DIM}_lambda${sanitized_lambda}"

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

        exp_root="onpolicy/scripts/results/${ENV_NAME}/${SCENARIO_NAME}/${ALGORITHM_NAME}/${exp_name}"
        latest_run_dir="$(find_latest_run_dir "${exp_root}")"
        if [[ -z "${latest_run_dir}" ]]; then
            echo "warning: could not resolve latest run dir for ${exp_name}; skipping final evals"
            continue
        fi

        run_final_evals "${exp_name}" "${seed}" "${latest_run_dir}"
    done
done
