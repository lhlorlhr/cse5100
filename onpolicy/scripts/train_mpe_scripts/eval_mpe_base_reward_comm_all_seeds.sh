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

ENV_NAME="${ENV_NAME:-MPE}"
SCENARIO_NAME="${SCENARIO_NAME:-simple_tag}"
ALGORITHM_NAME="${ALGORITHM_NAME:-mappo}"
USER_NAME="${USER_NAME:-kristin}"

NUM_AGENTS="${NUM_AGENTS:-4}"
NUM_GOOD_AGENTS="${NUM_GOOD_AGENTS:-1}"
NUM_ADVERSARIES="${NUM_ADVERSARIES:-3}"
NUM_LANDMARKS="${NUM_LANDMARKS:-2}"

COMM_DIMS=(${COMM_DIMS:-0 2 4 8 16 32 64})
SEEDS=(${SEEDS:-1 2 3})

EPISODE_LENGTH="${EPISODE_LENGTH:-100}"
FINAL_EVAL_EPISODES="${FINAL_EVAL_EPISODES:-200}"
RANDOM_BUFFER_STEPS="${RANDOM_BUFFER_STEPS:-2000}"
NOISE_SCALE="${NOISE_SCALE:-0.5}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

RUN_FIXED_OPP_EVAL="${RUN_FIXED_OPP_EVAL:-1}"
RUN_CUTOFF_EVAL="${RUN_CUTOFF_EVAL:-1}"
SKIP_MISSING="${SKIP_MISSING:-0}"
DRY_RUN="${DRY_RUN:-0}"

RESULT_ROOT="${RESULT_ROOT:-onpolicy/scripts/results/${ENV_NAME}/${SCENARIO_NAME}/${ALGORITHM_NAME}}"

run_cmd() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'CUDA_VISIBLE_DEVICES=%q' "${CUDA_DEVICE}"
        printf ' %q' "$@"
        printf '\n'
    else
        CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "$@"
    fi
}

check_python_env() {
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
PY
}

eval_fixed_opponent() {
    local exp_name="$1"
    local seed="$2"
    local comm_dim="$3"
    local model_dir="$4"

    local cmd=(
      "${PYTHON_BIN}" -m onpolicy.scripts.eval.eval_fixed_opponent_mpe
      --env_name "${ENV_NAME}"
      --scenario_name "${SCENARIO_NAME}"
      --algorithm_name "${ALGORITHM_NAME}"
      --experiment_name "${exp_name}"
      --user_name "${USER_NAME}"
      --model_dir "${model_dir}"
      --num_agents "${NUM_AGENTS}"
      --num_good_agents "${NUM_GOOD_AGENTS}"
      --num_adversaries "${NUM_ADVERSARIES}"
      --num_landmarks "${NUM_LANDMARKS}"
      --episode_length "${EPISODE_LENGTH}"
      --eval_episodes "${FINAL_EVAL_EPISODES}"
      --seed "${seed}"
      --fixed_opponent_policy all
      --share_policy
    )

    if [[ "${comm_dim}" -gt 0 ]]; then
        cmd+=(
          --use_simple_comm
          --comm_dim "${comm_dim}"
          --comm_target adversaries
        )
    fi

    run_cmd "${cmd[@]}"
}

eval_cutoff_intervention() {
    local exp_name="$1"
    local seed="$2"
    local comm_dim="$3"
    local model_dir="$4"

    if [[ "${comm_dim}" -le 0 ]]; then
        echo "skip cutoff-intervention eval for exp=${exp_name}, seed=${seed}: comm_dim=${comm_dim}"
        return
    fi

    run_cmd \
      "${PYTHON_BIN}" -m onpolicy.scripts.eval.eval_cutoff_intervention \
      --env_name "${ENV_NAME}" \
      --scenario_name "${SCENARIO_NAME}" \
      --algorithm_name "${ALGORITHM_NAME}" \
      --experiment_name "${exp_name}" \
      --user_name "${USER_NAME}" \
      --model_dir "${model_dir}" \
      --num_agents "${NUM_AGENTS}" \
      --num_good_agents "${NUM_GOOD_AGENTS}" \
      --num_adversaries "${NUM_ADVERSARIES}" \
      --num_landmarks "${NUM_LANDMARKS}" \
      --episode_length "${EPISODE_LENGTH}" \
      --eval_episodes "${FINAL_EVAL_EPISODES}" \
      --seed "${seed}" \
      --use_simple_comm \
      --comm_dim "${comm_dim}" \
      --comm_target adversaries \
      --fixed_opponent_policy all \
      --interventions vanilla zero random permuted noise \
      --random_buffer_steps "${RANDOM_BUFFER_STEPS}" \
      --noise_scale "${NOISE_SCALE}" \
      --share_policy
}

echo "root_dir: ${ROOT_DIR}"
echo "python_bin: ${PYTHON_BIN}"
echo "result_root: ${RESULT_ROOT}"
echo "comm_dims: ${COMM_DIMS[*]}"
echo "seeds: ${SEEDS[*]}"
echo "eval_episodes: ${FINAL_EVAL_EPISODES}"
echo "run_fixed_opp_eval: ${RUN_FIXED_OPP_EVAL}"
echo "run_cutoff_eval: ${RUN_CUTOFF_EVAL}"
echo "dry_run: ${DRY_RUN}"
echo "note: comm_dim=0 maps to base_reward_nocomm and runs fixed-opponent eval only"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "skip python env preflight in DRY_RUN mode"
else
    check_python_env
fi

for comm_dim in "${COMM_DIMS[@]}"; do
    if [[ "${comm_dim}" -eq 0 ]]; then
        exp_name="base_reward_nocomm"
    else
        exp_name="base_reward_comm${comm_dim}"
    fi

    for seed in "${SEEDS[@]}"; do
        run_name="run${seed}"
        model_dir="${RESULT_ROOT}/${exp_name}/${run_name}/models"

        if [[ ! -d "${model_dir}" ]]; then
            message="missing model_dir=${model_dir}"
            if [[ "${SKIP_MISSING}" == "1" ]]; then
                echo "skip: ${message}"
                continue
            fi
            echo "error: ${message}" >&2
            exit 1
        fi

        echo
        echo "evaluating exp=${exp_name}, seed=${seed}, run=${run_name}, comm_dim=${comm_dim}"

        if [[ "${RUN_FIXED_OPP_EVAL}" == "1" ]]; then
            echo "running fixed-opponent eval..."
            eval_fixed_opponent "${exp_name}" "${seed}" "${comm_dim}" "${model_dir}"
        fi

        if [[ "${RUN_CUTOFF_EVAL}" == "1" ]]; then
            echo "running cutoff-intervention eval..."
            eval_cutoff_intervention "${exp_name}" "${seed}" "${comm_dim}" "${model_dir}"
        fi
    done
done

echo
echo "Done."
