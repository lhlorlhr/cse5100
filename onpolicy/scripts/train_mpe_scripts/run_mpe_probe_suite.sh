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

MODEL_DIR="${MODEL_DIR:-onpolicy/scripts/results/MPE/simple_tag/mappo/base_reward_comm8/run1/models}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(dirname "${MODEL_DIR}")/probing_eval}"

NUM_GOOD_AGENTS="${NUM_GOOD_AGENTS:-1}"
NUM_ADVERSARIES="${NUM_ADVERSARIES:-3}"
NUM_LANDMARKS="${NUM_LANDMARKS:-2}"
EPISODE_LENGTH="${EPISODE_LENGTH:-100}"
SEED="${SEED:-1}"
COMM_DIM="${COMM_DIM:-8}"
COMM_TARGET="${COMM_TARGET:-adversaries}"
FIXED_OPPONENT_POLICY="${FIXED_OPPONENT_POLICY:-heuristic}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

N_EPISODES="${N_EPISODES:-500}"
DETERMINISTIC="${DETERMINISTIC:-0}"
TRAIN_RATIO="${TRAIN_RATIO:-0.7}"
VAL_RATIO="${VAL_RATIO:-0.15}"
SPLIT_SEED="${SPLIT_SEED:-1}"
MARGINAL_SEED="${MARGINAL_SEED:-1}"

RUN_SANITY="${RUN_SANITY:-1}"
RUN_COLLECT="${RUN_COLLECT:-1}"
RUN_PROBES="${RUN_PROBES:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
RUN_INFO="${RUN_INFO:-1}"
RUN_PLOTS="${RUN_PLOTS:-1}"

RANDOM_POLICY_SEEDS=(${RANDOM_POLICY_SEEDS:-1 2 3})
PER_WOLF_TASKS=(${PER_WOLF_TASKS:-sheep_rel_xy quadrant is_self_closest})
JOINT_TASKS=(${JOINT_TASKS:-closest_wolf_id})

DATASET_PATH="${DATASET_PATH:-${OUTPUT_ROOT}/dataset/probe_data.npz}"
PROBE_ROOT="${PROBE_ROOT:-${OUTPUT_ROOT}/probes}"
INFO_DIR="${INFO_DIR:-${OUTPUT_ROOT}/info}"
FIGURE_DIR="${FIGURE_DIR:-${OUTPUT_ROOT}/figures}"

check_python_stack() {
    "${PYTHON_BIN}" - <<'PY'
import sys

print(f"python_executable: {sys.executable}")
print(f"python_version: {sys.version.split()[0]}")

required = ["numpy", "pandas", "torch", "sklearn", "matplotlib"]
for name in required:
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "unknown")
        print(f"{name}: OK ({version})")
    except Exception as exc:
        raise SystemExit(
            "Preflight failed: required Python package is unavailable.\n"
            f"package: {name}\n"
            f"reason: {exc!r}\n"
            "Set PYTHON_BIN to the environment that can import the full stack."
        )
PY
}

maybe_flag() {
    local enabled="$1"
    local flag="$2"
    if [[ "${enabled}" == "1" ]]; then
        printf '%s\n' "${flag}"
    fi
}

run_probe() {
    local task="$1"
    local variant="$2"
    local wolf_id="${3:-}"

    local cmd=(
      "${PYTHON_BIN}" -m onpolicy.scripts.eval.train_mpe_probe
      --dataset_path "${DATASET_PATH}"
      --task "${task}"
      --probe_type linear
      --message_variant "${variant}"
      --split_seed "${SPLIT_SEED}"
      --marginal_seed "${MARGINAL_SEED}"
      --train_ratio "${TRAIN_RATIO}"
      --val_ratio "${VAL_RATIO}"
      --output_dir "${OUTPUT_ROOT}"
    )

    if [[ -n "${wolf_id}" ]]; then
        cmd+=(--wolf_id "${wolf_id}")
    fi

    if [[ "${variant}" == "random" ]]; then
        local random_seed="$4"
        cmd+=(--random_policy_seed "${random_seed}")
    fi

    "${cmd[@]}"
}

echo "root_dir: ${ROOT_DIR}"
echo "python_bin: ${PYTHON_BIN}"
echo "env: ${ENV_NAME}, scenario: ${SCENARIO_NAME}, algo: ${ALGORITHM_NAME}"
echo "model_dir: ${MODEL_DIR}"
echo "output_root: ${OUTPUT_ROOT}"
echo "comm_dim: ${COMM_DIM}"
echo "random_policy_seeds: ${RANDOM_POLICY_SEEDS[*]}"
echo "per_wolf_tasks: ${PER_WOLF_TASKS[*]}"
echo "joint_tasks: ${JOINT_TASKS[*]}"
echo "run_sanity: ${RUN_SANITY}"
echo "run_collect: ${RUN_COLLECT}"
echo "run_probes: ${RUN_PROBES}"
echo "run_summary: ${RUN_SUMMARY}"
echo "run_info: ${RUN_INFO}"
echo "run_plots: ${RUN_PLOTS}"

check_python_stack

if [[ "${RUN_SANITY}" == "1" ]]; then
    echo "running communication sanity inspection"
    sanity_cmd=(
      "${PYTHON_BIN}" -m onpolicy.scripts.eval.inspect_mpe_comm
      --env_name "${ENV_NAME}"
      --scenario_name "${SCENARIO_NAME}"
      --algorithm_name "${ALGORITHM_NAME}"
      --model_dir "${MODEL_DIR}"
      --num_good_agents "${NUM_GOOD_AGENTS}"
      --num_adversaries "${NUM_ADVERSARIES}"
      --num_landmarks "${NUM_LANDMARKS}"
      --episode_length "${EPISODE_LENGTH}"
      --seed "${SEED}"
      --use_simple_comm
      --comm_dim "${COMM_DIM}"
      --comm_target "${COMM_TARGET}"
      --share_policy
    )
    "${sanity_cmd[@]}"
fi

if [[ "${RUN_COLLECT}" == "1" ]]; then
    echo "collecting probe dataset"
    collect_cmd=(
      "${PYTHON_BIN}" -m onpolicy.scripts.eval.collect_mpe_probe_dataset
      --env_name "${ENV_NAME}"
      --scenario_name "${SCENARIO_NAME}"
      --algorithm_name "${ALGORITHM_NAME}"
      --model_dir "${MODEL_DIR}"
      --num_good_agents "${NUM_GOOD_AGENTS}"
      --num_adversaries "${NUM_ADVERSARIES}"
      --num_landmarks "${NUM_LANDMARKS}"
      --episode_length "${EPISODE_LENGTH}"
      --n_episodes "${N_EPISODES}"
      --seed "${SEED}"
      --fixed_opponent_policy "${FIXED_OPPONENT_POLICY}"
      --use_simple_comm
      --comm_dim "${COMM_DIM}"
      --comm_target "${COMM_TARGET}"
      --share_policy
      --output_dir "${OUTPUT_ROOT}"
      --random_policy_seeds "${RANDOM_POLICY_SEEDS[@]}"
    )

    if [[ "${DETERMINISTIC}" == "1" ]]; then
        collect_cmd+=(--deterministic)
    fi

    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${collect_cmd[@]}"
fi

if [[ "${RUN_PROBES}" == "1" ]]; then
    echo "training per-wolf probes"
    for task in "${PER_WOLF_TASKS[@]}"; do
        for wolf_id in 0 1 2; do
            run_probe "${task}" trained "${wolf_id}"
            run_probe "${task}" marginal "${wolf_id}"
            for random_seed in "${RANDOM_POLICY_SEEDS[@]}"; do
                run_probe "${task}" random "${wolf_id}" "${random_seed}"
            done
        done
    done

    echo "training joint probes"
    for task in "${JOINT_TASKS[@]}"; do
        run_probe "${task}" trained
        run_probe "${task}" marginal
        for random_seed in "${RANDOM_POLICY_SEEDS[@]}"; do
            run_probe "${task}" random "" "${random_seed}"
        done
    done
fi

if [[ "${RUN_SUMMARY}" == "1" ]]; then
    echo "summarizing probe runs"
    "${PYTHON_BIN}" -m onpolicy.scripts.eval.summarize_mpe_probes \
      --probe_root "${PROBE_ROOT}" \
      --output_dir "${PROBE_ROOT}"
fi

if [[ "${RUN_INFO}" == "1" ]]; then
    echo "running information-theoretic analysis"
    "${PYTHON_BIN}" -m onpolicy.scripts.eval.analyze_mpe_probe_info \
      --dataset_path "${DATASET_PATH}" \
      --output_dir "${INFO_DIR}"
fi

if [[ "${RUN_PLOTS}" == "1" ]]; then
    echo "plotting figures"
    "${PYTHON_BIN}" -m onpolicy.scripts.eval.plot_mpe_probe_figures \
      --probe_summary "${PROBE_ROOT}/probe_summary.csv" \
      --dataset_path "${DATASET_PATH}" \
      --info_dir "${INFO_DIR}" \
      --output_dir "${FIGURE_DIR}"
fi

echo "probe suite complete"
