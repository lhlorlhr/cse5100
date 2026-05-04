#!/usr/bin/env python
"""Collect per-step trajectories for post-hoc  emergence analysis.

This script intentionally does not modify eval_cutoff_intervention.py. It reuses
the policy-loading and intervention helpers from that script, then saves the
state needed by onpolicy/scripts/analysis/_analysis.py.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from onpolicy.config import get_config
from onpolicy.scripts.eval.eval_cutoff_intervention import (
    INTERVENTION_CHOICES,
    apply_comm_intervention,
    build_actions_env,
    build_communicator_metadata,
    build_policy,
    collect_message_buffer,
    configure_algorithm_flags,
    is_collision,
    load_policy_from_dir,
    make_env,
    maybe_set_proctitle,
    reset_predator_rnn_states,
)


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default="simple_tag")
    parser.add_argument("--num_landmarks", type=int, default=2)
    parser.add_argument("--num_good_agents", type=int, default=1)
    parser.add_argument("--num_adversaries", type=int, default=3)
    parser.add_argument(
        "--fixed_opponent_policy",
        type=str,
        default="heuristic",
        choices=["random", "heuristic", "all"],
    )
    parser.add_argument(
        "--interventions",
        nargs="+",
        default=["vanilla", "zero"],
        choices=INTERVENTION_CHOICES,
    )
    parser.add_argument(
        "--random_buffer_steps",
        type=int,
        default=2000,
        help="Vanilla rollout steps used only when --interventions includes random.",
    )
    parser.add_argument("--noise_scale", type=float, default=0.5)
    parser.add_argument(
        "--trajectory_path",
        type=str,
        default="",
        help="Defaults to <model_parent>/_analysis/_trajectories.pkl.",
    )
    return parser.parse_known_args(args)[0]


def resolve_output_path(all_args):
    if all_args.trajectory_path:
        trajectory_path = Path(all_args.trajectory_path).expanduser().resolve()
    else:
        trajectory_path = (
            Path(all_args.model_dir).resolve().parent
            / "_analysis"
            / "_trajectories.pkl"
        )
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    return trajectory_path


def extract_state(env, predator_ids, prey_ids):
    wolves = [env.world.agents[agent_id] for agent_id in predator_ids]
    sheep = [env.world.agents[agent_id] for agent_id in prey_ids]
    return {
        "wolf_positions": np.asarray([agent.state.p_pos.copy() for agent in wolves], dtype=np.float32),
        "wolf_velocities": np.asarray([agent.state.p_vel.copy() for agent in wolves], dtype=np.float32),
        "sheep_positions": np.asarray([agent.state.p_pos.copy() for agent in sheep], dtype=np.float32),
        "sheep_velocities": np.asarray([agent.state.p_vel.copy() for agent in sheep], dtype=np.float32),
    }


def extract_messages(actions_env, predator_ids, communicator_metadata):
    messages = []
    for agent_id in predator_ids:
        if agent_id not in communicator_metadata:
            messages.append(np.zeros((0,), dtype=np.float32))
            continue
        start, end = communicator_metadata[agent_id]["slice"]
        messages.append(np.asarray(actions_env[agent_id][start:end], dtype=np.float32).copy())
    return np.asarray(messages, dtype=np.float32)


def any_predator_captures_prey(env):
    for predator in [agent for agent in env.world.agents if agent.adversary]:
        for prey in [agent for agent in env.world.agents if not agent.adversary]:
            if is_collision(predator, prey):
                return True
    return False


def evaluate_episode_with_trace(
    env,
    all_args,
    predator_policy_map,
    fixed_opponent_policy,
    intervention_type,
    communicator_metadata,
    message_buffer,
    episode_idx,
    episode_seed,
):
    rng = np.random.default_rng(episode_seed)
    np.random.seed(episode_seed)
    torch.manual_seed(episode_seed)

    env.seed(episode_seed)
    env.reset()

    predator_ids = [i for i, agent in enumerate(env.world.agents) if agent.adversary]
    prey_ids = [i for i, agent in enumerate(env.world.agents) if not agent.adversary]
    predator_rnn_states, predator_masks = reset_predator_rnn_states(predator_ids, all_args)

    records = []
    episode_rewards = np.zeros(len(env.world.agents), dtype=np.float32)
    capture_step = None

    for step in range(all_args.episode_length):
        pre_state = extract_state(env, predator_ids, prey_ids)

        policy_actions_env = build_actions_env(
            env,
            all_args,
            predator_policy_map,
            fixed_opponent_policy,
            rng,
            predator_rnn_states,
            predator_masks,
        )
        policy_messages = extract_messages(policy_actions_env, predator_ids, communicator_metadata)

        applied_actions_env = apply_comm_intervention(
            policy_actions_env,
            communicator_metadata,
            intervention_type,
            rng,
            message_buffer=message_buffer,
            noise_scale=all_args.noise_scale,
        )
        applied_messages = extract_messages(applied_actions_env, predator_ids, communicator_metadata)

        _, rewards, _, _ = env.step(applied_actions_env)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        episode_rewards += rewards

        post_state = extract_state(env, predator_ids, prey_ids)
        captured_this_step = any_predator_captures_prey(env)
        if captured_this_step and capture_step is None:
            capture_step = step + 1

        records.append(
            {
                "episode": int(episode_idx),
                "episode_seed": int(episode_seed),
                "step": int(step),
                "opponent_policy": fixed_opponent_policy,
                "intervention": intervention_type,
                "predator_ids": list(predator_ids),
                "prey_ids": list(prey_ids),
                "pre_wolf_positions": pre_state["wolf_positions"],
                "pre_wolf_velocities": pre_state["wolf_velocities"],
                "pre_sheep_positions": pre_state["sheep_positions"],
                "pre_sheep_velocities": pre_state["sheep_velocities"],
                "post_wolf_positions": post_state["wolf_positions"],
                "post_wolf_velocities": post_state["wolf_velocities"],
                "post_sheep_positions": post_state["sheep_positions"],
                "post_sheep_velocities": post_state["sheep_velocities"],
                "policy_messages": policy_messages,
                "applied_messages": applied_messages,
                "rewards": rewards,
                "captured_this_step": bool(captured_this_step),
            }
        )

    if capture_step is None:
        capture_step = all_args.episode_length + 1

    return records, {
        "captured": bool(capture_step <= all_args.episode_length),
        "capture_step": int(capture_step),
        "predator_return": float(np.mean(episode_rewards[predator_ids])) if predator_ids else 0.0,
        "prey_return": float(np.mean(episode_rewards[prey_ids])) if prey_ids else 0.0,
    }


def build_predator_policy_map(env, all_args, predator_ids, device):
    if all_args.share_policy:
        share_observation_space = (
            env.share_observation_space[0]
            if all_args.use_centralized_V
            else env.observation_space[0]
        )
        shared_policy = build_policy(
            all_args,
            env.observation_space[0],
            share_observation_space,
            env.action_space[0],
            device,
        )
        load_policy_from_dir(shared_policy, all_args.model_dir)
        return {agent_id: shared_policy for agent_id in predator_ids}

    predator_policy_map = {}
    for agent_id in predator_ids:
        share_observation_space = (
            env.share_observation_space[agent_id]
            if all_args.use_centralized_V
            else env.observation_space[agent_id]
        )
        policy = build_policy(
            all_args,
            env.observation_space[agent_id],
            share_observation_space,
            env.action_space[agent_id],
            device,
        )
        load_policy_from_dir(policy, all_args.model_dir, separated_agent_id=agent_id)
        predator_policy_map[agent_id] = policy
    return predator_policy_map


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    configure_algorithm_flags(all_args)

    assert all_args.model_dir, "set model_dir first"
    assert all_args.env_name == "MPE", "This collector only supports MPE."
    assert all_args.scenario_name == "simple_tag", "This collector is intended for simple_tag."

    if all_args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)
    maybe_set_proctitle(all_args)

    env = make_env(all_args)
    predator_ids, communicator_metadata = build_communicator_metadata(env)
    if not communicator_metadata:
        print("No communicating predator agents found; collecting geometry-only vanilla trajectories.")
        non_vanilla = [name for name in all_args.interventions if name != "vanilla"]
        if non_vanilla:
            print(
                "Ignoring communication interventions for no-comm model: "
                + ", ".join(non_vanilla)
            )
            all_args.interventions = ["vanilla"]

    predator_policy_map = build_predator_policy_map(env, all_args, predator_ids, device)
    opponent_policies = (
        ["random", "heuristic"]
        if all_args.fixed_opponent_policy == "all"
        else [all_args.fixed_opponent_policy]
    )

    all_records = []
    summaries = {}

    for opponent_policy in opponent_policies:
        message_buffer = None
        if "random" in all_args.interventions:
            message_buffer = collect_message_buffer(
                env,
                all_args,
                predator_policy_map,
                opponent_policy,
                communicator_metadata,
                all_args.random_buffer_steps,
                all_args.seed + 100000,
            )

        summaries[opponent_policy] = {}
        for intervention in all_args.interventions:
            print(f"Collecting opponent={opponent_policy}, intervention={intervention}")
            episode_summaries = []
            for episode_idx in range(all_args.eval_episodes):
                episode_seed = all_args.seed + episode_idx
                records, episode_summary = evaluate_episode_with_trace(
                    env,
                    all_args,
                    predator_policy_map,
                    opponent_policy,
                    intervention,
                    communicator_metadata,
                    message_buffer,
                    episode_idx,
                    episode_seed,
                )
                all_records.extend(records)
                episode_summaries.append(episode_summary)

            summaries[opponent_policy][intervention] = {
                "n_episodes": int(len(episode_summaries)),
                "capture_rate": float(np.mean([s["captured"] for s in episode_summaries])),
                "mean_capture_step": float(np.mean([s["capture_step"] for s in episode_summaries])),
                "mean_predator_return": float(np.mean([s["predator_return"] for s in episode_summaries])),
                "mean_prey_return": float(np.mean([s["prey_return"] for s in episode_summaries])),
            }

    trajectory_path = resolve_output_path(all_args)
    payload = {
        "metadata": {
            "model_dir": str(Path(all_args.model_dir).resolve()),
            "seed": int(all_args.seed),
            "episode_length": int(all_args.episode_length),
            "eval_episodes": int(all_args.eval_episodes),
            "fixed_opponent_policy": all_args.fixed_opponent_policy,
            "interventions": list(all_args.interventions),
            "use_simple_comm": bool(all_args.use_simple_comm),
            "comm_dim": int(getattr(env.world, "dim_c", 0)),
            "comm_target": all_args.comm_target,
            "has_communication": bool(communicator_metadata),
            "predator_ids": list(predator_ids),
            "communicator_metadata": communicator_metadata,
            "summaries": summaries,
        },
        "records": all_records,
    }

    with open(trajectory_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    summary_path = trajectory_path.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload["metadata"], f, indent=2, ensure_ascii=False)

    print(f"Saved {len(all_records)} step records to: {trajectory_path}")
    print(f"Saved metadata summary to: {summary_path}")
    env.close()


if __name__ == "__main__":
    main(sys.argv[1:])
