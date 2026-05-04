#!/usr/bin/env python
"""Collect probing datasets from trained simple_tag communication checkpoints.

This script fixes the state distribution by rolling out the trained predator
policy, then evaluates trained and random-init communicators on the same local
observations. The environment transition always follows the trained policy.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from onpolicy.config import get_config
from onpolicy.scripts.eval.eval_cutoff_intervention import (
    action_to_env_action,
    build_policy,
    configure_algorithm_flags,
    load_policy_from_dir,
    make_env,
    maybe_set_proctitle,
    prey_action_to_env_action,
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
        choices=["random", "heuristic"],
        help="Sheep policy used during trained rollouts.",
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=500,
        help="Number of rollout episodes to collect.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Use greedy actions instead of sampling for trained/random actors.",
    )
    parser.add_argument(
        "--random_policy_seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Random seeds used to instantiate untrained control actors.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Defaults to <model_parent>/probing_eval.",
    )
    return parser.parse_known_args(args)[0]


def resolve_output_dir(all_args):
    if all_args.output_dir:
        output_dir = Path(all_args.output_dir).expanduser().resolve()
    else:
        output_dir = Path(all_args.model_dir).resolve().parent / "probing_eval"
    dataset_dir = output_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, dataset_dir


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


def build_random_predator_policy_maps(env, all_args, predator_ids, device, seeds):
    random_policy_maps = {}
    if all_args.share_policy:
        share_observation_space = (
            env.share_observation_space[0]
            if all_args.use_centralized_V
            else env.observation_space[0]
        )
        for seed in seeds:
            torch.manual_seed(seed)
            shared_policy = build_policy(
                all_args,
                env.observation_space[0],
                share_observation_space,
                env.action_space[0],
                device,
            )
            shared_policy.actor.eval()
            shared_policy.critic.eval()
            random_policy_maps[seed] = {agent_id: shared_policy for agent_id in predator_ids}
        return random_policy_maps

    for seed in seeds:
        torch.manual_seed(seed)
        policy_map = {}
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
            policy.actor.eval()
            policy.critic.eval()
            policy_map[agent_id] = policy
        random_policy_maps[seed] = policy_map
    return random_policy_maps


def get_comm_slice(action_space):
    if action_space.__class__.__name__ != "MultiDiscrete":
        return None
    sizes = action_space.high - action_space.low + 1
    if len(sizes) < 2:
        return None
    move_size = int(sizes[0])
    comm_size = int(sizes[1])
    return move_size, move_size + comm_size


def reset_rnn_states(agent_ids, all_args):
    rnn_states = {}
    masks = {}
    for agent_id in agent_ids:
        rnn_states[agent_id] = np.zeros(
            (1, all_args.recurrent_N, all_args.hidden_size), dtype=np.float32
        )
        masks[agent_id] = np.ones((1, 1), dtype=np.float32)
    return rnn_states, masks


def compute_angle_bin(rel_xy, num_bins):
    angle = np.arctan2(rel_xy[1], rel_xy[0])
    normalized = (angle + np.pi) / (2.0 * np.pi)
    return int(np.floor(normalized * num_bins)) % num_bins


def compute_distance_bin(distance):
    if distance < 0.5:
        return 0
    if distance < 1.0:
        return 1
    return 2


def compute_labels(env, predator_ids, prey_ids):
    sheep_agent = env.world.agents[prey_ids[0]]
    sheep_xy = np.asarray(sheep_agent.state.p_pos, dtype=np.float32).copy()
    sheep_vel = np.asarray(sheep_agent.state.p_vel, dtype=np.float32).copy()

    wolf_xy = []
    sheep_rel_xy = []
    quadrant = []
    octant = []
    distance = []
    distance_bin = []
    for agent_id in predator_ids:
        wolf_agent = env.world.agents[agent_id]
        wolf_pos = np.asarray(wolf_agent.state.p_pos, dtype=np.float32).copy()
        rel_xy = (sheep_xy - wolf_pos).astype(np.float32)
        dist = float(np.linalg.norm(rel_xy))

        wolf_xy.append(wolf_pos)
        sheep_rel_xy.append(rel_xy)
        quadrant.append(compute_angle_bin(rel_xy, 4))
        octant.append(compute_angle_bin(rel_xy, 8))
        distance.append(dist)
        distance_bin.append(compute_distance_bin(dist))

    distance = np.asarray(distance, dtype=np.float32)
    closest_wolf_id = int(np.argmin(distance))
    is_self_closest = np.zeros(len(predator_ids), dtype=np.int64)
    is_self_closest[closest_wolf_id] = 1

    return {
        "sheep_xy": sheep_xy,
        "sheep_vel": sheep_vel,
        "wolf_xy": np.asarray(wolf_xy, dtype=np.float32),
        "sheep_rel_xy": np.asarray(sheep_rel_xy, dtype=np.float32),
        "quadrant": np.asarray(quadrant, dtype=np.int64),
        "octant": np.asarray(octant, dtype=np.int64),
        "distance": distance,
        "distance_bin": np.asarray(distance_bin, dtype=np.int64),
        "is_self_closest": is_self_closest,
        "closest_wolf_id": closest_wolf_id,
    }


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    configure_algorithm_flags(all_args)

    assert all_args.model_dir, "set --model_dir first"
    assert all_args.env_name == "MPE", "This collector only supports MPE."
    assert all_args.scenario_name == "simple_tag", "This collector is intended for simple_tag."
    assert all_args.num_good_agents == 1, "Current collector assumes a single sheep."

    if all_args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    maybe_set_proctitle(all_args)

    env = make_env(all_args)
    output_dir, dataset_dir = resolve_output_dir(all_args)

    predator_ids = [i for i, agent in enumerate(env.world.agents) if agent.adversary]
    prey_ids = [i for i, agent in enumerate(env.world.agents) if not agent.adversary]
    comm_slice = get_comm_slice(env.action_space[predator_ids[0]])
    if comm_slice is None:
        raise RuntimeError("No communication branch found for predator agents.")
    comm_start, comm_end = comm_slice
    comm_dim_effective = comm_end - comm_start

    trained_policy_map = build_predator_policy_map(env, all_args, predator_ids, device)
    random_policy_maps = build_random_predator_policy_maps(
        env, all_args, predator_ids, device, all_args.random_policy_seeds
    )

    total_steps = all_args.n_episodes * all_args.episode_length
    wolf_obs = []
    sheep_xy = []
    sheep_vel = []
    wolf_xy = []
    sheep_rel_xy = []
    quadrant = []
    octant = []
    distance = []
    distance_bin = []
    is_self_closest = []
    closest_wolf_id = []
    episode_id = []
    timestep = []
    msg_trained = []
    msg_random = {seed: [] for seed in all_args.random_policy_seeds}

    step_counter = 0
    for episode_idx in range(all_args.n_episodes):
        episode_seed = all_args.seed + episode_idx
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        env.seed(episode_seed)
        obs = env.reset()

        trained_rnn_states, trained_masks = reset_rnn_states(predator_ids, all_args)
        random_rnn_states = {}
        random_masks = {}
        for seed in all_args.random_policy_seeds:
            random_rnn_states[seed], random_masks[seed] = reset_rnn_states(predator_ids, all_args)

        rng = np.random.default_rng(episode_seed)

        for step in range(all_args.episode_length):
            labels = compute_labels(env, predator_ids, prey_ids)
            actions_env = []
            trained_msgs_step = []
            random_msgs_step = {seed: [] for seed in all_args.random_policy_seeds}
            wolf_obs_step = []

            for agent_id, agent in enumerate(env.world.agents):
                action_space = env.action_space[agent_id]
                obs_i = np.asarray(obs[agent_id], dtype=np.float32)[None, :]

                if agent.adversary:
                    wolf_obs_step.append(obs_i[0].copy())

                    trained_policy = trained_policy_map[agent_id]
                    with torch.no_grad():
                        trained_action, next_rnn_state = trained_policy.act(
                            obs_i,
                            trained_rnn_states[agent_id],
                            trained_masks[agent_id],
                            deterministic=all_args.deterministic,
                        )
                    trained_rnn_states[agent_id] = next_rnn_state.detach().cpu().numpy()
                    trained_action = trained_action.detach().cpu().numpy()[0]
                    trained_env_action = action_to_env_action(trained_action, action_space)
                    trained_msgs_step.append(
                        np.asarray(trained_env_action[comm_start:comm_end], dtype=np.float32).copy()
                    )
                    actions_env.append(trained_env_action)

                    for seed in all_args.random_policy_seeds:
                        random_policy = random_policy_maps[seed][agent_id]
                        with torch.no_grad():
                            random_action, next_random_rnn = random_policy.act(
                                obs_i,
                                random_rnn_states[seed][agent_id],
                                random_masks[seed][agent_id],
                                deterministic=all_args.deterministic,
                            )
                        random_rnn_states[seed][agent_id] = next_random_rnn.detach().cpu().numpy()
                        random_action = random_action.detach().cpu().numpy()[0]
                        random_env_action = action_to_env_action(random_action, action_space)
                        random_msgs_step[seed].append(
                            np.asarray(random_env_action[comm_start:comm_end], dtype=np.float32).copy()
                        )
                else:
                    prey_action = prey_action_to_env_action(
                        agent,
                        env.world,
                        action_space,
                        all_args.fixed_opponent_policy,
                        rng,
                    )
                    actions_env.append(prey_action)

            wolf_obs.append(np.asarray(wolf_obs_step, dtype=np.float32))
            sheep_xy.append(labels["sheep_xy"])
            sheep_vel.append(labels["sheep_vel"])
            wolf_xy.append(labels["wolf_xy"])
            sheep_rel_xy.append(labels["sheep_rel_xy"])
            quadrant.append(labels["quadrant"])
            octant.append(labels["octant"])
            distance.append(labels["distance"])
            distance_bin.append(labels["distance_bin"])
            is_self_closest.append(labels["is_self_closest"])
            closest_wolf_id.append(labels["closest_wolf_id"])
            episode_id.append(episode_idx)
            timestep.append(step)
            msg_trained.append(np.asarray(trained_msgs_step, dtype=np.float32))
            for seed in all_args.random_policy_seeds:
                msg_random[seed].append(np.asarray(random_msgs_step[seed], dtype=np.float32))

            obs, _, _, _ = env.step(actions_env)
            step_counter += 1
            if step_counter % 5000 == 0:
                print(f"collected {step_counter}/{total_steps} timesteps")

    dataset_arrays = {
        "wolf_obs": np.asarray(wolf_obs, dtype=np.float32),
        "sheep_xy": np.asarray(sheep_xy, dtype=np.float32),
        "sheep_vel": np.asarray(sheep_vel, dtype=np.float32),
        "wolf_xy": np.asarray(wolf_xy, dtype=np.float32),
        "sheep_rel_xy": np.asarray(sheep_rel_xy, dtype=np.float32),
        "quadrant": np.asarray(quadrant, dtype=np.int64),
        "octant": np.asarray(octant, dtype=np.int64),
        "distance": np.asarray(distance, dtype=np.float32),
        "distance_bin": np.asarray(distance_bin, dtype=np.int64),
        "is_self_closest": np.asarray(is_self_closest, dtype=np.int64),
        "closest_wolf_id": np.asarray(closest_wolf_id, dtype=np.int64),
        "episode_id": np.asarray(episode_id, dtype=np.int64),
        "timestep": np.asarray(timestep, dtype=np.int64),
        "msg_trained": np.asarray(msg_trained, dtype=np.float32),
    }
    for seed in all_args.random_policy_seeds:
        dataset_arrays[f"msg_random_seed{seed}"] = np.asarray(msg_random[seed], dtype=np.float32)

    dataset_path = dataset_dir / "probe_data.npz"
    np.savez_compressed(dataset_path, **dataset_arrays)

    meta = {
        "model_dir": str(Path(all_args.model_dir).resolve()),
        "env_name": all_args.env_name,
        "scenario_name": all_args.scenario_name,
        "n_episodes": int(all_args.n_episodes),
        "episode_length": int(all_args.episode_length),
        "num_samples": int(total_steps),
        "predator_ids": predator_ids,
        "prey_ids": prey_ids,
        "comm_dim_effective": int(comm_dim_effective),
        "deterministic": bool(all_args.deterministic),
        "fixed_opponent_policy": all_args.fixed_opponent_policy,
        "random_policy_seeds": [int(seed) for seed in all_args.random_policy_seeds],
        "dataset_keys": sorted(dataset_arrays.keys()),
    }
    (dataset_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2))

    token_rows = []
    trained_tokens = np.argmax(dataset_arrays["msg_trained"], axis=-1)
    for wolf_idx in range(len(predator_ids)):
        counts = np.bincount(trained_tokens[:, wolf_idx], minlength=comm_dim_effective)
        probs = counts.astype(np.float64) / max(int(counts.sum()), 1)
        entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
        token_rows.append(
            {
                "wolf_id": wolf_idx,
                "entropy_bits": entropy,
                "counts": counts.tolist(),
                "probs": [float(x) for x in probs],
            }
        )
    (dataset_dir / "token_marginals.json").write_text(json.dumps(token_rows, indent=2))

    sanity_rows = []
    max_sanity_rows = min(20, total_steps)
    for idx in range(max_sanity_rows):
        row = {
            "episode_id": int(dataset_arrays["episode_id"][idx]),
            "timestep": int(dataset_arrays["timestep"][idx]),
            "closest_wolf_id": int(dataset_arrays["closest_wolf_id"][idx]),
        }
        for wolf_idx in range(len(predator_ids)):
            row[f"trained_token_wolf{wolf_idx}"] = int(trained_tokens[idx, wolf_idx])
            row[f"quadrant_wolf{wolf_idx}"] = int(dataset_arrays["quadrant"][idx, wolf_idx])
            row[f"is_self_closest_wolf{wolf_idx}"] = int(dataset_arrays["is_self_closest"][idx, wolf_idx])
        sanity_rows.append(row)
    pd.DataFrame(sanity_rows).to_csv(dataset_dir / "sanity_samples.csv", index=False)

    print(f"saved dataset to {dataset_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
