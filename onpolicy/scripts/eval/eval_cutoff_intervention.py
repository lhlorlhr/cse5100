#!/usr/bin/env python
"""
Evaluate zero-shot communication interventions on trained simple_tag checkpoints.

Example:
python -m onpolicy.scripts.eval.eval_cutoff_intervention \
    --env_name MPE \
    --scenario_name simple_tag \
    --algorithm_name mappo \
    --model_dir onpolicy/scripts/results/MPE/simple_tag/mappo/base_reward_comm8/run1/models \
    --use_simple_comm \
    --comm_dim 8 \
    --comm_target adversaries \
    --episode_length 100 \
    --eval_episodes 200 \
    --output_csv cutoff_intervention_summary.csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    import setproctitle
except ImportError:
    setproctitle = None

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.config import get_config
from onpolicy.envs.mpe.MPE_env import MPEEnv


INTERVENTION_CHOICES = ["vanilla", "zero", "random", "permuted", "noise"]


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default="simple_tag",
                        help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=2)
    parser.add_argument("--num_good_agents", type=int, default=1,
                        help="number of good agents (for simple_tag)")
    parser.add_argument("--num_adversaries", type=int, default=3,
                        help="number of adversaries (for simple_tag)")
    parser.add_argument(
        "--fixed_opponent_policy",
        type=str,
        default="heuristic",
        choices=["random", "heuristic", "all"],
        help="Which fixed prey policy to evaluate against.",
    )
    parser.add_argument(
        "--interventions",
        nargs="+",
        default=INTERVENTION_CHOICES,
        choices=INTERVENTION_CHOICES,
        help="Communication interventions to evaluate.",
    )
    parser.add_argument(
        "--random_buffer_steps",
        type=int,
        default=2000,
        help="Number of vanilla rollout steps used to build the random-message buffer.",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=0.5,
        help="Gaussian noise scale for the noise intervention.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Optional path for summary CSV. Defaults to <model_parent>/cutoff_intervention_eval/cutoff_intervention_summary.csv",
    )
    parser.add_argument(
        "--episode_csv",
        type=str,
        default="",
        help="Optional path for per-episode CSV. Defaults next to output_csv.",
    )
    all_args = parser.parse_known_args(args)[0]
    return all_args


def make_env(all_args):
    if all_args.env_name != "MPE":
        raise NotImplementedError("This eval script only supports MPE.")
    env = MPEEnv(all_args)
    env.seed(all_args.seed)
    return env


def maybe_set_proctitle(all_args):
    if setproctitle is None:
        return
    setproctitle.setproctitle(
        f"{all_args.algorithm_name}-{all_args.env_name}-{all_args.experiment_name}@{all_args.user_name}"
    )


def configure_algorithm_flags(all_args):
    if all_args.algorithm_name == "rmappo":
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError(
            f"Unsupported algorithm_name={all_args.algorithm_name} for MPE intervention eval."
        )


def load_policy_from_dir(policy, model_dir, separated_agent_id=None):
    if separated_agent_id is None:
        actor_path = Path(model_dir) / "actor.pt"
        critic_path = Path(model_dir) / "critic.pt"
    else:
        actor_path = Path(model_dir) / f"actor_agent{separated_agent_id}.pt"
        critic_path = Path(model_dir) / f"critic_agent{separated_agent_id}.pt"

    actor_state_dict = torch.load(actor_path, map_location="cpu", weights_only=True)
    policy.actor.load_state_dict(actor_state_dict)

    if critic_path.exists():
        critic_state_dict = torch.load(critic_path, map_location="cpu", weights_only=True)
        policy.critic.load_state_dict(critic_state_dict)

    policy.actor.eval()
    policy.critic.eval()


def build_policy(all_args, obs_space, cent_obs_space, act_space, device):
    return R_MAPPOPolicy(all_args, obs_space, cent_obs_space, act_space, device=device)


def is_collision(agent1, agent2):
    delta_pos = agent1.state.p_pos - agent2.state.p_pos
    dist = np.sqrt(np.sum(np.square(delta_pos)))
    dist_min = agent1.size + agent2.size
    return dist < dist_min


def action_to_env_action(action, action_space):
    action = np.asarray(action).reshape(-1)

    if action_space.__class__.__name__ == "Discrete":
        idx = int(action[0])
        return np.eye(action_space.n, dtype=np.float32)[idx]

    if action_space.__class__.__name__ == "MultiDiscrete":
        pieces = []
        for dim, high in enumerate(action_space.high):
            idx = int(action[dim])
            pieces.append(np.eye(high + 1, dtype=np.float32)[idx])
        return np.concatenate(pieces, axis=0)

    raise NotImplementedError(
        f"Unsupported action space: {action_space.__class__.__name__}"
    )


def prey_move_to_action_index(prey_agent, world, mode, rng):
    predators = [agent for agent in world.agents if agent.adversary]
    if len(predators) == 0:
        return 0

    if mode == "random":
        return int(rng.integers(0, 5))

    if mode != "heuristic":
        raise NotImplementedError(f"Unsupported fixed prey policy: {mode}")

    nearest_predator = min(
        predators,
        key=lambda predator: np.linalg.norm(predator.state.p_pos - prey_agent.state.p_pos)
    )
    delta = prey_agent.state.p_pos - nearest_predator.state.p_pos

    if np.allclose(delta, 0.0):
        return 0

    if abs(delta[0]) >= abs(delta[1]):
        return 2 if delta[0] > 0 else 1
    return 4 if delta[1] > 0 else 3


def prey_action_to_env_action(prey_agent, world, action_space, mode, rng):
    move_action = prey_move_to_action_index(prey_agent, world, mode, rng)

    if action_space.__class__.__name__ == "Discrete":
        return np.eye(action_space.n, dtype=np.float32)[move_action]

    if action_space.__class__.__name__ == "MultiDiscrete":
        pieces = []
        sizes = action_space.high - action_space.low + 1
        pieces.append(np.eye(sizes[0], dtype=np.float32)[move_action])
        for size in sizes[1:]:
            pieces.append(np.eye(size, dtype=np.float32)[0])
        return np.concatenate(pieces, axis=0)

    raise NotImplementedError(
        f"Unsupported prey action space: {action_space.__class__.__name__}"
    )


def get_comm_slice(action_space):
    if action_space.__class__.__name__ != "MultiDiscrete":
        return None

    sizes = action_space.high - action_space.low + 1
    if len(sizes) < 2:
        return None

    move_size = int(sizes[0])
    comm_size = int(sizes[1])
    return move_size, move_size + comm_size


def build_communicator_metadata(env):
    metadata = {}
    predator_ids = []
    for agent_id, agent in enumerate(env.world.agents):
        if not agent.adversary:
            continue
        predator_ids.append(agent_id)
        comm_slice = get_comm_slice(env.action_space[agent_id])
        if comm_slice is not None:
            metadata[agent_id] = {"slice": comm_slice}
    return predator_ids, metadata


def reset_predator_rnn_states(predator_ids, all_args):
    predator_rnn_states = {}
    predator_masks = {}
    for agent_id in predator_ids:
        predator_rnn_states[agent_id] = np.zeros(
            (1, all_args.recurrent_N, all_args.hidden_size), dtype=np.float32
        )
        predator_masks[agent_id] = np.ones((1, 1), dtype=np.float32)
    return predator_rnn_states, predator_masks


def build_actions_env(env, all_args, predator_policy_map, fixed_opponent_policy, rng,
                      predator_rnn_states, predator_masks):
    obs = env._get_obs  # keep reference local to avoid repeated attribute lookup
    actions_env = []

    for agent_id, agent in enumerate(env.world.agents):
        action_space = env.action_space[agent_id]

        if agent.adversary:
            policy = predator_policy_map[agent_id]
            agent_obs = np.asarray(obs(agent), dtype=np.float32)[None, :]
            with torch.no_grad():
                action, next_rnn_state = policy.act(
                    agent_obs,
                    predator_rnn_states[agent_id],
                    predator_masks[agent_id],
                    deterministic=True,
                )
            predator_rnn_states[agent_id] = next_rnn_state.detach().cpu().numpy()
            action = action.detach().cpu().numpy()[0]
            actions_env.append(action_to_env_action(action, action_space))
        else:
            actions_env.append(
                prey_action_to_env_action(agent, env.world, action_space, fixed_opponent_policy, rng)
            )

    return actions_env


def apply_comm_intervention(actions_env, communicator_metadata, intervention_type,
                            rng, message_buffer=None, noise_scale=0.5):
    modified = [np.array(action, copy=True) for action in actions_env]

    if intervention_type == "vanilla":
        return modified

    predator_ids = list(communicator_metadata.keys())
    if not predator_ids:
        return modified

    current_messages = []
    for agent_id in predator_ids:
        start, end = communicator_metadata[agent_id]["slice"]
        current_messages.append(modified[agent_id][start:end].copy())

    if intervention_type == "zero":
        for agent_id in predator_ids:
            start, end = communicator_metadata[agent_id]["slice"]
            modified[agent_id][start:end] = 0.0
        return modified

    if intervention_type == "random":
        if not message_buffer:
            raise ValueError("random intervention requires a non-empty message buffer.")
        for agent_id in predator_ids:
            start, end = communicator_metadata[agent_id]["slice"]
            msg = message_buffer[int(rng.integers(0, len(message_buffer)))]
            modified[agent_id][start:end] = msg
        return modified

    if intervention_type == "permuted":
        perm = rng.permutation(len(predator_ids))
        for out_idx, agent_id in enumerate(predator_ids):
            start, end = communicator_metadata[agent_id]["slice"]
            modified[agent_id][start:end] = current_messages[int(perm[out_idx])]
        return modified

    if intervention_type == "noise":
        for agent_id in predator_ids:
            start, end = communicator_metadata[agent_id]["slice"]
            msg = modified[agent_id][start:end]
            noisy_msg = msg + rng.normal(0.0, noise_scale, size=msg.shape).astype(np.float32)
            modified[agent_id][start:end] = np.clip(noisy_msg, 0.0, 1.0)
        return modified

    raise ValueError(f"Unknown intervention_type={intervention_type}")


def collect_message_buffer(env, all_args, predator_policy_map, fixed_opponent_policy,
                           communicator_metadata, num_steps, seed):
    rng = np.random.default_rng(seed)
    predator_ids = list(communicator_metadata.keys())
    predator_rnn_states, predator_masks = reset_predator_rnn_states(predator_ids, all_args)
    message_buffer = []

    env.seed(seed)
    env.reset()

    for step in range(num_steps):
        actions_env = build_actions_env(
            env,
            all_args,
            predator_policy_map,
            fixed_opponent_policy,
            rng,
            predator_rnn_states,
            predator_masks,
        )
        for agent_id in predator_ids:
            start, end = communicator_metadata[agent_id]["slice"]
            message_buffer.append(actions_env[agent_id][start:end].copy())

        env.step(actions_env)

        if (step + 1) % all_args.episode_length == 0:
            env.reset()
            predator_rnn_states, predator_masks = reset_predator_rnn_states(predator_ids, all_args)

    return message_buffer


def evaluate_episode(env, all_args, predator_policy_map, fixed_opponent_policy,
                     intervention_type, communicator_metadata, message_buffer,
                     episode_seed):
    rng = np.random.default_rng(episode_seed)
    np.random.seed(episode_seed)
    torch.manual_seed(episode_seed)

    env.seed(episode_seed)
    env.reset()

    predator_ids = [i for i, agent in enumerate(env.world.agents) if agent.adversary]
    prey_ids = [i for i, agent in enumerate(env.world.agents) if not agent.adversary]
    predator_rnn_states, predator_masks = reset_predator_rnn_states(predator_ids, all_args)

    episode_rewards = np.zeros(len(env.world.agents), dtype=np.float32)
    capture_step = None
    collision_steps = 0

    for step in range(all_args.episode_length):
        actions_env = build_actions_env(
            env,
            all_args,
            predator_policy_map,
            fixed_opponent_policy,
            rng,
            predator_rnn_states,
            predator_masks,
        )
        actions_env = apply_comm_intervention(
            actions_env,
            communicator_metadata,
            intervention_type,
            rng,
            message_buffer=message_buffer,
            noise_scale=all_args.noise_scale,
        )

        _, rewards, _, _ = env.step(actions_env)
        episode_rewards += np.asarray(rewards, dtype=np.float32).reshape(-1)

        captured_this_step = False
        for predator in [agent for agent in env.world.agents if agent.adversary]:
            for prey in [agent for agent in env.world.agents if not agent.adversary]:
                if is_collision(predator, prey):
                    captured_this_step = True
                    break
            if captured_this_step:
                break

        if captured_this_step:
            collision_steps += 1
            if capture_step is None:
                capture_step = step + 1

    if capture_step is None:
        capture_step = all_args.episode_length + 1

    predator_return = float(np.mean(episode_rewards[predator_ids])) if predator_ids else 0.0
    prey_return = float(np.mean(episode_rewards[prey_ids])) if prey_ids else 0.0

    return {
        "captured": bool(capture_step <= all_args.episode_length),
        "capture_step": int(capture_step),
        "collision_steps": int(collision_steps),
        "predator_return": predator_return,
        "prey_return": prey_return,
    }


def summarize_results(results, episode_length):
    capture_flags = np.asarray([r["captured"] for r in results], dtype=np.float32)
    capture_steps = np.asarray([r["capture_step"] for r in results], dtype=np.float32)
    collision_steps = np.asarray([r["collision_steps"] for r in results], dtype=np.float32)
    predator_returns = np.asarray([r["predator_return"] for r in results], dtype=np.float32)
    prey_returns = np.asarray([r["prey_return"] for r in results], dtype=np.float32)
    clipped_capture_steps = np.minimum(capture_steps, float(episode_length))

    summary = {
        "capture_rate": float(capture_flags.mean()) if len(results) > 0 else 0.0,
        "capture_rate_std": float(capture_flags.std()) if len(results) > 0 else 0.0,
        "mean_capture_step": float(capture_steps[capture_flags > 0].mean()) if np.any(capture_flags > 0) else None,
        "mean_capture_step_clipped": float(clipped_capture_steps.mean()) if len(results) > 0 else None,
        "mean_collision_steps": float(collision_steps.mean()) if len(results) > 0 else 0.0,
        "mean_predator_return": float(predator_returns.mean()) if len(results) > 0 else 0.0,
        "std_predator_return": float(predator_returns.std()) if len(results) > 0 else 0.0,
        "mean_prey_return": float(prey_returns.mean()) if len(results) > 0 else 0.0,
        "n_episodes": int(len(results)),
    }
    return summary


def resolve_output_paths(all_args):
    if all_args.output_csv:
        summary_path = Path(all_args.output_csv).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        base_dir = Path(all_args.model_dir).resolve().parent
        eval_dir = base_dir / "cutoff_intervention_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        summary_path = eval_dir / "cutoff_intervention_summary.csv"

    if all_args.episode_csv:
        episode_path = Path(all_args.episode_csv).expanduser().resolve()
        episode_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        episode_path = summary_path.with_name(f"{summary_path.stem}_episodes.csv")

    json_path = summary_path.with_suffix(".json")
    return summary_path, episode_path, json_path


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    configure_algorithm_flags(all_args)

    assert all_args.model_dir, "set model_dir first"
    assert all_args.env_name == "MPE", "This eval script only supports MPE."
    assert all_args.scenario_name == "simple_tag", "This eval script is intended for simple_tag."
    assert all_args.use_simple_comm, "Enable --use_simple_comm to match the trained communication checkpoint."
    assert all_args.comm_dim > 0, "comm_dim must be positive for communication intervention eval."

    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    maybe_set_proctitle(all_args)

    env = make_env(all_args)
    predator_ids, communicator_metadata = build_communicator_metadata(env)
    if not communicator_metadata:
        raise RuntimeError("No communicating predator agents were found in the reconstructed environment.")

    if all_args.share_policy:
        share_observation_space = (
            env.share_observation_space[0] if all_args.use_centralized_V else env.observation_space[0]
        )
        shared_policy = build_policy(
            all_args,
            env.observation_space[0],
            share_observation_space,
            env.action_space[0],
            device,
        )
        load_policy_from_dir(shared_policy, all_args.model_dir)
        predator_policy_map = {agent_id: shared_policy for agent_id in predator_ids}
    else:
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

    opponent_policies = (
        ["random", "heuristic"]
        if all_args.fixed_opponent_policy == "all"
        else [all_args.fixed_opponent_policy]
    )

    summary_rows = []
    episode_rows = []
    report = {
        "model_dir": str(Path(all_args.model_dir).resolve()),
        "seed": all_args.seed,
        "episode_length": all_args.episode_length,
        "eval_episodes": all_args.eval_episodes,
        "fixed_opponent_policy": all_args.fixed_opponent_policy,
        "interventions": list(all_args.interventions),
        "results": {},
    }

    for opponent_policy in opponent_policies:
        print(f"\nPreparing message buffer for opponent policy: {opponent_policy}")
        message_buffer = None
        if "random" in all_args.interventions:
            buffer_seed = all_args.seed + 100000
            message_buffer = collect_message_buffer(
                env,
                all_args,
                predator_policy_map,
                opponent_policy,
                communicator_metadata,
                all_args.random_buffer_steps,
                buffer_seed,
            )
            print(f"Collected {len(message_buffer)} historical messages for random replacement.")

        report["results"][opponent_policy] = {}

        for intervention in all_args.interventions:
            print(f"Evaluating opponent={opponent_policy}, intervention={intervention}")
            episode_results = []

            for episode_idx in range(all_args.eval_episodes):
                episode_seed = all_args.seed + episode_idx
                result = evaluate_episode(
                    env,
                    all_args,
                    predator_policy_map,
                    opponent_policy,
                    intervention,
                    communicator_metadata,
                    message_buffer,
                    episode_seed,
                )
                episode_results.append(result)
                episode_rows.append({
                    "opponent_policy": opponent_policy,
                    "intervention": intervention,
                    "episode_idx": episode_idx,
                    "episode_seed": episode_seed,
                    **result,
                })

            summary = summarize_results(episode_results, all_args.episode_length)
            summary_row = {
                "opponent_policy": opponent_policy,
                "intervention": intervention,
                **summary,
            }
            summary_rows.append(summary_row)
            report["results"][opponent_policy][intervention] = {
                "summary": summary,
                "episodes": episode_results,
            }

            print(
                f"capture_rate={summary['capture_rate']:.4f}, "
                f"mean_capture_step={summary['mean_capture_step']}, "
                f"mean_predator_return={summary['mean_predator_return']:.4f}"
            )

    summary_path, episode_path, json_path = resolve_output_paths(all_args)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(episode_rows).to_csv(episode_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nSaved summary CSV to: {summary_path}")
    print(f"Saved episode CSV to: {episode_path}")
    print(f"Saved JSON report to: {json_path}")

    env.close()


if __name__ == "__main__":
    main(sys.argv[1:])
