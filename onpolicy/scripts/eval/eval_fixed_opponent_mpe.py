#!/usr/bin/env python
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import setproctitle
except ImportError:
    setproctitle = None

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from onpolicy.config import get_config
from onpolicy.envs.mpe.MPE_env import MPEEnv
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='simple_tag', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=2)
    parser.add_argument('--num_good_agents', type=int,
                        default=1, help="number of good agents (for simple_tag)")
    parser.add_argument('--num_adversaries', type=int,
                        default=3, help="number of adversaries (for simple_tag)")

    parser.add_argument(
        "--fixed_opponent_policy",
        type=str,
        default="all",
        choices=["random", "heuristic", "all"],
        help="Which fixed prey policy to evaluate against."
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
            f"Unsupported algorithm_name={all_args.algorithm_name} for MPE fixed-opponent eval."
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


def predator_action_to_env_action(action, action_space):
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
        f"Unsupported action space for predator eval: {action_space.__class__.__name__}"
    )


def prey_move_to_action_index(prey_agent, world, mode):
    predators = [agent for agent in world.agents if agent.adversary]
    if len(predators) == 0:
        return 0

    if mode == "random":
        return int(np.random.randint(0, 5))

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


def prey_action_to_env_action(prey_agent, world, action_space, mode):
    move_action = prey_move_to_action_index(prey_agent, world, mode)

    if action_space.__class__.__name__ == "Discrete":
        return np.eye(action_space.n, dtype=np.float32)[move_action]

    if action_space.__class__.__name__ == "MultiDiscrete":
        # simple_tag uses movement + communication. We keep communication at 0.
        pieces = []
        move_high = action_space.high[0]
        comm_high = action_space.high[1]
        pieces.append(np.eye(move_high + 1, dtype=np.float32)[move_action])
        pieces.append(np.eye(comm_high + 1, dtype=np.float32)[0])
        return np.concatenate(pieces, axis=0)

    raise NotImplementedError(
        f"Unsupported action space for prey eval: {action_space.__class__.__name__}"
    )


def evaluate_episode(env, all_args, predator_policy_map, fixed_opponent_policy, episode_seed):
    np.random.seed(episode_seed)
    torch.manual_seed(episode_seed)

    obs = env.reset()
    world = env.world

    predator_ids = [i for i, agent in enumerate(world.agents) if agent.adversary]
    prey_ids = [i for i, agent in enumerate(world.agents) if not agent.adversary]

    predator_rnn_states = {}
    predator_masks = {}
    for agent_id in predator_ids:
        predator_rnn_states[agent_id] = np.zeros(
            (1, all_args.recurrent_N, all_args.hidden_size), dtype=np.float32
        )
        predator_masks[agent_id] = np.ones((1, 1), dtype=np.float32)

    episode_rewards = np.zeros(len(world.agents), dtype=np.float32)
    capture_step = None
    collision_steps = 0

    for step in range(all_args.episode_length):
        actions_env = []

        for agent_id, agent in enumerate(world.agents):
            action_space = env.action_space[agent_id]

            if agent.adversary:
                policy = predator_policy_map[agent_id]
                with torch.no_grad():
                    action, next_rnn_state = policy.act(
                        np.asarray(obs[agent_id])[None, :],
                        predator_rnn_states[agent_id],
                        predator_masks[agent_id],
                        deterministic=True,
                    )
                predator_rnn_states[agent_id] = next_rnn_state.detach().cpu().numpy()
                action = action.detach().cpu().numpy()[0]
                actions_env.append(predator_action_to_env_action(action, action_space))
            else:
                actions_env.append(
                    prey_action_to_env_action(agent, world, action_space, fixed_opponent_policy)
                )

        obs, rewards, dones, infos = env.step(actions_env)
        world = env.world
        episode_rewards += np.asarray(rewards, dtype=np.float32).reshape(-1)

        captured_this_step = False
        for predator in [agent for agent in world.agents if agent.adversary]:
            for prey in [agent for agent in world.agents if not agent.adversary]:
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
        "captured": capture_step <= all_args.episode_length,
        "capture_step": capture_step,
        "collision_steps": collision_steps,
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

    return {
        "capture_rate": float(capture_flags.mean()) if len(results) > 0 else 0.0,
        "mean_capture_step": float(capture_steps[capture_flags > 0].mean()) if np.any(capture_flags > 0) else None,
        "mean_capture_step_clipped": float(clipped_capture_steps.mean()) if len(results) > 0 else None,
        "mean_collision_steps": float(collision_steps.mean()) if len(results) > 0 else 0.0,
        "mean_predator_return": float(predator_returns.mean()) if len(results) > 0 else 0.0,
        "mean_prey_return": float(prey_returns.mean()) if len(results) > 0 else 0.0,
    }


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    configure_algorithm_flags(all_args)

    assert all_args.model_dir is not None and all_args.model_dir != "", "set model_dir first"
    assert all_args.env_name == "MPE", "This eval script only supports MPE."
    assert all_args.scenario_name == "simple_tag", "This eval script is intended for simple_tag."

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
    world = env.world
    predator_ids = [i for i, agent in enumerate(world.agents) if agent.adversary]

    if all_args.share_policy:
        share_observation_space = env.share_observation_space[0] if all_args.use_centralized_V else env.observation_space[0]
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
            share_observation_space = env.share_observation_space[agent_id] if all_args.use_centralized_V else env.observation_space[agent_id]
            policy = build_policy(
                all_args,
                env.observation_space[agent_id],
                share_observation_space,
                env.action_space[agent_id],
                device,
            )
            load_policy_from_dir(policy, all_args.model_dir, separated_agent_id=agent_id)
            predator_policy_map[agent_id] = policy

    opponent_policies = ["random", "heuristic"] if all_args.fixed_opponent_policy == "all" else [all_args.fixed_opponent_policy]

    base_dir = Path(all_args.model_dir).resolve().parent
    eval_dir = base_dir / "fixed_opponent_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    final_report = {
        "algorithm_name": all_args.algorithm_name,
        "scenario_name": all_args.scenario_name,
        "model_dir": str(Path(all_args.model_dir).resolve()),
        "seed": all_args.seed,
        "episode_length": all_args.episode_length,
        "eval_episodes": all_args.eval_episodes,
        "results": {},
    }

    for opponent_policy in opponent_policies:
        episode_results = []
        for episode_idx in range(all_args.eval_episodes):
            episode_seed = all_args.seed + episode_idx
            episode_result = evaluate_episode(
                env,
                all_args,
                predator_policy_map,
                opponent_policy,
                episode_seed,
            )
            episode_results.append(episode_result)

        summary = summarize_results(episode_results, all_args.episode_length)
        final_report["results"][opponent_policy] = summary
        final_report["results"][opponent_policy]["episodes"] = episode_results

        print(f"\nOpponent policy: {opponent_policy}")
        print(f"capture_rate: {summary['capture_rate']:.4f}")
        print(f"mean_capture_step: {summary['mean_capture_step']}")
        print(f"mean_capture_step_clipped: {summary['mean_capture_step_clipped']:.4f}")
        print(f"mean_collision_steps: {summary['mean_collision_steps']:.4f}")
        print(f"mean_predator_return: {summary['mean_predator_return']:.4f}")
        print(f"mean_prey_return: {summary['mean_prey_return']:.4f}")

    report_path = eval_dir / "fixed_opponent_metrics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print(f"\nSaved fixed-opponent eval report to: {report_path}")

    env.close()


if __name__ == "__main__":
    main(sys.argv[1:])
