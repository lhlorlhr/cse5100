#!/usr/bin/env python
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from onpolicy.config import get_config
from onpolicy.envs.mpe.MPE_env import MPEEnv
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default="simple_tag")
    parser.add_argument("--num_landmarks", type=int, default=2)
    parser.add_argument("--num_agents", type=int, default=4)
    parser.add_argument("--num_good_agents", type=int, default=1)
    parser.add_argument("--num_adversaries", type=int, default=3)
    parser.add_argument("--print_episodes", type=int, default=3,
                        help="How many episodes to print.")
    parser.add_argument("--print_steps", type=int, default=10,
                        help="How many steps to print per episode.")
    parser.add_argument("--deterministic", action="store_true", default=False,
                        help="Use greedy actions instead of sampling.")
    return parser.parse_known_args(args)[0]


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
            f"Unsupported algorithm_name={all_args.algorithm_name} for MPE communication inspection."
        )


def action_to_env_vector(action, action_space):
    action = np.asarray(action).reshape(-1)

    if action_space.__class__.__name__ == "Discrete":
        idx = int(action[0])
        return np.eye(action_space.n, dtype=np.float32)[idx]

    if action_space.__class__.__name__ == "MultiDiscrete":
        pieces = []
        sizes = (action_space.high - action_space.low + 1).astype(int)
        for dim, size in enumerate(sizes):
            idx = int(action[dim])
            pieces.append(np.eye(size, dtype=np.float32)[idx])
        return np.concatenate(pieces, axis=0)

    raise NotImplementedError(
        f"Unsupported action space: {action_space.__class__.__name__}"
    )


def get_comm_slice(action_space, agent):
    if getattr(agent, "silent", True):
        return None

    if action_space.__class__.__name__ != "MultiDiscrete":
        return None

    sizes = (action_space.high - action_space.low + 1).astype(int)
    if len(sizes) < 2:
        return None

    move_size = int(sizes[0])
    comm_size = int(sizes[1])
    return move_size, move_size + comm_size


def build_policy(all_args, obs_space, share_obs_space, act_space, device):
    return R_MAPPOPolicy(
        all_args,
        obs_space,
        share_obs_space,
        act_space,
        device=device,
    )


def load_actor_only(policy, actor_path, critic_path=None):
    actor_state_dict = torch.load(actor_path, map_location="cpu", weights_only=True)
    policy.actor.load_state_dict(actor_state_dict)
    policy.actor.eval()

    if critic_path is not None and critic_path.exists():
        critic_state_dict = torch.load(critic_path, map_location="cpu", weights_only=True)
        policy.critic.load_state_dict(critic_state_dict)
        policy.critic.eval()


def load_policies(all_args, env, device):
    model_dir = Path(all_args.model_dir)
    shared_actor_path = model_dir / "actor.pt"

    if shared_actor_path.exists():
        share_obs_space = env.share_observation_space[0] if all_args.use_centralized_V else env.observation_space[0]
        policy = build_policy(
            all_args,
            env.observation_space[0],
            share_obs_space,
            env.action_space[0],
            device,
        )
        load_actor_only(policy, shared_actor_path, model_dir / "critic.pt")
        return {agent_id: policy for agent_id in range(env.n)}, "shared"

    policies = {}
    for agent_id in range(env.n):
        actor_path = model_dir / f"actor_agent{agent_id}.pt"
        critic_path = model_dir / f"critic_agent{agent_id}.pt"
        if not actor_path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint for agent {agent_id}: {actor_path}"
            )

        share_obs_space = (
            env.share_observation_space[agent_id]
            if all_args.use_centralized_V
            else env.observation_space[agent_id]
        )
        policy = build_policy(
            all_args,
            env.observation_space[agent_id],
            share_obs_space,
            env.action_space[agent_id],
            device,
        )
        load_actor_only(policy, actor_path, critic_path)
        policies[agent_id] = policy

    return policies, "separated"


def print_space_summary(env):
    print("=== Action space summary ===")
    print(f"world.dim_c = {env.world.dim_c}")
    print(f"discrete_action_space = {env.discrete_action_space}")
    print(f"discrete_action_input = {env.discrete_action_input}")

    for agent_id, agent in enumerate(env.world.agents):
        action_space = env.action_space[agent_id]
        print(
            f"agent {agent_id}: adversary={agent.adversary}, silent={agent.silent}, "
            f"action_space={action_space.__class__.__name__}"
        )

        if action_space.__class__.__name__ == "MultiDiscrete":
            sizes = (action_space.high - action_space.low + 1).astype(int)
            print(f"  branch_sizes={sizes.tolist()} -> [movement, communication]")
            comm_slice = get_comm_slice(action_space, agent)
            if comm_slice is not None:
                print(
                    f"  message_type=discrete categorical one-hot, "
                    f"comm_id in [0, {sizes[1] - 1}], "
                    f"message_slice={comm_slice}"
                )
        elif action_space.__class__.__name__ == "Discrete":
            print("  no explicit communication branch")
        else:
            print(f"  unsupported summary path for {action_space}")


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    configure_algorithm_flags(all_args)

    assert all_args.env_name == "MPE", "This script only supports --env_name MPE."
    assert all_args.model_dir, "set --model_dir to a trained checkpoint directory"

    device = torch.device("cpu")
    env = MPEEnv(all_args)
    env.seed(all_args.seed)
    policies, policy_mode = load_policies(all_args, env, device)

    print_space_summary(env)
    print(f"policy_mode = {policy_mode}")
    print("\n=== Rollout samples ===")

    for ep in range(all_args.print_episodes):
        obs = env.reset()
        rnn_states = np.zeros(
            (env.n, all_args.recurrent_N, all_args.hidden_size), dtype=np.float32
        )
        masks = np.ones((env.n, 1), dtype=np.float32)

        print(f"\n[episode {ep}]")
        for step in range(all_args.print_steps):
            obs_batch = np.asarray(obs, dtype=np.float32)
            raw_actions = []
            next_rnn_states = np.zeros_like(rnn_states)
            for agent_id in range(env.n):
                policy = policies[agent_id]
                with torch.no_grad():
                    action, next_rnn_state = policy.act(
                        obs_batch[agent_id : agent_id + 1],
                        rnn_states[agent_id : agent_id + 1],
                        masks[agent_id : agent_id + 1],
                        deterministic=all_args.deterministic,
                    )
                raw_actions.append(action.detach().cpu().numpy()[0])
                next_rnn_states[agent_id] = next_rnn_state.detach().cpu().numpy()[0]

            rnn_states = next_rnn_states

            actions_env = []
            for agent_id, agent in enumerate(env.world.agents):
                env_action = action_to_env_vector(raw_actions[agent_id], env.action_space[agent_id])
                actions_env.append(env_action)

                comm_slice = get_comm_slice(env.action_space[agent_id], agent)
                if comm_slice is None:
                    continue

                comm_start, comm_end = comm_slice
                msg = env_action[comm_start:comm_end]
                print(
                    f"t={step} agent={agent_id} raw_action={raw_actions[agent_id].astype(int).tolist()} "
                    f"env_action={env_action.tolist()} msg={msg.tolist()} msg_argmax={int(np.argmax(msg))}"
                )

            obs, _, dones, _ = env.step(actions_env)
            masks = np.ones((env.n, 1), dtype=np.float32)
            done_mask = np.asarray(dones, dtype=bool)
            masks[done_mask] = 0.0
            rnn_states[done_mask] = 0.0


if __name__ == "__main__":
    main(sys.argv[1:])
