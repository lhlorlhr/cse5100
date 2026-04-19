#!/usr/bin/env python
import sys
import os
import socket
import numpy as np
from pathlib import Path
import re

import torch
try:
    import setproctitle
except ImportError:
    setproctitle = None

from onpolicy.config import get_config

from onpolicy.envs.mpe.MPE_env import MPEEnv
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv

def make_render_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int,
                        default=2, help="number of players")
    parser.add_argument('--num_good_agents', type=int,
                        default=1, help="number of good agents (for simple_tag)")
    parser.add_argument('--num_adversaries', type=int,
                        default=3, help="number of adversaries (for simple_tag)")
    parser.add_argument("--use_simple_comm", action="store_true", default=False,
                        help="Enable simple communication in simple_tag.")
    parser.add_argument("--comm_dim", type=int, default=2,
                        help="Communication channel size when use_simple_comm is enabled.")
    parser.add_argument("--comm_target", type=str, default="all",
                        choices=["all", "adversary", "good"],
                        help="Which team is allowed to send communication messages.")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def _actor_index(path: Path):
    match = re.search(r"actor_agent(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def _load_state_dict(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _infer_comm_target(speaker_flags, num_adversaries, num_agents):
    if not speaker_flags:
        return None
    if all(speaker_flags):
        return "all"
    if len(speaker_flags) == num_agents:
        if speaker_flags[:num_adversaries] == [True] * num_adversaries and speaker_flags[num_adversaries:] == [False] * (num_agents - num_adversaries):
            return "adversary"
        if speaker_flags[:num_adversaries] == [False] * num_adversaries and speaker_flags[num_adversaries:] == [True] * (num_agents - num_adversaries):
            return "good"
    return None


def _infer_checkpoint_layout(all_args):
    model_dir = Path(all_args.model_dir)
    shared_actor = model_dir / "actor.pt"
    separated_actor = model_dir / "actor_agent0.pt"

    if shared_actor.exists():
        all_args.share_policy = True
        actor_paths = [shared_actor]
    elif separated_actor.exists():
        all_args.share_policy = False
        actor_paths = sorted(model_dir.glob("actor_agent*.pt"), key=_actor_index)
    else:
        return all_args

    state_dicts = [_load_state_dict(path) for path in actor_paths]
    speaker_flags = [any(key.startswith("act.action_outs.") for key in state_dict.keys()) for state_dict in state_dicts]

    if any(speaker_flags):
        all_args.use_simple_comm = True
        speaker_state_dict = next(state_dict for state_dict in state_dicts if any(key.startswith("act.action_outs.") for key in state_dict.keys()))
        comm_head_key = "act.action_outs.1.linear.weight"
        if comm_head_key in speaker_state_dict:
            all_args.comm_dim = int(speaker_state_dict[comm_head_key].shape[0])
        inferred_target = _infer_comm_target(speaker_flags, all_args.num_adversaries, all_args.num_agents)
        if inferred_target is not None:
            all_args.comm_target = inferred_target
    else:
        all_args.use_simple_comm = False

    print(
        "Detected checkpoint layout: "
        f"{'shared' if all_args.share_policy else 'separated'}, "
        f"use_simple_comm={all_args.use_simple_comm}, "
        f"comm_dim={getattr(all_args, 'comm_dim', 'n/a')}, "
        f"comm_target={getattr(all_args, 'comm_target', 'n/a')}"
    )
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False.")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    all_args = _infer_checkpoint_layout(all_args)

    assert (all_args.share_policy == True and all_args.scenario_name == 'simple_speaker_listener') == False, (
        "The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")
    assert all_args.comm_dim > 0, "comm_dim must be positive."
    if all_args.share_policy and all_args.use_simple_comm and all_args.comm_target != "all":
        raise AssertionError("share_policy=True requires comm_target='all' so all agents have the same action space.")

    assert all_args.use_render, ("u need to set use_render be True")
    assert not (all_args.model_dir == None or all_args.model_dir == ""), ("set model_dir first")
    assert all_args.n_rollout_threads==1, ("only support to use 1 env to render.")
    
    # cuda
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

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if setproctitle is not None:
        setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
            str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_render_env(all_args)
    eval_envs = None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from onpolicy.runner.shared.mpe_runner import MPERunner as Runner
    else:
        from onpolicy.runner.separated.mpe_runner import MPERunner as Runner

    runner = Runner(config)
    runner.render()
    
    # post process
    envs.close()

if __name__ == "__main__":
    main(sys.argv[1:])
