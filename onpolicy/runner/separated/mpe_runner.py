import time
import os
import numpy as np
from itertools import chain
import torch

from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.separated.base_runner import Runner

try:
    import imageio
except ModuleNotFoundError:
    imageio = None

def _t2n(x):
    return x.detach().cpu().numpy()

class MPERunner(Runner):
    def __init__(self, config):
        super(MPERunner, self).__init__(config)
        self.use_comm_l1_penalty = self.all_args.use_simple_comm and self.all_args.use_comm_l1_penalty
        self.comm_l1_coef = self.all_args.comm_l1_coef
        self.agent_comm_mask = np.array(
            [space.__class__.__name__ == 'MultiDiscrete' and getattr(space, "shape", 0) > 1 for space in self.envs.action_space],
            dtype=bool,
        )
        self.comm_token_count = None
        for agent_id, can_comm in enumerate(self.agent_comm_mask):
            if can_comm:
                self.comm_token_count = int(self.envs.action_space[agent_id].high[1] + 1)
                break

    def _compute_comm_activity(self, actions):
        comm_activity = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.float32)
        if not self.use_comm_l1_penalty:
            return comm_activity

        for agent_id, can_comm in enumerate(self.agent_comm_mask):
            if can_comm:
                agent_actions = np.asarray(actions[agent_id])
                comm_activity[:, agent_id] = (agent_actions[:, 1] != 0).astype(np.float32)

        return comm_activity

    def _extract_comm_probs(self, agent_id, action_probs):
        if action_probs is None or not self.agent_comm_mask[agent_id]:
            return None

        move_branch_size = int(self.envs.action_space[agent_id].high[0] + 1)
        comm_branch_size = int(self.envs.action_space[agent_id].high[1] + 1)
        return action_probs[:, move_branch_size:move_branch_size + comm_branch_size]

    def _compute_vocab_and_entropy_stats(self, actions, comm_action_probs):
        if self.comm_token_count is None:
            return None

        full_counts = np.zeros(self.comm_token_count, dtype=np.float64)
        active_counts = np.zeros(max(self.comm_token_count - 1, 0), dtype=np.float64)
        active_entropies = []

        for agent_id, can_comm in enumerate(self.agent_comm_mask):
            if not can_comm:
                continue

            agent_actions = np.asarray(actions[agent_id])
            comm_ids = agent_actions[:, 1].astype(np.int64)
            full_counts += np.bincount(comm_ids, minlength=self.comm_token_count)

            active_ids = comm_ids[comm_ids > 0] - 1
            if active_counts.size > 0 and active_ids.size > 0:
                active_counts += np.bincount(active_ids, minlength=active_counts.size)

            probs = comm_action_probs[agent_id]
            if probs is None or active_counts.size == 0:
                continue

            active_mass = probs[:, 1:].sum(axis=1, keepdims=True)
            active_mask = (comm_ids > 0) & (active_mass.squeeze(-1) > 1e-12)
            if not np.any(active_mask):
                continue

            conditional_probs = probs[active_mask, 1:] / active_mass[active_mask]
            conditional_probs = np.clip(conditional_probs, 1e-12, 1.0)
            entropies = -(conditional_probs * np.log(conditional_probs)).sum(axis=1)
            active_entropies.extend(entropies.tolist())

        active_entropy = float(np.mean(active_entropies)) if active_entropies else 0.0

        return full_counts, active_counts, active_entropy
       
    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            episode_raw_rewards = []
            episode_shaped_rewards = []
            episode_comm_activity = []
            episode_comm_penalty = []
            episode_comm_full_counts = np.zeros(self.comm_token_count, dtype=np.float64) if self.comm_token_count is not None else None
            episode_comm_active_counts = np.zeros(max(self.comm_token_count - 1, 0), dtype=np.float64) if self.comm_token_count is not None else None
            episode_comm_entropy_values = []

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, comm_action_probs = self.collect(step)
                    
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                raw_rewards = np.asarray(rewards, dtype=np.float32)
                comm_activity = self._compute_comm_activity(actions)
                comm_penalty = self.comm_l1_coef * comm_activity[..., None]
                shaped_rewards = raw_rewards - comm_penalty

                episode_raw_rewards.append(raw_rewards)
                episode_shaped_rewards.append(shaped_rewards)
                if self.agent_comm_mask.any():
                    episode_comm_activity.append(np.mean(comm_activity[:, self.agent_comm_mask]))
                    episode_comm_penalty.append(np.mean(comm_penalty[:, self.agent_comm_mask]))
                else:
                    episode_comm_activity.append(0.0)
                    episode_comm_penalty.append(0.0)

                vocab_stats = self._compute_vocab_and_entropy_stats(actions, comm_action_probs)
                if vocab_stats is not None:
                    full_counts, active_counts, active_entropy = vocab_stats
                    if episode_comm_full_counts is not None:
                        episode_comm_full_counts += full_counts
                    if episode_comm_active_counts is not None:
                        episode_comm_active_counts += active_counts
                    episode_comm_entropy_values.append(active_entropy)

                data = obs, shaped_rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic 
                
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "MPE":
                    env_infos = {}
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for info in infos:
                            for count, info in enumerate(infos):
                                if 'individual_reward' in infos[count][agent_id].keys():
                                    idv_rews.append(infos[count][agent_id].get('individual_reward', 0))
                        train_infos[agent_id].update({'individual_rewards': np.mean(idv_rews)})
                        train_infos[agent_id].update({"average_episode_rewards": np.mean(np.asarray(episode_shaped_rewards)[:, :, agent_id]) * self.episode_length})
                        train_infos[agent_id].update({"average_episode_env_rewards": np.mean(np.asarray(episode_raw_rewards)[:, :, agent_id]) * self.episode_length})
                        train_infos[agent_id].update({"mean_step_comm_activity": np.mean(episode_comm_activity)})
                        train_infos[agent_id].update({"average_episode_comm_activity": np.mean(episode_comm_activity) * self.episode_length})
                        train_infos[agent_id].update({"average_episode_comm_penalty": np.mean(episode_comm_penalty) * self.episode_length})
                        env_infos[f'agent{agent_id}/individual_rewards'] = idv_rews

                    if episode_comm_full_counts is not None:
                        full_total = episode_comm_full_counts.sum()
                        if full_total > 0:
                            full_usage = episode_comm_full_counts / full_total
                        else:
                            full_usage = np.zeros_like(episode_comm_full_counts)
                        for token_id, value in enumerate(full_usage):
                            env_infos[f'comm/vocab_usage_full_token{token_id}'] = [float(value)]

                    if episode_comm_active_counts is not None and episode_comm_active_counts.size > 0:
                        active_total = episode_comm_active_counts.sum()
                        if active_total > 0:
                            active_usage = episode_comm_active_counts / active_total
                        else:
                            active_usage = np.zeros_like(episode_comm_active_counts)
                        for token_offset, value in enumerate(active_usage, start=1):
                            env_infos[f'comm/vocab_usage_active_token{token_offset}'] = [float(value)]

                    env_infos['comm/comm_entropy_active'] = [float(np.mean(episode_comm_entropy_values)) if episode_comm_entropy_values else 0.0]
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step])
            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action)
            # rearrange action
            if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i]+1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
            else:
                raise NotImplementedError

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append( _t2n(rnn_state_critic))

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(values).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        comm_action_probs = [None for _ in range(self.num_agents)]
        for agent_id, can_comm in enumerate(self.agent_comm_mask):
            if not can_comm:
                continue
            probs = self.trainer[agent_id].policy.get_probs(
                self.buffer[agent_id].obs[step],
                self.buffer[agent_id].rnn_states[step],
                self.buffer[agent_id].masks[step],
            )
            comm_action_probs[agent_id] = self._extract_comm_probs(agent_id, _t2n(probs))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, comm_action_probs

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            self.buffer[agent_id].insert(share_obs,
                                        np.array(list(obs[:, agent_id])),
                                        rnn_states[:, agent_id],
                                        rnn_states_critic[:, agent_id],
                                        actions[agent_id],
                                        action_log_probs[agent_id],
                                        values[:, agent_id],
                                        rewards[:, agent_id],
                                        masks[:, agent_id])

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(np.array(list(eval_obs[:, agent_id])),
                                                                                eval_rnn_states[:, agent_id],
                                                                                eval_masks[:, agent_id],
                                                                                deterministic=True)

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                if self.eval_envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i]+1)[eval_action[:, i]]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                    eval_action_env = np.squeeze(np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1)
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)
                
            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        
        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_train_infos.append({'eval_average_episode_rewards': eval_average_episode_rewards})
            print("eval average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))

        self.log_train(eval_train_infos, total_num_steps)  

    @torch.no_grad()
    def render(self):        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            episode_rewards = []
            obs = self.envs.reset()
            if self.all_args.save_gifs:
                image = self.envs.render('rgb_array')[0][0]
                all_frames.append(image)

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for step in range(self.episode_length):
                calc_start = time.time()
                
                temp_actions_env = []
                for agent_id in range(self.num_agents):
                    if not self.use_centralized_V:
                        share_obs = np.array(list(obs[:, agent_id]))
                    self.trainer[agent_id].prep_rollout()
                    action, rnn_state = self.trainer[agent_id].policy.act(np.array(list(obs[:, agent_id])),
                                                                        rnn_states[:, agent_id],
                                                                        masks[:, agent_id],
                                                                        deterministic=True)

                    action = action.detach().cpu().numpy()
                    # rearrange action
                    if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                        for i in range(self.envs.action_space[agent_id].shape):
                            uc_action_env = np.eye(self.envs.action_space[agent_id].high[i]+1)[action[:, i]]
                            if i == 0:
                                action_env = uc_action_env
                            else:
                                action_env = np.concatenate((action_env, uc_action_env), axis=1)
                    elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                        action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
                    else:
                        raise NotImplementedError

                    temp_actions_env.append(action_env)
                    rnn_states[:, agent_id] = _t2n(rnn_state)
                   
                # [envs, agents, dim]
                actions_env = []
                for i in range(self.n_rollout_threads):
                    one_hot_action_env = []
                    for temp_action_env in temp_actions_env:
                        one_hot_action_env.append(temp_action_env[i])
                    actions_env.append(one_hot_action_env)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = self.envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)

            episode_rewards = np.array(episode_rewards)
            for agent_id in range(self.num_agents):
                average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
                print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))
        
        if self.all_args.save_gifs:
            if imageio is None:
                raise ImportError("Saving GIFs requires imageio to be installed.")
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
