import time
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner

try:
    import imageio
except ModuleNotFoundError:
    imageio = None

def _t2n(x):
    return x.detach().cpu().numpy()

class MPERunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""
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
        comm_activity = np.zeros(actions.shape[:2], dtype=np.float32)
        if not self.use_comm_l1_penalty:
            return comm_activity

        for agent_id, can_comm in enumerate(self.agent_comm_mask):
            if can_comm:
                comm_activity[:, agent_id] = (actions[:, agent_id, 1] != 0).astype(np.float32)

        return comm_activity

    def _extract_comm_probs(self, action_probs):
        if self.comm_token_count is None:
            return None

        move_branch_size = int(self.envs.action_space[0].high[0] + 1)
        comm_branch_size = int(self.envs.action_space[0].high[1] + 1)
        return action_probs[:, :, move_branch_size:move_branch_size + comm_branch_size]

    def _compute_vocab_and_entropy_stats(self, actions, comm_action_probs):
        if self.comm_token_count is None or comm_action_probs is None:
            return None

        full_counts = np.zeros(self.comm_token_count, dtype=np.float64)
        active_counts = np.zeros(max(self.comm_token_count - 1, 0), dtype=np.float64)
        active_entropies = []

        for agent_id, can_comm in enumerate(self.agent_comm_mask):
            if not can_comm:
                continue

            comm_ids = actions[:, agent_id, 1].astype(np.int64)
            full_counts += np.bincount(comm_ids, minlength=self.comm_token_count)

            active_ids = comm_ids[comm_ids > 0] - 1
            if active_counts.size > 0 and active_ids.size > 0:
                active_counts += np.bincount(active_ids, minlength=active_counts.size)

            probs = comm_action_probs[:, agent_id, :]
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
                self.trainer.policy.lr_decay(episode, episodes)

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
                            if 'individual_reward' in info[agent_id].keys():
                                idv_rews.append(info[agent_id]['individual_reward'])
                        agent_k = 'agent%i/individual_rewards' % agent_id
                        env_infos[agent_k] = idv_rews

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

                train_infos["average_episode_rewards"] = np.mean(np.asarray(episode_shaped_rewards)) * self.episode_length
                train_infos["average_episode_env_rewards"] = np.mean(np.asarray(episode_raw_rewards)) * self.episode_length
                train_infos["mean_step_comm_activity"] = np.mean(episode_comm_activity)
                train_infos["average_episode_comm_activity"] = np.mean(episode_comm_activity) * self.episode_length
                train_infos["average_episode_comm_penalty"] = np.mean(episode_comm_penalty) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            raise NotImplementedError

        comm_action_probs = None
        if self.comm_token_count is not None:
            action_probs = self.trainer.policy.get_probs(
                np.concatenate(self.buffer.obs[step]),
                np.concatenate(self.buffer.rnn_states[step]),
                np.concatenate(self.buffer.masks[step]),
            )
            action_probs = np.array(np.split(_t2n(action_probs), self.n_rollout_threads))
            comm_action_probs = self._extract_comm_probs(action_probs)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, comm_action_probs

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i]+1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs
        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render('rgb_array')[0][0]
                all_frames.append(image)
            else:
                envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            episode_rewards = []
            
            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i]+1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render('human')

            episode_rewards = np.array(episode_rewards)
            # [episode_steps, n_envs, num_agents] -> [n_envs, num_agents]
            episode_returns = np.sum(episode_rewards, axis=0)
            agent_mean_returns = np.mean(episode_returns, axis=0)

            for agent_id in range(self.num_agents):
                print("eval average episode rewards of agent{}: {}".format(agent_id, agent_mean_returns[agent_id]))

            if hasattr(self.all_args, "num_adversaries") and 0 < self.all_args.num_adversaries < self.num_agents:
                num_adversaries = self.all_args.num_adversaries
                adv_ret = np.mean(agent_mean_returns[:num_adversaries])
                good_ret = np.mean(agent_mean_returns[num_adversaries:])
                print("predator(avg): {}, prey(avg): {}".format(adv_ret, good_ret))

            print("all-agents average episode rewards: {}".format(np.mean(agent_mean_returns)))

        if self.all_args.save_gifs:
            if imageio is None:
                raise ImportError("Saving GIFs requires imageio to be installed.")
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
