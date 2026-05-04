import numpy as np
from onpolicy.envs.mpe.core import World, Agent, Landmark
from onpolicy.envs.mpe.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        # set any world properties first
        use_simple_comm = getattr(args, "use_simple_comm", False)
        comm_dim = getattr(args, "comm_dim", 2)
        comm_target = getattr(args, "comm_target", "all")

        if use_simple_comm and comm_dim <= 0:
            raise ValueError("comm_dim must be a positive integer when --use_simple_comm is enabled.")
        if comm_target not in {"all", "adversaries", "good_agents"}:
            raise ValueError("comm_target must be one of: all, adversaries, good_agents.")

        world.dim_c = comm_dim if use_simple_comm else 0
        world.use_simple_comm = use_simple_comm
        world.comm_target = comm_target
        world.comm_has_null_action = use_simple_comm and getattr(args, "use_comm_l1_penalty", False)
        world.use_partial_obs = getattr(args, "use_partial_obs", False)
        world.partial_obs_radius = getattr(args, "partial_obs_radius", 1.0)
        if world.use_partial_obs and world.partial_obs_radius <= 0:
            raise ValueError("partial_obs_radius must be positive when --use_partial_obs is enabled.")
        num_good_agents = args.num_good_agents#1
        num_adversaries = args.num_adversaries#3
        num_agents = num_adversaries + num_good_agents
        num_landmarks = args.num_landmarks#2
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.adversary = True if i < num_adversaries else False
            agent.silent = not self._can_communicate(agent, world)
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world

    def _can_communicate(self, agent, world):
        if not getattr(world, "use_simple_comm", False):
            return False

        if world.comm_target == "all":
            return True
        if world.comm_target == "adversaries":
            return agent.adversary
        if world.comm_target == "good_agents":
            return not agent.adversary
        return False

    def reset_world(self, world):
        # random properties for agents
        world.assign_agent_colors()
        # random properties for landmarks
        world.assign_landmark_colors()
        # random properties for landmarks
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = 0.8 * np.random.uniform(-1, +1, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def _visible(self, agent, entity, world):
        if not getattr(world, "use_partial_obs", False):
            return True
        delta_pos = entity.state.p_pos - agent.state.p_pos
        return np.sqrt(np.sum(np.square(delta_pos))) <= world.partial_obs_radius


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        """
        Sheep (good agent) 的 reward
        被抓 -> 大额负奖励
        撞墙/landmark -> 小额负奖励
        远离wolves -> 正奖励 (shaping)
        """
        rew = 0
        shape = False  # 默认关闭shaping，遵循原始MADDPG设定；如果需要可以打开
        adversaries = self.adversaries(world)

        if shape:
            # Shaping: sheep距离最近的wolf越远越好
            # FIX: 用 agent (sheep自己) 计算距离，不要loop adv
            rew += 0.1 * min([
                np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
                for adv in adversaries
            ])

        if agent.collide:
            for adv in adversaries:
                if self.is_collision(adv, agent):
                    rew -= 10

        # Penalty for collision with landmarks
        for lm in world.landmarks:
            if self.is_collision(agent, lm):
                rew -= 1.0

        # Boundary penalty
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew


    def adversary_reward(self, agent, world):
        """
        Wolf (adversary) 的 reward - FIXED VERSION

        关键修正：
        1. Distance shaping: 只用 agent (当前wolf自己) 到sheep的距离，
        不要 loop 所有 adversaries 求和
        2. Collision reward: 只在 agent 自己抓到时才 +reward，
        避免 free-rider problem
        3. Landmark penalty: 保持不变
        """
        rew = 0
        shape = True
        agents = self.good_agents(world)  # sheep list (通常只有1只)

        # ===== FIX #1: Distance shaping (per-agent, not summed) =====
        # 原bug: for adv in adversaries: rew -= 0.1 * min(dist(a, adv))
        #        三只wolf都拿到 sum-of-distances，credit assignment崩溃
        # FIX:   只用agent自己到最近sheep的距离
        if shape:
            rew -= 0.1 * min([
                np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos)))
                for a in agents
            ])

        # ===== FIX #2: Collision reward (only self, not teammates) =====
        # 原bug: for ag in agents: for adv in adversaries:
        #        if collision(ag, adv): rew += 50
        #        所有 wolf 都拿 +50，无论是不是自己抓到的，free-rider problem
        # FIX:   只在 agent (当前wolf) 自己抓到sheep时 +reward
        if agent.collide:
            for ag in agents:
                if self.is_collision(ag, agent):
                    rew += 50.0  # 自己抓到，主奖励
            # 可选：team reward (撞到的wolf拿大份，没撞到的拿小份)
            # 这能鼓励coordination而不是单兵作战
            # 注释掉下面这段如果不想要 team reward
            else:
                for ag in agents:
                    for adv in self.adversaries(world):
                        if adv is not agent and self.is_collision(ag, adv):
                            rew += 10.0  # 队友抓到，分一点信用
                            break

        # ===== Landmark collision penalty (unchanged) =====
        for lm in world.landmarks:
            if self.is_collision(agent, lm):
                rew -= 1.0

        return rew



    # def agent_reward(self, agent, world):
    #     # Agents are negatively rewarded if caught by adversaries
    #     rew = 0
    #     shape = False #different from openai
    #     adversaries = self.adversaries(world)
    #     if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
    #         for adv in adversaries:
    #             rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
    #     if agent.collide:
    #         for a in adversaries:
    #             if self.is_collision(a, agent):
    #                 rew -= 10
    #     # penalty for collision （new）
    #     for lm in world.landmarks:
    #         if self.is_collision(agent, lm):
    #             rew -= 1.0

    #     # agents are penalized for exiting the screen, so that they can be caught by the adversaries
    #     def bound(x):
    #         if x < 0.9:
    #             return 0
    #         if x < 1.0:
    #             return (x - 0.9) * 10
    #         return min(np.exp(2 * x - 2), 10)
    #     for p in range(world.dim_p):
    #         x = abs(agent.state.p_pos[p])
    #         rew -= bound(x)

    #     return rew

    # def adversary_reward(self, agent, world):
    #     # Adversaries are rewarded for collisions with agents
    #     rew = 0
    #     shape = True #different from openai
    #     agents = self.good_agents(world)
    #     adversaries = self.adversaries(world)
    #     if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
    #         for adv in adversaries:
    #             rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
    #     if agent.collide:
    #         for ag in agents:
    #             for adv in adversaries:
    #                 if self.is_collision(ag, adv):
    #                     rew += 50.0
    #     # penalty for collision （new）
    #     for lm in world.landmarks:
    #         if self.is_collision(agent, lm):
    #             rew -= 1.0

    #     def bound(x):
    #         if x < 0.9:
    #             return 0
    #         if x < 1.0:
    #             return (x - 0.9) * 10
    #         return min(np.exp(2 * x - 2), 10)
    #     for p in range(world.dim_p):
    #         x = abs(agent.state.p_pos[p])
    #         rew -= bound(x)

    #     return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                if self._visible(agent, entity, world):
                    entity_pos.append(entity.state.p_pos - agent.state.p_pos)
                else:
                    entity_pos.append(np.zeros(world.dim_p))
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            if self._visible(agent, other, world):
                other_pos.append(other.state.p_pos - agent.state.p_pos)
            else:
                other_pos.append(np.zeros(world.dim_p))
            if not other.adversary:
                if self._visible(agent, other, world):
                    other_vel.append(other.state.p_vel)
                else:
                    other_vel.append(np.zeros(world.dim_p))
        obs = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel + comm)
        # Pad to uniform obs size across all agents (adversaries observe good agent vel, good agent does not)
        num_adversaries = sum(1 for a in world.agents if a.adversary)
        num_good = sum(1 for a in world.agents if not a.adversary)
        num_agents = len(world.agents)  #
        comm_dim_total = (num_agents - 1) * world.dim_c #
        max_obs_size = (4 + 2 * len([e for e in world.landmarks if not e.boundary]) #
                + 2 * (len(world.agents) - 1) + 2 * num_good + comm_dim_total)
        if len(obs) < max_obs_size:
            obs = np.concatenate([obs, np.zeros(max_obs_size - len(obs))])
        return obs
