import functools
import gymnasium
import numpy as np
from gymnasium.spaces import Discrete, Box
from pettingzoo import ParallelEnv
from gymnasium.utils import seeding

class RANParallelEnv(ParallelEnv):
    """
    A multi-agent reinforcement learning (MARL) environment wrapper for Radio Access Network (RAN)
    resource allocation, using the PettingZoo ParallelEnv API.

    In this environment, each agent represents a Base Station (gNB) that makes local decisions
    about resource allocation (e.g., resource block assignment, power control) to optimize
    network performance metrics like throughput and latency.

    Multi-Agent Execution Flow:
    1. Initialization: The environment is initialized with a configurable number of gNB agents.
    2. Reset: The environment state is reset, and initial observations are returned for all gNBs.
    3. Step: All gNB agents provide their actions simultaneously in each timestep.
       - The environment collects actions from all agents as a dictionary.
       - The global state transitions based on the joint action of all agents.
       - Rewards and new observations are computed for each agent based on the new state.
       - Terminations and truncations are checked for each agent.
    4. Observation/Action Spaces: Each agent has its own observation and action space,
       typically defined by its local network conditions and supported resource allocation actions.
    """

    metadata = {"render_modes": ["human"], "name": "ran_marl_v0"}

    def __init__(self, file_path, num_gnbs=3, grid_size=(14, 12), fft_size=64, alpha=1.0, beta=0.5):
        self.num_gnbs = num_gnbs
        self.possible_agents = [f"gnb_{i}" for i in range(num_gnbs)]
        self.agents = self.possible_agents[:]
        self.alpha, self.beta = alpha, beta
        self.noise_floor, self.coupling_factor = 1e-5, 0.2
        self.streamer = RANDataStreamer(filepath=file_path, num_gnbs=num_gnbs, grid_size=grid_size, fft_size=fft_size)
        self.action_spaces = {agent: Discrete(10) for agent in self.possible_agents}
        self.observation_spaces = {
            agent: Box(low=-1, high=1, shape=self.streamer.grid_size, dtype=np.float32) 
            for agent in self.possible_agents
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent): return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent): return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        observations = self.streamer.get_next_batch()
        return observations, {agent: {} for agent in self.agents}

    def _compute_rewards(self, observations, actions):
        obs_array = np.stack([observations[agent].cpu().numpy() if hasattr(observations[agent], 'cpu') else observations[agent] for agent in self.possible_agents])
        action_array = np.array([actions[agent] for agent in self.possible_agents]); power_levels = (action_array + 1) / 10.0
        signal_power = power_levels[:, np.newaxis, np.newaxis] * obs_array
        total_power = np.sum(power_levels)
        interference_power = self.coupling_factor * obs_array * (total_power - power_levels)[:, np.newaxis, np.newaxis]
        sinr = signal_power / (interference_power + self.noise_floor)
        throughput = np.sum(np.log2(1 + sinr), axis=(1, 2))
        other_obs_sum = np.sum(obs_array, axis=0) - obs_array
        interference_penalty = self.beta * np.sum(power_levels[:, np.newaxis, np.newaxis] * self.coupling_factor * other_obs_sum, axis=(1, 2))
        reward_values = self.alpha * throughput - interference_penalty
        rewards = {agent: float(reward_values[i]) for i, agent in enumerate(self.possible_agents)}
        metrics = {agent: {'throughput': float(throughput[i]), 'interference_penalty': float(interference_penalty[i])} for i, agent in enumerate(self.possible_agents)}
        return rewards, metrics

    def step(self, actions):
        observations = self.streamer.get_next_batch()
        if self.streamer.has_wrapped: print("[INFO] RANDataStreamer has wrapped around the data file.")
        rewards, metrics = self._compute_rewards(observations, actions)
        return observations, rewards, {a: False for a in self.agents}, {a: False for a in self.agents}, {a: metrics[a] for a in self.agents}

    def render(self): pass
    def close(self): pass

