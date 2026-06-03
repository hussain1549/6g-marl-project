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

    def __init__(self, num_gnbs=3, render_mode=None):
        """
        Initialize the RAN MARL environment.

        Args:
            num_gnbs (int): Number of base station agents (gNBs).
            render_mode (str, optional): The render mode to use.
        """
        self.possible_agents = [f"gnb_{i}" for i in range(num_gnbs)]
        self.render_mode = render_mode
        
        # Placeholder for internal state
        self.num_moves = 0
        self.max_moves = 100

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """
        Returns the observation space for a given agent.
        Placeholder: A Box space representing local network metrics.
        """
        # Example: [throughput, average_latency, connected_ues]
        return Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """
        Returns the action space for a given agent.
        Placeholder: A Discrete space for resource allocation levels.
        """
        # Example: 10 different levels of resource allocation/power
        return Discrete(10)

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state.

        Returns:
            observations (dict): Initial observations for each agent.
            infos (dict): Initial info for each agent.
        """
        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)
        
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        
        observations = {
            agent: self.observation_space(agent).sample() 
            for agent in self.agents
        }
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos

    def step(self, actions):
        """
        Performs a step in the environment using the provided actions.

        Args:
            actions (dict): Actions for each active agent.

        Returns:
            observations (dict): New observations for each agent.
            rewards (dict): Rewards for each agent.
            terminations (dict): Termination flags for each agent.
            truncations (dict): Truncation flags for each agent.
            infos (dict): Info for each agent.
        """
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # Update move count
        self.num_moves += 1
        
        # Placeholder logic: Transition state and calculate rewards
        # In a real RAN env, this would involve complex network simulation
        observations = {
            agent: self.observation_space(agent).sample() 
            for agent in self.agents
        }
        
        # Placeholder rewards (e.g., constant or based on action)
        rewards = {agent: 1.0 for agent in self.agents}
        
        terminations = {agent: False for agent in self.agents}
        
        env_truncation = self.num_moves >= self.max_moves
        truncations = {agent: env_truncation for agent in self.agents}
        
        infos = {agent: {} for agent in self.agents}

        if env_truncation:
            self.agents = []

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def render(self):
        """Renders the environment."""
        if self.render_mode is None:
            return
        print(f"Step: {self.num_moves}")

    def close(self):
        """Closes the environment."""
        pass
