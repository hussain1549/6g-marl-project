import torch
import numpy as np
from collections import deque
import random

class RANBuffer:
    def __init__(self, capacity=10000, num_agents=3, obs_shape=(14, 12)):
        self.buffer = deque(maxlen=capacity)
        self.num_agents, self.obs_shape = num_agents, obs_shape

    def add_transition(self, obs, actions, rewards, next_obs, terminations, truncations):
        dones = {a: terminations[a] or truncations[a] for a in obs.keys()}
        self.buffer.append({
            'obs': obs, 'actions': actions, 'rewards': rewards, 'next_obs': next_obs, 'dones': dones,
            'state': self._construct_global_state(obs, rewards),
            'next_state': self._construct_global_state(next_obs, rewards)
        })

    def _construct_global_state(self, observations, rewards):
        agent_ids = sorted(observations.keys())
        flat_obs = torch.cat([o.flatten() if isinstance(o, torch.Tensor) else torch.from_numpy(o).flatten() for o in [observations[aid] for aid in agent_ids]])
        total_r = sum(rewards.values()) if rewards else 0.0
        return torch.cat([flat_obs, torch.tensor([total_r, total_r/len(agent_ids)], dtype=torch.float32)])

    def get_batch(self, batch_size):
        samples = random.sample(self.buffer, min(len(self.buffer), batch_size))
        ids = sorted(samples[0]['obs'].keys())
        batch = {k: [] for k in ['obs', 'actions', 'rewards', 'next_obs', 'dones', 'state', 'next_state']}
        for s in samples:
            batch['obs'].append(torch.stack([s['obs'][i] for i in ids]))
            batch['next_obs'].append(torch.stack([s['next_obs'][i] for i in ids]))
            batch['actions'].append(torch.tensor([s['actions'][i] for i in ids]))
            batch['rewards'].append(torch.tensor([s['rewards'][i] for i in ids], dtype=torch.float32))
            batch['dones'].append(torch.tensor([float(s['dones'][i]) for i in ids]))
            batch['state'].append(s['state']); batch['next_state'].append(s['next_state'])
        return {k: torch.stack(v) for k, v in batch.items()}

class CTDEDataLoader:
    def __init__(self, capacity=10000, num_agents=3, obs_shape=(14, 12)):
        self.buffer = RANBuffer(capacity, num_agents, obs_shape)
    def collect_transition(self, *args): self.buffer.add_transition(*args)
    def sample_batch(self, batch_size): return self.buffer.get_batch(batch_size)
