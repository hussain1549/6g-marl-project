import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from env.ran_marl_env import RANParallelEnv
from pipeline.marl_dataloader import CTDEDataLoader

class ActorNetwork(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super(ActorNetwork, self).__init__()
        self.input_dim = obs_shape[0] * obs_shape[1]
        self.net = nn.Sequential(nn.Linear(self.input_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_actions), nn.Softmax(dim=-1))
    def forward(self, x): return self.net(x.view(x.size(0), -1))

class CriticNetwork(nn.Module):
    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x): return self.net(x)

def train_ctde(file_path, num_gnbs=3, epochs=500, batch_size=32):
    env = RANParallelEnv(file_path=file_path, num_gnbs=num_gnbs)
    obs_shape, n_actions = env.streamer.grid_size, env.action_space("gnb_0").n
    state_dim = num_gnbs * (obs_shape[0] * obs_shape[1]) + 2
    actors = {a: ActorNetwork(obs_shape, n_actions) for a in env.possible_agents}
    critic = CriticNetwork(state_dim)
    actor_opts = {a: optim.Adam(actors[a].parameters(), lr=1e-4) for a in env.possible_agents}
    critic_opt = optim.Adam(critic.parameters(), lr=5e-4)
    dataloader = CTDEDataLoader(num_agents=num_gnbs, obs_shape=obs_shape)
    history = []; obs, _ = env.reset()
    for t in range(epochs):
        actions = {a: torch.distributions.Categorical(actors[a](obs[a])).sample().item() for a in env.possible_agents}
        n_obs, rewards, term, trunc, infos = env.step(actions)
        dataloader.collect_transition(obs, actions, rewards, n_obs, term, trunc); obs = n_obs
        if len(dataloader.buffer.buffer) >= batch_size:
            batch = dataloader.sample_batch(batch_size); targets = batch['rewards'].mean(dim=1) + 0.99 * critic(batch['next_state']).squeeze(-1)
            critic_loss = F.mse_loss(critic(batch['state']).squeeze(-1), targets.detach())
            critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()
            adv = (targets - critic(batch['state']).squeeze(-1)).detach()
            for i, a in enumerate(env.possible_agents):
                log_p = torch.distributions.Categorical(actors[a](batch['obs'][:, i])).log_prob(batch['actions'][:, i])
                policy_loss = -(log_p * adv).mean(); actor_opts[a].zero_grad(); policy_loss.backward(); actor_opts[a].step()
        if (t + 1) % 100 == 0:
            mean_t = np.mean([infos[a]['throughput'] for a in env.possible_agents]); mean_i = np.mean([infos[a]['interference_penalty'] for a in env.possible_agents])
            history.append({'iteration': t+1, 'mean_reward': np.mean(list(rewards.values())), 'mean_throughput': mean_t, 'mean_interference': mean_i})
            pd.DataFrame(history).to_csv('logs/training_history.csv', index=False)

if __name__ == "__main__":
    iq = "tests/train_mock.bin"
    if not os.path.exists(iq): np.random.randn(100000).astype(np.float32).tofile(iq)
    try: train_ctde(iq)
    finally:
        if os.path.exists(iq): os.remove(iq)
