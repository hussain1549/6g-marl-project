import numpy as np
import pandas as pd
import os
from env.ran_marl_env import RANParallelEnv

class ProportionalFairScheduler:
    def __init__(self, num_agents, n_actions=10, window_size=20):
        self.num_agents, self.n_actions, self.window_size = num_agents, n_actions, window_size
        self.avg_throughput = {f"gnb_{i}": 1.0 for i in range(num_agents)}

    def select_actions(self, observations, agents):
        actions = {}
        for a in agents:
            r_curr = np.mean(observations[a].cpu().numpy() if hasattr(observations[a], 'cpu') else observations[a])
            actions[a] = int(np.clip((r_curr / self.avg_throughput[a]) * 10, 0, self.n_actions - 1))
        return actions

    def update_stats(self, throughputs):
        alpha = 1.0 / self.window_size
        for a, r in throughputs.items(): self.avg_throughput[a] = (1 - alpha) * self.avg_throughput[a] + alpha * r

def evaluate_baseline(file_path, num_gnbs=3, steps=500):
    env = RANParallelEnv(file_path=file_path, num_gnbs=num_gnbs)
    scheduler = ProportionalFairScheduler(num_gnbs)
    history = []; obs, _ = env.reset()
    for t in range(steps):
        actions = scheduler.select_actions(obs, env.possible_agents)
        obs, rewards, term, trunc, infos = env.step(actions)
        scheduler.update_stats({a: infos[a]['throughput'] for a in env.possible_agents})
        if (t + 1) % 100 == 0:
            m_t, m_i = np.mean([infos[a]['throughput'] for a in env.possible_agents]), np.mean([infos[a]['interference_penalty'] for a in env.possible_agents])
            history.append({'iteration': t+1, 'mean_reward': np.mean(list(rewards.values())), 'mean_throughput': m_t, 'mean_interference': m_i})
            pd.DataFrame(history).to_csv('logs/baseline_history.csv', index=False)

if __name__ == "__main__":
    mock_file = "tests/baseline_mock.bin"
    if not os.path.exists(mock_file): np.random.randn(100000).astype(np.float32).tofile(mock_file)
    try: evaluate_baseline(mock_file)
    finally:
        if os.path.exists(mock_file): os.remove(mock_file)
