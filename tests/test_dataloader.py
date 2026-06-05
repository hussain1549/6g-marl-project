import torch
from pipeline.marl_dataloader import CTDEDataLoader
def test_dataloader_scaling():
    obs_shape = (14, 12)
    for num_agents in [2, 3, 5]:
        loader = CTDEDataLoader(num_agents=num_agents, obs_shape=obs_shape)
        obs = {f"gnb_{i}": torch.randn(obs_shape) for i in range(num_agents)}
        loader.collect_transition(obs, obs, {f"gnb_{i}": 1.0 for i in range(num_agents)}, 
                                 obs, {f"gnb_{i}": False for i in range(num_agents)}, 
                                 {f"gnb_{i}": False for i in range(num_agents)})
        batch = loader.sample_batch(batch_size=1)
        assert batch['state'].shape == (1, num_agents * (14 * 12) + 2)
        print(f"Verified scaling for {num_agents} agents.")

if __name__ == "__main__":
    test_dataloader_scaling()
