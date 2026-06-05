import sys
import os
import numpy as np
import torch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from env.ran_marl_env import RANParallelEnv

def generate_mock_iq_data(filepath, n_samples=1000):
    data = np.random.randn(n_samples * 2).astype(np.float32)
    data.tofile(filepath)

def test_ran_env_integration():
    mock_file = "tests/mock_iq_env.bin"
    generate_mock_iq_data(mock_file, n_samples=2000)
    
    try:
        num_gnbs = 2
        grid_size = (14, 12)
        env = RANParallelEnv(file_path=mock_file, num_gnbs=num_gnbs, grid_size=grid_size)
        observations, _ = env.reset()
        
        # Step through to trigger wrap-around
        for _ in range(5):
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            env.step(actions)
        print("RANParallelEnv integration and wrap-around test successful.")
    finally:
        if os.path.exists(mock_file):
            os.remove(mock_file)

if __name__ == "__main__":
    test_ran_env_integration()
