import sys
import os
import numpy as np
import torch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from env.ran_marl_env import RANParallelEnv

def generate_mock_iq_data(filepath, n_samples=2000):
    data = np.random.randn(n_samples * 2).astype(np.float32)
    data.tofile(filepath)

def test_rewards():
    mock_file = "tests/mock_iq_reward.bin"
    generate_mock_iq_data(mock_file)
    
    try:
        env = RANParallelEnv(file_path=mock_file, num_gnbs=2)
        env.reset()
        
        # Compare low vs high power rewards
        _, r_low, _, _, _ = env.step({"gnb_0": 0, "gnb_1": 0})
        _, r_high, _, _, _ = env.step({"gnb_0": 9, "gnb_1": 9})
        
        assert r_low["gnb_0"] != r_high["gnb_0"]
        print("Wireless reward dynamic logic test successful.")
    finally:
        if os.path.exists(mock_file):
            os.remove(mock_file)

if __name__ == "__main__":
    test_rewards()
