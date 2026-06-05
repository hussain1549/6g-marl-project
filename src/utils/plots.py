import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_academic_plots(training_log='logs/training_history.csv', baseline_log='logs/baseline_history.csv'):
    if not os.path.exists(training_log) or not os.path.exists(baseline_log): return
    train_df, base_df = pd.read_csv(training_log), pd.read_csv(baseline_log)
    window = 10
    for df in [train_df, base_df]:
        df['throughput_smooth'] = df['mean_throughput'].rolling(window=window, min_periods=1).mean()
        df['interference_smooth'] = df['mean_interference'].rolling(window=window, min_periods=1).mean()
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({'font.size': 12, 'figure.dpi': 300, 'axes.labelsize': 14, 'axes.titlesize': 16})
    
    # Throughput Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_df['iteration'], train_df['throughput_smooth'], label='MARL (CTDE)', color='blue', linewidth=2.5)
    plt.plot(base_df['iteration'], base_df['throughput_smooth'], label='Baseline (PF)', color='red', linestyle='--', linewidth=2.0)
    plt.xlabel('Training Iteration Milestones'); plt.ylabel('Spectral Efficiency Log-Capacity'); plt.title('System Throughput Comparison'); plt.legend(); plt.tight_layout()
    os.makedirs('plots', exist_ok=True); plt.savefig('plots/throughput_comparison.pdf'); plt.savefig('plots/throughput_comparison.svg')
    
    # Interference Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_df['iteration'], train_df['interference_smooth'], label='MARL (CTDE)', color='purple', linewidth=2.5)
    plt.plot(base_df['iteration'], base_df['interference_smooth'], label='Baseline (PF)', color='darkorange', linestyle='--', linewidth=2.0)
    plt.xlabel('Training Iteration Milestones'); plt.ylabel('Inter-Cell Interference Overhead'); plt.title('Interference Mitigation Comparison'); plt.legend(); plt.tight_layout()
    plt.savefig('plots/interference_mitigation.pdf'); plt.savefig('plots/interference_mitigation.svg')

if __name__ == "__main__":
    generate_academic_plots()
7. Core Verification: tests/test_dataloader.py
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
