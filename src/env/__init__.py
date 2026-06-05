"""
Multi-Agent Reinforcement Learning Environments for 6G RAN Slicing.
Exposes the core PettingZoo ParallelEnv wrapper.
"""

from .ran_marl_env import RANParallelEnv

# Clearly defines the public API for the env package
__all__ = ["RANParallelEnv"]
