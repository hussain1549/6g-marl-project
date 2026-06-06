![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/pytorch-v2.0%2B-orange)
![RAN Architecture](https://img.shields.io/badge/O--RAN-KPM%20Compliant-purple)
![License](https://img.shields.io/badge/license-MIT-green)
# 6G Multi-Agent Reinforcement Learning (MARL) for Intelligent RAN Resource Allocation

An advanced, end-to-end Multi-Agent Reinforcement Learning (MARL) framework designed to optimize resource allocation, maximize spectral efficiency, and mitigate inter-cell interference in next-generation (5G/6G) Radio Access Networks (RAN). 

This repository implements a **Centralized Training with Decentralized Execution (CTDE)** architecture using a Multi-Agent Advantage Actor-Critic (MAA2C) approach. The framework is trained and validated on empirical, bare-metal network telemetry logs collected from an OpenAirInterface (OAI) 5G gNodeB deployment.

---

## 📌 Project Overview

As telecommunication systems move toward 6G, traditional heuristic resource scheduling algorithms struggle to manage high-density cell deployments, leading to severe Inter-Cell Interference (ICI). This project solves that problem by treating individual gNodeBs (base stations) as collaborative deep reinforcement learning agents. 

By utilizing a CTDE paradigm, the agents learn a globally coordinated scheduling policy during training by sharing a centralized critic network. During execution, the individual actor networks handle dynamic resource allocation independently in real-time, completely bypassing network communication overhead.

### 🔑 Key Features
* **CTDE Architecture:** Implements a centralized multi-agent actor-critic network utilizing PyTorch.
* **Empirical 5G/6G Grounding:** Replaces standard random noise simulations with an authentic O-RAN KPM (Key Performance Measurement) xApp dataset tracking real-world user throughput and Physical Resource Block (PRB) utilization profiles over a 5-week traffic logging window.
* **Proportional Fairness (PF) Baseline:** Includes a fully integrated traditional communication scheduling baseline to rigorously benchmark performance metrics.
* **Automated Data Pipeline:** Built-in sequential interleaving telemetry parser that converts raw text logs into high-throughput flat binary streams optimized for tensor dataloading.

---

## 🧬 System Architecture

The repository is built around a modular, highly scalable machine learning pipeline:

1. **Environment (`env/ran_marl_env.py`):** A parallel multi-agent environment tracking dynamic channel state matrices, calculating interference penalties, and serving physical throughput actions.
2. **Pipeline (`pipeline/marl_dataloader.py`):** An optimized experience replay buffer sampling continuous trajectories for centralized state evaluations.
3. **Core Model (`src/train.py`):** * **Actor Networks:** Light, fast neural layers running independently per agent, outputting a localized resource action distribution based on immediate observation space.
   * **Critic Network:** A larger centralized network evaluating global state vectors (concatenated agent grids + macro telemetry markers) to compute precise advantage signals.

---

## 📊 Experimental Results & Performance

The framework was trained over a 1,000-iteration milestone horizon against a traditional Proportional Fairness (PF) baseline. The results highlight massive empirical optimizations:

### 1. Inter-Cell Interference Mitigation
The MARL framework successfully learns collaborative spatial-frequency multiplexing patterns. The multi-agent coordination layers suppress cell collision frequencies down to a stable baseline of **~1.0**, whereas the traditional PF heuristic encounters unmanaged scheduling overlap, generating massive systemic interference overhead peaking near **15.0**.

### 2. Spectral Efficiency Log-Capacity
By actively learning dynamic traffic variations from the OAI KPM telemetry log, the MARL framework achieves superior network exploitation. The trained model reaches a stable spectral efficiency log-capacity exceeding **300**, delivering a massive performance lift over the static PF baseline which converges and plateaus early at **222**.

> *Note: Due to its deterministic and static nature, the traditional baseline algorithm achieves full operational convergence within 500 milestones, emphasizing the continuous learning and superior adaptive potential of the deep MARL framework.*

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.10+
* PyTorch
* Pandas
* Numpy

### Installation
1. Clone this repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/6g-marl-ran.git](https://github.com/YOUR_USERNAME/6g-marl-ran.git)
   cd 6g-marl-ran

Acknowledggments
Ground data provided via an open-source, bare-metal OpenAirInterface (OAI) 5G RAN setup using an O-RAN compliant KPM xApp logging structure.

---

### 💡 Recommendation for Your Repository Layout:
To make your profile look visually stunning right away, save your two generated evaluation figures (`interference_mitigation.pdf` and `throughput_comparison.pdf`) as `.png` files, place them inside a folder named `docs/` or `assets/`, and insert them directly into the **Experimental Results** section of the Markdown text using:
```markdown
![Throughput Comparison](docs/throughput_comparison.png)
![Interference Mitigation](docs/interference_mitigation.png)

### DATASET
Data set is picked from this website as a baseline to compare with our work> https://zenodo.org/records/17935683

## 📑 Citation

If you utilize this framework or the OAI KPM data pipeline configuration in your research, please cite this repository:

```bibtex
@misc{6gmarlran2026,
  author = {Hussain Ahmad},
  title = {6G Multi-Agent Reinforcement Learning for Intelligent RAN Resource Allocation},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{[https://github.com/hussain1549/6g-marl-ran](https://github.com/hussain1549/6g-marl-ran)}}
}
