<div align="center">

# Frame-Based Zero-Shot Semantic Channel Equalization for AI-Native Communications

<a href="https://arxiv.org/abs/2507.17835">
  <img alt="Paper" src="https://img.shields.io/badge/arXiv-2507.17835-b31b1b?logo=arxiv&logoColor=white">
</a>
<a href="https://pytorch.org/get-started/locally/">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.9-ee4c2c?logo=pytorch&logoColor=white">
</a>
<a href="https://www.pytorchlightning.ai/">
  <img alt="Lightning" src="https://img.shields.io/badge/Lightning-2.6-792ee5?logo=pytorchlightning&logoColor=white">
</a>
<a href="https://hydra.cc/">
  <img alt="Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd">
</a>
<a href="https://wandb.ai/site">
  <img alt="Weights & Biases" src="https://img.shields.io/badge/Tracking-W%26B-ffbe00?logo=weightsandbiases&logoColor=white">
</a>
</div>

## Abstract

In future AI-native wireless networks, the presence of mismatches between the latent spaces of independently designed and trained deep neural network (DNN) encoders may impede mutual understanding due to the emergence of semantic channel noise. This undermines the receiver's ability to interpret transmitted representations, thereby reducing overall system performance. To address this issue, we propose the Parseval Frame Equalizer (PFE), a zero-shot, frame-based semantic channel equalizer that aligns latent spaces of heterogeneous encoders without requiring system retraining. PFE enables dynamic signal compression and expansion, mitigating semantic noise while preserving performance on downstream tasks. Building on this capability, we introduce a dynamic optimization strategy that coordinates communication, computation, and learning resources to balance energy consumption, end-to-end (E2E) latency, and task performance in multi-agent semantic communication scenarios. Extensive simulations confirm the effectiveness of our approach in maintaining semantic consistency and meeting long-term constraints on latency and accuracy under diverse and time-varying network conditions.

## Installation

```
git clone https://github.com/SimoneFiorellino/TCCN-frame-based.git
cd TCCN-frame-based

uv sync
uv run pre-commit install
```

## BibTeX

```
@ARTICLE{fiorellino2025framebasedzeroshotsemanticchannel,
      title={Frame-Based Zero-Shot Semantic Channel Equalization for AI-Native Communications}, 
      author={Simone Fiorellino and Claudio Battiloro and Emilio Calvanese Strinati and Paolo Di Lorenzo},
      year={2026},
      journal={IEEE Transactions on Cognitive Communications and Networking}, 
}
```
