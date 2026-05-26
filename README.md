# Selective Deep Thinking for Language Agents

Anonymous submission.

## Table of Contents

- [Installation](#installation)
  - [1. Create Environment](#1-create-environment)
  - [2. Install ALFWorld](#2-install-alfworld)
- [Run Training](#run-training)
  - [Sel.-Action (RL-action)](#sel-action-rl-action)
  - [Sel.-Gate (RL-gate)](#sel-gate-rl-gate)
  - [Reward Design](#reward-design)
- [Key Hyperparameters](#key-hyperparameters)

---

## Installation

### 1. Create Environment

```bash
conda create -n verl-alfworld python=3.12 -y
conda activate verl-alfworld
```

Install the core ML stack:

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install vllm==0.8.5
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .
```

### 2. Install ALFWorld

```bash
pip install gymnasium==1.3.0
pip install alfworld==0.4.2
```

Download PDDL & game files and the pre-trained MaskRCNN detector
(stored in `~/.cache/alfworld/`):

```bash
alfworld-download -f
```

Verify with a quick smoke test:

```bash
alfworld-play-tw
```

---

## Run Training

All scripts are under `examples/grpo_trainer/`.
The two launcher scripts set mode-specific variables and start
`run_alfworld_grpo_selective.sh` in the background.

Set `MODEL_PATH` to the Qwen3-8B base model path before launching.
Logs and checkpoints are written to `${VERL_TRAIN_DIR:-<project-root>}/` (defaults to the project root directory).

### Sel.-Action (RL-action)

Thinking mode: `dt_action`. Trains with GRPO + LoRA on top of Qwen3-8B base.

```bash
export MODEL_PATH=/path/to/Qwen3-8B
bash examples/grpo_trainer/launch_grpo_selective_action.sh
```

Follow the log:

```bash
tail -f logs/grpo/grpo_selective_action_<timestamp>.log
```

### Sel.-Gate (RL-gate)

Thinking mode: `selective_gate`. Same setup, different thinking mode.

```bash
export MODEL_PATH=/path/to/Qwen3-8B
bash examples/grpo_trainer/launch_grpo_selective_gate.sh
```

Follow the log:

```bash
tail -f logs/grpo/grpo_selective_gate_<timestamp>.log
```

### Reward Design

The reward function is implemented in `agent_system/reward_manager/episode.py`.
It combines three components:

- **Task reward**: binary success signal (1.0 / 0.0) from the ALFWorld environment
- **Token-efficiency penalty**: once the training success rate exceeds 0.60, applies a penalty proportional to total token usage (`won × tokens/budget × coef`) to all successful episodes — fewer tokens yield a smaller penalty, creating a continuous gradient signal across the group
- **DT quality penalty**: penalises deep-thinking calls that exceed an adaptive reference count (initialised at 5, updated via a sliding window over winning episodes)

---

## Key Hyperparameters

| Parameter | Value |
|---|---|
| Base model | Qwen3-8B |
| LoRA rank / alpha | 32 / 64 |
| Train batch size | 8 |
| Group size | 8 |
| PPO mini-batch size | 32 |
| Learning rate | 1e-6 |
| KL coefficient | 0.01 (`low_var_kl`) |
| Invalid-action penalty | 0.2 |
| Max steps / episode | 30 |
| Total epochs | 150 |
| Hardware | 2 × NVIDIA A800 (80 GB) |

Full configuration: `examples/grpo_trainer/run_alfworld_grpo_selective.sh`.
