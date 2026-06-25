# DAPO Training with Log Probability QA Reward

This repository contains the implementation of **DAPO (Decoupled Alignment Policy Optimization)** training with a novel **Log Probability QA Reward** mechanism.

## Overview

The training framework uses GRPO (Group Relative Policy Optimization) for advantage estimation combined with PPO dual-clip policy loss. The key innovation is the log probability QA reward, which measures the improvement in predicting user intent when the model asks clarifying questions.

### Key Features

- **GRPO Advantage Estimation**: Group-normalized rewards for stable training
- **PPO Dual-Clip Loss**: Asymmetric clipping for better policy updates
- **Entropy Bonus**: Encourages exploration during training
- **Log Probability QA Reward**: Novel reward based on question-answer effectiveness
- **Flash Attention 2**: Memory-efficient attention for long sequences

## Project Structure

```
code-release/
├── scripts/
│   └── train_dapo.sh              # Main training script
├── reward/
│   ├── __init__.py
│   ├── simple_reward.py           # Simple reward function
│   ├── main_trainer.py            # Main trainer entry point
│   └── trainer_patch.py           # Trainer patch for QA reward
├── verl_recipe/
│   ├── __init__.py
│   ├── dapo_ray_trainer.py        # DAPO trainer (from verl)
│   └── config/
│       └── dapo_trainer.yaml      # DAPO configuration
├── data/
│   ├── train.parquet              # Training data
│   └── val.parquet                # Validation data
├── logs/                          # Log output directory
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- [verl](https://github.com/volcengine/verl) - Volcano Engine Reinforcement Learning framework
- Flash Attention 2
- Ray
- vLLM

## Installation

1. Install verl framework:
```bash
git clone https://github.com/volcengine/verl.git
cd verl
pip install -e .
```

2. Install additional dependencies:
```bash
pip install flash-attn --no-build-isolation
pip install ray vllm
```

## Configuration

Before running, you need to modify the following paths in `scripts/train_dapo.sh`:

```bash
# Model path - point to your model
MODEL_PATH="/path/to/your/model"

# Verl installation path
VERL_DIR="/path/to/verl"
```

## Usage

### Basic Training

Run with default configuration (Config A - conservative):
```bash
cd scripts
bash train_dapo.sh A
```

### Available Configurations

| Config | Learning Rate | Clip High | Entropy Coeff | Description |
|--------|--------------|-----------|---------------|-------------|
| A | 5e-7 | 0.28 | 0.001 | Conservative, stable |
| B | 1e-6 | 0.30 | 0.01 | Balanced |
| C | 2e-6 | 0.35 | 0.02 | Aggressive exploration |

### Run Different Configurations

```bash
# Conservative configuration
bash train_dapo.sh A

# Balanced configuration
bash train_dapo.sh B

# Aggressive configuration
bash train_dapo.sh C
```

### τ-Bench Base Qwen Baseline

To reproduce the paper's `Qwen3-8B / None` baseline, run the standalone
τ-Bench wrapper:

```bash
bash scripts/base_qwen_eval.sh retail airline
```

By default it uses:

- agent temperature `0.01`
- user temperature `1.0`
- seeds `0`, `1`, and `2`
- Qwen3 reasoning enabled by default at the vLLM server layer

The script writes seed-specific result directories under `results/` and a
summary file at `results/base_qwen_paper_summary.json`.

## Algorithm Details

### Loss Function

The total loss is:
```
L = L_PPO_DualClip(r, A) - λ_ent * H(π_θ)
```

Where:
- `L_PPO_DualClip`: PPO dual-clip policy gradient loss
- `H(π_θ)`: Policy entropy
- `λ_ent`: Entropy coefficient

### PPO Dual-Clip

The dual-clip mechanism uses:
- `clip_ratio_low = 0.2`: Lower clipping bound
- `clip_ratio_high = 0.28-0.35`: Upper clipping bound (configurable)
- `clip_ratio_c = 10.0`: Secondary clipping for negative advantages

### Advantage Estimation

Uses GRPO with group normalization:
```
A_i = (r_i - μ_group) / (σ_group + ε)
```

### Log Probability QA Reward

The reward measures the improvement in predicting user intent:
```
reward = log_prob(response | prompt_with_QA) - log_prob(response | prompt_without_QA)
```

This encourages the model to ask clarifying questions that actually help understand user intent.

## Output

Training outputs are saved to:
```
outputs/<config-name>/
├── train.log        # Full training log
├── metrics.log      # Key metrics extracted
└── checkpoints/     # Model checkpoints
```

## Metrics Logged

- `log_prob_reward/mean`: Mean log probability reward
- `log_prob_reward/valid_ratio`: Ratio of valid samples
- `actor/entropy`: Policy entropy
- `actor/pg_loss`: Policy gradient loss
- `actor/grad_norm`: Gradient norm
- `advantages/mean`: Mean advantage values

## Data Format

The training data should be in parquet format with the following fields:
- `prompt`: Input prompt text
- `reward_model.ground_truth_clean`: Ground truth for reward computation

## GPU Requirements

Default configuration is optimized for 4x RTX A6000 (48GB each). Adjust the following parameters for different hardware:

```bash
n_gpus_per_node=4
gpu_memory_utilization=0.60
ppo_micro_bsz_per_gpu=1
```

## Customization

### Custom Reward Function

Modify `reward/simple_reward.py` to implement your own reward logic:

```python
def compute_score(data_source, solution_str, ground_truth=None, extra_info=None, **kwargs):
    # Your custom reward logic here
    return score
```

### Custom User Response Simulation

Modify `reward/trainer_patch.py` function `simulate_user_response()` to use:
- A local LLM for response generation
- An external API
- Rule-based response generation

## Troubleshooting

### Out of Memory

Reduce batch sizes:
```bash
train_prompt_bsz=8
ppo_micro_bsz_per_gpu=1
```

### Slow Training

Increase generation batch size (may increase memory):
```bash
gen_prompt_bsz=2
```

### Ray Cluster Issues

Reset Ray cluster:
```bash
ray stop --force
ray start --head
```

## License

This project is licensed under the **Apache License 2.0**.

### Third-Party Code

This project includes code derived from the [verl](https://github.com/volcengine/verl) project:

- `verl_recipe/dapo_ray_trainer.py` - Derived from `recipe/dapo/dapo_ray_trainer.py`
- Copyright 2024 Bytedance Ltd. and/or its affiliates
- Licensed under Apache License 2.0

The verl framework is required as a dependency for core RL training functionality.

## Citation

If you use this code, please cite:

```bibtex
@misc{dapo-qa-reward,
  title={DAPO Training with Log Probability QA Reward},
  year={2024},
}
```

## Acknowledgments

- [verl](https://github.com/volcengine/verl) - The underlying RL training framework
- [vLLM](https://github.com/vllm-project/vllm) - Fast LLM inference
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) - Memory-efficient attention
