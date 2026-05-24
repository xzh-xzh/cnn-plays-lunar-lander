# Vision PPO Lunar Lander

[中文说明](README.zh-CN.md)

Vision-only PPO training for `LunarLander-v3`. The agent learns from rendered frames instead of low-dimensional environment state: RGB frames are resized, converted to grayscale, stacked over time, and passed into an actor-critic policy trained with clipped PPO and GAE.

## Features

- `TinyCNNGRU`: small CPU-friendly model for smoke tests.
- `TemporalResNetGRU`: ImageNet-pretrained ResNet18 frame encoder plus GRU temporal head.
- `Conv3DTransformerNet`: compact 3D-conv transformer alternative.
- YAML configs for smoke runs, ResNet-GRU training, and transformer training.
- JSONL metrics, checkpointing, optional MLflow logging, and live `rollout` / `update` progress bars.
- `SyncVectorEnv` by default for stable Windows execution.

## Installation

Python 3.11 is recommended.

```bash
conda create -n lunar python=3.11 -y
conda activate lunar
pip install -r requirements.txt
```

If Box2D installation fails on Linux, install system build tools first, or install `swig` through conda:

```bash
conda install -c conda-forge swig -y
pip install -r requirements.txt
```

Check GPU availability:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Quick Smoke Run

```bash
python train.py --config configs/smoke.yaml
```

This runs one small CPU rollout and PPO update to verify the environment, visual preprocessing, GAE, backpropagation, logging, and progress bars.

## Full Training

```bash
python train.py --config configs/temporal_resnet_gru.yaml --device cuda
```

Useful overrides:

```bash
python train.py --config configs/temporal_resnet_gru.yaml --rounds 10 --num-envs 4 --rollout-steps 512
python train.py --config configs/temporal_resnet_gru.yaml --device cpu --rounds 1 --num-envs 1 --rollout-steps 32
python train.py --config configs/smoke.yaml --mlflow
```

For remote server training, run inside `tmux`:

```bash
tmux new -s lunar
python -u train.py --config configs/temporal_resnet_gru.yaml --device cuda 2>&1 | tee train.log
```

Detach with `Ctrl+b`, then `d`; reattach with:

```bash
tmux attach -t lunar
```

## Current Default Model

`configs/temporal_resnet_gru.yaml` uses:

```yaml
model_name: "TemporalResNetGRU"
PRETRAINED_BACKBONE: true
FREEZE_RESNET: true
BACKBONE_LEARNING_RATE:
```

This means the ResNet18 backbone starts from ImageNet-pretrained weights and is frozen by default. Training updates the GRU, normalization layer, and actor/critic heads. To fine-tune the ResNet backbone, set:

```yaml
FREEZE_RESNET: false
BACKBONE_LEARNING_RATE: 0.00005
```

The first pretrained run may download torchvision ResNet18 weights if they are not already cached.

## Progress Output

Training alternates between:

- `rollout`: the current policy plays the environment and collects visual observations, actions, rewards, done flags, log-probs, and value estimates.
- `update`: PPO trains the neural network on the collected rollout data.

Example:

```text
round=0 rollout [##############################] 1024/1024 100.0%
round=0 update  [##############################] 128/128 100.0%
```

## Outputs

- Metrics: `runs/<run-name>/metrics.jsonl`
- Resolved config: `runs/<run-name>/config.json`
- Checkpoints: `checkpoints/<run-name>/<round>.pt`

Generated outputs, checkpoints, MLflow files, and local caches are ignored by git.

## Notes

`USE_TORCH_COMPILE` is disabled by default. On Windows, PyTorch Inductor requires a working Triton installation; without Triton, the first compiled forward pass fails.
