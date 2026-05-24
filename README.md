# Vision PPO Lunar Lander

[中文说明](README.zh-CN.md)

A compact vision-only PPO implementation for `LunarLander-v3`.

The agent observes rendered RGB frames, resizes them, converts them to grayscale, stacks recent frames, and trains an actor-critic policy with clipped PPO and GAE.

## What is included

- `TinyCNNGRU`: small visual model for smoke tests and CPU runs.
- `TemporalResNetGRU`: ResNet18 frame encoder plus GRU temporal aggregation.
- `Conv3DTransformerNet`: compact 3D-conv transformer policy.
- YAML-driven training configs.
- JSONL metrics and optional MLflow logging.
- Sync vector environments by default for stable Windows execution.

## Quick smoke run

```powershell
D:\xzh\anaconda3\envs\rl\python.exe train.py --config configs\smoke.yaml
```

This runs one small PPO rollout/update on CPU and writes logs under `runs/`.

## Full training

```powershell
D:\xzh\anaconda3\envs\rl\python.exe train.py --config configs\temporal_resnet_gru.yaml
```

Useful overrides:

```powershell
D:\xzh\anaconda3\envs\rl\python.exe train.py --config configs\temporal_resnet_gru.yaml --rounds 10 --device cpu
D:\xzh\anaconda3\envs\rl\python.exe train.py --config configs\smoke.yaml --mlflow
```

## Outputs

- Metrics: `runs/<run-name>/metrics.jsonl`
- Resolved config: `runs/<run-name>/config.json`
- Checkpoints: `checkpoints/<run-name>/<round>.pt`

## Notes

`USE_TORCH_COMPILE` is disabled by default. On Windows, PyTorch Inductor needs a working Triton install, and missing Triton causes the first compiled forward pass to fail.

`TemporalResNetGRU` uses ImageNet-pretrained ResNet18 by default. The first run may download torchvision weights if they are not already cached.

The ResNet backbone is frozen by default with `FREEZE_RESNET: true`; training updates the GRU, normalization layer, and actor/critic heads. Set `FREEZE_RESNET: false` and `BACKBONE_LEARNING_RATE` to a small value if you want to fine-tune the ResNet backbone.
