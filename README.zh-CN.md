# 视觉 PPO 月球登陆器

这是一个基于视觉输入的 `LunarLander-v3` PPO 实现。

智能体不读取环境的低维状态，而是读取渲染出来的 RGB 画面，将画面缩放、灰度化、堆叠最近若干帧，然后用 actor-critic 策略通过 clipped PPO 和 GAE 进行训练。

## 包含内容

- `TinyCNNGRU`：小型视觉模型，用于快速 smoke 测试和 CPU 运行。
- `TemporalResNetGRU`：ResNet18 单帧编码器 + GRU 时序聚合，是主要训练模型。
- `Conv3DTransformerNet`：轻量 3D 卷积 Transformer 策略网络。
- YAML 配置驱动训练参数。
- JSONL 指标日志，可选 MLflow 记录。
- 默认使用 `SyncVectorEnv`，在 Windows 上比异步子进程环境更稳定。

## 快速验证

在当前 `rl` conda 环境里运行：

```powershell
D:\xzh\anaconda3\envs\rl\python.exe train.py --config configs\smoke.yaml
```

这个配置会在 CPU 上跑一个很小的 rollout 和一次 PPO 更新，用来确认视觉观测、GAE、反向传播和日志链路都可用。

## 正式训练

```powershell
D:\xzh\anaconda3\envs\rl\python.exe train.py --config configs\temporal_resnet_gru.yaml
```

常用覆盖参数：

```powershell
D:\xzh\anaconda3\envs\rl\python.exe train.py --config configs\temporal_resnet_gru.yaml --rounds 10 --device cpu
D:\xzh\anaconda3\envs\rl\python.exe train.py --config configs\smoke.yaml --mlflow
```

## 输出文件

- 指标日志：`runs/<run-name>/metrics.jsonl`
- 实际配置：`runs/<run-name>/config.json`
- 模型权重：`checkpoints/<run-name>/<round>.pt`

## 主要配置

- `NUM_ENVS`：并行环境数量。
- `VECTOR_ENV`：`sync` 或 `async`。Windows 上建议保留 `sync`。
- `FRAME_SIZE`：渲染画面缩放后的边长。
- `STACK_SIZE`：堆叠帧数。
- `model_name`：可选 `TinyCNNGRU`、`TemporalResNetGRU`、`Conv3DTransformerNet`。
- `NUM_STEPS_PER_ROLLOUT`：每轮 PPO 采样步数。
- `PPO_EPOCHS`、`BATCH_SIZE`、`CLIP_EPSILON`、`GAE_LAMBDA`：PPO 训练超参数。

## 注意事项

`USE_TORCH_COMPILE` 默认关闭。当前 Windows 环境下 PyTorch Inductor 需要可用的 Triton；如果环境里没有 Triton，开启后会在第一次 compiled forward 时失败。

`TemporalResNetGRU` 默认使用 ImageNet 预训练 ResNet18。首次运行时，如果本机或服务器没有缓存 torchvision 权重，会自动下载一次。

ResNet backbone 默认冻结，也就是配置里的 `FREEZE_RESNET: true`。训练时只更新后面的 GRU、归一化层和 actor/critic 头部。如果要微调整个 ResNet，把 `FREEZE_RESNET` 改成 `false`，并给 `BACKBONE_LEARNING_RATE` 设置一个较小学习率。

如果只是确认项目能跑，请先使用 `configs/smoke.yaml`。如果要长时间训练，再使用 `configs/temporal_resnet_gru.yaml` 并根据显存调整 `NUM_ENVS`、`BATCH_SIZE` 和 `NUM_STEPS_PER_ROLLOUT`。
