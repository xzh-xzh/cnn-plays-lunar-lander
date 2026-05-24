# 视觉 PPO 月球登陆器

这是一个基于视觉输入的 `LunarLander-v3` PPO 项目。智能体不读取环境的低维状态，而是读取渲染出来的画面：RGB 帧会被缩放、灰度化、按时间堆叠，然后输入 actor-critic 策略网络，通过 clipped PPO 和 GAE 训练。

## 功能

- `TinyCNNGRU`：小型模型，用于 CPU smoke 测试。
- `TemporalResNetGRU`：ImageNet 预训练 ResNet18 单帧编码器 + GRU 时序头。
- `Conv3DTransformerNet`：轻量 3D 卷积 Transformer 备选模型。
- YAML 配置文件覆盖 smoke 测试、ResNet-GRU 训练和 transformer 训练。
- JSONL 指标日志、checkpoint、可选 MLflow，以及实时 `rollout` / `update` 进度条。
- 默认使用 `SyncVectorEnv`，在 Windows 上比异步子进程环境更稳定。

## 安装

推荐 Python 3.11。

```bash
conda create -n lunar python=3.11 -y
conda activate lunar
pip install -r requirements.txt
```

如果 Linux 上安装 Box2D 失败，先安装系统构建工具，或者用 conda 安装 `swig`：

```bash
conda install -c conda-forge swig -y
pip install -r requirements.txt
```

确认 GPU 是否可用：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 快速验证

```bash
python train.py --config configs/smoke.yaml
```

这个配置会在 CPU 上跑一个很小的 rollout 和一次 PPO 更新，用来确认环境、视觉预处理、GAE、反向传播、日志和进度条都能正常工作。

## 正式训练

```bash
python train.py --config configs/temporal_resnet_gru.yaml --device cuda
```

常用覆盖参数：

```bash
python train.py --config configs/temporal_resnet_gru.yaml --rounds 10 --num-envs 4 --rollout-steps 512
python train.py --config configs/temporal_resnet_gru.yaml --device cpu --rounds 1 --num-envs 1 --rollout-steps 32
python train.py --config configs/smoke.yaml --mlflow
```

远程服务器建议放在 `tmux` 里跑：

```bash
tmux new -s lunar
python -u train.py --config configs/temporal_resnet_gru.yaml --device cuda 2>&1 | tee train.log
```

按 `Ctrl+b`，再按 `d` 可以退出但不中断训练；重新进入：

```bash
tmux attach -t lunar
```

## 当前默认模型

`configs/temporal_resnet_gru.yaml` 默认使用：

```yaml
model_name: "TemporalResNetGRU"
PRETRAINED_BACKBONE: true
FREEZE_RESNET: true
BACKBONE_LEARNING_RATE:
```

含义是：ResNet18 使用 ImageNet 预训练权重，并且默认冻结。训练时只更新后面的 GRU、归一化层和 actor/critic 头部。如果要微调整个 ResNet backbone，改成：

```yaml
FREEZE_RESNET: false
BACKBONE_LEARNING_RATE: 0.00005
```

第一次使用预训练模型时，如果本机或服务器没有缓存 torchvision 的 ResNet18 权重，会自动下载一次。

## 进度输出

训练会在两个阶段之间循环：

- `rollout`：当前策略去玩环境，收集视觉观测、动作、奖励、结束标记、log-prob 和 value 估计。
- `update`：PPO 使用刚收集的数据更新神经网络。

示例：

```text
round=0 rollout [##############################] 1024/1024 100.0%
round=0 update  [##############################] 128/128 100.0%
```

## 输出文件

- 指标日志：`runs/<run-name>/metrics.jsonl`
- 实际配置：`runs/<run-name>/config.json`
- 模型权重：`checkpoints/<run-name>/<round>.pt`

生成的运行日志、checkpoint、MLflow 文件和本地缓存都已被 git 忽略。

## 注意事项

`USE_TORCH_COMPILE` 默认关闭。Windows 上 PyTorch Inductor 需要可用的 Triton；如果环境里没有 Triton，开启后会在第一次 compiled forward 时失败。
