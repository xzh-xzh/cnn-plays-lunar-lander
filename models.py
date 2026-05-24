from __future__ import annotations

import math

import torch
import torch.nn as nn


class TinyCNNGRU(nn.Module):
    """Fast visual policy used for smoke tests and CPU-friendly training."""

    def __init__(
        self,
        num_actions: int = 4,
        num_frames: int = 16,
        hidden_size: int = 256,
        num_layers: int = 1,
        *_: object,
        **__: object,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)
        self._init_heads()

    def _init_heads(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        x = x.squeeze(1)
        x = x.reshape(batch_size * self.num_frames, 1, x.size(2), x.size(3))
        features = self.backbone(x)
        features = features.reshape(batch_size, self.num_frames, -1)
        output, _ = self.gru(features)
        features = self.norm(output[:, -1, :])
        return self.actor(features), self.critic(features)


class TemporalResNetGRU(nn.Module):
    """ResNet18 frame encoder plus GRU temporal aggregation."""

    def __init__(
        self,
        num_actions: int = 4,
        num_frames: int = 16,
        hidden_size: int = 512,
        num_layers: int = 2,
        pretrained: bool = False,
        *_: object,
        **__: object,
    ) -> None:
        super().__init__()
        import torchvision

        self.num_frames = num_frames
        weights = (
            torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )
        self.backbone = torchvision.models.resnet18(weights=weights)
        self.backbone.fc = nn.Identity()

        if pretrained:
            grayscale_weight = self.backbone.conv1.weight.data.mean(
                dim=1, keepdim=True
            )
        else:
            grayscale_weight = None

        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        if grayscale_weight is not None:
            self.backbone.conv1.weight.data.copy_(grayscale_weight)

        self.gru = nn.GRU(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)
        self._init_recurrent_and_heads()

    def _init_recurrent_and_heads(self) -> None:
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        x = x.squeeze(1)
        x = x.reshape(batch_size * self.num_frames, 1, x.size(2), x.size(3))
        features = self.backbone(x)
        features = features.reshape(batch_size, self.num_frames, -1)
        output, _ = self.gru(features)
        features = self.norm(output[:, -1, :])
        return self.actor(features), self.critic(features)


class Conv3DTransformerNet(nn.Module):
    def __init__(self, num_actions: int = 4, num_frames: int = 16) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.embed_dim = 128

        self.conv3d_stem = nn.Conv3d(
            1, self.embed_dim, kernel_size=(4, 16, 16), stride=(4, 16, 16)
        )
        seq_len = (num_frames // 4) * 8 * 8
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, self.embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=4,
            dim_feedforward=256,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.actor = nn.Linear(self.embed_dim, num_actions)
        self.critic = nn.Linear(self.embed_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv3d_stem(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, : x.size(1)]
        x = self.transformer(x)
        x = self.norm(x.mean(dim=1))
        return self.actor(x), self.critic(x)


def build_model(
    model_name: str,
    num_actions: int,
    num_frames: int,
    hidden_size: int,
    num_layers: int,
    pretrained_backbone: bool,
) -> nn.Module:
    if model_name == "TinyCNNGRU":
        return TinyCNNGRU(num_actions, num_frames, hidden_size, num_layers)
    if model_name == "TemporalResNetGRU":
        return TemporalResNetGRU(
            num_actions,
            num_frames,
            hidden_size,
            num_layers,
            pretrained=pretrained_backbone,
        )
    if model_name == "Conv3DTransformerNet":
        return Conv3DTransformerNet(num_actions, num_frames)
    raise ValueError(f"Unknown model_name: {model_name}")
