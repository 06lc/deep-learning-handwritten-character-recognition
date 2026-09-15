"""识别模型定义和模型工厂。"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SafeBatchNorm1d(nn.BatchNorm1d):
    """论文的 BN；batch=1 时使用运行统计，避免单样本烟测崩溃。"""

    def forward(self, inputs: Tensor) -> Tensor:
        if self.training and inputs.shape[0] == 1:
            return F.batch_norm(
                inputs,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                self.momentum,
                self.eps,
            )
        return super().forward(inputs)

# 当前最佳模型 D1 使用的是 HCCR9Layer；下面的 ResidualBlock 只服务于旧的
# HandwrittenCNN 兼容模型，当前最佳模型不会走到这里。
class ResidualBlock(nn.Module):
    """两个卷积层和 shortcut 组成的残差块。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip: nn.Module = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(self.main(inputs) + self.skip(inputs))


class HandwrittenCNN(nn.Module):
    """原项目的轻量模型，保留用于旧 checkpoint 兼容。

    当前最佳 D1 模型不使用这个类
    """

    def __init__(self, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.features(inputs))


class PaperConvBlock(nn.Module):
    """ 3x3 卷积、BN、PReLU 组合。

    卷积负责从局部像素中提取笔画特征；BatchNorm 稳定数值分布；PReLU
    提供非线性，使网络可以表示弯钩、交叉和复杂部件等模式。
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.block(inputs)


class HCCR9Layer(nn.Module):
    """ HCCR-CNN9Layer 的 PyTorch 实现。

     7 个卷积层和 2 个全连接层  9-layer CNN。自适应池化保留了
    96x96 的默认输入，同时允许测试使用较小图片。
    """

    def __init__(self, num_classes: int, dropout: float = 0.5) -> None:
        super().__init__()
        if num_classes < 2:#确保模型至少有两个类别，否则分类任务没有意义。
            raise ValueError("num_classes must be at least 2")
        self.features = nn.Sequential(
            PaperConvBlock(1, 96),
            nn.MaxPool2d(3, stride=2, padding=1),
            PaperConvBlock(96, 128),
            nn.MaxPool2d(3, stride=2, padding=1),
            PaperConvBlock(128, 160),
            nn.MaxPool2d(3, stride=2, padding=1),
            PaperConvBlock(160, 256),
            PaperConvBlock(256, 256),
            nn.MaxPool2d(3, stride=2, padding=1),
            PaperConvBlock(256, 384),
            PaperConvBlock(384, 384),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            nn.Linear(384 * 3 * 3, 1024),
            SafeBatchNorm1d(1024),
            nn.PReLU(1024),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs 形状为 [batch, 1, 96, 96]。features 逐步降低空间尺寸、
        # 提高通道数，把像素变成越来越抽象的笔画和部件特征。
        return self.classifier(self.features(inputs))


class IdentityChannelAttention(nn.Module):
    """恒等初始化的通道注意力，便于从旧模型无损迁移。

    当前最佳 D1 模型没有启用注意力；这是后续增强模型实验使用的组件。
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.activation = nn.PReLU(hidden)
        self.fc2 = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        scale = 2.0 * torch.sigmoid(self.fc2(self.activation(self.fc1(self.pool(inputs)))))
        return inputs * scale


class HCCR9ResidualAttention(HCCR9Layer):
    """保持九层主体的残差注意力增强模型。

    当前最佳 D1 使用原始 HCCR9Layer，这个增强版本暂未用于正式成绩。
    """

    def __init__(self, num_classes: int, dropout: float = 0.5) -> None:
        super().__init__(num_classes, dropout)
        self.attention160 = IdentityChannelAttention(160)
        self.attention256 = IdentityChannelAttention(256)
        self.attention384 = IdentityChannelAttention(384)
        self.residual_gate256 = nn.Parameter(torch.zeros(()))
        self.residual_gate384 = nn.Parameter(torch.zeros(()))

    def forward(self, inputs: Tensor) -> Tensor:
        # 这里显式写出每个阶段，是为了在原九层网络上插入注意力和残差连接。
        # 两个 residual_gate 初始为 0，所以迁移训练刚开始时不会突然改变旧模型输出。
        x = self.features[1](self.features[0](inputs))
        x = self.features[3](self.features[2](x))
        x = self.attention160(self.features[4](x))
        x = self.features[5](x)
        stage_input = self.features[6](x)
        x = self.features[7](stage_input) + self.residual_gate256 * stage_input
        x = self.features[8](self.attention256(x))
        stage_input = self.features[9](x)
        x = self.features[10](stage_input) + self.residual_gate384 * stage_input
        return self.classifier(self.features[11](self.attention384(x)))


class HCCR9ResidualAttentionWide(nn.Module):
    """参数量受控的宽版九层残差注意力模型。

    这是预留的宽模型实验版本，当前最佳 D1 没有使用它。
    """

    def __init__(self, num_classes: int, dropout: float = 0.5) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.features = nn.Sequential(
            PaperConvBlock(1, 128),
            nn.MaxPool2d(3, stride=2, padding=1),
            PaperConvBlock(128, 160),
            nn.MaxPool2d(3, stride=2, padding=1),
            PaperConvBlock(160, 192),
            nn.MaxPool2d(3, stride=2, padding=1),
            PaperConvBlock(192, 320),
            PaperConvBlock(320, 320),
            nn.MaxPool2d(3, stride=2, padding=1),
            PaperConvBlock(320, 512),
            PaperConvBlock(512, 512),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.attention192 = IdentityChannelAttention(192)
        self.attention320 = IdentityChannelAttention(320)
        self.attention512 = IdentityChannelAttention(512)
        self.residual_gate320 = nn.Parameter(torch.zeros(()))
        self.residual_gate512 = nn.Parameter(torch.zeros(()))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 1280),
            SafeBatchNorm1d(1280),
            nn.PReLU(1280),
            nn.Dropout(dropout),
            nn.Linear(1280, num_classes),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.features[1](self.features[0](inputs))
        x = self.features[3](self.features[2](x))
        x = self.attention192(self.features[4](x))
        x = self.features[5](x)
        stage_input = self.features[6](x)
        x = self.features[7](stage_input) + self.residual_gate320 * stage_input
        x = self.features[8](self.attention320(x))
        stage_input = self.features[9](x)
        x = self.features[10](stage_input) + self.residual_gate512 * stage_input
        return self.classifier(self.features[11](self.attention512(x)))


def create_model(name: str, num_classes: int) -> nn.Module:
    """根据 checkpoint/CLI 名称构建模型。

    训练、评估、预测必须调用同一个工厂，否则同一个 checkpoint 可能被错误地
    载入到另一种网络结构中。
    """

    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    normalized = name.lower().strip().replace("-", "_")
    if normalized in {"hccr_cnn9", "hccr_cnn9layer", "hccr9"}:
        return HCCR9Layer(num_classes)
    # 当前最佳 D1 不走以下增强模型分支；它们保留给后续对照实验。
    if normalized in {"hccr_cnn9_ra", "hccr9_ra"}:
        return HCCR9ResidualAttention(num_classes)
    if normalized in {"hccr_cnn9_ra_wide", "hccr9_ra_wide"}:
        return HCCR9ResidualAttentionWide(num_classes)
    # 旧轻量 CNN 仅用于兼容旧 checkpoint,当前最佳 D1 不使用。
    if normalized in {"cnn", "handwritten_cnn"}:
        return HandwrittenCNN(num_classes)
    raise ValueError(
        f"unknown model '{name}'; choose hccr_cnn9, hccr_cnn9_ra, "
        "hccr_cnn9_ra_wide or cnn"
    )


MODEL_BUILDERS: dict[str, Callable[[int], nn.Module]] = {
    "hccr_cnn9": HCCR9Layer,
    "hccr_cnn9_ra": HCCR9ResidualAttention,
    "hccr_cnn9_ra_wide": HCCR9ResidualAttentionWide,
    "cnn": HandwrittenCNN,
}
