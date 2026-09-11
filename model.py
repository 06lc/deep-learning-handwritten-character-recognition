"""识别模型定义和模型工厂。"""

from __future__ import annotations

from collections.abc import Callable

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
    """原项目的轻量模型，保留用于旧 checkpoint 兼容。"""

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
    """论文中的 3x3 卷积、BN、PReLU 组合。"""

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
    """论文 HCCR-CNN9Layer 的 PyTorch 实现。

    论文将 7 个卷积层和 2 个全连接层合称 9-layer CNN。自适应池化保留了
    96x96 的论文默认输入，同时允许测试使用较小图片。
    """

    def __init__(self, num_classes: int, dropout: float = 0.5) -> None:
        super().__init__()
        if num_classes < 2:
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
        return self.classifier(self.features(inputs))


def create_model(name: str, num_classes: int) -> nn.Module:
    """根据 checkpoint/CLI 名称构建模型。"""

    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    normalized = name.lower().strip().replace("-", "_")
    if normalized in {"hccr_cnn9", "hccr_cnn9layer", "hccr9"}:
        return HCCR9Layer(num_classes)
    if normalized in {"cnn", "handwritten_cnn"}:
        return HandwrittenCNN(num_classes)
    raise ValueError(f"unknown model '{name}'; choose hccr_cnn9 or cnn")


MODEL_BUILDERS: dict[str, Callable[[int], nn.Module]] = {
    "hccr_cnn9": HCCR9Layer,
    "cnn": HandwrittenCNN,
}
