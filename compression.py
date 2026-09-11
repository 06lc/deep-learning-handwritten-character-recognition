"""GSLRE、ADW 剪枝和可移植权重量化工具。"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.utils import prune


@dataclass(frozen=True, slots=True)
class CompressionStats:
    total_parameters: int
    zero_parameters: int
    sparsity: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _weight_modules(model: nn.Module) -> Iterator[nn.Module]:
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, "weight"):
            yield module


def compression_stats(model: nn.Module) -> CompressionStats:
    weights = [module.weight.detach() for module in _weight_modules(model)]
    if not weights:
        return CompressionStats(0, 0, 0.0)
    total = sum(weight.numel() for weight in weights)
    zeros = sum(int(torch.count_nonzero(weight == 0)) for weight in weights)
    return CompressionStats(total, zeros, zeros / total)


class GSLREConv2d(nn.Module):
    """空间低秩卷积：先 Kx1，再 1xK。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        rank: int | None = None,
        stride: int = 1,
        padding: int | None = None,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("GSLREConv2d expects an odd kernel size")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.rank = rank or max(1, min(in_channels, out_channels) // 2)
        padding = kernel_size // 2 if padding is None else padding
        self.vertical = nn.Conv2d(
            in_channels,
            self.rank,
            (kernel_size, 1),
            stride=stride,
            padding=(padding, 0),
            bias=False,
        )
        self.horizontal = nn.Conv2d(
            self.rank,
            out_channels,
            (1, kernel_size),
            padding=(0, padding),
            bias=False,
        )

    @classmethod
    def from_conv(cls, layer: nn.Conv2d, rank: int) -> GSLREConv2d:
        if layer.kernel_size[0] != layer.kernel_size[1] or layer.kernel_size[0] % 2 == 0:
            raise ValueError("only odd square convolutions can be factorized")
        if layer.groups != 1 or layer.dilation != (1, 1):
            raise ValueError("grouped and dilated convolutions are not supported")
        result = cls(
            layer.in_channels,
            layer.out_channels,
            layer.kernel_size[0],
            rank,
            layer.stride[0],
            layer.padding[0],
        ).to(device=layer.weight.device, dtype=layer.weight.dtype)
        with torch.no_grad():
            # 用平均通道的空间核做 SVD 初始化，随后由监督微调恢复通道特征。
            spatial = layer.weight.detach().mean(dim=1)
            matrix = spatial.reshape(
                layer.out_channels * layer.kernel_size[0], layer.kernel_size[0]
            )
            u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
            rank = min(result.rank, singular.numel())
            left = u[:, :rank] * singular[:rank].sqrt()
            right = singular[:rank].sqrt().unsqueeze(1) * vh[:rank]
            result.horizontal.weight[:, :rank].copy_(
                left.reshape(layer.out_channels, layer.kernel_size[0], rank)
                .permute(0, 2, 1)
                .unsqueeze(2)
            )
            result.vertical.weight[:rank].copy_(
                right.reshape(rank, 1, layer.kernel_size[0], 1).repeat(1, layer.in_channels, 1, 1)
                / max(1.0, layer.in_channels**0.5)
            )
            if rank < result.rank:
                result.horizontal.weight[rank:].zero_()
                result.vertical.weight[rank:].zero_()
        return result

    def forward(self, inputs: Tensor) -> Tensor:
        return self.horizontal(self.vertical(inputs))


def replace_conv_with_gslre(model: nn.Module, rank_ratio: float = 0.5) -> int:
    """递归替换 3x3 卷积，返回替换层数量。"""

    if not 0 < rank_ratio <= 1:
        raise ValueError("rank_ratio must be in (0, 1]")
    replaced = 0
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Conv2d) and child.kernel_size == (3, 3) and child.groups == 1:
            rank = max(1, round(min(child.in_channels, child.out_channels) * rank_ratio))
            setattr(model, name, GSLREConv2d.from_conv(child, rank))
            replaced += 1
        else:
            replaced += replace_conv_with_gslre(child, rank_ratio)
    return replaced


def apply_adw_pruning(
    model: nn.Module, target_sparsity: float = 0.5, steps: int = 5
) -> CompressionStats:
    """按层渐进阈值应用 ADW 风格的固定 mask。"""

    if not 0 <= target_sparsity < 1:
        raise ValueError("target_sparsity must be in [0, 1)")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    for module in _weight_modules(model):
        if hasattr(module, "weight_orig"):
            prune.remove(module, "weight")
        absolute = module.weight.detach().abs()
        mask = torch.ones_like(absolute, dtype=torch.bool)
        for step in range(1, steps + 1):
            threshold = torch.quantile(absolute.flatten(), target_sparsity * step / steps)
            mask &= absolute > threshold
        if not bool(mask.any()):
            mask.view(-1)[int(module.weight.detach().abs().argmax())] = True
        prune.custom_from_mask(module, name="weight", mask=mask)
    return compression_stats(model)


def finalize_pruning(model: nn.Module) -> CompressionStats:
    for module in _weight_modules(model):
        if hasattr(module, "weight_orig"):
            prune.remove(module, "weight")
    return compression_stats(model)


def _kmeans_centers(values: Tensor, clusters: int, iterations: int = 8) -> Tensor:
    sample = values.detach().flatten()
    if sample.numel() > 100_000:
        sample = sample[:: max(1, sample.numel() // 100_000)][:100_000]
    minimum, maximum = sample.min(), sample.max()
    if minimum == maximum:
        return minimum.reshape(1)
    centers = torch.linspace(minimum, maximum, clusters, device=sample.device)
    for _ in range(iterations):
        boundaries = (centers[:-1] + centers[1:]) / 2
        assignment = torch.bucketize(sample.contiguous(), boundaries)
        sums = torch.zeros_like(centers)
        counts = torch.zeros_like(centers)
        sums.scatter_add_(0, assignment, sample)
        counts.scatter_add_(0, assignment, torch.ones_like(sample))
        centers = torch.where(counts > 0, sums / counts, centers)
    return centers


def quantize_state_dict(
    state_dict: dict[str, Tensor], clusters: int = 256
) -> tuple[dict[str, Tensor], dict[str, dict[str, Any]]]:
    """将卷积/全连接权重保存为 uint8 索引和量化中心。"""

    if not 2 <= clusters <= 256:
        raise ValueError("clusters must be between 2 and 256")
    plain: dict[str, Tensor] = {}
    quantized: dict[str, dict[str, Any]] = {}
    for key, value in state_dict.items():
        if not (value.is_floating_point() and value.ndim >= 2 and key.endswith(".weight")):
            plain[key] = value.detach().cpu()
            continue
        flat_value = value.detach().float().cpu().flatten()
        centers = _kmeans_centers(flat_value, clusters).cpu()
        zero_mask = flat_value == 0
        zero_index = None
        if bool(zero_mask.any()):
            centers[int(torch.argmin(centers.abs()))] = 0
            centers = centers.sort().values
            zero_index = int(torch.argmin(centers.abs()))
        boundaries = (centers[:-1] + centers[1:]) / 2
        indices = torch.bucketize(flat_value, boundaries).to(torch.uint8)
        if zero_index is not None:
            indices[zero_mask] = zero_index
        quantized[key] = {
            "indices": indices,
            "centers": centers.to(torch.float16),
            "shape": tuple(value.shape),
        }
    return plain, quantized


def dequantize_state_dict(
    plain: dict[str, Tensor], quantized: dict[str, dict[str, Any]]
) -> dict[str, Tensor]:
    state = dict(plain)
    for key, packed in quantized.items():
        state[key] = packed["centers"].float()[packed["indices"].long()].reshape(packed["shape"])
    return state


def quantized_payload(model_state: dict[str, Tensor], clusters: int = 256) -> dict[str, Any]:
    plain, quantized = quantize_state_dict(model_state, clusters)
    return {"model_state": plain, "quantized_state": quantized, "quantization_clusters": clusters}


def model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
