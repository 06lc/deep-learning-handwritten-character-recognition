from __future__ import annotations

import torch

from compression import (
    GSLREConv2d,
    apply_adw_pruning,
    compression_stats,
    dequantize_state_dict,
    finalize_pruning,
    quantize_state_dict,
    replace_conv_with_gslre,
)
from model import HCCR9Layer


def test_gslre_conv_preserves_shape() -> None:
    layer = GSLREConv2d(4, 6, rank=2)
    assert layer(torch.randn(2, 4, 16, 16)).shape == (2, 6, 16, 16)


def test_gslre_replaces_hccr_convolutions() -> None:
    model = HCCR9Layer(3)
    assert replace_conv_with_gslre(model, 0.5) == 7
    assert any(isinstance(module, GSLREConv2d) for module in model.modules())


def test_adw_pruning_and_quantization_round_trip() -> None:
    model = HCCR9Layer(3)
    apply_adw_pruning(model, target_sparsity=0.3)
    stats = finalize_pruning(model)
    assert 0.0 < stats.sparsity < 1.0
    plain, packed = quantize_state_dict(model.state_dict(), clusters=8)
    restored = dequantize_state_dict(plain, packed)
    assert set(restored) == set(model.state_dict())
    assert compression_stats(model).total_parameters == stats.total_parameters
    for key, value in model.state_dict().items():
        if key in packed and bool((value == 0).any()):
            assert torch.equal(
                restored[key][value == 0], torch.zeros_like(restored[key][value == 0])
            )
