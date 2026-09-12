from __future__ import annotations

import torch

from model import (
    HandwrittenCNN,
    HCCR9Layer,
    HCCR9ResidualAttention,
    HCCR9ResidualAttentionWide,
    create_model,
)
from train import warm_start_model


def test_hccr9layer_returns_logits_for_96px_input() -> None:
    model = HCCR9Layer(num_classes=7)
    outputs = model(torch.randn(1, 1, 96, 96))
    assert outputs.shape == (1, 7)


def test_model_factory_builds_paper_model() -> None:
    assert isinstance(create_model("hccr_cnn9", 7), HCCR9Layer)


def test_handwritten_cnn_returns_logits_for_each_class() -> None:
    model = HandwrittenCNN(num_classes=7)
    inputs = torch.randn(2, 1, 64, 64)

    outputs = model(inputs)

    assert outputs.shape == (2, 7)
    assert outputs.dtype == torch.float32


def test_handwritten_cnn_supports_backpropagation() -> None:
    model = HandwrittenCNN(num_classes=3)
    inputs = torch.randn(2, 1, 64, 64)
    targets = torch.tensor([0, 2])

    loss = torch.nn.functional.cross_entropy(model(inputs), targets)
    loss.backward()

    assert any(parameter.grad is not None for parameter in model.parameters())


def test_residual_attention_warm_start_preserves_baseline_logits() -> None:
    baseline = HCCR9Layer(num_classes=7).eval()
    enhanced = HCCR9ResidualAttention(num_classes=7).eval()
    report = warm_start_model(enhanced, baseline.state_dict())
    inputs = torch.randn(2, 1, 96, 96)

    with torch.inference_mode():
        baseline_logits = baseline(inputs)
        enhanced_logits = enhanced(inputs)

    assert report["coverage"] > 0.99
    assert torch.allclose(enhanced_logits, baseline_logits, atol=1e-6, rtol=1e-5)


def test_wide_residual_attention_stays_below_parameter_limit() -> None:
    baseline = HCCR9Layer(num_classes=3926)
    wide = HCCR9ResidualAttentionWide(num_classes=3926)

    assert sum(parameter.numel() for parameter in wide.parameters()) <= 2 * sum(
        parameter.numel() for parameter in baseline.parameters()
    )
    assert wide(torch.randn(1, 1, 96, 96)).shape == (1, 3926)
