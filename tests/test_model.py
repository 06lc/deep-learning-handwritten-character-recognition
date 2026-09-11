from __future__ import annotations

import torch

from model import HandwrittenCNN


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
