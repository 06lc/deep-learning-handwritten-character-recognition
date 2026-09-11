from __future__ import annotations

import torch

from model import HandwrittenCNN, HCCR9Layer, create_model


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
