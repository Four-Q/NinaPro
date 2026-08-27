import pytest
import torch

from src.models import NinaProCSNN


def make_small_model(**options):
    defaults = {
        "feature_channels": (8, 16, 32),
        "dropout_rate": 0.0,
    }
    defaults.update(options)
    return NinaProCSNN(**defaults)


@pytest.mark.parametrize("time_steps", [40, 80, 120])
def test_csnn_accepts_available_time_steps(time_steps):
    model = make_small_model()
    logits = model(torch.rand(2, 32, time_steps))

    assert logits.shape == (2, 12)
    assert torch.isfinite(logits).all()


def test_csnn_reshapes_polarities_without_reordering_channels():
    model = make_small_model().eval()
    inputs = torch.arange(32 * 8, dtype=torch.float32).reshape(1, 32, 8)
    captured = []

    handle = model.blocks[0].conv.register_forward_pre_hook(
        lambda module, arguments: captured.append(arguments[0].detach().clone())
    )
    try:
        model(inputs)
    finally:
        handle.remove()

    assert len(captured) == 1
    assert torch.equal(captured[0], inputs.reshape(1, 2, 16, 8))


def test_csnn_backward_reaches_convolution_and_classifier():
    model = make_small_model().train()
    logits = model(torch.rand(3, 32, 40))
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2]))
    loss.backward()

    assert model.blocks[0].conv.weight.grad is not None
    assert torch.isfinite(model.blocks[0].conv.weight.grad).all()
    assert model.classifier.weight.grad is not None
    assert torch.isfinite(model.classifier.weight.grad).all()


def test_csnn_auto_reset_prevents_state_leakage_between_batches():
    model = make_small_model(auto_reset=True).eval()
    inputs = torch.rand(2, 32, 40)

    first = model(inputs)
    second = model(inputs)

    assert torch.equal(first, second)


@pytest.mark.parametrize(
    "inputs, error_type",
    [
        (torch.rand(2, 16, 40), ValueError),
        (torch.ones(2, 32, 40, dtype=torch.uint8), TypeError),
        (torch.rand(2, 2, 16, 40), ValueError),
    ],
)
def test_csnn_rejects_invalid_inputs(inputs, error_type):
    model = make_small_model()

    with pytest.raises(error_type):
        model(inputs)


@pytest.mark.parametrize(
    "options",
    [
        {"input_channels": 31},
        {"feature_channels": (8, 16)},
        {"feature_channels": (8, 0, 32)},
        {"dropout_rate": 1.0},
        {"tau": 1.0},
        {"backend": "invalid"},
    ],
)
def test_csnn_rejects_invalid_configuration(options):
    with pytest.raises((TypeError, ValueError)):
        NinaProCSNN(**options)
