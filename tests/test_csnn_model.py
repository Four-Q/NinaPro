import pytest
import torch

from spikingjelly.activation_based import neuron

from src.models import NinaProCSNN


def make_small_model(**options):
    defaults = {
        "feature_channels": (8, 16, 32),
        "dropout_rate": 0.0,
    }
    defaults.update(options)
    return NinaProCSNN(**defaults)


@pytest.mark.parametrize("input_channels, polarity_channels", [(16, 1), (32, 2)])
@pytest.mark.parametrize("time_steps", [40, 80, 120])
def test_csnn_accepts_encodings_and_available_time_steps(
    input_channels,
    polarity_channels,
    time_steps,
):
    model = make_small_model(
        input_channels=input_channels,
        polarity_channels=polarity_channels,
    )
    logits = model(torch.rand(2, input_channels, time_steps))

    assert logits.shape == (2, 12)
    assert torch.isfinite(logits).all()


def test_csnn_applies_conv1d_to_all_16_electrodes_without_reordering():
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
    expected = (
        inputs.reshape(1, 2, 16, 8)
        .permute(3, 0, 1, 2)
        .reshape(8, 2, 16)
    )
    assert torch.equal(captured[0], expected)
    assert model.blocks[0].conv.kernel_size == (3,)
    assert model.blocks[0].conv.padding == (1,)


@pytest.mark.parametrize(
    "learnable_tau, expected_type",
    [
        (False, neuron.LIFNode),
        (True, neuron.ParametricLIFNode),
    ],
)
def test_csnn_switches_between_fixed_and_learnable_tau(
    learnable_tau,
    expected_type,
):
    model = make_small_model(learnable_tau=learnable_tau)

    assert all(isinstance(block.neuron, expected_type) for block in model.blocks)


def test_csnn_preserves_all_electrodes_until_readout():
    model = make_small_model().eval()
    captured_shapes = []
    handles = [
        block.neuron.register_forward_hook(
            lambda module, arguments, output: captured_shapes.append(
                tuple(output.shape)
            )
        )
        for block in model.blocks
    ]
    try:
        model(torch.rand(2, 32, 40))
    finally:
        for handle in handles:
            handle.remove()

    assert captured_shapes == [
        (40, 2, 8, 16),
        (40, 2, 16, 16),
        (40, 2, 32, 16),
    ]


def test_learnable_tau_receives_gradients():
    model = make_small_model(learnable_tau=True).train()
    logits = model(torch.rand(3, 32, 40))
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2])).backward()

    for block in model.blocks:
        assert block.neuron.w.grad is not None
        assert torch.isfinite(block.neuron.w.grad).all()


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
        {"learnable_tau": 1},
        {"backend": "invalid"},
    ],
)
def test_csnn_rejects_invalid_configuration(options):
    with pytest.raises((TypeError, ValueError)):
        NinaProCSNN(**options)
