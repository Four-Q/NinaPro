import pytest
import torch

from spikingjelly.activation_based import layer, neuron, surrogate

from src.models import DropoutPLIFSNN


def test_default_dropout_plif_configuration_and_parameter_count():
    model = DropoutPLIFSNN()

    assert model.input_projection.in_features == 16
    assert model.input_projection.out_features == 128
    assert model.hidden_projection.in_features == 128
    assert model.hidden_projection.out_features == 128
    assert model.classifier.out_features == 12
    assert isinstance(model.lif1, neuron.ParametricLIFNode)
    assert isinstance(model.lif2, neuron.ParametricLIFNode)
    assert isinstance(model.dropout1, layer.Dropout)
    assert isinstance(model.dropout2, layer.Dropout)
    assert isinstance(model.lif1.surrogate_function, surrogate.ATan)
    assert isinstance(model.lif2.surrogate_function, surrogate.ATan)
    assert model.lif1.step_mode == "m"
    assert model.lif2.step_mode == "m"
    assert model.dropout1.step_mode == "m"
    assert model.dropout2.step_mode == "m"
    assert float(1.0 / model.lif1.w.detach().sigmoid()) == pytest.approx(2.0)
    assert float(1.0 / model.lif2.w.detach().sigmoid()) == pytest.approx(2.0)
    assert model.dropout1.p == pytest.approx(0.3)
    assert model.dropout2.p == pytest.approx(0.3)
    assert model.lif1.w.requires_grad
    assert model.lif2.w.requires_grad
    assert sum(parameter.numel() for parameter in model.parameters()) == 20238


def test_forward_returns_logits_for_variable_time_lengths():
    model = DropoutPLIFSNN(
        input_channels=8,
        hidden_size=24,
        num_classes=5,
        tau=3.0,
        dropout_rate=0.2,
    ).eval()

    short_logits = model(torch.randn(3, 8, 6))
    long_logits = model(torch.randn(2, 8, 25))

    assert short_logits.shape == (3, 5)
    assert long_logits.shape == (2, 5)
    assert torch.isfinite(short_logits).all()
    assert torch.isfinite(long_logits).all()


def test_dropout_mask_is_shared_across_time_steps():
    torch.manual_seed(13)
    model = DropoutPLIFSNN(hidden_size=16, dropout_rate=0.3).train()
    spike_sequence = torch.ones(12, 4, 16)

    dropped_sequence = model.dropout1(spike_sequence)

    assert model.dropout1.mask.shape == (4, 16)
    for time_step in range(1, spike_sequence.shape[0]):
        assert torch.equal(dropped_sequence[0], dropped_sequence[time_step])


def test_reset_clears_plif_state_and_dropout_masks():
    model = DropoutPLIFSNN(hidden_size=16).train()
    model(torch.randn(4, 16, 12))

    assert isinstance(model.lif1.v, torch.Tensor)
    assert isinstance(model.lif2.v, torch.Tensor)
    assert isinstance(model.dropout1.mask, torch.Tensor)
    assert isinstance(model.dropout2.mask, torch.Tensor)

    model.reset()

    assert model.lif1.v == 0.0
    assert model.lif2.v == 0.0
    assert model.dropout1.mask is None
    assert model.dropout2.mask is None


def test_cross_entropy_backward_updates_linear_and_plif_parameters():
    torch.manual_seed(19)
    model = DropoutPLIFSNN(hidden_size=32)
    inputs = torch.randn(8, 16, 40)
    targets = torch.arange(8) % 12

    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()

    assert model.classifier.weight.grad.abs().sum() > 0
    assert model.lif1.w.grad is not None
    assert model.lif2.w.grad is not None
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("dropout_rate", [-0.1, 1.0, True, "0.3"])
def test_invalid_dropout_rate_is_rejected(dropout_rate):
    with pytest.raises(ValueError):
        DropoutPLIFSNN(dropout_rate=dropout_rate)
