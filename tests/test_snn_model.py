import pytest
import torch

from spikingjelly.activation_based import surrogate

from src.models.snn import NinaProSNN


def test_default_model_configuration_and_parameter_count():
    model = NinaProSNN()

    assert model.input_projection.in_features == 16
    assert model.input_projection.out_features == 128
    assert model.hidden_projection.in_features == 128
    assert model.hidden_projection.out_features == 128
    assert model.classifier.in_features == 128
    assert model.classifier.out_features == 12
    assert model.lif1.step_mode == "m"
    assert model.lif2.step_mode == "m"
    assert model.lif1.backend == "torch"
    assert model.lif2.backend == "torch"
    assert model.lif1.tau == 2.0
    assert model.lif2.tau == 2.0
    assert model.lif1.detach_reset is True
    assert model.lif2.detach_reset is True
    assert isinstance(model.lif1.surrogate_function, surrogate.ATan)
    assert isinstance(model.lif2.surrogate_function, surrogate.ATan)
    assert sum(parameter.numel() for parameter in model.parameters()) == 20236


def test_forward_accepts_data_pipeline_layout_and_returns_logits():
    model = NinaProSNN()
    x = torch.randn(7, 16, 40)

    logits = model(x)

    assert logits.shape == (7, 12)
    assert logits.dtype == x.dtype
    assert torch.isfinite(logits).all()


def test_forward_uses_emg_samples_as_snn_time_steps():
    model = NinaProSNN(hidden_size=8)
    x = torch.arange(2 * 16 * 5, dtype=torch.float32).reshape(2, 16, 5)
    captured_inputs = []

    def capture_input(module, inputs):
        del module
        captured_inputs.append(inputs[0].detach().clone())

    handle = model.input_projection.register_forward_pre_hook(capture_input)
    model(x)
    handle.remove()

    assert captured_inputs[0].shape == (5, 2, 16)
    assert torch.equal(captured_inputs[0], x.permute(2, 0, 1))


def test_auto_reset_makes_independent_forwards_repeatable():
    torch.manual_seed(7)
    model = NinaProSNN(hidden_size=16, auto_reset=True).eval()
    x = torch.randn(4, 16, 12)

    first_logits = model(x)
    second_logits = model(x)

    assert torch.equal(first_logits, second_logits)
    assert isinstance(model.lif1.v, torch.Tensor)
    assert isinstance(model.lif2.v, torch.Tensor)

    model.reset()
    assert model.lif1.v == 0.0
    assert model.lif2.v == 0.0


def test_cross_entropy_backward_produces_finite_gradients():
    torch.manual_seed(11)
    model = NinaProSNN(hidden_size=32)
    x = torch.randn(8, 16, 40)
    y = torch.arange(8) % 12

    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()

    assert torch.isfinite(loss)
    assert model.classifier.weight.grad.abs().sum() > 0
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_model_supports_configurable_dimensions_and_time_length():
    model = NinaProSNN(
        input_channels=8,
        hidden_size=24,
        num_classes=5,
        tau=3.0,
    )

    short_logits = model(torch.randn(3, 8, 6))
    long_logits = model(torch.randn(2, 8, 25))

    assert short_logits.shape == (3, 5)
    assert long_logits.shape == (2, 5)
    assert model.lif1.tau == 3.0


@pytest.mark.parametrize(
    "keyword, value, exception",
    [
        ("input_channels", 0, ValueError),
        ("hidden_size", -1, ValueError),
        ("num_classes", 1.5, ValueError),
        ("tau", 1.0, ValueError),
        ("auto_reset", 1, TypeError),
        ("backend", "invalid", ValueError),
    ],
)
def test_invalid_model_configuration_is_rejected(keyword, value, exception):
    options = {keyword: value}
    with pytest.raises(exception):
        NinaProSNN(**options)


def test_invalid_input_is_rejected():
    model = NinaProSNN()

    with pytest.raises(TypeError):
        model([1, 2, 3])
    with pytest.raises(ValueError):
        model(torch.randn(16, 40))
    with pytest.raises(ValueError):
        model(torch.randn(2, 8, 40))
    with pytest.raises(ValueError):
        model(torch.empty(0, 16, 40))
    with pytest.raises(ValueError):
        model(torch.empty(2, 16, 0))
    with pytest.raises(TypeError):
        model(torch.ones(2, 16, 40, dtype=torch.int64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="当前环境没有可用 CUDA")
def test_model_runs_on_cuda_when_available():
    model = NinaProSNN().cuda()
    x = torch.randn(4, 16, 40, device="cuda")

    logits = model(x)

    assert logits.is_cuda
    assert logits.shape == (4, 12)
