import pytest
import torch

from primed import PRIMED, load_model, save_weights
from primed.weights import architecture


def tiny_model(num_classes=4, num_inputs=3):
    torch.manual_seed(0)
    return PRIMED(num_classes=num_classes, num_inputs=num_inputs, channels=(32, 64), num_res_blocks=1)


def tiny_batch(num_inputs=3, batch=2):
    inputs = torch.rand(batch, num_inputs, 1, 16, 16, 16)
    prior = inputs.mean(dim=1)
    target = torch.rand(batch, 1, 16, 16, 16)
    target_class = torch.randint(0, num_inputs + 1, (batch,))
    return prior, inputs, target, target_class


def test_loss_is_finite_and_trains_the_network():
    model = tiny_model()
    prior, inputs, target, target_class = tiny_batch()
    loss = model(prior, inputs, target, target_class)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.net.parameters())


def test_untrained_model_returns_the_prior():
    model = tiny_model().eval()
    prior, inputs, _, target_class = tiny_batch()
    prediction = model.synthesize(prior, inputs, target_class)
    assert prediction.shape == prior.shape
    assert torch.allclose(prediction, prior.clamp(0, 1), atol=1e-5)


def test_synthesis_is_clamped():
    model = tiny_model().eval()
    with torch.no_grad():
        for p in model.net.parameters():
            p.add_(torch.randn_like(p))
    prior, inputs, _, target_class = tiny_batch()
    prediction = model.synthesize(prior, inputs, target_class)
    assert prediction.min() >= 0 and prediction.max() <= 1


@pytest.mark.parametrize("num_classes,num_inputs", [(4, 3), (3, 2)])
def test_weights_round_trip(tmp_path, num_classes, num_inputs):
    model = tiny_model(num_classes, num_inputs).eval()
    path = str(tmp_path / "weights.safetensors")
    save_weights(model.net.state_dict(), path, {"step": 1})
    loaded = load_model(path)
    expected = {"num_classes": num_classes, "num_inputs": num_inputs, "channels": [32, 64], "num_res_blocks": 1, "class_dim": 8}
    assert architecture(loaded.net.state_dict()) == expected
    prior, inputs, _, target_class = tiny_batch(num_inputs, batch=1)
    assert torch.equal(model.synthesize(prior, inputs, target_class), loaded.synthesize(prior, inputs, target_class))
