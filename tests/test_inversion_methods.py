import torch

from codec_attribute_control.head import MLPF0Head
from codec_attribute_control.inversion_methods import ProbeAdapter, available_methods, invert


def test_linear_capability_discovered_and_exact_inverse():
    head = MLPF0Head(input_dim=3, hidden_sizes=[], context_size=1)
    with torch.no_grad():
        head.projection.weight.copy_(torch.tensor([[2., 0., 0.]]))
        head.projection.bias.zero_()
    adapter = ProbeAdapter(head)
    assert available_methods(adapter) == ("linear", "adam", "jacobian")
    x = torch.zeros(1, 4, 3)
    target = torch.ones(1, 4)
    result = invert(adapter, x, target, method="linear", regularization=1e-6)
    assert torch.allclose(result.final_prediction, target, atol=1e-5)
    assert torch.allclose(result.latents[..., 1:], torch.zeros_like(result.latents[..., 1:]))
    assert torch.allclose(x, torch.zeros_like(x))


def test_nonlinear_and_context_heads_do_not_claim_exact_linear_inverse():
    nonlinear = MLPF0Head(input_dim=3, hidden_sizes=[8])
    contextual = MLPF0Head(input_dim=3, context_size=3)
    for head in (nonlinear, contextual):
        assert available_methods(ProbeAdapter(head)) == ("adam", "jacobian")


def test_system_head_is_automatically_unwrapped():
    class System(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = MLPF0Head(input_dim=3)
    adapter = ProbeAdapter(System())
    assert available_methods(adapter)[0] == "linear"


def test_jacobian_fits_signed_targets_and_masks_unvoiced_frames():
    head = MLPF0Head(input_dim=2, hidden_sizes=[], context_size=1)
    with torch.no_grad():
        head.projection.weight.copy_(torch.tensor([[2.0, 0.0]]))
        head.projection.bias.zero_()
    latents = torch.zeros(1, 3, 2)
    target = torch.tensor([[-1.0, float("nan"), 1.0]])
    result = invert(ProbeAdapter(head), latents, target, method="jacobian")
    assert torch.allclose(result.final_prediction, torch.tensor([[-1.0, 0.0, 1.0]]), atol=3e-4)
    assert torch.count_nonzero(latents) == 0
