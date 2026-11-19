import torch

from codec_attribute_control.code_head import GlobalContextPitchCodeHead, PitchCodeHead


def test_cnn_output_and_gradients():
    head = GlobalContextPitchCodeHead(codebook_size=16, hidden_size=8, dilations=(1, 2))
    f0 = torch.tensor([[100., 110., 120., 0., 0.], [200., 220., 240., 260., 280.]])
    logits = head(f0, lengths=torch.tensor([3, 5]))
    assert logits.shape == (2, 5, 16)
    logits[:, :3].sum().backward()
    assert all(p.grad is not None for p in head.parameters())


def test_padding_does_not_affect_valid_predictions():
    torch.manual_seed(0)
    head = GlobalContextPitchCodeHead(codebook_size=8, hidden_size=8, dilations=(1, 2)).eval()
    short = torch.tensor([[100., 120., 110.]])
    padded = torch.tensor([[100., 120., 110., 999., 999.]])
    with torch.no_grad():
        a = head(short, torch.tensor([3]))
        b = head(padded, torch.tensor([3]))
    torch.testing.assert_close(a, b[:, :3], atol=1e-5, rtol=1e-5)


def test_mlp_accepts_lengths_for_shared_training_interface():
    head = PitchCodeHead(codebook_size=16)
    assert head(torch.ones(2, 5), lengths=torch.tensor([3, 5])).shape == (2, 5, 16)
