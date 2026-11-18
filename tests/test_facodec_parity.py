"""Opt-in pretrained FACodec parity tests.

Run with FACODEC_PARITY=1 uv run --extra train --extra facodec pytest -s tests/test_facodec_parity.py
These tests download the pretrained weights; ordinary CI skips them.
"""
import os

import pytest
import torch
import torch.nn.functional as F

from codec_attribute_control.facodec import FACodec, HOP

pytestmark = pytest.mark.skipif(
    os.environ.get("FACODEC_PARITY") != "1",
    reason="Requires pretrained FACodec weights; set FACODEC_PARITY=1",
)


@pytest.fixture(scope="module")
def codec():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(20261118)
    return FACodec(device=device)


def _samples(lengths):
    generator = torch.Generator().manual_seed(20261118)
    recordings = [torch.randn(n, generator=generator) * 0.08 for n in lengths]
    padded = torch.nn.utils.rnn.pad_sequence(recordings, batch_first=True)
    return recordings, padded, torch.tensor(lengths)


def _stats(reference, actual):
    reference, actual = reference.detach().float().reshape(-1), actual.detach().float().reshape(-1)
    error = (reference - actual).abs()
    return (error.mean().item(), error.max().item(),
            F.cosine_similarity(reference[None], actual[None]).item())


def _compare(codec, lengths, label, *, strict):
    recordings, padded, sizes = _samples(lengths)
    references = [codec.encode(x) for x in recordings]
    batched = codec.encode_batch(padded, sizes)
    print(f"\\n{label}: lengths={lengths}")
    print(f"{'item':<24} {'MAE':>12} {'max':>12} {'cosine':>12} {'IDs %':>10}")
    for i, (recording, reference) in enumerate(zip(recordings, references)):
        frames = (len(recording) + HOP - 1) // HOP
        ref_pre = codec.encoder(F.pad(recording, (0, (-len(recording)) % HOP))[None, None].to(codec.device))
        batch_pre = codec.encoder(
            F.pad(padded, (0, (-padded.shape[1]) % HOP)).unsqueeze(1).to(codec.device)
        )[i:i + 1, :, :frames]
        for name, ref, got in [
            ("pre_vq", ref_pre, batch_pre),
            *[(key, getattr(reference, key), getattr(batched, key)[i:i + 1, :, :frames])
              for key in ("prosody", "content", "residual", "combined")],
        ]:
            mae, maximum, cosine = _stats(ref, got)
            print(f"{i}:{name:<22} {mae:12.3g} {maximum:12.3g} {cosine:12.6f}")
            if strict:
                torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-4)
        ref_ids = reference.codes[:, 0, :frames]
        got_ids = batched.codes[:, i, :frames]
        agreement = (ref_ids == got_ids).float().mean().item() * 100
        print(f"{i}:{'quantizer IDs':<22} {'':>12} {'':>12} {'':>12} {agreement:9.3f}%")
        if strict:
            assert torch.equal(ref_ids, got_ids)
        mae, maximum, cosine = _stats(reference.speaker, batched.speaker[i:i + 1])
        print(f"{i}:{'speaker':<22} {mae:12.3g} {maximum:12.3g} {cosine:12.6f}")
        if strict:
            torch.testing.assert_close(batched.speaker[i:i + 1], reference.speaker,
                                       atol=1e-5, rtol=1e-4)
        # Separate boundary from interior; the final 8 frames are diagnostic.
        ref = reference.prosody[0]
        got = batched.prosody[i, :, :frames]
        for region, slc in (("interior", slice(None, max(0, frames - 8))),
                            ("last_8", slice(max(0, frames - 8), None))):
            if ref[:, slc].numel():
                mae, maximum, cosine = _stats(ref[:, slc], got[:, slc])
                print(f"{i}:prosody_{region:<14} {mae:12.3g} {maximum:12.3g} {cosine:12.6f}")


def test_batch_one_parity(codec):
    _compare(codec, [16000 + 37], "batch_one", strict=True)


def test_equal_length_parity(codec):
    _compare(codec, [16000, 16000, 16000, 16000], "equal_length", strict=True)


def test_variable_length_diagnostics(codec):
    _compare(codec, [16000, 12037, 8000, 15091], "variable_length", strict=False)


def test_batch_composition_diagnostics(codec):
    # Same first utterance, different padding contexts.
    _compare(codec, [12037, 12037], "same_length_partner", strict=True)
    _compare(codec, [12037, 24000], "longer_partner", strict=False)
