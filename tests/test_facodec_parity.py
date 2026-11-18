"""Opt-in pretrained FACodec parity tests.

Run with FACODEC_PARITY=1 uv run --extra train --extra facodec pytest -s tests/test_facodec_parity.py
These tests download the pretrained weights; ordinary CI skips them.
"""
import os

import pytest
import torch
import torch.nn.functional as F

from codec_attribute_control.facodec import FACodec, HOP
from codec_attribute_control.facodec_encoder import FACodecContentEncoder

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


@torch.inference_mode()
def _compare(codec, lengths, label, *, strict):
    recordings, padded, sizes = _samples(lengths)
    references = [codec.encode(x) for x in recordings]
    batched = FACodecContentEncoder(codec=codec).encode_waveforms(padded, sizes)
    print(f"\\n{label}: lengths={lengths}")
    print(f"{'item':<24} {'MAE':>12} {'max':>12} {'cosine':>12} {'IDs %':>10}")
    probe = torch.nn.Linear(256, 1, device=codec.device).eval()
    with torch.no_grad():
        probe.weight.fill_(1.0 / 256)
        probe.bias.zero_()
    failures = []
    for i, (recording, reference) in enumerate(zip(recordings, references)):
        frames = (len(recording) + HOP - 1) // HOP
        ref_pre = codec.encoder(F.pad(recording, (0, (-len(recording)) % HOP))[None, None].to(codec.device))
        batch_pre = codec.encoder(
            F.pad(recording, (0, (-len(recording)) % HOP))[None, None].to(codec.device)
        )
        for name, ref, got in [
            ("pre_vq", ref_pre, batch_pre),
            *[(key, getattr(reference, key), (batched.values if key == 'combined' else batched.streams[key])[i:i + 1, :frames].transpose(1, 2))
              for key in ("prosody", "content", "residual", "combined")],
        ]:
            mae, maximum, cosine = _stats(ref, got)
            print(f"{i}:{name:<22} {mae:12.3g} {maximum:12.3g} {cosine:12.6f}")
            if strict and name != "pre_vq":
                try:
                    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-4)
                except AssertionError as exc:
                    failures.append(f'{i}:{name}: {str(exc).splitlines()[0]}')
        ref_prediction = probe(reference.prosody.transpose(1, 2))
        batched_prediction = probe(batched.prosody[i:i + 1, :frames])
        mae, maximum, cosine = _stats(ref_prediction, batched_prediction)
        print(f'{i}:fixed_probe MAE={mae:.3g} max={maximum:.3g} cosine={cosine:.6f}')
        if strict:
            try:
                torch.testing.assert_close(batched_prediction, ref_prediction, atol=1e-5, rtol=1e-4)
            except AssertionError as exc:
                failures.append(f'{i}:fixed_probe: {str(exc).splitlines()[0]}')
        ref_ids = reference.codes[:, 0, :frames]
        got_ids = batched.codes[:, i, :frames]
        agreement = (ref_ids == got_ids).float().mean().item() * 100
        print(f"{i}:{'quantizer IDs':<22} {'':>12} {'':>12} {'':>12} {agreement:9.3f}%")
        if strict:
            if not torch.equal(ref_ids, got_ids):
                failures.append(f'{i}:quantizer IDs: {agreement:.3f}% agreement')
        mae, maximum, cosine = _stats(reference.speaker, batched.speaker[i:i + 1])
        print(f"{i}:{'speaker':<22} {mae:12.3g} {maximum:12.3g} {cosine:12.6f}")
        # Transformer batched floating-point execution can change speaker embeddings
        # slightly even when VQ IDs match; report this separately.
        if not strict and os.environ.get('FACODEC_PARITY_MAX_PROSODY_MAE'):
            limit = float(os.environ['FACODEC_PARITY_MAX_PROSODY_MAE'])
            assert _stats(reference.prosody, batched.prosody[i:i + 1, :frames].transpose(1, 2))[0] <= limit
        # Separate boundary from interior; the final 8 frames are diagnostic.
        ref = reference.prosody[0]
        got = batched.prosody[i, :frames].transpose(0, 1)
        for region, slc in (("interior", slice(None, max(0, frames - 8))),
                            ("last_8", slice(max(0, frames - 8), None))):
            if ref[:, slc].numel():
                mae, maximum, cosine = _stats(ref[:, slc], got[:, slc])
                print(f"{i}:prosody_{region:<14} {mae:12.3g} {maximum:12.3g} {cosine:12.6f}")

    if failures:
        pytest.fail('Strict parity mismatches:\\n' + '\\n'.join(failures), pytrace=False)


def test_batch_one_parity(codec):
    _compare(codec, [16000 + 37], "batch_one", strict=True)


def test_equal_length_parity(codec):
    _compare(codec, [16000, 16000, 16000, 16000], "equal_length", strict=True)


def test_variable_length_diagnostics(codec):
    _compare(codec, [16000, 12037, 8000, 15091], "variable_length", strict=True)


def test_batch_composition_diagnostics(codec):
    # Same first utterance, different padding contexts.
    _compare(codec, [12037, 12037], "same_length_partner", strict=True)
    _compare(codec, [12037, 24000], "longer_partner", strict=True)
