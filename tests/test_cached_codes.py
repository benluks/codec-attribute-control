import torch

from codec_attribute_control.code_system import PitchToProsodyCodeSystem


def test_cached_codes_preserve_values_and_lengths():
    class FakeResource:
        values = torch.tensor([[[0], [1023], [7]]], dtype=torch.uint16)
        lengths = torch.tensor([3])

    class TestSystem(PitchToProsodyCodeSystem):
        def get_resource(self, batch, name):
            assert name == "prosody_codes"
            return FakeResource()

    system = TestSystem(head=torch.nn.Identity(), code_source="cached")
    codes, lengths = system(object())
    assert codes.dtype == torch.long
    assert codes.tolist() == [[0, 1023, 7]]
    assert lengths.tolist() == [3]


def test_existing_code_vectors_collate_as_sequences(tmp_path):
    from dataclasses import replace
    from importlib.resources import files

    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from quick_convert.data.resources.base import ResourceCollection
    from quick_convert.data.resources.loader import load_resource
    from quick_convert.data.types import AudioBatch, AudioSample

    config = OmegaConf.load(
        files("codec_attribute_control").joinpath("configs/codes/cached.yaml")
    )
    config.codes_cache_root = str(tmp_path)
    (provider,) = instantiate(config.resource_providers)
    split = tmp_path / "test-clean"
    split.mkdir()
    samples = []
    for name, codes in [("a", [1, 1023]), ("b", [4])]:
        torch.save(torch.tensor(codes, dtype=torch.uint16), split / f"{name}.pt")
        sample = AudioSample(
            utt_id=name, path=split / f"{name}.flac", split="test-clean"
        )
        resources = ResourceCollection.from_refs([load_resource(provider(sample))])
        samples.append(replace(sample, resources=resources))
    batch = AudioBatch.from_samples(samples)
    resource = batch.resources["prosody_codes"]
    assert resource.values.tolist() == [[1, 1023], [4, 0]]
    assert resource.lengths.tolist() == [2, 1]
