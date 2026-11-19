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
