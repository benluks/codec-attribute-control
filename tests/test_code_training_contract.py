from codec_attribute_control.code_training import PitchCodeTrainingModule


def test_pitch_code_training_module_is_concrete():
    assert not bool(getattr(PitchCodeTrainingModule, "__abstractmethods__", None))
