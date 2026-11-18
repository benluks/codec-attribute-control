import numpy as np

from codec_attribute_control.plotting import pitch_curves_hz


def test_pitch_curves_are_hz_and_break_at_unvoiced_frames():
    reference = np.array([100., 110., 0., 120., 130.])
    # Zero z-score corresponds to the geometric mean of voiced F0.
    actual, reconstructed = pitch_curves_hz(reference, np.zeros(5), 5)
    assert np.allclose(actual[[0, 1, 3, 4]], reference[[0, 1, 3, 4]])
    assert np.isnan(actual[2]) and np.isnan(reconstructed[2])
    assert np.allclose(reconstructed[[0, 1, 3, 4]], np.exp(np.log(reference[[0, 1, 3, 4]]).mean()))


def test_inverse_transform_recovers_reference_contour():
    from codec_attribute_control.targets import f0_target

    reference = np.array([100., 110., 0., 120., 130.])
    z, _ = f0_target(reference)
    actual, predicted = pitch_curves_hz(reference, z, len(reference))
    np.testing.assert_allclose(predicted[[0, 1, 3, 4]], actual[[0, 1, 3, 4]], rtol=1e-6)
