from audiolib.samples import piano


def test_piano():
    """
    Test that the piano sample is loaded
    """
    piano_array, sampling_rate = piano()
    assert sampling_rate == 16000
    assert piano_array.shape[0] == 64000