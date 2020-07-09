import os
import numpy as np
import pickle
from scipy.io import wavfile
from audiolib.audio import Audio
from audiolib.spectrogram import Spectrogram


DIR = os.path.dirname(os.path.abspath(__file__))

class TestSpectrogram:

    def test_load(self):
        """
        Test that the test data loads

        """
        # Normal spectrogram
        with open(os.path.join(DIR, 'data/spectrogram.npy'), 'rb') as f:
            spec = pickle.load(f)
        assert isinstance(spec, np.ndarray)
        assert isinstance(spec, Spectrogram)
        assert spec.cqt == False

        # CQT spectrogram
        with open(os.path.join(DIR, 'data/spectrogram_cqt.npy'), 'rb') as f:
            spec = pickle.load(f)
        assert isinstance(spec, np.ndarray)
        assert isinstance(spec, Spectrogram)
        assert spec.cqt == True

    def test_filter_harmonics(self):
        """
        Test that the harmonics are filtered

        """
        with open(os.path.join(DIR, 'data/spectrogram.npy'), 'rb') as f:
            spec = pickle.load(f)
            spec.fundamental_freq = 55

        spec_filtered = spec.filter_harmonics(neighbour_radius=0)
        harmonic_step = int(55 * spec.shape[0] / spec.nyquist)
        assert spec_filtered[harmonic_step, 0] == spec[harmonic_step, 0] != 0
        assert spec_filtered[harmonic_step * 2, 0] == spec[harmonic_step * 2, 0]
        assert spec_filtered[harmonic_step + 1, 0] == 0

        spec_filtered = spec.filter_harmonics(neighbour_radius=1)
        assert spec_filtered[harmonic_step + 1, 0] == spec[harmonic_step + 1, 0]

    def test_to_audio(self):
        """
        Test that the spectrogram converts back to the original audio

        """
        # Load test data generated with:
        # with open('tests/data/spectrogram.npy', 'wb') as f:
        #     pickle.dump(Audio(*piano()).to_spectrogram(time_intervals=10), f)
        with open(os.path.join(DIR, 'data/spectrogram.npy'), 'rb') as f:
            spec = pickle.load(f)
        # Test function call
        recovered_audio = spec.to_audio()
        # Load original audio
        sampling_rate, audio = wavfile.read(os.path.join(DIR, 'data/piano.wav'))
        audio = audio[:recovered_audio.shape[0]]
        # Assert equal
        assert isinstance(recovered_audio, Audio)
        np.testing.assert_array_almost_equal(np.asarray(recovered_audio), np.asarray(audio), decimal=0)
