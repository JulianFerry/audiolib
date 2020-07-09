import os
import numpy as np
from scipy.io import wavfile
from audiolib.audio import Audio
from audiolib.spectrogram import Spectrogram


DIR = os.path.dirname(os.path.abspath(__file__))
sampling_rate, audio = wavfile.read(os.path.join(DIR, 'data/piano.wav'))

class TestAudio:

    piano = Audio(audio, sampling_rate)

    def test_init(self):
        """
        Test that the class has been initalised accordingly

        """
        assert isinstance(self.piano, np.ndarray)
        assert self.piano.shape[0] == 64000
        assert self.piano.sampling_rate == 16000
        assert self.piano.nyquist == 8000
        assert self.piano.duration == 4
        
    def test_trim(self):
        """
        Test that the audio is trimmed

        """
        piano_trimmed = self.piano.trim(start=1, end=2)
        assert piano_trimmed.shape[0] == 16000
        assert piano_trimmed[0] == self.piano[16000]

    def test_get_n_fft(self):
        """
        Test that n_fft gets calculated

        """
        # Should return the number of samples
        n_fft1 = self.piano._get_n_fft(resolution=1, mode='max')
        assert n_fft1 == 64000

        # Should return half the number of samples
        n_fft2 = self.piano._get_n_fft(resolution=0.5, mode='max')
        assert n_fft2 == 32000

        # Should return the largest power of 2 smaller than the number of samples
        n_fft3 = self.piano._get_n_fft(resolution=1, mode='fast')
        assert np.log2(n_fft3) % 1 == 0
        assert 2 * n_fft3 > self.piano.shape[0]

    def test_get_cqt_params(self):
        """
        Test that the cqt parameters get calculated

        """
        # One octave
        cqt_params = self.piano._get_cqt_params(
            time_intervals=1, note_resolution=1, fmin=100, fmax=200)
        assert cqt_params['n_bins'] == 12
        assert cqt_params['bins_per_octave'] == 12
        assert cqt_params['hop_length'] == 64000

        # One octave, ten time intervals
        cqt_params = self.piano._get_cqt_params(
            time_intervals=10, note_resolution=1, fmin=100, fmax=200)
        assert cqt_params['hop_length'] == 6400

        # Two octaves
        cqt_params = self.piano._get_cqt_params(
            time_intervals=1, note_resolution=1, fmin=100, fmax=400)
        assert cqt_params['n_bins'] == 24
        assert cqt_params['bins_per_octave'] == 12

        # One octave, double note resolution
        cqt_params = self.piano._get_cqt_params(
            time_intervals=1, note_resolution=2, fmin=100, fmax=200)
        assert cqt_params['n_bins'] == 24
        assert cqt_params['bins_per_octave'] == 24

    def test_to_spectrogram(self):
        """
        Test that the audio converts to a spectrogram

        """
        spec = self.piano.to_spectrogram(time_intervals=1, resolution=1, mode='max')
        assert isinstance(spec, Spectrogram)
        assert spec.cqt == False
        assert spec.sampling_rate == self.piano.sampling_rate
        assert spec.params['hop_length'] == self.piano.shape[0] + 1

    def test_to_cqt_spectrogram(self):
        """
        Test that the audio converts to a CQT spectrogram

        """
        spec = self.piano._to_cqt_spectrogram(time_intervals=1, resolution=1)
        assert isinstance(spec, Spectrogram)
        assert spec.cqt == True
        assert spec.sampling_rate == self.piano.sampling_rate
        assert spec.params['hop_length'] % 2 == 0
        assert spec.params['bins_per_octave'] == 12
