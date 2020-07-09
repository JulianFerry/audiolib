try:
    import importlib.resources as pkg_resources  # py3.7
except ImportError:
    import importlib_resources as pkg_resources
from scipy.io import wavfile


def piano():
    with pkg_resources.path(__name__, 'piano.wav') as piano_file:
        sampling_rate, audio = wavfile.read(piano_file)
        return audio, sampling_rate
