import json
import re
import librosa
import librosa.display
from functools import partial
from pathlib import Path
from scipy.io import wavfile
from warnings import warn

# Fix circular imports with absolute import
from importlib import import_module
_audio = import_module(f'audiolib.audio')


class AudioDataset:
    """
    Loads audio file names and metadata from an NSynth dataset.

    Parameters
    ----------
    path: str or pathlib.Path
        Path to the NSynth dataset folder

    Attributes
    ----------
    path
        Path to the root directory of the dataset
    examples
        Metadata supplied with the dataset
    file_names
        Dictionary of file names, with instrument name as the key
    file_names_nested
        Nested version of file_names, with instrument ID as the nested key
    unique_instruments
        List of unique instrument names in the dataset
    file_counts
        Number of files found for each instrument

    Methods
    -------
    __init__
    load_file
        Load audio file from the dataset as an `AudioFile` object

    """

    def __init__(
        self,
        path: str = '../data/raw/nsynth-train/'
    ):
        """
        Load the file names and metadata from a root directory.

        Parameters
        ----------
        See AudioDataset class docstring

        """
        self.path = path if type(path) is Path else Path(path)
        # Metadata
        self.examples = json.load(open(self.path/'examples.json'))
        # File names per instrument
        self.file_names = {}
        self.file_names_nested = {}
        re_instrument = partial(re.findall, r'(^.*?)_\d')
        for file in self.examples.keys():
            prefix = self.examples[file]['instrument_str']
            instrument = re_instrument(prefix)[0]
            (self.file_names.setdefault(instrument, [])
                            .append(file))
            (self.file_names_nested.setdefault(instrument, {})
                                   .setdefault(prefix, [])
                                   .append(file))
        # Unique instruments
        self.unique_instruments = list(self.file_names.keys())
        # File counts
        self.file_counts = {instr: len(self.file_names[instr])
                            for instr in self.unique_instruments}

    def _check_instrument(
        self,
        instrument: str
    ):
        """
        Check that the instrument name exists in the dataset

        """
        if instrument not in self.unique_instruments:
            raise(ValueError('Instrument {} is not one of {}'.format(
                instrument, self.unique_instruments)))

    def load_file(
        self,
        file_name: str = None,
        instrument: str = None,
        file_index: int = None
    ):
        """
        Loads an audio file from the dataset as an `AudioFile` object.

        There are two ways of loading a file:

        * Either specify `file_name` to fetch an NSynth file by name
        * Alternatively, restrict files to those starting with the `instrument` string
          and use `file_index` to reference the file index within that subset of files

        Parameters
        ----------
        file_name: str
            Name of the audio file
        instrument: str
            Must match one of the AudioDataset's unique_instruments
        file_index: int
            Index of the file within that instrument's file_names

        Returns
        -------
        audiofile: audiolib.nsynth.AudioFile

        """
        if file_name is not None:
            file = Path(file_name + '.wav')
        elif instrument is not None and file_index is not None:
            self._check_instrument(instrument)
            file = Path(self.file_names[instrument][file_index] + '.wav')
        else:
            raise(ValueError(
                'A file_name or both instrument and file_index must be specified'))
        path = self.path/'audio'/file.name
        info = self.examples[file.stem]
        return AudioFile(path, info)


class AudioFile:
    """
    Loads audio data from a .wav file.

    The main way to interact with the AudioFile is via its `audio` attribute: an
    `audiolib.audio.Audio` object which defines methods for audio waveform analysis.

    Parameters
    ----------
    path: str
        Path to the audio .wav file
    info: dict
        NSynth metadata (from examples.json)

    Attributes
    ----------
    path
        Path to the audio file
    info
        Metadata supplied by the audio dataset
    audio
        `Audio` object which defines methods to analyse audio waveforms.
    
    Methods
    -------
    __init__
    reload
        Reload Audio raw data (removes any previously applied processing)

    """

    def __init__(
        self,
        path: str,
        info: dict
    ):
        """
        Load audio data from a .wav file.

        Parameters
        ----------
        See AudioFile class docstring

        """
        self.path = path
        self.info = info
        if self.info.get('pitch') is None:
            warn('No pitch information found. Some functionality will not work')
        self.reload()

    def reload(self):
        """
        Reload raw audio data from the AudioFile's path.

        Automatically called on object creation.

        """
        sampling_rate, audio = wavfile.read(self.path)
        if self.info.get('pitch'):
            fundamental_freq = librosa.midi_to_hz(self.info['pitch'])
        else:
            fundamental_freq = None
        self.audio = _audio.Audio(audio, sampling_rate, fundamental_freq)
