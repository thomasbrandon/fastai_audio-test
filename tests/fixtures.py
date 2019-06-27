import pytest
from unittest.mock import Mock
import numpy as np
import soundfile as sf

from fastai.data_block import ItemLists
from fastai_audio import AudioList, AudioItem

@pytest.fixture
def make_sine():
    def _make(sr=22050, n_samples=22050, duration=None, freq=440.0):
        if duration:
            n_samples = duration * sr
        samples = (np.sin(2 * np.pi * np.arange(n_samples) * freq / sr) * 0.8 ).astype(np.float32)
        return samples
    return _make

@pytest.fixture
def make_sine_file(tmp_path_factory, make_sine):
    def_tmp_dir = tmp_path_factory.mktemp('sines')

    def _make(sr=22050, n_samples=22050, duration=None, freq=440.0, format='wav', fn=None, tmp_dir=None):
        samples = make_sine(sr=sr, n_samples=n_samples, duration=duration, freq=freq)
        if duration: n_samples = duration * sr
        if not tmp_dir: tmp_dir = def_tmp_dir
        if not fn: fn = f'{sr}_{n_samples}.{format}'
        if not (tmp_dir/fn).exists(): sf.write(tmp_dir/fn, samples, sr)
        return (tmp_dir/fn, samples)
    return _make

@pytest.fixture
def make_sample_folder(tmp_path_factory, make_sine_file):
    def _make(n=5, format='wav'):
        tmp_dir = tmp_path_factory.mktemp('samples')
        files = [make_sine_file(format=format, fn=f'sample{i}.{format}', tmp_dir=tmp_dir) for i in range(n)]
        return (tmp_dir, files)
    return _make

@pytest.fixture
def make_mock_audio_list():
    '''Creates a mock `AudioList`. Returns `(Audioist,items)`.'''
    def make(n=5):
        items = [Mock(spec=AudioItem, name=f'item{i}') for i in range(n)]
        al = AudioList(np.array(items), check_info=False)
        al.get = Mock(side_effect=items)
        return (al,items)
    return make

@pytest.fixture
def mock_audio_list(make_mock_audio_list):
    return make_mock_audio_list()

@pytest.fixture
def make_audio_list(make_sample_folder):
    def make(n=5, format='wav'):
        (folder,files) = make_sample_folder(n=n, format=format)
        al = AudioList.from_folder(folder)
        return al
    return make
