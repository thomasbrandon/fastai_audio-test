import pytest
from torch.testing import assert_allclose
from .fixtures import *
from math import ceil
from unittest.mock import Mock, sentinel

from fastai.basics import *
from fastai_audio.data import *
from fastai_audio.transform import *

class TestToFreq():
    def test_defaults(self):
        '''Verify defaults are the same as pytorch'''
        def test(func, vals):
            co = func.__code__
            defs = func.__defaults__
            defs = {k:v for k,v in zip(co.co_varnames[co.co_argcount - len(defs):co.co_argcount], defs)}
            for k in defs:
                if k in vals: assert vals[k] == defs[k]
        import librosa
        tf = ToFreq()
        test(torch.stft, tf.stft_args)

    def test_window(self, mocker):
        def check_win(tf, exp):
            win = to_np(tf.window['cpu'])
            assert win is exp or np.array_equal(win, exp)
        n_fft=16
        retwin = np.zeros(16)
        mock_gw = mocker.patch('librosa.filters.get_window', return_value=retwin)
        # Default should be hann
        tf = ToFreq(n_fft=n_fft)
        mock_gw.assert_called_once_with('hann', n_fft)
        check_win(tf, retwin)
        mock_gw.reset_mock()
        # Strings and tuples should call get_window
        for winparam in ['hahn', ('win',1)]:
            tf = ToFreq(n_fft=n_fft, window=winparam)
            mock_gw.assert_called_once_with(winparam, n_fft)
            check_win(tf, retwin)
            mock_gw.reset_mock()
        mocker.stopall()
        # numpy array and tensor
        tf = ToFreq(n_fft=n_fft, window=retwin)
        check_win(tf, retwin)
        tf = ToFreq(n_fft=n_fft, window=tensor(retwin))
        check_win(tf, retwin)
        # torch window function
        tf = ToFreq(n_fft=n_fft, window=torch.hann_window)
        check_win(tf, to_np(torch.hann_window(n_fft)))
        # Bad window function
        with pytest.raises(TypeError, match=r".*Got unexpected window type after creation.*"):
            ToFreq(window=lambda x: "Bad")

    @pytest.mark.parametrize("n_samples,n_fft,hop_length", [
        (2048,2048,512),(2048,2048,1024),
        (2048,1000,500),(2048,1000,250),
        (2047,2048,512),(2049,2048,512)])
    def test_output_shape(self, make_sine, n_samples, n_fft, hop_length):
        samples = Tensor(make_sine(sr=16000, n_samples=n_samples), device="cpu")
        assert len(samples) == n_samples
        ad = AudioTimeData(samples, 16000)
        tf = ToFreq(n_fft=n_fft, hop_length=hop_length)
        res = tf.process(ad)
        n_frames = (n_samples // hop_length) + 1
        assert res.shape == (ceil(1+n_fft/2), # Frequencies
                             n_frames) # Frames

    def test_info(self):
        tf = ToFreq(n_fft=1024, hop_length=1024)
        info = tf.process_info(AudioDataInfo({'kind':AudioDataKind.TIME, 'rate': 2048}))
        assert info.rate == 2048/1024
        assert info.power == 1

class TestTransforms():
    def test_process_one(self):
        tfms,items = [],[Mock(spec=AudioData)]
        for i in range(4):
            tfm = Mock(spec=AudioTransform)
            item = Mock(spec=AudioData)
            tfm.process.return_value = item
            tfms.append(tfm)
            items.append(item)
        
        ats = AudioTransforms(tfms)
        assert ats.process_one(items[0]) is items[-1]
        for i in range(4):
            tfms[i].process.assert_called_once_with(items[i])

    def test_process_info(self):
        tfms,infos = [],[Mock(spec=dict)]
        for i in range(4):
            tfm = Mock(spec=AudioTransform)
            info = Mock(spec=dict)
            tfm.process_info.return_value = info
            tfms.append(tfm)
            infos.append(info)
        
        ats = AudioTransforms(tfms)
        assert ats.process_info(infos[0]) == infos[-1]
        for i in range(4):
            tfms[i].process_info.assert_called_once_with(infos[i])
        
    def test_callable(self):
        class Tfm(AudioTransform):
            def __init__(self):
                self.process = Mock()
                self.process.return_value = sentinel.proced
                self.process_info = Mock()
            def process(self,ad): pass
            def process_info(self,info): pass
        tfm = Tfm() #Instantiate class
        ad = Mock(spec=AudioData)
        assert tfm(ad) is sentinel.proced # Call instance
        tfm.process.assert_called_once_with(ad)
