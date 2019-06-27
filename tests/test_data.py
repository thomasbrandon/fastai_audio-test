import pytest
from unittest.mock import Mock, sentinel, call
from .fixtures import *
from fastai_audio import *
import numpy as np
import matplotlib


class TestAudioTimeData():
    def test_load(self, make_sine_file):
        fn,sig = make_sine_file(sr=22050, duration=1.0)
        atd = AudioTimeData.load(fn)
        assert atd.rate == 22050
        assert atd.duration == 1.0
        assert len(atd) == 22050
        assert atd.channels == 1
        assert atd.shape == (1,22050)
        np.testing.assert_allclose(atd.sig.squeeze(), sig, atol=1e-4)

class TestAudioList():
    def test_from_folder(self, mocker, make_sample_folder):
        (folder,files) = make_sample_folder()
        mock_open = mocker.patch('fastai_audio.data.AudioList.open', return_value=sentinel.open_ret)
        al = AudioList.from_folder(folder)
        assert len(al) == len(files)
        for i in range(len(al)): assert al.get(i) == sentinel.open_ret
        assert mock_open.call_count == len(files)
        for (f,_) in files:
            mock_open.assert_any_call(f)

    @pytest.mark.parametrize('num_items,max_cols', [(3,5),(5,5),(5,3),(11,5)])
    def test_show_xys(self, mocker, make_mock_audio_list, num_items, max_cols):
        (al,items) = make_mock_audio_list(n=num_items)
        ys = [Mock(name=f'y{i}') for i in range(len(items))]
        axes = None
        def sp(rows,cols,**kwargs):
            assert cols <= max_cols
            assert rows*cols >= num_items
            fig = Mock(spec=matplotlib.figure.Figure)
            nonlocal axes
            axes = [Mock(spec=matplotlib.axes.Axes, name=f'axes{i}') for i in range(rows*cols)]
            return (fig,axes)
        mock_sp = mocker.patch('matplotlib.pyplot.subplots', side_effect=sp)
        AudioList.show_xys(al, items, ys, max_cols=max_cols)
        mock_sp.assert_called_once()
        for x,y,ax in zip(items,ys,axes):
            x.show.assert_called_with(ax)
            ax.set_title.assert_called_with(str(y))
        for ax in axes[num_items:]: ax.axis.assert_called_once_with('off')


def test_AudioDataInfo():
    adi = AudioDataInfo()
    assert list(adi.keys()) == []
    adi = AudioDataInfo({'val1':0})
    assert list(adi.keys()) == ['val1']
    assert adi['val1'] == 0
    assert adi.val1 == 0
    adi2 = adi.update({'val1': 1})
    assert adi2.val1 == 1
    assert adi.val1 == 0
    with pytest.raises(TypeError, match=".*Do not update values on an AudioDataInfo attribute.*"):
        adi.val1 = 1
    with pytest.raises(TypeError, match=".*Do not update values on an AudioDataInfo attribute.*"):
        adi['val1'] = 0


#FIXME: Not really testing our code
def test_transform(mock_audio_list):
    (al,items) = mock_audio_list
    for it in items: it.apply_tfms.return_value = sentinel.retval
    lls = al.split_none().label_const(0)
    tfm = Mock(spec=AudioTransform)
    tfm.process.retval = sentinel.proc_ret
    tfm.use_on_y = False #TODO: Probably shouldn't be required by ImageLists.transform
    tfmed = lls.transform(([tfm],None))
    for (it,_) in tfmed.train:
        assert it == sentinel.retval
    for it in items:
        it.apply_tfms.assert_called_once_with([tfm])

