from .data import AudioData, AudioTimeData, AudioFreqData, AudioDataKind, AudioDataInfo
from fastai.basics import *
from abc import ABC, abstractmethod
from math import floor
from warnings import warn
from typing import Iterable, Generator, Union
import torch.nn.functional as F
import librosa
from .torchaudio_contrib import magphase, amplitude_to_db, MelFilterbank, apply_filterbank

__all__ = ['AudioTransform','AudioTransforms','ToDevice','ToFreq','ToDb','ToMel','PadTo','PadTrim','ToMono','Resample','Mask']

TIME,FREQ = AudioDataKind.TIME,AudioDataKind.FREQ
    
#TODO: Use decorator to extract parameters from librosa/torchaudio
#TODO: Track rates used and counts

class PerDeviceTensor():
    def __init__(self, item:Union[Tensor]):
        if not item.device.type == 'cpu': item = item.to('cpu')
        self.item = item
        self.dev_items = {}

    def __getitem__(self, dev):
        if dev == 'cpu' or torch.device(dev).type == 'cpu': return self.item
        it = self.dev_items.get(dev, None)
        if it is None:
            it = self.item.to(dev)
            self.dev_items[dev] = it
        return it

def _extract_kwargs(kwargs, arg_names):
    '''Extract `arg_names` arguments from `kwargs`. Returns tuple of (extracted,other).'''
    ex,kw= {},{}
    for k,v in kwargs.items:
        (ex if k in arg_names else kw)[k] = v
    return (ex,kw)

class AudioTransform(ABC):
    '''Base class for audio transforms. Can be either time-domain or frequency domain based on `kind`. A `kind` of `None` indicates it can be applied to either.'''

    def __init__(self, kind:AudioDataKind, out_kind:AudioDataKind=None, use_on_y:bool=False):
        self.kind,self.out_kind = kind,ifnone(out_kind,kind)
        self.use_on_y = use_on_y

    def __call__(self, data:AudioData)->AudioData:
        return self.process(data)

    @abstractmethod
    def process(self, data:AudioData)->AudioData:
        raise NotImplementedError()

    def process_info(self, info:dict)->dict:
        '''Verify the input `info` and update it based on this transform.'''
        if self.kind is not None and info.kind != self.kind:
            sk = 'untyped' if self.kind is None else self.kind.display
            ik = 'untyped' if info.kind is None else info.kind.display
            raise ValueError(f"Applying {self.__class__.__name__} which is a {sk} transform to {ik} data.")
        if self.out_kind is not None: info = info.update({'kind': self.out_kind})
        return info


def _check_transforms(tfms: TfmList):
    '''Check a set of transforms and ensure they can be applied.'''
    pass

def get_transforms(xtra_tfms:Optional[TfmList]=None, reorder=False)->Collection[AudioTransform]:
    tfms = []
    
    return tfms
    
TfmList = Union[AudioTransform, Collection[AudioTransform]]

class AudioTransforms():
    "Groups a set of `AudioTransform`s and applies them to items."

    #TODO: Allow specification of processing device(s)

    def __init__(self, tfms:TfmList):
        self.tfms = listify(tfms)
        
    def process_one(self, item:AudioData)->AudioData:
        for tfm in self.tfms:
            item = tfm.process(item)
        return item

    def process(self, items:Iterable[AudioData])->Generator[AudioData,None,None]:
        for it in items:
            yield self.process_one(it)

    def process_info(self, info:AudioDataInfo)->AudioDataInfo:
        for tfm in self.tfms:
            info = tfm.process_info(info)
        return info

# Basic transforms

class ToDevice(AudioTransform):
    # TODO: Set default to non_blocking, requires that contiguous data which torchaudio.load doesn't provide
    def __init__(self, device, non_blocking=False):
        super().__init__(None, use_on_y=True)
        if isinstance(device, str): device = torch.device(device)
        self.device,self.non_blocking = device,non_blocking

    def process(self, data:AudioData)->AudioData:
        data.sig = data.sig.to(self.device, non_blocking=self.non_blocking)
        return data

class ToFreq(AudioTransform):
    '''Converts time-domain data into frequency domain. See `torch.stft` and also `librosa.core.stft`. Parameters from both are accepted where they diverge.
       TODO: Note unique args and different default window to torch
    '''

    def __init__(self, n_fft:int=2048, hop_length:int=None, power:float=1, center=True, pad_mode='reflect',
                 win_length=None, window='hann', normalized=False, keep_phase=False, **kwargs):
        super().__init__(kind=TIME, out_kind=FREQ)

        # Also accept librosa pad_mode names
        if pad_mode == 'edge': pad_mode = 'replicate'
        if pad_mode == 'wrap': pad_mode = 'circular'
        self.n_fft,self.hop_length,self.center,self.win_length = n_fft,hop_length,center,win_length
        self.pad_mode,self.normalized,self.power,self.keep_phase = pad_mode,normalized,power,keep_phase

        if not isinstance(window, (np.ndarray, Tensor)):
            try:
                window = librosa.filters.get_window(window, ifnone(win_length, n_fft))
            except (TypeError,ValueError,librosa.ParameterError) as exc:
                raise ValueError("Error creating window. See librosa.filter.get_window for allowable methods.") from exc
        if isinstance(window, np.ndarray):
            window = as_tensor(window, dtype=torch.float32)
        if isinstance(window, torch.Tensor):
            if not window.dtype == torch.float32: window = window.to(dtype=torch.float32, device='cpu')
            self.window = PerDeviceTensor(window)
        else:
            raise TypeError(f"Got unexpected window type after creation. Expected numpy.ndarray or torch.Tensor. Got {type(window)}.")

    @property
    def stft_args(self):
        return {'n_fft': self.n_fft, 'hop_length': self.hop_length, 'center': self.center, 'win_length': self.win_length,
                'normalized': self.normalized, 'pad_mode': self.pad_mode}

    def process(self, data:AudioTimeData)->AudioFreqData:
        sig = data.sig
        window = self.window[sig.device] # Create device copy if needed
        stft = torch.stft(sig, window=window, **self.stft_args)
        mag,phase = magphase(stft, power=self.power)
        if data.rate == AudioDataInfo.MULTI: rate = AudioDataInfo.MULTI
        else: rate = data.rate / ifnone(self.hop_length, floor(self.n_fft / 4))
        if not self.keep_phase:
            phase = None
        return AudioFreqData(mag, rate, phase=phase)

    def process_info(self, info:AudioDataInfo)->AudioDataInfo:
        info = super().process_info(info)
        newrate = (info.rate / self.hop_length) if info.rate is not AudioDataInfo.MULTI else AudioDataInfo.MULTI
        return info.update({'orig_rate': info.rate, 'rate': newrate, 'power': self.power, 'n_fft': self.n_fft})

class ToDb(AudioTransform):
    '''Convert a signal from amplitude to dB. See `torchaudio_contrib.functional.amplitude_to_db`.'''

    #TODO: Option to use mean as ref to normalise
    def __init__(self, ref:float=1.0, amin:float=1e-7):
        super().__init__(kind=None)
        self.ref,self.amin = ref,amin

    def process(self, data:AudioData)->AudioData:
        data.sig = amplitude_to_db(data.sig, ref=self.ref, amin=self.amin)
        return data

    def process_info(self, info:AudioDataInfo)->AudioDataInfo:
        return super().process_info(info).update({'y_scale': 'log'})

class ToMel(AudioTransform):
    '''Converts a spectrogram to mel-scaling. See `librosa.feature.melspectrogram`'''

    def __init__(self, n_mels:int=128, fmin:float=0.0, fmax:float=None, htk:bool=False):
        super().__init__(kind=FREQ)
        self.n_mels,self.fmin,self.fmax,self.htk = n_mels,fmin,fmax,htk
        self._args = {'num_bands': n_mels, 'min_freq': fmin, 'max_freq': fmax, 'htk': htk}
        self._melfb = None

    @staticmethod
    def create_filter(rate, n_fft, **args):
        return MelFilterbank(num_bins=n_fft//2+1, sample_rate=rate, **args).get_filterbank()

    def process(self, data:AudioFreqData)->AudioFreqData:
        melfb = self._melfb[data.sig.device] # Get per device filter
        data.sig = apply_filterbank(data.sig, melfb)
        return data

    def process_info(self, info:AudioDataInfo)->AudioDataInfo:
        info = super().process_info(info)
        if 'n_fft' not in info: raise TypeError("Trying to apply ToMel before data has been converted to frequency domain. Add a ToFreq transform first.")
        if info.rate == AudioDataInfo.MULTI: raise TypeError("Trying to apply ToMel when multiple sample rates are present. Resample first.")
        if self._melfb is None:
            melfb = ToMel.create_filter(info.orig_rate, info.n_fft, **self._args)
            self._melfb = PerDeviceTensor(tensor(melfb))
        return info.update({'y_scale': 'mel', 'fmin': self.fmin, 'fmax': self.fmax})

# TODO: Is duration samples or seconds?
class PadTo(AudioTransform):
    '''Pad the input to be `to` samples long.'''
    def __init__(self, to, mode='constant', value=0):
        super().__init__(TIME)
        self.to,self.mode,self.value = to,mode,value

    def process(self, data:AudioTimeData):
        pad = self.to - data.sig.shape[-1]
        data.sig = F.pad(data.sig, (0,pad), mode=self.mode, value=self.value)
        return data

    def process_info(self, info:AudioDataInfo)->AudioDataInfo:
        return super().process_info(info).update({'duration': self.to})

# Port of scipy.signal.resample to PyTorch
def resample_torch(x:Tensor, num:int, dim:int=-1)->Tensor:
    dim = (x.dim() + dim) if dim < 0 else dim
    X = torch.rfft(x, 1, onesided=False)
    Nx = X.shape[dim]
    sl = [slice(None)] * X.ndimension()
    newshape = list(X.shape)
    newshape[dim] = num
    N = int(np.minimum(num, Nx))
    Y = torch.zeros(newshape, dtype=X.dtype, device=X.device)
    sl[dim] = slice(0, (N + 1) // 2)
    Y[sl] = X[sl]
    sl[dim] = slice(-(N - 1) // 2, None)
    Y[sl] = X[sl]
    if N % 2 == 0:  # special treatment if low number of points is even. So far we have set Y[-N/2]=X[-N/2]
        if N < Nx:  # if downsampling
            sl[dim] = slice(N//2, N//2+1,None)  # select the component at frequency N/2
            Y[sl] += X[sl] # add the component of X at N/2
        elif N < num:  # if upsampling
            sl[dim] = slice(num-N//2,num-N//2+1,None)  # select the component at frequency -N/2
            Y[sl] /= 2  # halve the component at -N/2
            temp = Y[sl]
            sl[dim] = slice(N//2,N//2+1,None)  # select the component at +N/2
            Y[sl] = temp # set that equal to the component at -N/2
    y = torch.irfft(Y, 1, onesided=False) * (float(num) / float(Nx))
    return y

class Resample(AudioTransform):
    def __init__(self, rate):
        super().__init__(TIME)
        self.rate = rate

    def process(self, data:AudioData):
        num = int(data.shape[-1] / data.rate * self.rate)
        data.sig = resample_torch(data.sig, num)
        data.rate = self.rate
        return data

    def process_info(self, info):
        return super().process_info(info).update({'rate': self.rate})

class ToMono(AudioTransform):
    def __init__(self):
        super().__init__(None)

    def process(self, data:AudioData):
        if data.channels > 1: data.sig = data.sig.mean(-2, keepdim=True)
        return data

    def process_info(self, info):
        return super().process_info(info).update({'channels': 1})
    
# TODO: Implement
class Trim(AudioTransform):
    def __init__(self, length):
        super().__init__(None)

    def process(self, data:AudioData):
        raise NotImplementedError()

    def process_info(self, info):
        raise NotImplementedError()

class PadTrim(AudioTransform):
    '''Either pad or trim the input to be `to` samples long.'''
    def __init__(self, to, mode='constant', value=0):
        super().__init__(TIME)
        self.to,self.mode,self.value = to,mode,value

    def process(self, data:AudioTimeData):
        if data.sig.shape[-1] < self.to:
            pad = self.to - data.sig.shape[-1]
            data.sig = F.pad(data.sig, (0,pad), mode=self.mode, value=self.value)
        else: data.sig = data.sig[...,:self.to]
        return data

    def process_info(self, info:AudioDataInfo)->AudioDataInfo:
        return super().process_info(info).update({'duration': self.to})

# class Normalize(AudioTransform):
#     def __init__(self, mean, std):
#         super().__init__(TIME)
#         self.mean,self.std = mean,std

def check_transform_info(info: AudioDataInfo, tfms:Collection[AudioTransform]):
    msgs = []
    def msg(prob, opt=None, tfm=None):
        msg = f"{prob} present in output. To resolve this "
        if opt: msg += f"set {opt} when creating the AudioList"
        if opt and tfm: msg += " or "
        if tfm: msg += f"add a {tfm} transform."
        msgs.append(msg)
    def has(tfm_cls):
        for tfm in tfms:
            if isinstance(tfm, tfm_cls): return True
        return False
    if info.channels == AudioDataInfo.MULTI and not has(ToMono):
        msg("Both mono and stereo", "mono", "ToMono")
    if info.duration == AudioDataInfo.MULTI and not has((PadTo,Trim,PadTrim)):
        msg("Multiple lengths", "duration", "PadTo or Trim")
    det = (" To see details on files causing issues run read_data_info(print=True) on the data bunch. " + 
           "To skip this check set check_info=False when creating the AudioList.")
    if msgs:
        msgs.insert(0, "Input data is not uniform." + det)
        raise ValueError('\n'.join(msgs))
    if info.rate == AudioDataInfo.MULTI and not has(Resample):
        warn("Multiple sample rates present in input. You probably want to add a Resample transform." + det)


# Data Augmentations

def mask_(sig, n_f, max_f, n_t, max_t, val):
    '''Mask `sig` inplace with `n_f` frequency masks with size from `(0, max_f)`
        and `n_t` time masks with size from `(0, max_t), masking to `val`.'''
    for f in np.random.randint(max_f + 1, size=n_f):
        f0 = np.random.randint(sig.shape[-2] - f)
        sig[...,f0:f0+f,:] = val
    for t in np.random.randint(max_t + 1, size=n_t):
        t0 = np.random.randint(sig.shape[-1] - t)
        sig[...,:,t0:t0+t] = val
    return sig

class Mask(AudioTransform):
    ''' The SpecAugment augmentation from https://arxiv.org/pdf/1904.08779.pdf
        The idea here is to mask particular time or frequency bands.
    '''

    def __init__(self, n_f, max_f, n_t, max_t, val=0):
        '''Create a SpecAugment schedule with  `n_f` frequency masks with size from `(0, max_f)`
           and `n_t` time masks with size from `(0, max_t), masking to `val`.'''
        super().__init__(FREQ)
        self.n_f,self.max_f,self.n_t,self.max_t,self.val = n_f,max_f,n_t,max_t,val

    def process(self, data:AudioFreqData):
        mask_(data.sig, self.n_f, self.max_f, self.n_t, self.max_t, self.val)
        return data
