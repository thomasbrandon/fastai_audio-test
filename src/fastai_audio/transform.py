from .data import AudioList, AudioItem, AudioData, AudioTimeData, AudioFreqData, AudioDataKind, AudioDataInfo
from fastai.basics import *
from abc import ABC, abstractmethod
from math import floor
from warnings import warn
from typing import Iterable, Generator, Union, Optional
import torch.nn.functional as F
import librosa
from .torchaudio_contrib import magphase, amplitude_to_db, MelFilterbank, apply_filterbank

__all__ = ['AudioTransform','AudioTransforms','ToDevice','ToFreq','ToDb','ToLog','ToMel','PadTo','PadTrim',
           'ToMono','Resample','Mask','AudioStatistics','StatsRecorder','RecordStats','collect_stats','Normalize']

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
        return super().process_info(info).update({'y_scale': 'db'})

class ToLog(AudioTransform):
    '''Convert a signal to logirthmic scale computing `log10(scale * S + eps)` for signal S.'''

    def __init__(self, scale:float=1.0, eps:float=1e-7):
        super().__init__(kind=None)
        self.scale,self.eps = scale,eps

    def process(self, data:AudioData)->AudioData:
        if self.scale != 1.0: data.sig.mul_(self.scale)
        data.sig.add_(self.eps).log10_()
        return data

    def process_info(self, info:AudioDataInfo)->AudioDataInfo:
        return super().process_info(info).update({'y_scale': 'log'})

# TODO: Fix it so doesn't error if procesS_info is not called before process
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

# Statistics and Normalisation

@dataclass
class AudioStatistics():
    n_items: int
    mean: Tensor
    var: Tensor
    std: Tensor
    min: Optional[Tensor]
    max: Optional[Tensor]

    @property
    def values(self):
        vals = {}
        agg = len(self.mean.shape) > 1
        for n in ['mean','var','std']:
            v = getattr(self,n)
            if agg: v = v.mean()
            vals[n] = v.item()
        if self.min is not None:
            vals['min'] = (self.min.min() if agg else self.min).item()
            vals['max'] = (self.max.max() if agg else self.max).item()
        return vals

    def __str__(self):
        if self.n_items == 0: return "No statistics collected."
        agg = len(self.mean.shape) > 1
        s = (f"Statistics of {self.n_items} items"
            + (f" (aggregate of {'x'.join(map(str,self.mean.shape))} data)"
               if agg else '')
            + ":\n")
        vals = self.values
        # Align values on decimal point
        fvs = {n: (     f'{v:3.3f}' if 1e4>abs(v)>1e-4
                   else f'{v:.3e}'
                  ).split('.') for n,v in vals.items()}
        mdp = max([len(v[0]) for v in fvs.values()])
        ml = max(map(len, fvs.keys()))
        s += '\n'.join([f" {n:>{ml}}: {w:>{mdp}}.{d}" for n,(w,d) in fvs.items()])
        return s

class StatsRecorder:
    '''Records mean and variance of the final dimension over other dimensions across items. So collecting across `(m,n,o)` sized
       items will collect `(m,n)` sized statistics.

       Uses the algorithm from Chan, Golub, and LeVeque in "Algorithms for computing the sample variance: analysis and recommendations":

       `variance = variance1 + variance2 + n/(m*(m+n)) * pow(((m/n)*t1 - t2), 2)`

       This combines the variance for 2 blocks: block 1 having `n` elements with `variance1` and a sum of `t1` and block 2 having `m` elements
       with `variance2` and a sum of `t2`. The algorithm is proven to be numerically stable but there is a reasonable loss of accuracy (~0.1% error).

       Note that collecting minimum and maximum values is reasonably innefficient, adding about 80% to the running time, and hence is disabled by default.
    '''
    def __init__(self, record_range=False):
        self.n_items,self.n,self._range = 0,0,record_range
        self.nvar,self.sum,self.min,self.max = None,None,None,None
    
    def update(self, data):
        self.n_items += 1
        with torch.no_grad():
            new_n,new_var,new_sum = data.shape[-1],data.var(-1),data.sum(-1)
            if self.n == 0:
                self.n = new_n
                self._shape = data.shape[:-1]
                self._sum = new_sum
                self._nvar = new_var.mul_(new_n)
                if self._range:
                    self.min = data.min(-1)[0]
                    self.max = data.max(-1)[0]
            else:
                assert data.shape[:-1] == self._shape, "Mismatched shapes, expected {self._shape} but got {data._shape[:-1]}."
                ratio = self.n / new_n
                t = (self._sum / ratio).sub_(new_sum).pow_(2)
                self._nvar.add_(new_n, new_var).add_(ratio / (self.n + new_n), t)
                self._sum.add_(new_sum)
                self.n += new_n
                if self._range:
                    self.min = torch.min(self.min, data.min(-1)[0])
                    self.max = torch.max(self.max, data.max(-1)[0])

    @property
    def mean(self): return self._sum / self.n if self.n > 0 else None
    @property
    def var(self): return self._nvar / self.n if self.n > 0 else None
    @property
    def std(self): return self.var.sqrt() if self.n > 0 else None

    @property
    def stats(self):
        return AudioStatistics(
            self.n_items, self.mean, self.var, self.std,
            self.min if self._range else None,
            self.max if self._range else None,
        )

class RecordStats(AudioTransform):
    '''A transform that records statistics (mean, variance, range) of items. Items can be any shape
       but must all be the same shape, statistics are computed across the last dimension.'''
    def __init__(self, record_range=False):
        super().__init__(kind=None)
        self.stats = StatsRecorder(record_range=record_range)

    def process(self, data:AudioData)->AudioData:
        self.stats.update(data.sig)
        return data

AudioCollection = Union[AudioList,Collection[Union[AudioItem,AudioData,Tensor]]]

def tfmed_items(items: AudioCollection, tfms: TfmList=None, progress=False):
    if progress: items = progress_bar(items)
    for it in items:
        if isinstance(it, AudioItem): 
            yield it.apply_tfms(tfms).data
        else:
            if isinstance(it, AudioData): it = it.sig
            assert isinstance(it, Tensor), "Items should be a collection of tensors or audio items."
            for tfm in ifnone(tfms, []): it = tfm(it)
            yield it

def collect_stats(data: AudioCollection, tfms: TfmList=None,
                  record_range=False, progress=True) -> AudioStatistics:
    '''Collect statistics on the items in `data` (e.g. an `AudioList`), after applying optional `tfms`.'''
    rec = StatsRecorder(record_range=record_range)
    for it in tfmed_items(data, tfms, progress):
        rec.update(it)
    return rec.stats

class Normalize(AudioTransform):
    def __init__(self, stats:Union[AudioStatistics,Tuple[Union[float,Tensor],Union[float,Tensor]]]):
        super().__init__(None)
        if isinstance(stats, AudioStatistics):
            mean,std = stats.mean,stats.std
        else:
            mean,std = Tensor(stats[0]),Tensor(stats[1])
        # Need singleton final dimension for broadcasting
        if mean.ndim != 0 and mean.shape[-1] != 1: mean = mean[...,None]
        if  std.ndim != 0 and  std.shape[-1] != 1: std  =  std[...,None]
        self.mean,self.std = PerDeviceTensor(mean),PerDeviceTensor(std)

    def process(self, data:AudioData)->AudioData:
        dev = data.sig.device
        data.sig.sub_(self.mean[dev])\
                .div_(self.std[dev])
        return data

    def process_info(self, info):
        return super().process_info(info).update({'norm_stats':(self.mean,self.std)})
