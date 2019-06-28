from fastai.basics import *
from abc import ABC, abstractmethod
from typing import Union
import librosa
import torchaudio
from librosa.display import specshow, waveplot
from IPython.display import display, Audio
import mimetypes
from warnings import warn

__all__ = ['AudioItem','AudioDataBunch','AudioList','AudioData','AudioTimeData','AudioFreqData','AudioDataKind','AudioDataInfo','ReadInfoProcessor']

AUDIO_EXTENSIONS = tuple(str.lower(k) for k,v in mimetypes.types_map.items() 
                         if v.startswith('audio/'))

# TODO: Handle resampling
# TODO: Handle stereo

def _print_trace(msg):
    import traceback as tb
    tr = ''.join(tb.format_stack(limit=8))
    print(f'{msg}\n{tr}\n\n')
    
class AudioDataKind(IntEnum):
    TIME=0
    FREQ=1

    @classmethod
    def register_item(cls, val, item_cls):
        if not hasattr(cls, '_item_classes'):
            setattr(cls, '_item_classes', {})
        cls._item_classes[val] = item_cls
    @property
    def item_cls(self):
        return AudioDataKind._item_classes[self]

    @property
    def display(self):
        return 'time-domain' if self is TIME else 'frequency-domain'


class AudioDataInfo(dict):
    '''Stores information on the nature of audio data as it is inputted and transformed. Both indexed and dotted access to members is provided. Data is immutable, use `update` to get a new instance with updated values.'''
    MULTI = 'multi-valued' # Tag for items with multiple values

    def __init__(self, vals:dict={}):
        super().__init__()
        if vals: super().update(vals)

    def __getattr__(self, name):
        if name in self: return self[name]
        else: raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        raise TypeError("Do not update values on an AudioDataInfo attribute, instead use the update method to get an updated instance.")

    def __delattr__(self, name):
        if name in self: del self[name]
        else: raise AttributeError("No such attribute: " + name)

    def __setitem__(self, key, item):
        raise TypeError("Do not update values on an AudioDataInfo attribute, instead use the update method to get an updated instance.")

    def update(self, new_info:dict):
        new_vals = self.copy()
        new_vals.update(new_info)
        return AudioDataInfo(new_vals)

class AudioData(ABC):
    '''Base class for audio data.'''
    def __init__(self, kind:AudioDataKind, sig, rate):
        self.kind = kind
        self.sig = sig
        self.rate = rate

    @abstractmethod
    def show(self, ax:plt.Axes=None, **kwargs):
        '''Plot this data on the `ax` axes, or the current axes if `None`.'''
        pass

    @abstractmethod
    def hear(self):
        '''Get this item as playable `IPython.display` object.'''
        pass
    
    @classmethod
    def reconstruct(cls, t:Tensor, info:AudioDataInfo):
        return cls(t, info.rate)

    def __len__(self): return self.sig.shape[-1]

    @property
    def shape(self): return self.sig.shape
    @property
    def channels(self): return 1 if len(self.shape) == 1 else self.shape[0]   
    @property
    def duration(self): return self.shape[-1]/self.rate
    @property
    def tensor(self):
        return as_tensor(self.sig)

    
class AudioTimeData(AudioData):
    '''Represents a time-domain audio signal'''
    def __init__(self, sig, rate):
        super().__init__(AudioDataKind.TIME, sig, rate)
        
    def __str__(self): return f'{self.__class__.__name__}: {self.duration}s ({len(self)}); {self.channels}ch; {self.rate/1000:.0f}kHz'

    @classmethod
    def load(cls, fileName, duration:Optional[float]=None, offset:float=0.0, **kwargs):
        p = Path(fileName)
        if not p.exists():
            raise FileNotFoundError(f'File not found: {p.absolute()}')
        # TODO: Support num_frames and offset
        offset = ifnone(offset, 0.0)
        #signal,samplerate = librosa.load(p, sr=resample, mono=mono, duration=duration, offset=offset, **kwargs)
        sig,rate = torchaudio.load(p)
        #librosa.load(p, sr=resample, mono=mono, duration=duration, offset=offset, **kwargs)
        if sig.ndim == 1: sig = sig[None,:] # Add channel dimension to mono audio
        return cls(as_tensor(sig), rate)

    # TODO: Support both playback and image    
    def show(self, ax:plt.Axes=None, **kwargs):
        '''Plot this data on the `ax` axes, or the current axes if `None`. Additional `kwargs` are passed to `librosa.display.waveplot`'''
        return waveplot(self.sig.numpy(), sr=self.rate, ax=ax, **kwargs)

    def hear(self):
        return Audio(data=self.sig, rate=self.rate)

AudioDataKind.register_item(AudioDataKind.TIME, AudioTimeData)

class AudioFreqData(AudioData):
    '''Represents a frequency-domain audio signal.'''
    def __init__(self, sig, rate, phase=None):
        super().__init__(AudioDataKind.FREQ, sig, rate)
        self.phase = phase

    def show(self, ax:plt.Axes=None, cmap:str=None, **kwargs):
        '''Plot this data on the `ax` axes, or the current axes if `None`. Additional `kwargs` are passed to `librosa.display.specshow`'''
        cmap = ifnone(cmap, defaults.cmap)
        data = self.sig # channels x width x height
        #TODO: Implement multi-channel display
        if self.channels != 1: raise NotImplementedError("Dsiaply of multi-channel audio not supported")
        data = self.sig[0,:,:]
        return specshow(data.numpy(), sr=self.rate, ax=ax, **kwargs)

    def hear(self):
        # TODO: Do an istft?
        return None

AudioDataKind.register_item(AudioDataKind.FREQ, AudioFreqData)
    
class AudioItem(ItemBase):
    '''Represents an audio item. This can be either time- or frequency-domain data.'''
    def __init__(self, item:AudioTimeData, **kwargs):
        self.item = item
        self.kwargs = kwargs

    def __str__(self):
        return f'{self.__class__.__name__}: data={self.item}'
    def __len__(self): return len(self.data)
    #def _repr_html_(self): return f'{self.__str__()}<br />{self.item.display._repr_html_()}'
    
    def show(self, title:Optional[str]=None, ax:plt.Axes=None, info:AudioDataInfo=None, **kwargs):
        "Show sound on `ax` with `title`"
        # TODO: Use info to set options
        ax = ifnone(ax, plt.gca())
        res = self.item.show(ax, **kwargs)
        if title: ax.set_title(title)
        return res

    def hear(self, title=None):
        # TODO: Show title with widget
        if title is not None: print(title)
        disp = self.item.hear()
        if disp: display(disp)

    def apply_tfms(self, tfms):
        for tfm in tfms:
            self.item = tfm(self.item)
        return self

    @property
    def data(self): return self.item.tensor
    @property
    def shape(self): return self.item.shape
    @property
    def rate(self): return self.item.rate
    @property
    def channels(self): return self.item.channels
    @property
    def duration(self): return self.item.duration

class AudioDataBunch(DataBunch):
    def hear_ex(self, rows:int=3, ds_type:DatasetType=DatasetType.Valid, **kwargs):
        batch = self.dl(ds_type).dataset[:rows]
        self.train_ds.hear_xys(batch.x, batch.y, **kwargs)

    @property
    def output_info(self):
        return self.train_dl.x.output_info

    @classmethod
    def create(cls, train_ds:Dataset, valid_ds:Dataset, test_ds:Optional[Dataset]=None, path:PathOrStr='.', bs:int=64,
               val_bs:int=None, num_workers:int=defaults.cpus, dl_tfms:Optional[Collection[Callable]]=None,
               device:torch.device=None, collate_fn:Callable=data_collate, no_check:bool=False, **kwargs)->'DataBunch':
        "Create an `AudioDataBunch` from self, `path` will override `self.path`, `kwargs` are passed to `DataBunch.create`."
        # Can't use pin_memory in dataloader if data is already on GPU
        from .transform import ToDevice
        todev = [tfm for tfm in ifnone(train_ds.tfms, []) if isinstance(tfm, ToDevice)]
        if todev and todev[-1].device.type == 'cuda':
            kwargs['pin_memory'] = False
        data = super().create(train_ds, valid_ds, test_ds=test_ds, path=path, bs=bs, val_bs=val_bs, num_workers=num_workers,
                              device=device, collate_fn=collate_fn, no_check=no_check, **kwargs)
        return data

class AudioLabelList(LabelList):
    def transform(self, tfms, tfm_y=None, **kwargs):
        self.x.check_transforms(tfms)
        return super().transform(tfms, tfm_y=tfm_y, **kwargs)

class AudioLabelLists(LabelLists):
    def databunch(self, path:PathOrStr=None, bs:int=64, val_bs:int=None, num_workers:int=defaults.cpus,
                  dl_tfms:Optional[Collection[Callable]]=None, device:torch.device=None, collate_fn:Callable=data_collate,
                  no_check:bool=False, **kwargs)->'AudioDataBunch':
        "Create an `AudioDataBunch` from self, `path` will override `self.path`, `kwargs` are passed to `DataBunch.create`."
        # TODO: Investigate circular dependency issue here and try to move this checking out
        from .transform import ToDevice
        use_gpu = ToDevice in map(lambda t: t.__class__, self.tfms)
        if use_gpu and num_workers != 0: # TODO: See why this doesn't work
            warn("GPU transforms cannot be used with multiple workers, overriding num_workers.")
            num_workers = 0
        super().databunch(path=path, bs=bs, val_bs=val_bs, num_workers=num_workers, dl_tfms=dl_tfms, device=device,
                          collate_fn=collate_fn, no_check=no_check, **kwargs)
        

PreProcessors = Union[PreProcessor, Collection[PreProcessor]]

class ReadInfoProcessor(PreProcessor):
    "Read metadata from items and adds it to the items."

    # Default keys to collect from sox_signal_info_t and sox_encoding_info_t
    SIG_KEYS = ['rate','channels','precision','length']
    ENC_KEYS = []

    def __init__(self, ds=None, fn_col:Union[int,str]=0, progress:Union[bool,progress_bar]=False,
                 sig_keys:Collection[str]=None, enc_keys:Collection[str]=None):
        self.fn_col = fn_col
        self.sig_keys = ifnone(sig_keys, ReadInfoProcessor.SIG_KEYS)
        self.enc_keys = ifnone(enc_keys, ReadInfoProcessor.ENC_KEYS)
        self.progress = progress

    def process(self, ds:Collection):
        # TODO: Need to use items instead
        df = ds.inner_df
        if df is None:
            raise TypeError("No inner_df on dataset.")
        if not isinstance(df, pd.DataFrame):
            raise ValueError('inner_df is not a DataFrame.')
        fn_col = self.fn_col if isinstance(self.fn_col, int) else df.columns.get_loc(self.fn_col)
        path = ifnone(getattr(ds, 'path', None), Path('.'))
        info = {k: [None]*len(df) for k in self.sig_keys + self.enc_keys + ['error']}
        items = df.iloc[:,fn_col]
        if self.progress == True:
            items = master_bar(items, total=len(df))
        elif isinstance(self.progress, master_bar):
            items = self.progress.add_child(items, total=len(df))
        for i,fn in enumerate(items):
            if not os.path.isabs(fn): fn = (path/fn).absolute()
            if not os.path.exists(fn):
                info['error'][i] = 'File not found'
                continue
            try:
                si,ei = torchaudio.info(str(fn))
                for k in self.sig_keys: info[k][i] = getattr(si, k, None)
                for k in self.enc_keys: info[k][i] = getattr(ei, k, None)
            except RuntimeError as exc:
                info['error'][i] = str(exc)
        for k,v in info.items(): df[k] = array(v)
        ds.inner_df = df

class AudioList(ItemList):
    _bunch = AudioDataBunch
    
    # TODO: Implement duration and offset    
    def __init__(self, items:Iterator, path:PathOrStr='.', label_cls:Callable=None, inner_df:Any=None,
                 x:'ItemList'=None, ignore_empty:bool=False, duration:Optional[float]=None, offset:float=0.0,
                 data_info:AudioDataInfo=None, check_info=True, **kwargs):
        super().__init__(items, path=path, label_cls=label_cls, inner_df=inner_df, x=x, ignore_empty=ignore_empty, **kwargs)
        self._label_list = AudioLabelList
        self.duration,self.offset,self.data_info,self.check_info = duration,offset,data_info,check_info
        if data_info is None and check_info: self.data_info = self.read_data_info()
        self.copy_new.extend(['duration','offset','data_info','check_info'])
        #_processor = [AudioProcessor]

    # TODO: Use torchaudio.info
    def read_data_info(self, num_samples=25)->AudioDataInfo:
        '''Gets AudioDataInfo by checking `num_samples` files.'''
        #TODO: Add print=True option
        rates,channels,durations = set(),set(),set()
        for idx in range(0, len(self), max(1, len(self) // num_samples)):
            it = self.get(idx)
            rates.add(it.rate)
            channels.add(it.channels)
            durations.add(it.duration)
        info = {'kind': AudioDataKind.TIME}
        for n,v in (('rate',rates), ('channels',channels), ('duration',durations)):
            info[n] = v.pop() if len(v) == 1 else AudioDataInfo.MULTI
        return AudioDataInfo(info)

    def _check_info(self, data:AudioTimeData):
        for k in ('rate','channels','duration'):
            if self.output_info[k] != AudioDataInfo.MULTI and self.output_info[k] != getattr(data, k):
                self.output_info = self.output_info.update({k: AudioDataInfo.MULTI})

    def open(self, fn:PathOrStr)->AudioItem:
        atd = AudioTimeData.load(fn, duration=self.duration, offset=self.offset)
        return AudioItem(atd)
   
    def get(self, i)->AudioItem:
        it = super().get(i)
        # Why was this here?
        # if hasattr(it, '__len__') and len(it) == 2: #data,sr
        #     if not self.resample: self._check_sample_rate(it[1])
        #     return AudioItem(AudioTimeData(it[0], it[1]))
        res = self.open(it)
        return res

    def check_transforms(self, tfms):
        if tfms:
            info = self.data_info
            for tfm in ifnone(tfms, []):
                info = tfm.process_info(info)
            self.output_info = info
            from .transform import check_transform_info
            check_transform_info(info, tfms)
        else:
            self.output_info = self.data_info

    def reconstruct(self, t:Tensor):
        cls = self.output_info.kind.item_cls
        return cls.reconstruct(t, self.data_info)

    def show_xys(self, xs, ys, itemsize:int=3, figsize:Optional[Tuple[int,int]]=None, max_cols:int=5, **kwargs):
        "Show the `xs` (inputs) and `ys` (targets) on a figure of `figsize`."
        # TODO: Output table so can both show and have playback
        cols = max(math.ceil(math.sqrt(len(xs))), max_cols)
        rows = math.ceil(len(xs)/cols)
        axs = subplots(rows, cols, imgsize=itemsize, figsize=figsize)
        # TODO: Support non-=string ys?
        for x,y,ax in zip(xs, ys, axs.flatten()): AudioItem(x).show(ax=ax, title=str(y), **kwargs)
        for ax in axs.flatten()[len(xs):]: ax.axis('off')
        plt.tight_layout()
    
    def show_xyzs(self, xs, ys, zs, imgsize:int=4, figsize:Optional[Tuple[int,int]]=None, **kwargs):
        raise NotImplementedError()# TODO: Implement
            
    # TODO: example with from_folder
    @classmethod
    def from_folder(cls, path:PathOrStr='.', extensions:Collection[str]=None, duration:Optional[float]=None,
                    offset:float=0.0, **kwargs)->'AudioItemList':
        "Get the list of files in `path` that have an audio suffix. `recurse` determines if we search subfolders."
        extensions = ifnone(extensions, AUDIO_EXTENSIONS)
        res = super().from_folder(path=path, extensions=extensions, duration=duration, offset=offset, **kwargs)
        return res
    
    @classmethod
    def from_df(cls, df:DataFrame, path:PathOrStr, cols:IntsOrStrs=0, folder:PathOrStr=None, suffix:str='',
                duration:Optional[float]=None, offset:float=0.0, processor:PreProcessors=None, **kwargs)->'AudioItemList':
        "Get the filenames in `cols` of `df` with `folder` in front of them, `suffix` at the end."
        suffix = suffix or ''
        #TODO: Need to pass empty data_info to avoid check which will fail. Instead move the folder/suffix handling to fastai, either ItemList or a FileList subclass
        res = super().from_df(df, path=path, cols=cols, duration=duration, offset=offset,
                              data_info=AudioDataInfo(), processor=processor, **kwargs)
        pref = f'{res.path}{os.path.sep}'
        if folder is not None: pref += f'{folder}{os.path.sep}'
        res.items = np.char.add(np.char.add(pref, res.items.astype(str)), suffix)
        res.data_info = res.read_data_info() # See above
        return res
