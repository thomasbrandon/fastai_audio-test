from .data import AudioDataBunch
from fastai.basics import *
from fastai.vision import cnn_learner
from torch.nn import Conv2d, Sequential, Module

__all__ = ['adapt_conv','adapt_model','audio_cnn_learner']

def adapt_conv(conv: Conv2d, n_channels:int, pretrained:bool=False,
               init:Optional[Callable]=None, padding_mode:str='zeros'):
    '''Create a new layer that adapts `conv` to accept `n_channels` inputs.
       Copies existing weights if `pretrained` or initialises them with `init`.'''
    if conv.in_channels == n_channels: return conv # No need to adapt
    args = {n: getattr(conv, n) for n in ['kernel_size','stride','padding','dilation','groups']}
    bias = conv.bias is not None
    if 'padding_mode' in Conv2d.__constants__: # Padding mode added in PyTorch 1.1
        args['padding_mode'] = ifnone(padding_mode, conv.padding_mode)
    new_conv = Conv2d(n_channels, conv.out_channels, bias=bias, **args)
    if pretrained:
        exp_shape = (conv.out_channels, conv.in_channels, *conv.kernel_size)
        assert conv.weight.shape == exp_shape, f"Unexpected weights shape, expected {exp_shape}, got {conv.weight.shape}."
        #TODO: Allow other methods of calculating weights, e.g. other channel, mean or perhaps diversity of activations
        new_conv.weight.data[...] = conv.weight.data[:,0:1,:,:]
        if bias: new_conv.bias.data = conv.bias.data
    elif init: init_default(new_conv, init)
    new_conv.to(conv.weight.device)
    return new_conv

def adapt_model(model:Union[Module,Sequential], n_channels:int, name:str='conv1',
                   pretrained:bool=False, init:Optional[Callable]=None, padding_mode:str='zeros'):
    '''Adapt a convolutional model to `n_channels` inputs and copy weights if `pretrained` or initialise with `init`.'''
    # Find direct parent of first conv layer. Could be either a Sequential or a custom Module (but not the Conv itself)
    while (isinstance(model, Sequential) and 
           isinstance(model[0], (Sequential,Module)) and
           not isinstance(model[0], Conv2d)):
        model = model[0]
    if isinstance(model, Sequential) and isinstance(model[0], Conv2d):
        conv1 = model[0]
        def update(conv): model[0] = conv
    elif isinstance(model, Module) and hasattr(model, name):
        conv1 = getattr(model, name)
        update = partial(setattr, model, name)
    else: raise TypeError(f"Could not locate first convolution layer. If it is a named layer then pass it's name, otherwise use adapt_conv.")
    update(adapt_conv(conv1, n_channels, pretrained=pretrained, init=init, padding_mode=padding_mode))

def audio_cnn_learner(data:AudioDataBunch, base_arch:Callable, cut:Union[int,Callable]=None, pretrained:bool=False,
                      lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5, custom_head:Optional[nn.Module]=None,
                      split_on:Optional[SplitFuncOrIdxList]=None, bn_final:bool=False, init=nn.init.kaiming_normal_,
                      concat_pool:bool=True, padding_mode:str='zeros', **kwargs:Any)->Learner:
    '''Create a learner to apply a CNN model to audio spectrograms.'''
    learn = cnn_learner(data, base_arch, cut=cut, pretrained=pretrained, lin_ftrs=lin_ftrs, ps=ps,
                        custom_head=custom_head, split_on=split_on, bn_final=bn_final, init=init,
                        concat_pool=concat_pool, **kwargs)
    adapt_model(learn.model, data.output_info.channels, pretrained=pretrained, init=init, padding_mode=padding_mode)
    learn.unfreeze() # Model shouldn't be frozen, unlike vision
    return learn
