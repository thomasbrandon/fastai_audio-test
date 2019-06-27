import pytest
from fastai_audio import *
from unittest.mock import Mock

@pytest.mark.parametrize('n_channels', [1,2,3,4])
@pytest.mark.parametrize('pretrained,init', [(True,False),(False,True),(False,False)])
@pytest.mark.parametrize('bias', [True,False])
@pytest.mark.parametrize('sizes', [(3,3),(5,5)])
def test_adapt_conv(n_channels, pretrained, init, bias, sizes):
    attrs = {n: sizes for n in ['kernel_size','stride','padding']}
    conv = torch.nn.Conv2d(3, 10, bias=bias, **attrs)
    wgts = torch.rand_like(conv.weight.data)
    if pretrained: conv.weight.data = wgts
    init_fn = Mock() if init else None
    if bias: conv.bias.data = biases = torch.rand_like(conv.bias.data)
    new_conv = adapt_conv(conv, n_channels, pretrained=pretrained, init=init_fn)
    assert(new_conv.in_channels == n_channels)
    for a in attrs: assert getattr(conv, a) == getattr(new_conv, a)
    if pretrained:
        assert torch.equal(*torch.broadcast_tensors(new_conv.weight.data, wgts[:,0:1,:,:]))
    if init: init_fn.assert_called_with(new_conv.weight)

#TODO: Test adapt_channels and audio_cnn_learner

def test_adapt_channels():
    raise NotImplementedError()

def test_audio_cnn_learner():
    raise NotImplementedError()