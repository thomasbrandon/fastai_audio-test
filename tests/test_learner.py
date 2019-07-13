import pytest
from fastai_audio import *
from unittest.mock import Mock

@pytest.mark.parametrize('orig_channels', [1,3])
@pytest.mark.parametrize('n_channels', [1,2,3,4])
@pytest.mark.parametrize('pretrained,init', [(True,False),(False,True),(False,False)])
@pytest.mark.parametrize('bias', [True,False])
@pytest.mark.parametrize('sizes', [(3,3),(5,5)])
def test_adapt_conv(orig_channels, n_channels, pretrained, init, bias, sizes):
    attrs = {n: sizes for n in ['kernel_size','stride','padding']} # Nonsensical, just testing parameters used
    conv = torch.nn.Conv2d(orig_channels, 10, bias=bias, **attrs)
    wgts = torch.rand_like(conv.weight.data)
    if pretrained: conv.weight.data = wgts
    init_fn = Mock() if init else None
    if bias: conv.bias.data = biases = torch.rand_like(conv.bias.data)
    new_conv = adapt_conv(conv, n_channels, pretrained=pretrained, init=init_fn)
    if n_channels == orig_channels:
        assert new_conv is conv
    else:
        assert(new_conv.in_channels == n_channels)
        for a in attrs: assert getattr(conv, a) == getattr(new_conv, a)
        if pretrained:
            assert torch.equal(*torch.broadcast_tensors(new_conv.weight.data, wgts[:,0:1,:,:]))
        if init: init_fn.assert_called_with(new_conv.weight)

@pytest.mark.parametrize('nest', [0,1,3])
def test_adapt_channels_seq(mocker, nest):
    ac = mocker.patch('fastai_audio.learner.adapt_conv', return_value=Mock(spec=nn.Conv2d))
    conv1 = Mock(spec=nn.Conv2d)
    mdl = nn.Sequential(conv1, nn.Conv2d(20, 40, (3,3)))
    for _ in range(nest): mdl = nn.Sequential(mdl)
    adapt_model(mdl, 1)
    ac.assert_called_once()
    assert ac.call_args[0] == (conv1, 1)

@pytest.mark.parametrize('nest', [0,1,3])
def test_adapt_channels_mod(mocker, nest):
    ac = mocker.patch('fastai_audio.learner.adapt_conv', return_value=Mock(spec=nn.Conv2d))
    conv1 = Mock(spec=nn.Conv2d)
    class Mod(nn.Module):
        def __init__(self, conv):
            super().__init__()
            self.conv1 = conv
            self.conv2 = nn.Conv2d(20, 40, (3,3))   
    mdl = Mod(conv1)
    for _ in range(nest): mdl = nn.Sequential(mdl)
    adapt_model(mdl, 1)
    ac.assert_called_once()
    assert ac.call_args[0] == (conv1, 1)
