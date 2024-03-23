from fastai.basics import defaults
from fastai.layers import ConvLayer, NormType, SimpleSelfAttention, AvgPool, SEModule, MaxPool, AdaptiveAvgPool, Flatten
from fastai.vision.models.xresnet import init_cnn, ResBlock

from fastxtend.vision.models.attention_modules import *
from fastxtend.imports import *


class imagenorm(nn.Module):
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize=normalize
    
    def forward(self, x:Tensor):
        with torch.no_grad():
            if self.normalize: return x/255.0
            else: return x


class DuelingHead(nn.Module):
    def __init__(self, n_in, n_out, dueling=False, act_cls=nn.ReLU):
        super().__init__()
        self.dueling = dueling
        self.flatten = nn.Flatten()
        self.action_value = nn.Sequential(
            nn.Linear(n_in, 256),
            act_cls(),
            nn.Linear(256, n_out),
        )
        self.state_value = nn.Sequential(
            nn.Linear(n_in, 256),
            act_cls(),
            nn.Linear(256, 1),
        )
        init_cnn(self)

    def forward(self, x:Tensor):
        x = self.flatten(x)
        action_value = self.action_value(x)
        if self.dueling:
            state_value = self.state_value(x)
            action_score_centered = action_value - action_value.mean(dim=-1, keepdim=True)
            return state_value + action_score_centered
        else:
            return action_value


class XResNet(nn.Sequential):
    "A flexible version of fastai's XResNet"
    def __init__(self, block, expansion, layers, p=0.0, c_in=3, n_out=1000, stem_szs=(32,32,64), 
                 block_szs=[64,128,256,512], widen=1.0, sa=False, act_cls=defaults.activation, ndim=2, 
                 ks=3, stride=2, stem_layer=ConvLayer, stem_pool=MaxPool, head_pool=AdaptiveAvgPool, 
                 custom_head=None, normalize=True, **kwargs):
        store_attr('block,expansion,act_cls,ndim,ks')
        if ks % 2 == 0: raise Exception('kernel size has to be odd!')
        stem_szs = [c_in, *stem_szs]
        stem = [stem_layer(stem_szs[i], stem_szs[i+1], ks=ks, stride=stride if i==0 else 1,
                           act_cls=act_cls, ndim=ndim)
                for i in range(len(stem_szs)-1)]

        assert len(layers) == len(block_szs), 'Length of `layers` must match `block_szs`'
        block_szs = [int(o*widen) for o in block_szs]
        block_szs = [stem_szs[-1]//expansion] + block_szs
        stem_pool = stem_pool(ks=ks, stride=stride, padding=ks//2, ndim=ndim)
        if not is_listy(stem_pool): stem_pool = [stem_pool]
        blocks    = self._make_blocks(layers, block_szs, sa, stride, **kwargs)

        head = custom_head(block_szs[-1]*expansion, n_out, act_cls=act_cls)
        if not is_listy(head): head = [head]
        body = nn.Sequential(imagenorm(normalize), *stem, *stem_pool, *blocks)
        init_cnn(body)
        super().__init__(*list(body), *[head_pool(sz=1, ndim=ndim)], *head)

    def _make_blocks(self, layers, block_szs, sa, stride, **kwargs):
        return [self._make_layer(ni=block_szs[i], nf=block_szs[i+1], blocks=l,
                                 stride=1 if i==0 else stride, sa=sa and i==len(layers)-4, **kwargs)
                for i,l in enumerate(layers)]

    def _make_layer(self, ni, nf, blocks, stride, sa, **kwargs):
        return nn.Sequential(
            *[self.block(self.expansion, ni if i==0 else nf, nf, stride=stride if i==0 else 1,
                      sa=sa and i==(blocks-1), act_cls=self.act_cls, ndim=self.ndim, ks=self.ks, **kwargs)
              for i in range(blocks)])

    def _make_head(self, ni, head_pool, ndim, p, n_out):
        return [head_pool(sz=1, ndim=ndim), nn.Flatten(), nn.Dropout(p), nn.Linear(ni, n_out)]


def dqn_resnet9(n_out=12, dueling=False, norm='instance', **kwargs):
    norm_type=NormType.Instance if norm == 'instance' else NormType.Batch
    custom_head = partial(DuelingHead, dueling=dueling)
    return XResNet(ResBlock, 1, [1, 1, 1], block_szs=[64,128,256], n_out=n_out, c_in=4, norm_type=norm_type, custom_head=custom_head, **kwargs)