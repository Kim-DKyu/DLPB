import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
import numpy as np

__all__ = ['densenetd40k12']


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function

class ConvNormAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, norm_layer=nn.BatchNorm2d, stride=1, padding=0, groups=1, act=True):
        super(ConvNormAct, self).__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False, groups=groups),
            norm_layer(out_ch),
            nn.ReLU(inplace=True) if act else nn.Identity()
        )

class BottleNeck(nn.Module):
    factor = 4
    def __init__(self, in_channels, out_channels, stride, norm_layer, downsample=None, groups=1, base_width=64):
        super(BottleNeck, self).__init__()
        self.width = width = int(out_channels * (base_width / 64.0)) * groups
        self.out_channels = out_channels * self.factor
        self.conv1 = ConvNormAct(in_channels, width, 1, norm_layer)
        self.conv2 = ConvNormAct(width, width, 3, norm_layer, stride, 1, groups=groups)
        self.conv3 = ConvNormAct(width, self.out_channels, 1, norm_layer, act=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample if downsample else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return self.relu(self.downsample(x) + self.conv3(out))

class ClassAttn(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., dim_embed=128):
        super().__init__()
        self.dim_embed = dim_embed
        self.num_heads = num_heads
        head_dim = self.dim_embed // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, self.dim_embed , bias=qkv_bias)
        self.k = nn.Linear(dim, self.dim_embed , bias=qkv_bias)
        self.v = nn.Linear(dim, self.dim_embed , bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim_embed , dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        C = self.dim_embed
        q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls

class GroupConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, drop=0., groups=1):
        super().__init__()
        self.groups = groups

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=True, groups=groups)
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=True, groups=groups)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x_ndim = x.ndim
        if x_ndim == 3:
            x = x.permute(0, 2, 1)
            x = x.unsqueeze(-1)
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = channel_shuffle(x, self.groups)
        x = self.fc2(x)
        if x_ndim == 3:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1)
        return x

def channel_shuffle(x, group):
    batchsize, num_channels, height, width = x.data.size()
    assert num_channels % group == 0
    group_channels = num_channels // group

    x = x.reshape(batchsize, group_channels, group, height, width)
    x = x.permute(0, 2, 1, 3, 4)
    x = x.reshape(batchsize, num_channels, height, width)

    return x

class LayerScaleBlockClassAttn(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add CA and LayerScale
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_block=ClassAttn,
            mlp_block=GroupConvMlp, mlp_block_groups=2, init_values=1e-4, dim_embed=128):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, dim_embed=dim_embed)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mlp_block(in_features=dim // 1, hidden_features=mlp_hidden_dim // 1, act_layer=act_layer,
                             out_features=dim // 1, drop=drop, groups=mlp_block_groups)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x, x_cls):
        u = torch.cat((x_cls, x), dim=1)
        x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
        x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
        return x_cls

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)

class ILR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, num_branches):
        ctx.num_branches = num_branches
        return input

    @staticmethod
    def backward(ctx, grad_output):
        num_branches = ctx.num_branches
        return grad_output / num_branches, None


class DenseNet(nn.Module):
    def __init__(self, growth_rate=12, block_config=(16, 16, 16), branches=3, gram_dim=64,
                 bpscale=False, avg=False,
                 compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0,
                 num_classes=10, small_inputs=True, efficient=False, ind=False):

        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = 8 if small_inputs else 7
        self.avg = avg
        self.ind = ind
        self.bpscale = bpscale
        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            if i != len(block_config) - 1:
                block = _DenseBlock(
                    num_layers=num_layers,
                    num_input_features=num_features,
                    bn_size=bn_size,
                    growth_rate=growth_rate,
                    drop_rate=drop_rate,
                    efficient=efficient,
                )
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate

                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)
            else:
                for i in range(2):
                    setattr(self, 'layer3_' + str(i), _DenseBlock(
                        num_layers=num_layers,
                        num_input_features=num_features,
                        bn_size=bn_size,
                        growth_rate=growth_rate,
                        drop_rate=drop_rate,
                        efficient=efficient,
                    ))

        num_features = num_features + num_layers * growth_rate
        for i in range(2):
            setattr(self, 'norm_final_' + str(i), nn.BatchNorm2d(num_features))
            setattr(self, 'relu_final_' + str(i), nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_features, num_classes)

        self.branches = branches
        self.gram_contraction = nn.ModuleList()
        self.gram_layer = nn.ModuleList()
        self.gram_embedding = nn.ModuleList()

        self.ga = nn.ModuleList()  # class attention layers
        self.fc = nn.ModuleList()
        self.gram_dim = gram_dim
        self.dim_embed = 128
        self.gram_embedding_gropus = 1
        gram_layer = True

        for i in range(self.branches):
            self.gram_contraction.append(
                nn.Sequential(nn.Conv2d(132, self.gram_dim, kernel_size=1, stride=1,
                                        padding=0, bias=True, groups=1),
                              nn.BatchNorm2d(self.gram_dim)))

            if gram_layer:
                self.gram_layer.append(
                    BottleNeck(self.gram_dim, self.gram_dim // BottleNeck.factor, stride=1, norm_layer=nn.BatchNorm2d))

            else:
                self.gram_layer.append(nn.Identity())

            self.gram_embedding.append(
                nn.Sequential(nn.Conv2d(((self.gram_dim + 1) * self.gram_dim // 2), 132,
                                        kernel_size=1, stride=1, padding=0, bias=True,
                                        groups=self.gram_embedding_gropus),
                              nn.BatchNorm2d(132)))
            self.ga.append(LayerScaleBlockClassAttn(132, num_heads=1, mlp_block_groups=1,
                                                    dim_embed=self.dim_embed))
            self.fc.append(nn.Linear(132, num_classes))

        self.gram_index = np.zeros(((self.gram_dim + 1) * self.gram_dim // 2))
        count = 0
        for i in range(self.gram_dim):
            for j in range(self.gram_dim):
                if j >= i:
                    self.gram_index[count] = (i * self.gram_dim) + j
                    count += 1

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_gram(self, xbp, B, C):
        xbp = xbp/xbp.size()[2] # torch.Size([128, 192, 14, 14])

        if self.training and B < 128:
            xbp = xbp.to(dtype=torch.float64, memory_format=torch.contiguous_format)

        xbp = torch.reshape(xbp, (xbp.size()[0], xbp.size()[1], xbp.size()[2] * xbp.size()[3])) # torch.Size([128, 192, 196])
        xbp = torch.bmm(xbp, torch.transpose(xbp, 1, 2)) / (xbp.size()[2]) # torch.Size([128, 192, 192])
        xbp = torch.reshape(xbp, (B, C ** 2)) # torch.Size([128, 36864])
        xbp = xbp[:, self.gram_index] # torch.Size([128, 18528])

        xbp = torch.nn.functional.normalize(xbp)
        xbp = xbp.float()
        xbp = torch.reshape(xbp, (xbp.size()[0], xbp.size()[1], 1, 1)) # torch.Size([128, 18528, 1, 1])
        return xbp

    def forward(self, input):
        x = self.features(input)
        input = x

        x = self.layer3_0(input)
        x = self.norm_final_0(x)
        x = self.relu_final_0(x)

        x_out = []
        for k in range(self.branches):
            gram = self.gram_contraction[k](x)  # Bx192x14x14
            gram = self.gram_layer[k](gram)
            B, C, _, _ = gram.shape
            gram = self.get_gram(gram, B, C)
            gram = self.gram_embedding[k](gram)

            gram = gram.view(gram.size()[0], gram.size()[1], -1)
            gram = gram.permute(0, 2, 1)
            gram = self.ga[k](x.view(x.size()[0], x.size()[1], -1).permute(0, 2, 1), gram)
            gram = gram.view(gram.size(0), -1)
            gram = self.fc[k](gram)
            x_out.append(gram)

        x_branch = self.layer3_1(input)
        x_branch = self.norm_final_1(x_branch)
        x_branch = self.relu_final_1(x_branch)
        x_branch = self.avgpool(x_branch)
        x_branch = x_branch.view(x_branch.size(0), -1)
        x_branch = self.classifier(x_branch)

        return x_out, x_branch

model_config = {
    'densenetd40k12_branch_3_dlpb': {
        'parameter': dict(growth_rate=12, block_config=[6, 6, 6], gram_dim=64, branches=3),
        'etc': {},
    },
}

def create_densenet(model_name, num_classes):
    config = model_config[model_name]['parameter']
    return DenseNet(num_classes=num_classes, **config)


if __name__ == '__main__':
    net = create_densenet('ga_densenetd40k12_branch_3', num_classes=100)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
