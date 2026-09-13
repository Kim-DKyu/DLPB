from torch import nn
import numpy as np
import torch
from functools import partial
class ClassAttn(nn.Module):
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

class ConvNormAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, norm_layer=nn.BatchNorm2d, stride=1, padding=0, groups=1, act=True):
        super(ConvNormAct, self).__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False, groups=groups),
            norm_layer(out_ch),
            nn.ReLU(inplace=True) if act else nn.Identity()
        )


class SEUnit(nn.Sequential):
    def __init__(self, ch, norm_layer, r=16):
        super(SEUnit, self).__init__(
            nn.AdaptiveAvgPool2d(1), # squeeze
            ConvNormAct(ch, ch//r, 1, norm_layer), nn.Conv2d(ch//r, ch, 1, bias=True), nn.Sigmoid(), # excitation
        )
    def forward(self, x):
        out = super(SEUnit, self).forward(x)
        return out * x


class BasicBlock(nn.Module):
    factor = 1
    def __init__(self, in_channels, out_channels, stride, norm_layer, downsample=None, groups=1, base_width=64,
                 drop_path_rate=0.0, se=False):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, 3, norm_layer, stride, 1)
        self.conv2 = ConvNormAct(out_channels, out_channels, 3, norm_layer, 1, 1, act=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample if downsample else nn.Identity()
        self.drop_path = StochasticDepth(drop_path_rate)
        self.se = SEUnit(out_channels, norm_layer) if se else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        return self.relu(self.downsample(x) + self.drop_path(self.se(self.conv2(out))))


class BottleNeck(nn.Module):
    factor = 4
    def __init__(self, in_channels, out_channels, stride, norm_layer, downsample=None, groups=1, base_width=64,
                 drop_path_rate=0.0, se=False):
        super(BottleNeck, self).__init__()
        self.width = width = int(out_channels * (base_width / 64.0)) * groups
        self.out_channels = out_channels * self.factor
        self.conv1 = ConvNormAct(in_channels, width, 1, norm_layer)
        self.conv2 = ConvNormAct(width, width, 3, norm_layer, stride, 1, groups=groups)
        self.conv3 = ConvNormAct(width, self.out_channels, 1, norm_layer, act=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample if downsample else nn.Identity()
        self.drop_path = StochasticDepth(drop_path_rate)
        self.se = SEUnit(self.out_channels, norm_layer) if se else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return self.relu(self.downsample(x) + self.drop_path(self.se(self.conv3(out))))


class StochasticDepth(nn.Module):
    def __init__(self, prob, mode='row'):
        super(StochasticDepth, self).__init__()
        self.prob = prob
        self.survival = 1.0 - prob
        self.mode = mode

    def forward(self, x):
        if self.prob == 0.0 or not self.training:
            return x
        else:
            shape = [x.size(0)] + [1] * (x.ndim - 1) if self.mode == 'row' else [1]
            return x * x.new_empty(shape).bernoulli_(self.survival).div_(self.survival)


class ResNet(nn.Module):
    def __init__(self,
                 nblock,
                 block = BottleNeck,
                 norm_layer: nn.Module = nn.BatchNorm2d,
                 channels=[16, 32, 64, 64],
                 # channels=[64, 128, 256, 512],
                 strides=[1, 2, 2, 1],
                 groups=1,
                 base_width=64,
                 zero_init_last=True,
                 num_classes=100,
                 in_channels=3,
                 drop_path_rate=0.0,
                 se=False,
                 gram_dim=32,
                 branches=5,
                 stage3_naggre = 2) -> None:
        super(ResNet, self).__init__()
        self.groups = groups
        self.num_classes = num_classes
        self.base_width = base_width
        self.norm_layer = norm_layer
        self.in_channels = channels[0]
        self.out_channels = channels[-1] * block.factor
        self.num_block = sum(nblock)
        self.cur_block = 0
        self.drop_path_rate = drop_path_rate
        self.se = se

        self.stage3_naggre = stage3_naggre
        self.avg_pool_down = nn.AdaptiveAvgPool2d(output_size=16)
        self.upsample_f = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False)
        self.bn1 = self.norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.make_layer(block=block, nblock=nblock[0], channels=channels[0], stride=strides[0])
        self.layer2 = self.make_layer(block=block, nblock=nblock[1], channels=channels[1], stride=strides[1])
        self.layer3 = self.make_layer(block=block, nblock=nblock[2], channels=channels[2], stride=strides[2], branch_ch=True)
        self.layer4 = self.make_layer(block=block, nblock=nblock[3], channels=channels[3], stride=strides[3], branch_ch=True)

        self.layer3_branch = self.make_layer(block=block, nblock=nblock[2], channels=channels[2], stride=strides[2], branch_ch=True)
        self.classifier_single = nn.Linear(self.out_channels, self.num_classes)

        self.flatten = nn.Flatten()
        self.last_pool = nn.AdaptiveAvgPool2d((1, 1))

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
            self.gram_contraction.append(nn.Sequential(nn.Conv2d(256, self.gram_dim, kernel_size=1, stride=1,
                                                                 padding=0, bias=True, groups=1),
                                                       nn.BatchNorm2d(self.gram_dim)))

            if gram_layer:
                self.gram_layer.append(block(self.gram_dim, self.gram_dim//block.factor, stride=1, norm_layer=norm_layer))

            else:
                self.gram_layer.append(nn.Identity())

            self.gram_embedding.append(nn.Sequential(nn.Conv2d(((self.gram_dim + 1) * self.gram_dim // 2), 256,
                                                               kernel_size=1, stride=1, padding=0, bias=True,
                                                               groups=self.gram_embedding_gropus), nn.BatchNorm2d(256)))
            self.ga.append(LayerScaleBlockClassAttn(256, num_heads=1, mlp_block_groups=1, dim_embed=self.dim_embed))
            self.fc.append(nn.Linear(256, num_classes))
        self.gram_index = np.zeros(((self.gram_dim + 1) * self.gram_dim // 2))
        count = 0
        for i in range(self.gram_dim):
            for j in range(self.gram_dim):
                if j >= i:
                    self.gram_index[count] = (i * self.gram_dim) + j
                    count += 1

        self.init_weight(zero_init_last)

    def register_layer(self):
        for i, layer in enumerate(self.layers):
            exec('self.layer{} = {}'.format(i + 1, 'layer'))

    def get_drop_path_rate(self):
        drop_path_rate = self.drop_path_rate * (self.cur_block / self.num_block)
        self.cur_block += 1
        return drop_path_rate

    def make_layer(self, block, nblock: int, channels: int, stride: int, branch_ch: bool = False) -> nn.Sequential:
        if branch_ch:
            if nblock!=1:
                self.in_channels = 32 * block.factor
            else:
                self.in_channels = 1344 #1792 1344

        if self.in_channels != channels * block.factor or stride != 1:
            downsample = ConvNormAct(self.in_channels, channels * block.factor, 1, self.norm_layer, stride, act=False)
        else:
            downsample = None

        layers = []
        for i in range(nblock):
            if i == 1:
                stride = 1
                downsample = None
                self.in_channels = channels * block.factor
            layers.append(block(in_channels=self.in_channels, out_channels=channels, stride=stride,
                                norm_layer=self.norm_layer, downsample=downsample, groups=self.groups,
                                base_width=self.base_width, drop_path_rate=self.get_drop_path_rate(), se=self.se))
        return nn.Sequential(*layers)

    def get_gram(self, xbp, B, C):

        xbp = xbp/xbp.size()[2] # torch.Size([128, 192, 14, 14])

        if self.training and B < 128:
            xbp = xbp.to(dtype=torch.float64, memory_format=torch.contiguous_format)

        xbp = torch.reshape(xbp, (xbp.size()[0], xbp.size()[1], xbp.size()[2] * xbp.size()[3]))
        xbp = torch.bmm(xbp, torch.transpose(xbp, 1, 2)) / (xbp.size()[2])
        xbp = torch.reshape(xbp, (B, C ** 2))
        xbp = xbp[:, self.gram_index]

        xbp = torch.nn.functional.normalize(xbp)
        xbp = xbp.float()
        xbp = torch.reshape(xbp, (xbp.size()[0], xbp.size()[1], 1, 1))
        return xbp

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x_cat = []
        xs_stage1 = []
        xs_stage2 = []
        xs_stage3 = []

        for i, layer in enumerate(self.layer1):
            x = layer(x)
            if (i + 1) % (len(self.layer1) // (self.stage3_naggre + 1)) == 0 and len(xs_stage1) < (self.stage3_naggre):
                xs_stage1.append(x)

        x_cat.append(x)

        for i, layer in enumerate(self.layer2):
            x = layer(x)
            if (i + 1) % (len(self.layer2) // (self.stage3_naggre + 1)) == 0 and len(xs_stage2) < (self.stage3_naggre):
                xs_stage2.append(x)

        x_cat.append(x)
        input = x

        for i, layer in enumerate(self.layer3):
            x = layer(x)
            if (i + 1) % (len(self.layer3) // (self.stage3_naggre + 1)) == 0 and len(xs_stage3) < (self.stage3_naggre):
                xs_stage3.append(x)
        x_cat.append(x)

        x = torch.cat((self.avg_pool_down(xs_stage1[0]),self.avg_pool_down(xs_stage1[1]), self.avg_pool_down(x_cat[0]),
                       xs_stage2[0], xs_stage2[1], x_cat[1],
                       self.upsample_f(x_cat[2]),self.upsample_f(xs_stage3[0]),self.upsample_f(xs_stage3[1])),dim=1)
        x = self.layer4(x)
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

        x_branch = self.layer3_branch(input)
        x_branch = self.flatten(self.last_pool(x_branch))
        x_single_logit = self.classifier_single(x_branch)

        return x_out, x_single_logit

    def init_weight(self, zero_init_last=True):
        for m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_last:
            for m in self.named_modules():
                if isinstance(m, BottleNeck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

model_config = {
    'resnet110_gram_dim_64_branch_3_dlpb': {
        'parameter': dict(nblock=[12, 12, 12, 1], block=BottleNeck, gram_dim=64, branches=3),
        'etc': {},
    },
}

def create_resnet(model_name, num_classes):
    config = model_config[model_name]['parameter']
    return ResNet(num_classes=num_classes, **config)

if __name__ == '__main__':
    import torch
    model_name = 'ga_resnet110_gram_dim_64_branch_3'
    x = torch.rand(2, 3, 32, 32)
    config = model_config[model_name]['parameter']
    model = ResNet(num_classes=100, **config)
    print(f'param count: {sum([m.numel() for m in model.parameters()])}')
    y = model(x)
