import math
import torch
from torch import nn
import torch.nn.functional as F

# split an original model into a base and a head
class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()

        self.base = base
        self.head = head
        
    def forward(self, x):
        rep = self.base(x)
        out = self.head(rep)

        return out

class CNN1D(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, dropout=0.2, num_classes=10):
        super(CNN1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.global_maxpool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # UNSW 和 KDD 通用

    def forward(self, x):
        x = x.unsqueeze(1)  # 增加一个维度以适配 1D CNN 的输入要求
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.global_maxpool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.dropout(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)    

        return x


class CNN2D(nn.Module):
    def __init__(self, hidden_dim, dropout=0.2, num_classes=10, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.global_maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    

@torch.no_grad()
def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0,
                  a: float = -2.0, b: float = 2.0) -> torch.Tensor:
    """
    Pure torch truncated normal initializer (no timm dependency).
    Implementation mirrors the standard inverse-CDF method used in common init code.

    Fills `tensor` with values drawn from N(mean, std^2) truncated to [a, b].
    """
    if std <= 0:
        raise ValueError("trunc_normal_: std must be > 0")

    # Helper: standard normal CDF
    def norm_cdf(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    device = tensor.device
    dtype = tensor.dtype

    a_t = torch.tensor(a, device=device, dtype=dtype)
    b_t = torch.tensor(b, device=device, dtype=dtype)
    mean_t = torch.tensor(mean, device=device, dtype=dtype)
    std_t = torch.tensor(std, device=device, dtype=dtype)

    low = norm_cdf((a_t - mean_t) / std_t)
    high = norm_cdf((b_t - mean_t) / std_t)

    # Uniform in [2*low-1, 2*high-1], then erfinv to get truncated standard normal
    tensor.uniform_(2.0 * low - 1.0, 2.0 * high - 1.0)
    tensor.erfinv_()

    tensor.mul_(std_t * math.sqrt(2.0)).add_(mean_t)
    tensor.clamp_(min=a, max=b)
    return tensor


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def reparam(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(
            w.size(1) * self.c.groups, w.size(0), w.shape[2:],
            stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
            groups=self.c.groups, device=c.weight.device
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Linear(nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def reparam(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop
        if self.drop > 0:
            self.forward = self.forward_drop
        else:
            self.forward = self.forward_deploy

    def forward_drop(self, x):
        return x + self.m(x) * torch.rand(
            x.size(0), 1, 1, 1, device=x.device
        ).ge_(self.drop).div(1 - self.drop).detach()

    def forward_deploy(self, x):
        return x + self.m(x)

    @torch.no_grad()
    def reparam(self):
        # NOTE: this is for deployment reparameterization. Not needed for MNIST training.
        # Original code had an impossible condition (Conv2d_BN AND Identity simultaneously).
        if isinstance(self.m, Conv2d_BN):
            m = self.m.reparam()
            # Only add identity if it is depthwise and square odd kernel.
            if m.groups == m.in_channels and m.in_channels == m.out_channels and m.kernel_size[0] == m.kernel_size[1]:
                k = m.kernel_size[0]
                if k % 2 == 1:
                    center = k // 2
                    identity = torch.zeros_like(m.weight)
                    identity[:, 0, center, center] = 1.0
                    m.weight += identity
            return m
        return self


class FFN(torch.nn.Module):
    def __init__(self, ed, h, act_layer=nn.GELU):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = act_layer()
        self.pw2 = Conv2d_BN(h, ed)

    def forward(self, x):
        return self.pw2(self.act(self.pw1(x)))


class Classfier(nn.Module):
    def __init__(self, dim, num_classes, distillation=True):
        super().__init__()
        self.classifier = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation
        if distillation:
            self.classifier_dist = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()

    def forward(self, x):
        if self.distillation:
            x = self.classifier(x), self.classifier_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.classifier(x)
        return x

    @torch.no_grad()
    def reparam(self):
        classifier = self.classifier.reparam()
        if self.distillation:
            classifier_dist = self.classifier_dist.reparam()
            classifier.weight += classifier_dist.weight
            classifier.bias += classifier_dist.bias
            classifier.weight /= 2
            classifier.bias /= 2
            return classifier
        return classifier


class StemLayer(nn.Module):
    def __init__(self, inc, ouc, ks=3, ps=16, act_layer=nn.ReLU):
        super().__init__()
        pad = 0 if (ks % 2) == 0 else ks // 2

        # blocks ~= log2(patch_size) for common settings (ps=16 -> 4 blocks)
        blocks = math.ceil(ps ** 0.5)
        dims = [inc] + [x.item() for x in ouc // 2 ** torch.arange(blocks - 1, -1, -1)]
        stem = [
            nn.Sequential(
                Conv2d_BN(dims[i], dims[i + 1], ks=ks, stride=2, pad=pad),  # FIX: use pad
                act_layer()
            )
            for i in range(blocks)
        ]
        self.stem = nn.Sequential(*stem)

    def forward(self, x):
        return self.stem(x)


class SSHA(nn.Module):
    def __init__(self, dim, qk_dim, pdim, sr=2, dcons=True, inp_group=1):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim
        self.split_index = (qk_dim, qk_dim, pdim, dim - pdim)
        self.pre_norm = nn.GroupNorm(1, dim)
        self.in_proj = Conv2d_BN(dim, qk_dim * 2 + dim, 3, sr, 1, groups=inp_group)
        if sr > 1:
            self.ups = nn.ConvTranspose2d(
                dim, dim, sr * (2 if dcons else 1),
                stride=sr, padding=sr // 2 if dcons else 0, groups=dim
            )
        else:
            self.ups = nn.Identity()

        self.out_proj = nn.Sequential(nn.GELU(), Conv2d_BN(dim, dim, 1, 1))

    def forward(self, x):
        x = self.pre_norm(x)
        q, k, v, u = self.in_proj(x).split(self.split_index, dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        B, _, H, W = u.shape
        attn = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)
        out = self.out_proj(self.ups(torch.cat((attn, u), dim=1)))
        return out


class LRSHA(nn.Module):
    def __init__(self, dim, qk_dim, pdim, sr=2, inp_group=1):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim
        self.split_index = (qk_dim, qk_dim, pdim, dim - pdim)
        self.pre_norm = nn.GroupNorm(1, dim)
        self.in_proj = Conv2d_BN(dim, (qk_dim * 2) + dim, 3, 1, 1, groups=inp_group)
        if sr > 1:
            self.k = Conv2d_BN(qk_dim, qk_dim, sr, sr, groups=qk_dim)
            self.v = Conv2d_BN(pdim, pdim, sr, sr, groups=pdim)
        else:
            self.k = nn.Identity()
            self.v = nn.Identity()
        self.out_proj = nn.Sequential(nn.ReLU(), Conv2d_BN(dim, dim, 1, 1))

    def forward(self, x):
        x = self.pre_norm(x)
        q, k, v, u = self.in_proj(x).split(self.split_index, dim=1)
        q, k, v = q.flatten(2), self.k(k).flatten(2), self.v(v).flatten(2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        B, _, H, W = u.shape
        attn = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)
        out = self.out_proj(torch.cat((attn, u), dim=1))
        return out


class PatchMerging(nn.Module):
    def __init__(self, inc, ouc, ks=3, act_layer=nn.ReLU):
        super().__init__()
        self.token_mix = nn.Sequential(
            Conv2d_BN(inc, inc, ks=3, stride=2, pad=1, groups=inc),
            act_layer(),
            Conv2d_BN(inc, ouc, ks=1, stride=1)
        )
        self.channel_mix = Residual(nn.Sequential(
            Conv2d_BN(ouc, ouc * 2, ks=1, stride=1, pad=0),
            act_layer(),
            Conv2d_BN(ouc * 2, ouc, ks=1, stride=1, pad=0)
        ))

    def forward(self, x):
        return self.channel_mix(self.token_mix(x))


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio, qk_dim, att_cr, att_sr, att_ipg, type, act_layer=nn.ReLU):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        pdim = int(dim * att_cr)

        if type == 'c':
            spatial_mix = Conv2d_BN(dim, dim, 3, 1, 1, groups=dim)
            self.block = nn.Sequential(
                Residual(spatial_mix),
                Residual(FFN(dim, hidden_dim, act_layer))
            )
        elif type == 'a':
            spatial_mix = LRSHA(dim, qk_dim, pdim, sr=att_sr, inp_group=att_ipg)
            self.block = nn.Sequential(
                Residual(spatial_mix),
                Residual(FFN(dim, hidden_dim, act_layer))
            )
        else:
            raise ValueError(f"Unknown block type: {type}")

    def forward(self, x):
        return self.block(x)


class Stage(nn.Module):
    def __init__(self, dim, depth, mlp_ratio, qk_dim, att_cr, att_sr, att_ipg, type, act_layer=nn.ReLU):
        super().__init__()
        self.blocks = nn.Sequential(*[
            Block(dim, mlp_ratio, qk_dim, att_cr, att_sr, att_ipg, type, act_layer)
            for _ in range(depth)
        ])

    def forward(self, x):
        return self.blocks(x)


class MicroViT(nn.Module):
    def __init__(self,
                 in_chans=1,
                 num_classes=10,
                 dims=[128, 256, 320],
                 depths=[2, 5, 5],
                 type=['c', 'c', 'a'],
                 qk_dim=[0, 0, 16],
                 attn_sr=[0, 0, 1],
                 attn_cr=[0, 0, 0.215],
                 attn_ipg=[0, 0, 32],
                 patch_size=4,              # recommended for MNIST
                 mlp_ratio=2,
                 act_layer=nn.GELU,
                 distillation=False,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.dims = list(dims)  # FIX: used by feature_dim

        if not isinstance(depths, (list, tuple)):
            depths = [depths]
        if not isinstance(dims, (list, tuple)):
            dims = [dims]

        num_stage = len(depths)
        self.num_stage = num_stage

        stages = []
        stages.append(StemLayer(in_chans, dims[0], ps=patch_size, act_layer=act_layer))

        for i_stage in range(num_stage):
            stages.append(Stage(
                dim=dims[i_stage],
                depth=depths[i_stage],
                mlp_ratio=mlp_ratio,
                qk_dim=qk_dim[i_stage],
                att_cr=attn_cr[i_stage],
                att_ipg=attn_ipg[i_stage],
                att_sr=attn_sr[i_stage],
                act_layer=act_layer,
                type=type[i_stage]
            ))
            if i_stage < (num_stage - 1):
                stages.append(PatchMerging(dims[i_stage], dims[i_stage + 1], act_layer=act_layer))
                stages.append(nn.Sequential(
                    Residual(Conv2d_BN(dims[i_stage + 1], dims[i_stage + 1], 3, 1, 1, groups=dims[i_stage + 1])),
                    Residual(FFN(dims[i_stage + 1], dims[i_stage + 1] * 2, act_layer=act_layer)),
                ))

        self.stages = nn.Sequential(*stages)
        self.avgpool_pre_head = nn.AdaptiveAvgPool2d(1)
        self.fc = Classfier(dims[-1], num_classes, distillation)

        self.apply(self.cls_init_weights)

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def feature_dim(self) -> int:
        return self.dims[-1]

    def extract_features(self, x):
        x = self.stages(x)
        x = self.avgpool_pre_head(x).flatten(1)
        return x

    def forward(self, x):
        x = self.stages(x)
        x = self.avgpool_pre_head(x).flatten(1)
        x = self.fc(x)
        return x
