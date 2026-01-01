import random
from timm.models.layers import DropPath, to_2tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


##########################################################################
# Basic modules
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer


def default_conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), stride=stride, bias=bias)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.PReLU(), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            if i == 0:
                m.append(conv(n_feats, 64, kernel_size, bias=bias))
            else:
                m.append(conv(64, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

    
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y 
    

# class mergeblock(nn.Module):
#     def __init__(self, n_feat, kernel_size, bias):
#         super(mergeblock, self).__init__()
#         self.conv_block = conv(n_feat * 2, n_feat, kernel_size, bias=bias)

#     def forward(self, x, bridge):
#         out = torch.cat([x, bridge], 1)
#         out = self.conv_block(out)
#         return out + x
class mergeblock(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, subspace_dim=16):
        super(mergeblock, self).__init__()
        self.conv_block = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.num_subspace = subspace_dim
        self.subnet = conv(n_feat * 2, self.num_subspace, kernel_size, bias=bias)

    def forward(self, x, bridge):
        out = torch.cat([x, bridge], 1)
        b_, c_, h_, w_ = bridge.shape
        sub = self.subnet(out)
        V_t = sub.view(b_, self.num_subspace, h_*w_)
        V_t = V_t / (1e-6 + torch.abs(V_t).sum(axis=2, keepdims=True))
        V = V_t.permute(0, 2, 1)
        mat = torch.matmul(V_t, V)
        try:
            mat_inv = torch.inverse(mat)
        except:
            mat_inv = torch.linalg.pinv(mat)
        project_mat = torch.matmul(mat_inv, V_t)
        bridge_ = bridge.view(b_, c_, h_*w_)
        project_feature = torch.matmul(project_mat, bridge_.permute(0, 2, 1))
        bridge = torch.matmul(V, project_feature).permute(0, 2, 1).view(b_, c_, h_, w_)
        out = torch.cat([x, bridge], 1)
        out = self.conv_block(out)
        return out+x
    

class SKFF(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size, n_feats, H, W = inp_feats[1].shape

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)

        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V
    

class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)
        self.sk = SKFF(out_channels, height=2, reduction=8, bias=False)

    def forward(self, x):
        x1 = self.relu(self.conv(x))
        output = self.sk((x, x1))
        return output
    

class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x1 = x1 + x
        return x1, img
    

class SK_RDB(nn.Module):
    def __init__(self, n_feat, num_layers=3):
        super(SK_RDB, self).__init__()
        self.proj = nn.Conv2d(3, n_feat, 1, 1, 0)
        self.identity = nn.Conv2d(n_feat * 2, n_feat, 1, 1, 0)
        self.layers = nn.Sequential(
            *[DenseLayer(n_feat, n_feat) for _ in range(num_layers)]
        )
        self.lff = nn.Conv2d(n_feat * 2, 3, kernel_size=1)

    def forward(self, x, x_img):
        proj = self.proj(x_img)    
        feat = torch.cat([x, proj], 1)
        feat = self.identity(feat)
        feat = self.layers(feat)
        out = self.lff(torch.cat([proj, feat], 1)) + x_img
        feat = feat + x
        return feat, out
    
    
##########################################################################
## U-Net    
class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff, depth=5):
        super(Encoder, self).__init__()
        self.body=nn.ModuleList()#[]
        self.depth=depth
        for i in range(depth-1):
            self.body.append(UNetConvBlock(in_size=n_feat+scale_unetfeats*i, out_size=n_feat+scale_unetfeats*(i+1), downsample=True, relu_slope=0.2, use_csff=csff, use_HIN=True))
        self.body.append(UNetConvBlock(in_size=n_feat+scale_unetfeats*(depth-1), out_size=n_feat+scale_unetfeats*(depth-1), downsample=False, relu_slope=0.2, use_csff=csff, use_HIN=True))

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        res=[]
        if encoder_outs is not None and decoder_outs is not None:
            for i, down in enumerate(self.body):
                if (i+1) < self.depth:
                    x, x_up = down(x,encoder_outs[i],decoder_outs[-i-1])
                    res.append(x_up)
                else:
                    x = down(x)
        else:
            for i, down in enumerate(self.body):
                if (i+1) < self.depth:
                    x, x_up = down(x)
                    res.append(x_up)
                else:
                    x = down(x)
        return res, x


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_csff=False, use_HIN=False):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_csff = use_csff

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.csff_dec = nn.Conv2d(in_size, out_size, 3, 1, 1)
            self.phi = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.gamma = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size//2, affine=True)
        self.use_HIN = use_HIN

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, enc=None, dec=None):
        out = self.conv_1(x)

        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

        out += self.identity(x)
        if enc is not None and dec is not None:
            assert self.use_csff
            skip_ = F.leaky_relu(self.csff_enc(enc) + self.csff_dec(dec), 0.1, inplace=True)
            out = out*torch.sigmoid(self.phi(skip_)) + self.gamma(skip_) + out
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out
    

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(out_size*2, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out
    

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x
    

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
    

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate != 1:
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                     stride=win_size)  # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0, 2, 1).contiguous().view(-1, C, win_size, win_size)  # B' ,C ,Wh ,Ww
        windows = windows.permute(0, 2, 3, 1).contiguous()  # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)  # B' ,Wh ,Ww ,C
    return windows


def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate != 1:
        x = windows.permute(0, 5, 3, 4, 1, 2).contiguous()  # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                   stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class ConvProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout=0.,
                 last_stage=False, bias=True):
        super().__init__()

        inner_dim = dim_head * heads
        self.heads = heads
        pad = (kernel_size - q_stride) // 2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad, bias)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad, bias)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad, bias)

    def forward(self, x, attn_kv=None):
        b, n, c, h = *x.shape, self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))

        attn_kv = x if attn_kv is None else attn_kv
        x = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)
        attn_kv = rearrange(attn_kv, 'b (l w) c -> b c l w', l=l, w=w)
        # print(attn_kv)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)

        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        return q, k, v


class PModule(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1)
        #self.selayer = SELayer(hidden_dim//2)
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim//2, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x, img_size=(128, 128)):
        # bs x hw x c
        hh,ww = img_size[0],img_size[1]
        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=ww)

        x1,x2 = self.dwconv(x).chunk(2, dim=1)
        x3 = x1 * x2
        #x4=self.selayer(x3)
        # flaten
        x3 = rearrange(x3, ' b c h w -> b (h w) c', h=hh, w=ww)
        y = self.linear2(x3)

        return y
    

class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_, 1, 1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q, k, v
    

class WindowAttention(nn.Module):
    def __init__(self, dim, win_size, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if token_projection == 'conv':
            self.qkv = ConvProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        elif token_projection == 'linear':
            self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        else:
            raise Exception("Projection error!")

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        q, k, v = self.qkv(x)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class ShuffleBlock(nn.Module):
    def __init__(self, dim, num_heads, win_size=8,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_projection='linear'):
        super(ShuffleBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.win_size = win_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.Pmodule = PModule(dim=dim, hidden_dim=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.norm2 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            token_projection=token_projection)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        shortcut = x
        img_size = (H, W)

        # Attention
        x = self.norm2(x)
        x = x.view(B, H, W, C)
        
        shuffle_indices = self._get_shuffle_indices(B, H, W)
        x = self._shuffle_and_attend(x, shuffle_indices)
        
        x = shortcut + self.drop_path(x)

        # MLP
        x = x + self.drop_path(self.Pmodule(self.norm1(x), img_size=img_size))
        
        return rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

    def _get_shuffle_indices(self, B, H, W):
        return [
            (torch.randperm(H), torch.randperm(W))
            for _ in range(B)
        ]

    def _shuffle_and_attend(self, x, shuffle_indices):
        B, H, W, C = x.shape
        
        # Shuffle
        x = torch.stack([
            x[b, h_idx[:, None], w_idx, :] 
            for b, (h_idx, w_idx) in enumerate(shuffle_indices)
        ])

        # Window partition and attention
        x = window_partition(x, self.win_size)
        x = x.view(-1, self.win_size * self.win_size, C)
        x = self.attn(x)
        x = x.view(-1, self.win_size, self.win_size, C)
        x = window_reverse(x, self.win_size, H, W)

        # Reverse shuffle
        return torch.stack([
            x[b, h_idx.argsort()[:, None], w_idx.argsort(), :]
            for b, (h_idx, w_idx) in enumerate(shuffle_indices)
        ]).view(B, H * W, C)
    

class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=5):
        super(Decoder, self).__init__()
        
        self.body=nn.ModuleList()
        self.skip_conv=nn.ModuleList()#[]
        for i in range(depth-1):
            self.body.append(UNetUpBlock(in_size=n_feat+scale_unetfeats*(depth-i-1), out_size=n_feat+scale_unetfeats*(depth-i-2), relu_slope=0.2))
            self.skip_conv.append(nn.Conv2d(n_feat+scale_unetfeats*(depth-i-1), n_feat+scale_unetfeats*(depth-i-2), 3, 1, 1))
            
    def forward(self, x, bridges):
        res=[]
        for i,up in enumerate(self.body):
            x=up(x,self.skip_conv[i](bridges[-i-1]))
            res.append(x)
        return res
    
    
##########################################################################
## DGUNet_plus  
class DocUnfold(nn.Module):
    def __init__(self, n_feat=40, scale_unetfeats=20, kernel_size=3, reduction=4, bias=False, act=nn.PReLU()):
        super(DocUnfold, self).__init__()
        # Extract Shallow Features
        self.shallow_feat1 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat3 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat4 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat5 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat6 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat7 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        
        # Gradient Descent Module (GDM)
        # self.phi_0 = ResBlock(default_conv,3,3)
        # self.phi_1 = ResBlock(default_conv,3,3)
        # self.phi_2 = ResBlock(default_conv,3,3)
        # self.phi_3 = ResBlock(default_conv,3,3)
        # self.phi_4 = ResBlock(default_conv,3,3)
        # self.phi_5 = ResBlock(default_conv,3,3)
        # self.phi_6 = ResBlock(default_conv,3,3)

        # self.phit_0 = ResBlock(default_conv,3,3)
        # self.phit_1 = ResBlock(default_conv,3,3)
        # self.phit_2 = ResBlock(default_conv,3,3)
        # self.phit_3 = ResBlock(default_conv,3,3)
        # self.phit_4 = ResBlock(default_conv,3,3)
        # self.phit_5 = ResBlock(default_conv,3,3)
        # self.phit_6 = ResBlock(default_conv,3,3)

        self.phi_0 = ShuffleBlock(dim=3, num_heads=1)
        self.phi_1 = ShuffleBlock(dim=3, num_heads=1)
        self.phi_2 = ShuffleBlock(dim=3, num_heads=1)
        self.phi_3 = ShuffleBlock(dim=3, num_heads=1)
        self.phi_4 = ShuffleBlock(dim=3, num_heads=1)
        self.phi_5 = ShuffleBlock(dim=3, num_heads=1)
        self.phi_6 = ShuffleBlock(dim=3, num_heads=1)
        
        self.phit_0 = ResBlock(default_conv,3,3)
        self.phit_1 = ResBlock(default_conv,3,3)
        self.phit_2 = ResBlock(default_conv,3,3)
        self.phit_3 = ResBlock(default_conv,3,3)
        self.phit_4 = ResBlock(default_conv,3,3)
        self.phit_5 = ResBlock(default_conv,3,3)
        self.phit_6 = ResBlock(default_conv,3,3)

        self.r0 = nn.Parameter(torch.Tensor([0.5]))
        self.r1 = nn.Parameter(torch.Tensor([0.5]))
        self.r2 = nn.Parameter(torch.Tensor([0.5]))
        self.r3 = nn.Parameter(torch.Tensor([0.5]))
        self.r4 = nn.Parameter(torch.Tensor([0.5]))
        self.r5 = nn.Parameter(torch.Tensor([0.5]))
        self.r6 = nn.Parameter(torch.Tensor([0.5]))

        # Informative Proximal Mapping Module (IPMM)
        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=4, csff=False)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=4)

        self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=4, csff=True)
        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=4)

        self.stage3_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=4, csff=True)
        self.stage3_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=4)

        self.stage4_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=4, csff=True)
        self.stage4_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=4)
        
        self.stage5_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=4, csff=True)
        self.stage5_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=4)
        
        self.stage6_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=4, csff=True)
        self.stage6_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=4)

        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)
        self.sam23 = SAM(n_feat, kernel_size=1, bias=bias)
        self.sam34 = SAM(n_feat, kernel_size=1, bias=bias)
        self.sam45 = SAM(n_feat, kernel_size=1, bias=bias)
        self.sam56 = SAM(n_feat, kernel_size=1, bias=bias)
        self.sam67 = SAM(n_feat, kernel_size=1, bias=bias)
        
        self.merge12=mergeblock(n_feat, 3, True)
        self.merge23=mergeblock(n_feat, 3, True)
        self.merge34=mergeblock(n_feat, 3, True)
        self.merge45=mergeblock(n_feat, 3, True)
        self.merge56=mergeblock(n_feat, 3, True)
        self.merge67=mergeblock(n_feat, 3, True)
        
        self.tail = conv(n_feat, 3, kernel_size, bias=bias)

    def forward(self, img):
        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------
        ## GDM
        phixsy_1 = self.phi_0(img) - img
        x1_img = img - self.r0*self.phit_0(phixsy_1)
        # PMM
        x1 = self.shallow_feat1(x1_img)
        feat1,feat_fin1 = self.stage1_encoder(x1)
        res1 = self.stage1_decoder(feat_fin1,feat1)
        x2_samfeats, stage1_img = self.sam12(res1[-1], x1_img)

        ##-------------------------------------------
        ##-------------- Stage 2---------------------
        ##-------------------------------------------
        ## GDM
        phixsy_2 = self.phi_1(stage1_img) - img
        x2_img = stage1_img - self.r1*self.phit_1(phixsy_2)
        x2 = self.shallow_feat2(x2_img)
        ## PMM
        x2_cat = self.merge12(x2, x2_samfeats)
        feat2,feat_fin2 = self.stage2_encoder(x2_cat, feat1, res1)
        res2 = self.stage2_decoder(feat_fin2,feat2)
        x3_samfeats, stage2_img = self.sam23(res2[-1], x2_img)

        ##-------------------------------------------
        ##-------------- Stage 3---------------------
        ##-------------------------------------------
        ## GDM
        phixsy_3 = self.phi_2(stage2_img) - img
        x3_img = stage2_img - self.r2*self.phit_2(phixsy_3)
        ## PMM
        x3 = self.shallow_feat3(x3_img)
        x3_cat = self.merge23(x3, x3_samfeats)
        feat3,feat_fin3 = self.stage3_encoder(x3_cat, feat2, res2)
        res3 = self.stage3_decoder(feat_fin3,feat3)
        x4_samfeats, stage3_img = self.sam34(res3[-1], x3_img)

        ##-------------------------------------------
        ##-------------- Stage 4---------------------
        ##-------------------------------------------
        ## GDM
        phixsy_4 = self.phi_3(stage3_img) - img
        x4_img = stage3_img - self.r3*self.phit_3(phixsy_4)
        ## PMM
        x4 = self.shallow_feat4(x4_img)
        x4_cat = self.merge34(x4, x4_samfeats)
        feat4,feat_fin4 = self.stage4_encoder(x4_cat, feat3, res3)
        res4 = self.stage4_decoder(feat_fin4,feat4)
        x5_samfeats, stage4_img = self.sam45(res4[-1], x4_img)
        
        ##-------------------------------------------
        ##-------------- Stage 5---------------------
        ##-------------------------------------------
        ## GDM
        phixsy_5 = self.phi_4(stage4_img) - img
        x5_img = stage4_img - self.r4*self.phit_4(phixsy_5)
        ## PMM
        x5 = self.shallow_feat5(x5_img)
        x5_cat = self.merge45(x5, x5_samfeats)
        feat5,feat_fin5 = self.stage5_encoder(x5_cat, feat4, res4)
        res5 = self.stage5_decoder(feat_fin5,feat5)
        x6_samfeats, stage5_img = self.sam56(res5[-1], x5_img)
        
        ##-------------------------------------------
        ##-------------- Stage 6---------------------
        ##-------------------------------------------
        ## GDM
        phixsy_6 = self.phi_5(stage5_img) - img
        x6_img = stage5_img - self.r5*self.phit_5(phixsy_6)
        ## PMM
        x6 = self.shallow_feat6(x6_img)
        x6_cat = self.merge56(x6, x6_samfeats)
        feat6,feat_fin6 = self.stage6_encoder(x6_cat, feat5, res5)
        res6 = self.stage6_decoder(feat_fin6,feat6)
        x7_samfeats, stage6_img = self.sam67(res6[-1], x6_img)

        ##-------------------------------------------
        ##-------------- Stage 7---------------------
        ##-------------------------------------------
        ## GDM
        phixsy_7 = self.phi_6(stage6_img) - img
        x7_img = stage6_img - self.r6*self.phit_6(phixsy_7)
        ## PMM
        x7 = self.shallow_feat7(x7_img)
        x7_cat = self.merge67(x7, x7_samfeats)
        stage7_img = self.tail(x7_cat)+ img

        return [stage7_img, stage6_img, stage5_img, stage4_img, stage3_img, stage2_img, stage1_img]
    

if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256)
    model = DocUnfold()
    res = model(x)
    for temp in res:
        print(temp.shape)
    # calculate parameter and flops using thop
    from thop import profile
    macs, params = profile(model, inputs=(x,))
    print('macs: ', macs, 'params: ', params)
    print('macs: ', macs / 10 ** 9, 'G, params: ', params / 10 ** 6, 'M')
