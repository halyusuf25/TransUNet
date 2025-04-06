# swin_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import LayerNorm
import ml_collections

def _pair(x):
    if isinstance(x, (tuple, list)):
        return x
    return (x, x)

class SwinPatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class WindowAttention(nn.Module):
    """ Window-based Multi-head Self-Attention """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, dropout_rate=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = _pair(window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinMLP(nn.Module):
    """ MLP as used in Swin Transformer """
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout_rate=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SwinBlock(nn.Module):
    """ Swin Transformer Block """
    def __init__(self, dim, num_heads, window_size=7, mlp_ratio=4., 
                 qkv_bias=True, dropout_rate=0.0):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, 
                                  qkv_bias=qkv_bias, dropout_rate=dropout_rate)
        self.norm2 = LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SwinMLP(in_features=dim, hidden_features=mlp_hidden_dim,
                          dropout_rate=dropout_rate)
        
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + x
        
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x
        return x

class SwinEncoder(nn.Module):
    """ Swin Transformer Encoder """
    def __init__(self, config, img_size, vis=False):
        super().__init__()
        self.vis = vis
        self.patch_embed = SwinPatchEmbed(img_size=img_size, 
                                        patch_size=config.patch_size,
                                        in_chans=3, 
                                        embed_dim=config.hidden_size)
        
        self.layers = nn.ModuleList([
            SwinBlock(dim=config.hidden_size, 
                     num_heads=config.transformer["num_heads"],
                     window_size=config.window_size,
                     mlp_ratio=config.transformer["mlp_ratio"],
                     qkv_bias=True,
                     dropout_rate=config.transformer["dropout_rate"])
            for _ in range(config.transformer["num_layers"])
        ])
        
        self.norm = LayerNorm(config.hidden_size, eps=1e-6)
        
    def forward(self, x):
        x = self.patch_embed(x)
        attn_weights = []
        
        for layer in self.layers:
            x = layer(x)
            if self.vis:
                attn_weights.append(None)
            
        x = self.norm(x)
        return x, attn_weights

class SwinTransformer(nn.Module):
    """ Swin Transformer backbone """
    def __init__(self, config, img_size=224, vis=False):
        super().__init__()
        self.encoder = SwinEncoder(config, img_size, vis)
        self.config = config
        
    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        encoded, attn_weights = self.encoder(x)
        features = None
        return encoded, attn_weights, features

# Decoder Components (moved from vit_seg_modeling.py)
class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        super(Conv2dReLU, self).__init__(conv, bn, relu)

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,  # Swin-Tiny: 96
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels  # (256, 128, 64, 16)
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        skip_channels = [0, 0, 0, 0]  # No skip connections for basic Swin

        # Use only 2 blocks for Swin with patch_size=4 (56 -> 112 -> 224)
        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) 
            for in_ch, out_ch, sk_ch in zip(in_channels[:2], out_channels[:2], skip_channels[:2])
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # e.g., [B, 3136, 96] for 56x56
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))  # 56x56
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)  # [B, 96, 56, 56]
        x = self.conv_more(x)  # [B, 512, 56, 56]
        for i, decoder_block in enumerate(self.blocks):
            skip = None  # No skip connections
            x = decoder_block(x, skip=skip)  # 56->112, 112->224
        return x  # [B, 128, 224, 224]

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)

class SwinTransNet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=9, zero_head=False, vis=False):
        super(SwinTransNet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.transformer = SwinTransformer(config, img_size=img_size, vis=vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config.decoder_channels[1],  # 128 after 2 blocks
            out_channels=config.n_classes,
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, attn_weights, features = self.transformer(x)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits

    def load_from(self, weights=None):
        pass

def get_swin_tiny_config():
    config = ml_collections.ConfigDict()
    config.patch_size = 4
    config.hidden_size = 96
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_ratio = 4.0
    config.transformer.num_heads = 3
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.window_size = 7
    
    config.classifier = 'seg'
    config.representation_size = None
    config.pretrained_path = None
    
    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.n_skip = 0
    config.skip_channels = [512, 256, 64, 16]
    config.activation = 'softmax'
    return config        

