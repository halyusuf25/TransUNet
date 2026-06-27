# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2

from .attention import SHSAttention, TopkAttention, ATSAttention
from .swin_transformer_official import SwinTransformer
from torchvision.models.efficientnet import MBConvConfig, MBConv
from .efficientnetpp import EfficientNetppDecoderBlock
import yaml
from easydict import EasyDict as edict
from .lib import topk_indices
from .se_block import SELayer

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.args = config
        self.num_attention_heads = config.transformer["num_heads"] # default = 12
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads) #default = 64
        self.all_head_size = self.num_attention_heads * self.attention_head_size # default = 768

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states,):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        if self.args.verbose:
            print(f"Attention input hidden_states shape: {hidden_states.shape}")
            print(f"Query, Key, Value shapes after linear projection: {mixed_query_layer.shape}, {mixed_key_layer.shape}, {mixed_value_layer.shape}")
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        if self.args.verbose:
            print(f"Query, Key, Value shapes after transpose for scores: {query_layer.shape}, {key_layer.shape}, {value_layer.shape}")
            
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.args.verbose:
            print(f"Raw attention scores shape (before scaling): {attention_scores.shape}")
            
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        # weights = attention_probs if self.vis else None
        # weights = attention_probs #always output attention score
        attention_probs = self.attn_dropout(attention_probs)

        if self.args.verbose:
            print(f"Attention probabilities shape: {attention_probs.shape}")
        context_layer = torch.matmul(attention_probs, value_layer)
        if self.args.verbose:
            print(f"Context layer shape (before merging heads): {context_layer.shape}")
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, attention_probs

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)
        with open("networks/swin_large_patch4_window7_224_22k.yaml", "r") as f:
            self.swin_config = edict(yaml.safe_load(f))

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            if self.config.use_swin:
                self.hybrid_model = SwinTransformer(img_size=self.swin_config.DATA.IMG_SIZE,
                                patch_size=self.swin_config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=self.swin_config.MODEL.SWIN.IN_CHANS,
                                num_classes=self.swin_config.MODEL.NUM_CLASSES,
                                embed_dim=self.swin_config.MODEL.SWIN.EMBED_DIM,
                                depths=self.swin_config.MODEL.SWIN.DEPTHS,
                                num_heads=self.swin_config.MODEL.SWIN.NUM_HEADS,
                                window_size=self.swin_config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=self.swin_config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=self.swin_config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=self.swin_config.MODEL.SWIN.QK_SCALE,
                                drop_rate=self.swin_config.MODEL.DROP_RATE,
                                drop_path_rate=self.swin_config.MODEL.DROP_PATH_RATE,
                                ape=self.swin_config.MODEL.SWIN.APE,
                                norm_layer=nn.LayerNorm,
                                patch_norm=self.swin_config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=True,
                                fused_window_process=False)
                
                checkpoint_path = self.config.swin_pretrained_path
                checkpoint = torch.load(checkpoint_path, map_location='cpu')

                if "model" in checkpoint:
                    state_dict = checkpoint["model"]
                else:
                    state_dict = checkpoint

                model_dict = self.hybrid_model.state_dict()
                filtered_dict = {}

                for k, v in state_dict.items():
                    if k in model_dict and v.shape == model_dict[k].shape:
                        filtered_dict[k] = v
                    else:
                        logger.warning(f"Skipping checkpoint key '{k}': shape mismatch or key not in model. "
                                     f"Expected shape {model_dict[k].shape if k in model_dict else 'N/A'}, "
                                     f"got shape {v.shape}")

                self.hybrid_model.load_state_dict(filtered_dict, strict=False)

                self.change_channel = nn.Linear(self.swin_config.MODEL.SWIN.EMBED_DIM * 2 ** (self.swin_config.MODEL.SWIN.DEPTHS.__len__() -1), config.hidden_size)
                # in_channels = self.swin_config.hidden_size * 16
                # self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
                self.patch_embeddings = nn.Identity()
            else:
                self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
                in_channels = self.hybrid_model.width * 16
                self.patch_embeddings = Conv2d(in_channels=in_channels,
                                            out_channels=config.hidden_size,
                                            kernel_size=patch_size,
                                            stride=patch_size)
                
                self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        if self.hybrid:
            if self.config.use_swin:
                x, _ , features = self.hybrid_model(x) # x: [B, N, hidden]
                # embeddings = x
                return self.change_channel(x), features
            
            x, features = self.hybrid_model(x)    
            x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
            x = x.flatten(2)
            x = x.transpose(-1, -2)  # (B, n_patches, hidden)

            embeddings = x + self.position_embeddings
            embeddings = self.dropout(embeddings)
        else:
            features = None
        
        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis, alternate_partial_attn=False):
        super(Block, self).__init__()
        self.args = config
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        
        self.use_shsa = config.use_shsa
        self.topk_attn = config.topk_attn
         
        if self.use_shsa:
            self.attn = SHSAttention(config, vis, alternate_partial_attn=alternate_partial_attn)
        elif config.use_ats:
            if self.args.verbose:
                print(f"Using Adaptive Token Sampling (ATS) for attention with keep_rate={self.topk_attn}.")
            self.attn = ATSAttention(config, config.hidden_size, keep_rate=self.topk_attn)
        elif self.topk_attn > 0.0:
            if self.args.verbose:
                print(f"Using Top-k Attention with keep_rate={self.topk_attn}.")
            self.attn = TopkAttention(config, config.hidden_size, keep_rate=self.topk_attn)
        else:
            self.attn = Attention(config, vis)    

    def forward(self, x, return_indices=False):
        # Pre-norm
        if self.args.verbose:
            print(f"Block input x shape: {x.shape}")
        # Multi-head self-attention with residual
        h = x
        x = self.attention_norm(x)
        if self.args.topk_attn > 0.0:
            # x: [B, N, D] -> [B, k, D], weights: [B, H, k, N], topk_idx: [B, k]
            x, weights, topk_idx = self.attn(x, return_indices=True)

            # Residual must be reduced to the same selected tokens: [B, k, D]
            gather_index = topk_idx.unsqueeze(-1).expand(-1, -1, h.size(-1))
            h = torch.gather(h, dim=1, index=gather_index)
        else:
            x, weights = self.attn(x)

        x = x + h
        if self.args.verbose:
            print(f"Block output x shape after attention and residual: {x.shape}")
            print(f"Block attention score shape: {weights.shape}")
        # FFN with residual
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        if self.args.verbose:
            print(f"Block output x shape after FFN and residual: {x.shape}")
        
        if return_indices:
            return x, weights, topk_idx if self.args.topk_attn > 0.0 else None
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            if not self.use_shsa and self.topk_attn <= 0.0:
                query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
                key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
                value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
                out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

                query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
                key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
                value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
                out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

                self.attn.query.weight.copy_(query_weight)
                self.attn.key.weight.copy_(key_weight)
                self.attn.value.weight.copy_(value_weight)
                self.attn.out.weight.copy_(out_weight)
                self.attn.query.bias.copy_(query_bias)
                self.attn.key.bias.copy_(key_bias)
                self.attn.value.bias.copy_(value_bias)
                self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.args = config
        self.vis = vis
        self.layer = nn.ModuleList()
        self.SELayer = nn.ModuleList() if self.args.use_se_block else None
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for i in range(config.transformer["num_layers"]):
            if config.use_alternate_shsa:
                alternate_partial_attn = (i % 2 == 1) # alternate every other layer
                logger.info(f"Using alternate partial attention in layer {i}: {alternate_partial_attn}")
            else: 
                alternate_partial_attn = False  

            layer = Block(config, vis, alternate_partial_attn=alternate_partial_attn)
            
            self.layer.append(copy.deepcopy(layer))
            
            if self.SELayer is not None:
                se = SELayer(config.hidden_size)
                self.SELayer.append(copy.deepcopy(se))

    

    def forward(self, hidden_states):
        attn_weights = []
        kept_indices = None  # absolute indices into the original (pre-prune) sequence
        absolute_indices = None
        # layer_block_id = 0
        if self.args.verbose:
            print(f"input shape for the Encoder Transformer Layers: {hidden_states.shape}")

        if self.args.topk_attn > 0.0:
            B, N0, _ = hidden_states.shape
            absolute_indices = torch.arange(
                N0,
                device=hidden_states.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(B, N0)
            if self.args.verbose:
                print(f"Encoder absolute kept_indices initialized with shape: {absolute_indices.shape}")

        se_layers = getattr(self, "SELayer", None)
        drop_se = getattr(self.args, "drop_se_block", False)
        use_se = se_layers is not None and not drop_se
        if not use_se:
            se_layers = [None] * len(self.layer)
        se_scale = []
        
        # for layer_block in self.layer:
        for layer_block_id, (layer_block, se_layer) in enumerate(zip(self.layer, se_layers)):
            if self.args.verbose:
                print(f"Encoder layer#{layer_block_id} input hidden_states shape: {hidden_states.shape}")

            if self.args.topk_attn > 0.0:
                hidden_states, attn, local_idx = layer_block(hidden_states, return_indices=True)
                local_idx = local_idx.to(device=absolute_indices.device, dtype=torch.long)
                if self.args.verbose:
                    print(
                        f"Encoder layer#{layer_block_id} local_idx min/max: "
                        f"{local_idx.min().item()}/{local_idx.max().item()} "
                        f"within current token range [0, {absolute_indices.size(1) - 1}]"
                )
                absolute_indices = torch.gather(
                    absolute_indices,
                    dim=1,
                    index=local_idx,
                )
                if self.args.verbose:
                    print(
                        f"Encoder layer#{layer_block_id} absolute kept_indices min/max: "
                        f"{absolute_indices.min().item()}/{absolute_indices.max().item()} "
                        f"within original patch range [0, {N0 - 1}]"
                )
                kept_indices = absolute_indices
            else:
                hidden_states, attn = layer_block(hidden_states)
            
            if self.args.verbose:
                print(f"Encoder layer#{layer_block_id} output hidden_states shape: {hidden_states.shape}")
                print(f"Encoder layer#{layer_block_id} attention weights shape: {attn.shape}")
                if kept_indices is not None:
                    print(f"Encoder layer#{layer_block_id} absolute kept_indices shape: {kept_indices.shape}")

            
            attn_weights.append(attn)
            
            if use_se and se_layer is not None:
                # se_in = hidden_states.transpose(1, 2).unsqueeze(2)  # [B, C, 1, N]
                # se_out, _ = se_layer(se_in)
                # hidden_states = se_out.squeeze(2).transpose(1, 2)   # [B, N, C]
                # se_in = hidden_states.transpose(1, 2).unsqueeze(2)  # [B, C, 1, N]
                h_se, scale = se_layer(hidden_states)
                if self.args.verbose:
                    print(f"SE Layer scale shape at Encoder layer#{layer_block_id}: {scale.shape}")
                    print(f"Encoder layer#{layer_block_id} hidden_states shape after SE block: {h_se.shape}")
                    
                se_scale.append(scale)
        
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights, kept_indices, se_scale
    
class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.config = config
        self.embeddings = Embeddings(config, img_size=img_size)
        # self.embeddings = SwinTransformer(get_swin_tiny_config(), img_size=img_size, vis=vis)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        if self.config.verbose:
            print(f"Backbone input shape: {input_ids.shape}")
            #torch.Size([2, 3, 224, 224]) = [batch, channels, height, width] → batch size 2, RGB 3 channels, 224×224 input images.


        embedding_output, features = self.embeddings(input_ids)
        # embedding_output, _, features = self.embeddings(input_ids)
        # encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        # return encoded, attn_weights, features
        orig_n_patches = embedding_output.size(1)
        if self.config.verbose:
            print(f"Embedding output and Input to encoder transformer layer(0) shape: {embedding_output.shape}")
            
        encoded, attn_weights, kept_indices, se_scale = self.encoder(embedding_output)  # (B, n_patch, hidden)
        # output ={ # to be used instead of the current return statement
        #     "encoded": encoded,
        #     "attn_weights": attn_weights,
        #     "features": features,
        #     "kept_indices": kept_indices,
        #     "orig_n_patches": orig_n_patches,
        #     "se_scale": se_scale,
        # }
        return encoded, attn_weights, features, kept_indices, orig_n_patches, se_scale

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

        bn = nn.BatchNorm2d(out_channels)

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

class MBConvDecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio=4, drop_rate=0.0):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        config = MBConvConfig(
            kernel=3, expand_ratio=expand_ratio,
            input_channels=in_ch, out_channels=out_ch,
            stride=1, num_layers=1
        )
        self.mbconv = MBConv(config, stochastic_depth_prob=0.1, norm_layer=torch.nn.BatchNorm2d)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        return self.mbconv(x)

class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        if self.config.use_swin:
            self.conv_more_pre = nn.ConvTranspose2d(
                config.hidden_size,
                config.hidden_size,
                kernel_size=2,
                stride=2,
            )
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        # if self.config.use_swin and not self.config.use_efficientnet:
        #     self.conv_more_skip = Conv2dReLU(
        #         config.hidden_size,
        #         head_channels,
        #         kernel_size=3,
        #         padding=1,
        #         use_batchnorm=True,
        #     )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if config.verbose:
            print(f"skip channels before n_skip adjustment: {self.config.n_skip}")
        
        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]
            Warning("n_skip is set to 0, no skip connection is used in the decoder.")

        if config.verbose:
            print(f"skip channels after n_skip adjustment: {skip_channels}")
        
        if self.config.use_efficientnet:
            blocks = [
            EfficientNetppDecoderBlock(in_ch, sk_ch, out_ch) for in_ch, sk_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        else:
            blocks = [
                DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
            ]

        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        if self.config.verbose:
            print(f"Decoder input hidden_states shape: {hidden_states.size()}")
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        if self.config.verbose:
            print(f"Decoder input hidden_states permuted shape: {x.size()}")
        x = x.contiguous().view(B, hidden, h, w)
        if self.config.verbose:
            print(f"Decoder input hidden_states reshaped to image grid shape: {x.size()}")
        if self.config.use_swin:
            x = self.conv_more_pre(x)
        x = self.conv_more(x)
        # if self.config.use_swin:
        #     x = self.conv_more2(x)
        # if self.config.use_swin and not self.config.use_efficientnet and features is not None:
        #     features_new = []
        #     for feature in features:
        #         B, n_patch, hidden = feature.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        #         h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        #         feature = feature.permute(0, 2, 1)
        #         feature = feature.contiguous().view(B, hidden, h, w)
        #         features_new.append(feature)
        #         feature = self.conv_more_skip(feature)

        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            if self.config.verbose:
                print(f"Decoder block {i}, x shape: {x.shape}, skip shape: {skip.shape if skip is not None else None}, features shape: {features[i].shape if features is not None and i < len(features) else None}")
            x = decoder_block(x, skip=skip)

        if self.config.verbose:
            print(f"Decoder output shape: {x.shape}")
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.args = config
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        x, attn_weights, features, kept_indices, orig_n_patches, se_scale = self.transformer(x)  # (B, n_patch, hidden)
        # If pruning occurred, scatter tokens back to the original grid length
        if kept_indices is not None:
            if x.dim() != 3:
                raise RuntimeError(f"Expected encoded tokens to be [B, K, C], got {tuple(x.shape)}.")
            if kept_indices.dim() != 2:
                raise RuntimeError(f"Expected kept_indices to be [B, K], got {tuple(kept_indices.shape)}.")
            B, K, C = x.size()
            if kept_indices.shape != (B, K):
                raise RuntimeError(
                    f"Expected kept_indices shape {(B, K)} to match encoded tokens, got {tuple(kept_indices.shape)}."
                )
            kept_indices = kept_indices.to(device=x.device, dtype=torch.long)
            if kept_indices.numel() == 0:
                raise RuntimeError("kept_indices must not be empty before decoder scatter.")
            
            if self.args.verbose:
                min_kept = kept_indices.min().item()
                max_kept = kept_indices.max().item()
                if min_kept < 0 or max_kept >= orig_n_patches:
                    raise RuntimeError(
                        f"kept_indices out of range for scatter: min={min_kept}, max={max_kept}, "
                        f"orig_n_patches={orig_n_patches}."
                    )
                else:
                    print(f"Decoder scatter kept_indices min/max: {min_kept}/{max_kept}, orig_n_patches={orig_n_patches}.")
            
            # total_n = orig_n_patches
            full = x.new_zeros(B, orig_n_patches, C)
            scatter_index = kept_indices.unsqueeze(-1).expand(-1, -1, C)  # (B, K, C)
            full.scatter_(1, scatter_index, x)
            if self.args.verbose:
                print(f"Decoder scatter kept_indices shape: {kept_indices.shape}, full sequence shape: {full.shape}")
            x = self.decoder(full, features)
        else:
            x = self.decoder(x, features)

        if self.args.verbose:
            print(f"Segmentation head input shape: {x.size()}")
        logits = self.segmentation_head(x)
        if self.args.verbose:
            print(f"Segmentation head output logits shape: {logits.size()}")
        
        # output = { # to be used instead of the current return statement
        #     "logits": logits,
        #     "attn_weights": attn_weights,
        #     "features": features,
        #     "kept_indices": kept_indices,
        #     "orig_n_patches": orig_n_patches,
        #     "se_scale": se_scale,
        # }
        return logits, attn_weights, features, se_scale

    def load_from(self, weights):
        with torch.no_grad():
            
            if not self.config.use_swin:
                res_weight = weights
                self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
                self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

                self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
                self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

                posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

                posemb_new = self.transformer.embeddings.position_embeddings
                if posemb.size() == posemb_new.size():
                    self.transformer.embeddings.position_embeddings.copy_(posemb)
                elif posemb.size()[1]-1 == posemb_new.size()[1]:
                    posemb = posemb[:, 1:]
                    self.transformer.embeddings.position_embeddings.copy_(posemb)
                else:
                    logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                    ntok_new = posemb_new.size(1)
                    if self.classifier == "seg":
                        _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    gs_old = int(np.sqrt(len(posemb_grid)))
                    gs_new = int(np.sqrt(ntok_new))
                    print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                    posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                    zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                    posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                    posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                    posemb = posemb_grid
                    self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

                # Encoder whole
                if not self.config.use_se_block:
                    for bname, block in self.transformer.encoder.named_children():
                        for uname, unit in block.named_children():
                            unit.load_from(weights, n_block=uname)

                if self.transformer.embeddings.hybrid:
                    self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                    gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                    gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                    self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                    self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                    for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                        for uname, unit in block.named_children():
                            unit.load_from(res_weight, n_block=bname, n_unit=uname)

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}
