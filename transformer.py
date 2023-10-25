from functools import partial
import math
import torch


import torch.nn as nn
from typing import Callable, List, Optional, NamedTuple
from torchvision.models import VisionTransformer
from torchvision.ops.misc import Conv2dNormActivation

class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    padding: int = 0
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU

class GrayscaleVisionTransformer(VisionTransformer):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(
            nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
    ):
        super().__init__(image_size=image_size, patch_size=patch_size,
                         num_layers=num_layers, num_heads=num_heads, hidden_dim=hidden_dim, mlp_dim=mlp_dim, dropout=dropout, attention_dropout=attention_dropout, num_classes=num_classes, representation_size=representation_size, norm_layer=norm_layer)

        if conv_stem_configs is not None:
            # As per https://arxiv.org/abs/2106.14881
            seq_proj = nn.Sequential()
            prev_channels = 1
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                seq_proj.add_module(f"conv_dropout_{i}", nn.Dropout(p=dropout))
                seq_proj.add_module(
                    f"conv_bn_relu_{i}",
                    Conv2dNormActivation(
                        in_channels=prev_channels,
                        out_channels=conv_stem_layer_config.out_channels,
                        kernel_size=conv_stem_layer_config.kernel_size,
                        padding=conv_stem_layer_config.padding,
                        stride=conv_stem_layer_config.stride,
                        norm_layer=conv_stem_layer_config.norm_layer,
                        activation_layer=conv_stem_layer_config.activation_layer,
                    ),
                )
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.add_module(f"conv_dropout_last", nn.Dropout(p=dropout))
            seq_proj.add_module(
                "conv_last", nn.Conv2d(
                    in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
            )
            self.conv_proj: nn.Module = seq_proj
        else:
            self.conv_proj = nn.Conv2d(
                in_channels=1, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
            )

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * \
                self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight,
                                  std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        # if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
        #     fan_in = self.heads.pre_logits.in_features
        #     nn.init.trunc_normal_(
        #         self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
        #     nn.init.zeros_(self.heads.pre_logits.bias)

        # if isinstance(self.heads.head, nn.Linear):
        #     nn.init.zeros_(self.heads.head.weight)
        #     nn.init.zeros_(self.heads.head.bias)
