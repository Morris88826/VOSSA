import typing as tp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import StreamingConv1d, TransposedLayerNorm
# from .conv2d import StreamingConv2d
from .streaming import StreamingContainer, StreamingAdd
# from .frame_exciter import FrameRateExciter, ExciteFuse
from ..utils.compile import torch_compile_lazy


# DropPath copied from timm library
def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """  # noqa: E501

    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""  # noqa: E501

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """  # noqa: E501

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x


class ChannelRMSNorm1d(nn.Module):
    """Streaming-friendly per-channel RMSNorm over time: (B,C,T) -> (B,C,T)"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(1, dim, 1))
        self.eps = eps
    def forward(self, x):
        rms = torch.sqrt((x**2).mean(dim=1, keepdim=True) + self.eps)  # (B,1,T)
        return self.gain * (x / rms)
        

# ConvNeXt Block copied from https://github.com/fishaudio/fish-diffusion/blob/main/fish_diffusion/modules/convnext.py
class ConvNeXtBlock(StreamingContainer):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        kernel_size (int): Kernel size for depthwise conv. Default: 7.
        dilation (int): Dilation for depthwise conv. Default: 1.
    """  # noqa: E501

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.1,
        layer_scale_init_value: float = 1e-6,
        mlp_ratio: float = 4.0,
        kernel_size: int = 7,
        dilation: int = 1,
        norm: str = "none",
        norm_params: tp.Dict[str, tp.Any] = {},
        causal: bool = False,
        pad_mode: str = "reflect",
    ):
        super().__init__()

        self.dwconv = StreamingConv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            norm="weight_norm",
            norm_kwargs=norm_params,
            causal=causal,
            pad_mode=pad_mode,
            groups=dim,
        )

        # self.norm = LayerNorm(dim, eps=1e-6) -> using batch norm instead of layer norm
        self.norm = nn.BatchNorm1d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, int(mlp_ratio * dim)
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.add = StreamingAdd()

    def forward(self, x, apply_residual: bool = True):
        input = x

        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)
        x = self.drop_path(x)

        if apply_residual:
            # x = input + x
            x = self.add(input, x)

        return x
    

class ChannelRMSNorm2d(nn.Module):
    # x: (B, C, F, T); normalize across C only, per (F,T)
    def __init__(self, C, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, C, 1, 1))
            self.bias   = nn.Parameter(torch.zeros(1, C, 1, 1))
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)  # over C only
        y = x / rms
        if self.affine:
            y = y * self.weight + self.bias
        return y


# class ConvNeXtBlock2d(StreamingContainer):
#     r"""ConvNeXt-style block for 2D with streaming along time (W).

#     Structure:
#       DwConv(StreamingConv2d, groups=dim) -> BatchNorm2d -> 1x1 Conv -> GELU -> 1x1 Conv
#       + residual (DropPath)

#     Args:
#         dim (int): Number of input channels.
#         drop_path (float): Stochastic depth rate.
#         layer_scale_init_value (float): Init value for Layer Scale (per-channel).
#         mlp_ratio (float): Expansion ratio for the pointwise MLP (1x1 convs).
#         kernel_size (int | tuple[int,int]): Depthwise kernel (kh, kw). Default 7.
#         dilation (int | tuple[int,int]): Depthwise dilation (dh, dw). Default 1.
#         norm (str): Normalization reparam for the depthwise conv weights (e.g., "weight_norm").
#         norm_params (dict): Extra args for the normalization reparam.
#         causal (bool): Causal along W (time). Height is symmetric.
#         pad_mode (str): Padding mode for F.pad (e.g., "reflect", "replicate", "constant").
#     """

#     def __init__(
#         self,
#         dim: int,
#         drop_path: float = 0.1,
#         layer_scale_init_value: float = 1e-6,
#         mlp_ratio: float = 4.0,
#         kernel_size: int | tuple[int, int] = 7,
#         dilation: int | tuple[int, int] = 1,
#         norm: str = "none",
#         norm_params: tp.Dict[str, tp.Any] = {},
#         block_norm: str = "batch_norm", # or "rms_norm"
#         causal: bool = False,
#         pad_mode: str = "reflect",
#     ):
#         super().__init__()

#         # Normalize tuple params
#         if isinstance(kernel_size, int):
#             kh = kw = kernel_size
#         else:
#             kh, kw = kernel_size

#         if isinstance(dilation, int):
#             dh = dw = dilation
#         else:
#             dh, dw = dilation

#         # Depthwise streaming conv (2D), causal along W only
#         self.dwconv = StreamingConv2d(
#             in_channels=dim,
#             out_channels=dim,
#             kernel_size=(kh, kw),
#             stride=(1, 1),
#             dilation=(dh, dw),
#             groups=dim,
#             bias=True,
#             causal=causal,
#             norm=norm,
#             norm_kwargs=norm_params,
#             pad_mode=pad_mode,
#         )

#         # BatchNorm2d (parity with your BatchNorm1d in 1D block)
#         self.norm = nn.BatchNorm2d(dim, eps=1e-6) if block_norm == "batch_norm" else ChannelRMSNorm2d(dim)

#         hidden_dim = int(mlp_ratio * dim)
#         # Pointwise MLP with 1x1 Conv2d (channels-last Linear equivalent)
#         self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=True)
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True)

#         # Layer scale (per-channel, broadcast to H and W)
#         self.gamma = (
#             nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True)
#             if layer_scale_init_value > 0
#             else None
#         )

#         self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
#         self.add = StreamingAdd()  # should already support broadcasting for NCHW

#     def forward(self, x: torch.Tensor, apply_residual: bool = True) -> torch.Tensor:
#         """
#         x: [B, C, T] or [B, C, H, T]
#         Returns same rank as input.
#         """
#         identity = x

#         # Depthwise streaming conv2d (handles [B,C,T] internally by adding H=1)
#         x = self.dwconv(x)
#         # If input was [B,C,T], dwconv returns [B,C,T]; BatchNorm2d expects NCHW,
#         # but StreamingConv2d ensures NCHW inside and squeezes back. So we re-ensure NCHW for BN:
#         if x.dim() == 3:
#             B, C, T = x.shape
#             x_bn = x.view(B, C, 1, T)
#             x_bn = self.norm(x_bn)
#             x = x_bn.view(B, C, T)
#         else:
#             x = self.norm(x)

#         # Pointwise MLP path (1x1 convs) — operate NCHW
#         if x.dim() == 3:
#             # lift to NCHW for pointwise convs
#             B, C, T = x.shape
#             x_nchw = x.view(B, C, 1, T)
#             x_nchw = self.pwconv1(x_nchw)
#             x_nchw = self.act(x_nchw)
#             x_nchw = self.pwconv2(x_nchw)
#             # layer scale
#             if self.gamma is not None:
#                 x_nchw = self.gamma * x_nchw
#             # drop path
#             x_nchw = self.drop_path(x_nchw)
#             # back to [B,C,T]
#             x = x_nchw.view(B, C, T)
#         else:
#             # already NCHW
#             x = self.pwconv1(x)
#             x = self.act(x)
#             x = self.pwconv2(x)
#             if self.gamma is not None:
#                 x = self.gamma * x
#             x = self.drop_path(x)

#         if apply_residual:
#             x = self.add(identity, x)

#         return x



class InterpDWConv1d(StreamingContainer):
    def __init__(
        self, 
        in_ch, 
        out_ch, 
        r,
        kernel_size: int,
        causal=True, 
        norm="none", 
        norm_params={},
        pad_mode="constant",
    ):
        super().__init__()
        self.r = r
        self.dw = StreamingConv1d(
            in_ch,
            in_ch,
            kernel_size,
            norm=norm,
            norm_kwargs=norm_params,
            causal=causal,
            pad_mode=pad_mode,
        )
        self.pw = nn.Conv1d(in_ch, out_ch, 1)
        self.act = nn.GELU()
        self.causal = causal

    def forward(self, x):
        x = torch.repeat_interleave(x, self.r, dim=-1)  # nearest
        x = self.dw(x)
        x = self.act(x)
        return self.pw(x)




# class ConvNeXtWavDecoder(StreamingContainer):
#     """ConvNeXt decoder.

#     Args:
#         channels (int): Audio channels.
#         dimension (int): Intermediate representation dimension.
#         n_filters (int): Base width for the model.
#         n_residual_layers (int): nb of residual layers.
#         ratios (Sequence[int]): kernel size and stride ratios.
#         activation (str): Activation function.
#         activation_params (dict): Parameters to provide to the activation function.
#         final_activation (str): Final activation function after all convolutions.
#         final_activation_params (dict): Parameters to provide to the activation function.
#         norm (str): Normalization method.
#         norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
#         kernel_size (int): Kernel size for the initial convolution.
#         last_kernel_size (int): Kernel size for the initial convolution.
#         residual_kernel_size (int): Kernel size for the residual layers.
#         dilation_base (int): How much to increase the dilation with each layer.
#         causal (bool): Whether to use fully causal convolution.
#         pad_mode (str): Padding mode for the convolutions.
#         true_skip (bool): Whether to use true skip connection or a simple.
#             (streamable) convolution as the skip connection in the residual network blocks.
#         compress (int): Reduced dimensionality in residual branches (from Demucs v3).
#         disable_norm_outer_blocks (int): Number of blocks for which we don't apply norm.
#             For the decoder, it corresponds to the N last blocks.
#         trim_right_ratio (float): Ratio for trimming at the right of the transposed convolution under the causal setup.
#             If equal to 1.0, it means that all the trimming is done at the right.
#     """

#     def __init__(
#         self,
#         channels: int = 1,
#         dimension: int = 128,
#         n_filters: int = 32,
#         n_residual_layers: int = 3,
#         ratios: tp.List[int] = [8, 5, 4, 2],
#         activation: str = "ELU",
#         activation_params: dict = {"alpha": 1.0},
#         final_activation: tp.Optional[str] = None,
#         final_activation_params: tp.Optional[dict] = None,
#         norm: str = "none",
#         norm_params: tp.Dict[str, tp.Any] = {},
#         kernel_size: int = 7,
#         last_kernel_size: int = 7,
#         residual_kernel_size: int = 3,
#         dilation_base: int = 2,
#         causal: bool = False,
#         pad_mode: str = "reflect",
#         true_skip: bool = True,
#         compress: int = 2,
#         disable_norm_outer_blocks: int = 0,
#         trim_right_ratio: float = 1.0,
#         device=None,
#         dtype=None,
#     ):
#         super().__init__()
#         self.dimension = dimension
#         self.channels = channels
#         self.n_filters = n_filters
#         self.ratios = ratios
#         del ratios
#         self.n_residual_layers = n_residual_layers
#         self.hop_length = int(np.prod(self.ratios))
#         self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
#         self.disable_norm_outer_blocks = disable_norm_outer_blocks
#         assert (
#             self.disable_norm_outer_blocks >= 0 and self.disable_norm_outer_blocks <= self.n_blocks
#         ), (
#             "Number of blocks for which to disable norm is invalid."
#             "It should be lower or equal to the actual number of blocks in the network and greater or equal to 0."
#         )

#         self.device = device
#         self.dtype = dtype

#         self.frame_exciter = FrameRateExciter(
#             sr=16000,
#             hop=self.hop_length,
#             n_harm=6,
#             noise_color="lp1p",
#             lp_cutoff_hz=7000.0,
#             f0_voicing_hz=20.0,
#             device=device,
#             dtype=dtype,
#         )
#         self.excite_fuse = ExciteFuse(
#             d_hidden=dimension,
#             n_harm=6
#         )

#         act = getattr(nn, activation)
#         mult = int(2 ** len(self.ratios))
#         model: tp.List[nn.Module] = [
#             StreamingConv1d(
#                 dimension,
#                 mult * n_filters,
#                 kernel_size,
#                 norm=(
#                     "none" if self.disable_norm_outer_blocks == self.n_blocks else norm
#                 ),
#                 norm_kwargs=norm_params,
#                 causal=causal,
#                 pad_mode=pad_mode,
#             )
#         ]

#         # Upsample to raw audio scale
#         for i, ratio in enumerate(self.ratios):
#             block_norm = (
#                 "none"
#                 if self.disable_norm_outer_blocks >= self.n_blocks - (i + 1)
#                 else norm
#             )
#             # Add upsampling layers
#             model += [
#                 act(**activation_params),
#                 InterpDWConv1d(
#                     mult * n_filters,
#                     mult * n_filters // 2,
#                     r=ratio,
#                     kernel_size=ratio * 2 + 1,
#                     causal=causal,
#                     norm=block_norm,
#                     norm_params=norm_params,
#                     pad_mode=pad_mode
#                 ),
#             ]
#             # Add residual layers
#             for j in range(n_residual_layers):
#                 model += [
#                     ConvNeXtBlock(
#                         dim = mult * n_filters // 2,
#                         drop_path = 0.0,
#                         mlp_ratio = 4.0,
#                         kernel_size = residual_kernel_size,
#                         dilation = dilation_base**j,
#                         norm=block_norm,
#                         norm_params=norm_params,
#                         causal=causal,
#                         pad_mode=pad_mode,
#                     ),
#                     ConvNeXtBlock(
#                         dim = mult * n_filters // 2,
#                         drop_path = 0.0,
#                         mlp_ratio = 4.0,
#                         kernel_size = residual_kernel_size,
#                         dilation = 1,
#                         norm=block_norm,
#                         norm_params=norm_params,
#                         causal=causal,
#                         pad_mode=pad_mode,
#                     )
#                 ]

#             mult //= 2

#         # Add final layers
#         model += [
#             act(**activation_params),
#             StreamingConv1d(
#                 n_filters,
#                 channels,
#                 last_kernel_size,
#                 norm="none" if self.disable_norm_outer_blocks >= 1 else norm,
#                 norm_kwargs=norm_params,
#                 causal=causal,
#                 pad_mode=pad_mode,
#             ),
#         ]
#         # Add optional final activation to decoder (eg. tanh)
#         if final_activation is not None:
#             final_act = getattr(nn, final_activation)
#             final_activation_params = final_activation_params or {}
#             model += [final_act(**final_activation_params)]
#         self.model = nn.Sequential(*model)

#     @torch_compile_lazy
#     def forward(self, z, f0_hz, A_p, A_ap):
#         exc_feats = self.frame_exciter(f0_hz, A_p, A_ap)
#         # print( "exc_feats", exc_feats.shape, "z", z.shape )
#         z = self.excite_fuse(z, exc_feats)

#         y = self.model(z)
#         return y