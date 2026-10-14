import typing as tp

import torch
from torch import nn
from .streaming import StreamingContainer
from ..utils.compile import torch_compile_lazy
from .transformer import StreamingTransformer
from .conv import StreamingConv1d
from .convnext import ConvNeXtBlock

class CondLNWithFusion(nn.Module):
    def __init__(self, c_in=512, c_cond=192, out_channels=512, eps=1e-5, zero_init=True):
        super().__init__()
        self.norm = nn.GroupNorm(1, c_in, eps=eps, affine=False)          # LN over channels per time step
        self.to_affine = nn.Conv1d(c_cond, 2 * c_in, kernel_size=1)       # -> γ, β
        self.gate = nn.Conv1d(c_cond, 1, kernel_size=1)                   # scalar gate per time step
        self.proj = nn.Conv1d(c_in + c_cond, out_channels, kernel_size=1) # mixes [y, gated cond]

        if zero_init:
            nn.init.zeros_(self.to_affine.weight); nn.init.zeros_(self.to_affine.bias)
            nn.init.zeros_(self.gate.weight);      nn.init.zeros_(self.gate.bias)

    def forward(self, x, cond):
        # x:    (B,256,T), cond: (B,192,T)
        # print(x.shape, cond.shape)

        B, C, T = x.shape
        assert cond.shape[0] == B and cond.shape[2] == T

        x_n = self.norm(x.float()).to(x.dtype)
        gamma, beta = torch.chunk(self.to_affine(cond), 2, dim=1)  # (B,512,T) each
        y = (1.0 + gamma) * x_n + beta                            # (B,512,T)

        g = torch.sigmoid(self.gate(cond))                        # (B,1,T)
        cond_norm = nn.functional.group_norm(cond, 1, eps=1e-5)   # stabilize scale
        fused = torch.cat([y, g * cond_norm], dim=1)              # (B,448,T)

        return self.proj(fused) 


class VariancePredictor(StreamingContainer):
    def __init__(
        self,
        in_channels, 
        out_channels, 
        kernel_size,
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        dropout: float = 0.0,
        norm: str = "none",
        norm_params: tp.Dict[str, tp.Any] = {},
        causal: bool = False,
        pad_mode: str = "reflect",
    ):
        super().__init__()
        act = getattr(nn, activation)

        self.model = nn.Sequential(
            StreamingConv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
            act(**activation_params),
            nn.Dropout(dropout),
            nn.BatchNorm1d(out_channels),
            StreamingConv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
            act(**activation_params),
            nn.Dropout(dropout),
            nn.BatchNorm1d(out_channels),
            StreamingConv1d(
                out_channels,
                1,
                kernel_size=1,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    

class PitchEnergyPredictor(StreamingContainer):
    def __init__(
        self,
        in_channels: int,
        v_channels: int,
        var_kernel_size: int = 3,
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        dropout=0.0,
        norm: str = "none",
        norm_params: tp.Dict[str, tp.Any] = {},
        causal: bool = False,
        pad_mode: str = "reflect",
    ):
        super().__init__()
        act = getattr(nn, activation)

        self.pitchPred = VariancePredictor(
            in_channels=in_channels,
            out_channels=v_channels,
            kernel_size=var_kernel_size,
            activation=activation,
            activation_params=activation_params,
            dropout=dropout,
            norm=norm,
            norm_params=norm_params,
            causal=causal,
            pad_mode=pad_mode,
        )
        
        self.energyPred = VariancePredictor(
            in_channels=in_channels,
            out_channels=v_channels,
            kernel_size=var_kernel_size,
            activation=activation,
            activation_params=activation_params,
            dropout=dropout,
            norm=norm,
            norm_params=norm_params,
            causal=causal,
            pad_mode=pad_mode,
        )

    def _average_over_durations(self, values, durs):
        """Average values over durations.
        Arguments
        ---------
        values: torch.Tensor
            shape: [B, 1, T_de]
        durs: torch.Tensor
            shape: [B, T_en]
        Returns
        ---------
        avg: torch.Tensor
            shape: [B, 1, T_en]
        """
        avg_batch = []
        for i, (d, v) in enumerate(zip(durs, values)):
            v = v.unsqueeze(0)
            durs_cums_ends = torch.cumsum(d.unsqueeze(0), dim=1).long()
            durs_cums_starts = torch.nn.functional.pad(durs_cums_ends[:, :-1], (1, 0))
            values_nonzero_cums = torch.nn.functional.pad(
                torch.cumsum(v != 0.0, dim=2), (1, 0)
            )
            values_cums = torch.nn.functional.pad(torch.cumsum(v, dim=2), (1, 0))

            bs, length = durs_cums_ends.size()
            n_formants = v.size(1)
            dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, length)
            dce = durs_cums_ends[:, None, :].expand(bs, n_formants, length)

            values_sums = (
                torch.gather(values_cums, 2, dce) - torch.gather(values_cums, 2, dcs)
            ).float()
            values_nelems = (
                torch.gather(values_nonzero_cums, 2, dce)
                - torch.gather(values_nonzero_cums, 2, dcs)
            ).float()

            avg = torch.where(
                values_nelems == 0.0, values_nelems, values_sums / values_nelems
            )
            avg_batch.append(avg.squeeze(0).repeat_interleave(d, dim=-1))
        return torch.stack(avg_batch, dim=0)


    def _get_durations(self, tokens):
        durations = [
            torch.unique_consecutive(tokens[i], return_counts=True)[1]
            for i in range(len(tokens))
        ]
        return durations

    @torch_compile_lazy
    def forward(
        self,
        x: torch.Tensor,
        pitch: torch.Tensor=None, 
        energy: torch.Tensor=None, 
        tokens: torch.Tensor=None,
        pitch_rate: float = 1.0,
        energy_rate: float = 1.0,
    ):
        """Forward pass of the speaker variance adapter.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, T].
            embeds (torch.Tensor): Speaker embedding of shape [B, speaker_dim].
            pitch (torch.Tensor): Pitch tensor of shape [B, 1, T].
            energy (torch.Tensor): Energy tensor of shape [B, 1, T].
            tokens (torch.Tensor): Token tensor of shape [B, T].
            return_variance (bool): Whether to return the variance predictions.

        Returns:
            torch.Tensor: Output tensor of shape [B, C, T].
        """
        # Get pitch and energy predictions
        predicted_pitch = self.pitchPred(x) * pitch_rate
        predicted_energy = self.energyPred(x) * energy_rate

        avg_pitch, avg_energy = None, None
        # Apply pitch and energy prediction
        if pitch is not None and energy is not None:
            if tokens is not None:
                durs = self._get_durations(tokens)
                # Get the average over durations
                avg_pitch = self._average_over_durations(pitch.unsqueeze(1), durs)
                avg_energy = self._average_over_durations(energy.unsqueeze(1), durs)
            else:
                avg_pitch, avg_energy = pitch, energy

        return predicted_pitch, avg_pitch, predicted_energy, avg_energy

class FrameSynthesizerWOExciter(StreamingContainer):
    """Transformer with optional projections of the input and output to different dimensions when needed.
    Supports multiple outputs.

    Args:
        input_dimension (int): dimension of the input.
        output_dimensions (tuple[int]): dimensions of the outputs.
        d_model (int): inner dimension of the Transformer.
        conv_layout (bool): If True, expects `[B, C, T]` shaped tensors, otherwise, `[B, T, C]`.
            Similarly, the output will have the same layout.
    """

    def __init__(
        self,
        input_dimension: int = 512,
        output_dimension: int = 512,
        timbre_cond_dim: int = 192,
        timbre_dimension: int = 512,
        d_model: int = 512,
        num_heads: int = 8,
        b1_layers: int = 3,
        b2_layers: int = 3,
        b3_layers: int = 2,
        causal: bool = True,
        context: int = 250,
        right_context: int = 0,
        conv_layout: bool = True,
        max_period: int = 10000,
        positional_embedding: str = "rope",
        dim_feedforward: int = 2048,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()

        self.timbre_cond_dim = timbre_cond_dim
        self.b1_transformer = StreamingTransformer(
            d_model=d_model, 
            num_heads=num_heads,
            num_layers=b1_layers,
            dim_feedforward=dim_feedforward,
            causal=causal,
            context=context,
            right_context=right_context,
            positional_embedding=positional_embedding,
            max_period=max_period,
            device=device,
            dtype=dtype,
            **kwargs
        )
        self.b1_ln = CondLNWithFusion(c_in=d_model, c_cond=timbre_cond_dim, out_channels=d_model, zero_init=True)
        self.b2_transformer = StreamingTransformer(
            d_model=d_model, 
            num_heads=num_heads,
            num_layers=b2_layers,
            dim_feedforward=dim_feedforward,
            causal=causal,
            context=context,
            right_context=right_context,
            positional_embedding=positional_embedding,
            max_period=max_period,
            device=device,
            dtype=dtype,
            **kwargs
        )
        self.b2_ln = CondLNWithFusion(c_in=d_model, c_cond=timbre_cond_dim, out_channels=d_model, zero_init=True)
        self.b3_transformer = StreamingTransformer(
            d_model=d_model, 
            num_heads=num_heads,
            num_layers=b3_layers,
            dim_feedforward=dim_feedforward,
            causal=causal,
            context=context,
            right_context=right_context,
            positional_embedding=positional_embedding,
            max_period=max_period,
            device=device,
            dtype=dtype,
            **kwargs
        )
        self.b3_ln = CondLNWithFusion(c_in=d_model, c_cond=timbre_cond_dim, out_channels=d_model, zero_init=True)

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.conv_layout = conv_layout
        self.input_proj = None
        if d_model != input_dimension:
            self.input_proj = nn.Linear(input_dimension, d_model, bias=False)

        if d_model == output_dimension:
            self.output_proj = nn.Identity()
        else:
            self.output_proj = nn.Linear(d_model, output_dimension, bias=False)

    def forward(
            self, 
            x: torch.Tensor, 
            timbre: torch.Tensor,
            *args, 
            **kwargs
        ):

        B, _, T = x.shape
        if self.conv_layout:
            x = x.transpose(1, 2)
        if self.input_proj is not None:
            x = self.input_proj(x)
        z1 = self.b1_transformer(x, *args, **kwargs)
        z1 = self.b1_ln(z1.transpose(1, 2), timbre)

        z2 = self.b2_transformer(z1.transpose(1, 2), *args, **kwargs)
        z = self.b2_ln(z2.transpose(1, 2), timbre)

        z3 = self.b3_transformer(z.transpose(1, 2), *args, **kwargs)
        z = self.b3_ln(z3.transpose(1, 2), timbre)

        y = self.output_proj(z.transpose(1, 2)).transpose(1, 2)
        
        return y, None


class ConvNeXtHead(StreamingContainer):
    def __init__(
        self,
        input_channels: int = 512,
        timbre_channels: int = 192,
        dim: int = 512,
        mlp_ratio: float = 3.0,
        kernel_size: int = 7,
        dilation: int = 1,
        norm: str = "none",
        norm_params: tp.Dict[str, tp.Any] = {},
        causal: bool = False,
        pad_mode: str = "reflect",
        b1_layers: int = 3,
        b2_layers: int = 2,
        layer_scale_init_value: float = 1e-6
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = StreamingConv1d(
                input_channels,
                dim,
                kernel_size=kernel_size,
                dilation=dilation,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )
        self.norm = nn.BatchNorm1d(dim, eps=1e-6) #nn.LayerNorm(dim, eps=1e-6)

        layer_scale_init_value = layer_scale_init_value or 1 / (b1_layers + b2_layers)
        self.b1_blocks = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    mlp_ratio=mlp_ratio,
                    layer_scale_init_value=layer_scale_init_value,
                    kernel_size=kernel_size,
                    norm=norm,
                    norm_params=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                )
                for _ in range(b1_layers)
            ]
        )
        self.cln = CondLNWithFusion(c_in=dim, c_cond=timbre_channels, out_channels=dim, zero_init=True)
        self.b2_blocks = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    mlp_ratio=mlp_ratio,
                    layer_scale_init_value=layer_scale_init_value,
                    kernel_size=kernel_size,
                    norm=norm,
                    norm_params=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                )
                for _ in range(b2_layers)
            ]
        )
        self.final_layer_norm = nn.BatchNorm1d(dim, eps=1e-6) #nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, timbre: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.norm(x)
        for conv_block in self.b1_blocks:
            x = conv_block(x)
        x = self.cln(x, timbre)
        for conv_block in self.b2_blocks:
            x = conv_block(x)
        x = self.final_layer_norm(x)
        return x