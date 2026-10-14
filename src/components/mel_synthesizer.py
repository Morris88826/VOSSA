import typing as tp

import torch
from torch import nn
from torch.nn import functional as F

from .streaming import StreamingContainer, StreamingAdd
from ..utils.compile import torch_compile_lazy
from .transformer import StreamingTransformer
from .conv import StreamingConv1d
from .convnext import ConvNeXtBlock, ChannelRMSNorm1d
from .tv_timbre import StreamingGlobalOnlyTVTimbre
from .frame_synthesizer import CondLNWithFusion, PitchEnergyPredictor

class PostNet(StreamingContainer):
    def __init__(
        self,
        input_channels: int = 80,
        dim: int = 256,
        mlp_ratio: float = 3.0,
        kernel_size: int = 7,
        dilation: int = 1,
        norm: str = "weight_norm",
        norm_params: tp.Dict[str, tp.Any] = {},
        causal: bool = True,
        pad_mode: str = "constant",
        layers: int = 3,
        smoothing_layer: bool = False,
        layer_scale_init_value: float = 1e-6
    ):
        super().__init__()
        self.smoothing_layer = smoothing_layer
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
        self.norm = ChannelRMSNorm1d(dim) #nn.BatchNorm1d(dim, eps=1e-6) #nn.LayerNorm(dim, eps=1e-6)

        layer_scale_init_value = layer_scale_init_value or 1 / (layers)
        self.blocks = nn.ModuleList(
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
                for _ in range(layers)
            ]
        )
        self.final_norm = ChannelRMSNorm1d(dim) #nn.BatchNorm1d(dim, eps=1e-6) #nn.LayerNorm(dim, eps=1e-6)
        self.output_proj = StreamingConv1d(
                dim,
                input_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                norm="none",
                causal=causal,
                pad_mode=pad_mode,
            )

        if self.smoothing_layer:
            self.smoothing = StreamingConv1d(
                input_channels,
                input_channels,
                kernel_size=13,
                dilation=1,
                norm="none",
                causal=causal,
                pad_mode=pad_mode,
            )
        

        self.add = StreamingAdd()

        self.apply(self._init_weights)

        nn.init.constant_(self.output_proj.conv.conv.weight, 0.0)
        if self.output_proj.conv.conv.bias is not None:
            nn.init.constant_(self.output_proj.conv.conv.bias, 0.0)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mel_in = x
        x = self.embed(x)
        x = self.norm(x)
        for conv_block in self.blocks:
            x = conv_block(x)
        x = self.final_norm(x)
        delta = self.output_proj(x)
        mel_out = self.add(mel_in, delta)

        if self.smoothing_layer:
            mel_out = self.smoothing(mel_out)
        return mel_out

class MelSynthesizer(StreamingContainer):
    """Three-stage streaming Transformer that maps content features to mel features,
    conditioned on a precomputed timbre embedding via conditional LayerNorm after the
    first two stages: b1 -> CondLN -> b2 -> CondLN -> b3.

    Args:
        input_dimension (int): dimension of the input content features.
        output_dimension (int): dimension of the output features.
        timbre_cond_dim (int): channel dimension of the `timbre` conditioning tensor.
        timbre_dimension (int): unused; accepted for config compatibility.
        d_model (int): inner dimension of the Transformer.
        b1_layers, b2_layers, b3_layers (int): number of layers in each stage.
        conv_layout (bool): If True, expects `[B, C, T]` shaped tensors, otherwise, `[B, T, C]`.
            The output is always `[B, C, T]`.

    Forward returns `(y, None)`.
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

    def _first_stage(self, x: torch.Tensor, timbre: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Input projection + b1 + CondLN. Returns `[B, d_model, T]`."""
        if self.conv_layout:
            x = x.transpose(1, 2)
        if self.input_proj is not None:
            x = self.input_proj(x)
        z1 = self.b1_transformer(x, *args, **kwargs)
        return self.b1_ln(z1.transpose(1, 2), timbre)

    def _remaining_stages(self, z1: torch.Tensor, timbre: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """b2 + CondLN + b3 + output projection. Takes and returns `[B, C, T]`."""
        z2 = self.b2_transformer(z1.transpose(1, 2), *args, **kwargs)
        z = self.b2_ln(z2.transpose(1, 2), timbre)

        z = self.b3_transformer(z.transpose(1, 2), *args, **kwargs)
        return self.output_proj(z).transpose(1, 2)

    def forward(
            self, 
            x: torch.Tensor, 
            timbre: torch.Tensor,
            *args, 
            **kwargs
        ):
        z1 = self._first_stage(x, timbre, *args, **kwargs)
        y = self._remaining_stages(z1, timbre, *args, **kwargs)
        return y, None

class ProsodyMelSynthesizer(MelSynthesizer):
    """`MelSynthesizer` with a pitch/energy predictor between b1 and b2. Pitch and
    energy are embedded and added to the b1 output: the ground-truth averages when both
    `pitch` and `energy` are given (training), otherwise the predictions (inference).

    Takes the same constructor arguments as `MelSynthesizer`.

    Forward returns `(y, (predicted_pitch, avg_pitch, predicted_energy, avg_energy))`.
    """

    def __init__(self, d_model: int = 512, causal: bool = True, **kwargs):
        super().__init__(d_model=d_model, causal=causal, **kwargs)

        self.pitch_energy_predictor = PitchEnergyPredictor(
            in_channels=d_model,
            v_channels=d_model,
            var_kernel_size=7,
            activation="ELU",
            activation_params={"alpha": 1.0},
            dropout=0.1,
            norm="weight_norm",
            norm_params={},
            causal=causal,
            pad_mode="constant",
        )
        self.pitchEmbed = StreamingConv1d(
            1,
            d_model,
            kernel_size=7,
            norm="weight_norm",
            norm_kwargs={},
            causal=causal,
            pad_mode="constant",
        )

        self.energyEmbed = StreamingConv1d(
            1,
            d_model,
            kernel_size=7,
            norm="weight_norm",
            norm_kwargs={},
            causal=causal,
            pad_mode="constant",
        )

        self.pitch_add = StreamingAdd()
        self.energy_add = StreamingAdd()

    def forward(
            self, 
            x: torch.Tensor, 
            timbre: torch.Tensor,
            pitch: torch.Tensor=None, 
            energy: torch.Tensor=None, 
            tokens: torch.Tensor=None,
            pitch_rate: float = 1.0,
            energy_rate: float = 1.0,
            *args, 
            **kwargs
        ):
        z1 = self._first_stage(x, timbre, *args, **kwargs)

        predicted_pitch, avg_pitch, predicted_energy, avg_energy = self.pitch_energy_predictor(
            z1, pitch, energy, tokens, pitch_rate, energy_rate
        )

        if pitch is not None and energy is not None:
            pitch = self.pitchEmbed(avg_pitch)
            energy = self.energyEmbed(avg_energy)
        else:
            pitch = self.pitchEmbed(predicted_pitch)
            energy = self.energyEmbed(predicted_energy)

        # Add pitch and energy to the input
        z1 = self.pitch_add(z1, pitch)
        z1 = self.energy_add(z1, energy)

        y = self._remaining_stages(z1, timbre, *args, **kwargs)
        return y, (predicted_pitch, avg_pitch, predicted_energy, avg_energy)

class MelSynthesizerWithTimbreEncoder(StreamingContainer):
    """All-in-one variant of `ProsodyMelSynthesizer` that encodes timbre internally from a
    global speaker embedding, with optional language embedding and discrete-code input.

    Args:
        input_dimension (int): dimension of the input content features.
        output_dimension (int): dimension of the output features.
        timbre_dimension (int): dimension of the global speaker embedding passed as `timbre`.
        timbre_cond_dim (int): channel dimension of the encoded timbre used for conditioning.
        d_model (int): inner dimension of the Transformer.
        b1_layers, b2_layers, b3_layers (int): number of layers in each stage.
        conv_layout (bool): If True, expects `[B, C, T]` shaped tensors, otherwise, `[B, T, C]`.
            The output is always `[B, C, T]`.
        use_tv_timbre (bool): If True, encode timbre with `StreamingGlobalOnlyTVTimbre`
            (content-dependent); otherwise project the normalized global embedding and
            broadcast it over time.
        use_lang_emb (bool): If True, fuse a language embedding (from `lang_ids`) into the input.
        input_type (str): "latents" for continuous input, "code" for codebook indices.
        use_pitch_energy_predictor (bool): If True, add a pitch/energy predictor between b1 and b2.

    Forward returns `(y, (predicted_pitch, avg_pitch, predicted_energy, avg_energy))` when
    the predictor is enabled, otherwise `(y, None)`.
    """

    def __init__(
        self,
        input_dimension: int = 512,
        output_dimension: int = 512,
        timbre_dimension: int = 704,
        timbre_cond_dim: int = 192,
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
        use_tv_timbre: bool = True,
        use_lang_emb: bool = False,
        num_languages: int = 12,
        input_type: str = "latents",
        num_codes: int = 1024,
        use_pitch_energy_predictor: bool = True,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()

        self.use_tv_timbre = use_tv_timbre
        self.timbre_cond_dim = timbre_cond_dim
        self.use_lang_emb = use_lang_emb
        self.input_type = input_type
        self.use_pitch_energy_predictor = use_pitch_energy_predictor

        if input_type == "code":
            self.input_embbedder = nn.Embedding(num_codes, input_dimension)  # Example: 1024 codebook size

        if use_tv_timbre:
            self.timbre_encoder = StreamingGlobalOnlyTVTimbre(
                content_dim=input_dimension,
                g_dim=timbre_dimension,
                d_out=timbre_cond_dim,
            )
        else:
            self.timbre_encoder = nn.Conv1d(
                timbre_dimension, timbre_cond_dim, kernel_size=1
            )

        if self.use_lang_emb:
            self.lang_emb = nn.Embedding(num_languages, input_dimension)  # Example: 10 languages
            self.input_lng_emb_proj = nn.Conv1d(2*input_dimension, input_dimension, kernel_size=1)


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

        if self.use_pitch_energy_predictor:
            self.pitch_energy_predictor = PitchEnergyPredictor(
                in_channels=d_model,
                v_channels=d_model,
                var_kernel_size=7,
                activation="ELU",
                activation_params={"alpha": 1.0},
                dropout=0.1,
                norm="weight_norm",
                norm_params={},
                causal=causal,
                pad_mode="constant",
            )
            self.pitchEmbed = StreamingConv1d(
                1,
                d_model,
                kernel_size=7,
                norm="weight_norm",
                norm_kwargs={},
                causal=causal,
                pad_mode="constant",
            )

            self.energyEmbed = StreamingConv1d(
                1,
                d_model,
                kernel_size=7,
                norm="weight_norm",
                norm_kwargs={},
                causal=causal,
                pad_mode="constant",
            )

            self.pitch_add = StreamingAdd()
            self.energy_add = StreamingAdd()

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
            lang_ids: torch.Tensor=None,
            pitch: torch.Tensor=None, 
            energy: torch.Tensor=None, 
            tokens: torch.Tensor=None,
            pitch_rate: float = 1.0,
            energy_rate: float = 1.0,
            *args, 
            **kwargs
        ):

        B, _, T = x.shape

        if self.input_type == "code":
            if x.dim() == 3:
                x = x.squeeze(1)
            x = self.input_embbedder(x.long()).transpose(1, 2)  # Assuming x is [B, T] of token indices

        if self.use_lang_emb and lang_ids is not None:
            lang_embedding = self.lang_emb(lang_ids).unsqueeze(-1).expand(-1, -1, T)  # [B, C, T]
            x = torch.cat([x, lang_embedding], dim=1)  # Concatenate on channel dimension
            x = self.input_lng_emb_proj(x)

        if self.use_tv_timbre:
            timbre = self.timbre_encoder(x, timbre)
        else:
            timbre = self.timbre_encoder(F.normalize(timbre, dim=-1).unsqueeze(-1)).expand(B, self.timbre_cond_dim, T)
        
        if self.conv_layout:
            x = x.transpose(1, 2)
        if self.input_proj is not None:
            x = self.input_proj(x)
        z1 = self.b1_transformer(x, *args, **kwargs)
        z1 = self.b1_ln(z1.transpose(1, 2), timbre)


        if self.use_pitch_energy_predictor:
            # Pitch/Energy Prediction
            predicted_pitch, avg_pitch, predicted_energy, avg_energy = self.pitch_energy_predictor(
                z1, pitch, energy, tokens, pitch_rate, energy_rate
            )

            if pitch is not None and energy is not None:
                pitch = self.pitchEmbed(avg_pitch)
                energy = self.energyEmbed(avg_energy)
            else:
                pitch = self.pitchEmbed(predicted_pitch)
                energy = self.energyEmbed(predicted_energy)

            # Add pitch and energy to the input
            z1 = self.pitch_add(z1, pitch)
            z1 = self.energy_add(z1, energy)


        z2 = self.b2_transformer(z1.transpose(1, 2), *args, **kwargs)
        z = self.b2_ln(z2.transpose(1, 2), timbre)

        z = self.b3_transformer(z.transpose(1, 2), *args, **kwargs)
        y = self.output_proj(z).transpose(1, 2)

        if self.use_pitch_energy_predictor:
            return y, (predicted_pitch, avg_pitch, predicted_energy, avg_energy)

        return y, None