"""Stateless streaming step for VOSSA.

`VossaStreamingStep` turns a trained `VOSSASynthesizer` into one pure function

    (chunk, spk_k, spk_v, spk_g, *state) -> (audio, ready, *next_state)

Every cache (conv tails, conv-transpose overlap-add tails, attention KV caches, RoPE
offsets, latent history) is an explicit tensor that is passed in and returned, so the
modules themselves hold no streaming state. The step only uses fixed shapes and no
data-dependent Python control flow, so it can be CUDA Graphed (and traced for export).

Timing: frames are `hop_length` = 320 samples (50 Hz). One chunk is `chunk_frames`
latent frames. The encoder transformer looks `lookahead` frames into the future, so the
output runs `lag_chunks = ceil(lookahead / chunk_frames)` chunks behind the input. Until
enough frames have been seen, `ready` is False, the output is zeros and only the
CNN-encoder caches advance; every later stage keeps its initial state.
"""

from dataclasses import dataclass
import math
import typing as tp

from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as F

from src.components.conv import StreamingConv1d, StreamingConvTranspose1d
from src.components.convnext import ConvNeXtBlock
from src.components.mel_synthesizer import MelSynthesizer, PostNet
from src.components.seanet import SEANetDecoder, SEANetEncoder, SEANetResnetBlock
from src.components.streaming import StreamingAdd
from src.components.transformer import (
    ProjectedTransformer,
    StreamingMultiheadAttention,
    StreamingTransformer,
    StreamingTransformerLayer,
)
from src.components.tv_timbre import StreamingGlobalOnlyTVTimbre, rope_apply, slerp

State = tp.Dict[str, torch.Tensor]

# Modules handled explicitly below. Anything else reached while walking a network must
# not contain any of these, otherwise its streaming state would silently be dropped.
_STATEFUL = (StreamingConv1d, StreamingConvTranspose1d, StreamingMultiheadAttention, StreamingAdd)


@dataclass(frozen=True)
class StateSpec:
    name: str
    shape: tp.Tuple[int, ...]  # without the batch dimension
    dtype: torch.dtype
    batched: bool = True


class VossaStreamingStep(nn.Module):
    """Pure streaming step around a trained `VOSSASynthesizer` (in eval mode).

    Args:
        model: the synthesizer. Its parameters are used as is, not copied.
        chunk_frames: latent frames per step (5 frames = 1600 samples = 100 ms).
    """

    def __init__(self, model: nn.Module, chunk_frames: int = 5, lookahead = None):
        super().__init__()
        self.model = model.eval()
        self.chunk_frames = chunk_frames

        enc = model.encoder.net
        self.hop_length = int(math.prod(enc.encoder.ratios))
        self.chunk_samples = chunk_frames * self.hop_length
        self.lookahead = self._right_context(enc.encoder_transformer) if lookahead is None else lookahead
        self.lag_chunks = -(-self.lookahead // chunk_frames)
        # Latent history: the current chunk sits `lag_chunks` chunks back, so that its
        # `lookahead` future frames are already available.
        self.history_frames = chunk_frames * (self.lag_chunks + 1)
        self.latent_dim = enc.dimension

        self._names = {mod: name for name, mod in model.named_modules()}
        self._always_advance = ("history", "encoded_seen", self._names[enc.encoder] + ".")
        self.state_specs = self._build_state_specs()
        self.state_names = [spec.name for spec in self.state_specs]

    # ------------------------------------------------------------------ state

    @staticmethod
    def _right_context(projected: ProjectedTransformer) -> int:
        return projected.transformer.layers[0].self_attn.right_context or 0

    def _build_state_specs(self) -> tp.List[StateSpec]:
        specs = [
            StateSpec("encoded_seen", (1,), torch.long, batched=False),
            StateSpec("history", (self.latent_dim, self.history_frames), torch.float32),
        ]
        for name, mod in self.model.named_modules():
            if isinstance(mod, StreamingConv1d):
                raw = mod.conv.conv
                assert mod.causal and mod.pad_mode == "constant", (
                    f"{name}: only causal, zero-padded convs can start from a zero tail."
                )
                specs.append(StateSpec(f"{name}.tail", (raw.in_channels, mod._padding_total), torch.float32))
            elif isinstance(mod, StreamingConvTranspose1d):
                raw = mod.convtr.convtr
                assert mod.causal and mod.trim_right_ratio == 1.0, name
                partial = raw.kernel_size[0] - raw.stride[0]
                specs.append(StateSpec(f"{name}.partial", (raw.out_channels, partial), torch.float32))
            elif isinstance(mod, StreamingTransformer):
                specs.append(StateSpec(f"{name}.offset", (1,), torch.long, batched=False))
                for i, layer in enumerate(mod.layers):
                    attn = layer.self_attn
                    assert attn.context is not None and not attn.weights_per_step, name
                    # The first query of a chunk needs `context` keys behind it.
                    capacity = attn.context + self.chunk_frames - 1
                    shape = (attn.num_heads, capacity, attn.embed_dim // attn.num_heads)
                    specs.append(StateSpec(f"{name}.layers.{i}.k", shape, attn.in_proj_weight.dtype))
                    specs.append(StateSpec(f"{name}.layers.{i}.v", shape, attn.in_proj_weight.dtype))
            elif isinstance(mod, StreamingGlobalOnlyTVTimbre):
                specs.append(StateSpec(f"{name}.pos_offset", (1,), torch.long, batched=False))
        return specs

    def initial_state(self, batch_size: int = 1) -> tp.Tuple[torch.Tensor, ...]:
        device = next(self.parameters()).device
        return tuple(
            torch.zeros(((batch_size,) if spec.batched else ()) + spec.shape, dtype=spec.dtype, device=device)
            for spec in self.state_specs
        )

    # ---------------------------------------------------------------- speaker

    @torch.no_grad()
    def prepare_speaker(self, global_embed: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Precompute the speaker-only part of the TV-timbre module once per session.

        Args:
            global_embed: `[B, g_dim]` global speaker embedding from `get_speaker_embeddings`.
        Returns:
            (spk_k, spk_v, spk_g): projected cross-attention keys/values `[B, H, N, d]` and
            the projected global timbre `[B, d_out]`.
        """
        tv = self.model.tvtimbre_encoder
        cross = tv.cross
        B = global_embed.shape[0]
        K_seq, V_seq = tv.gtm(global_embed)
        spk_k = cross._shape(cross.k_proj(K_seq), B)
        spk_v = cross._shape(cross.v_proj(V_seq), B)
        spk_g = tv.speaker_proj(F.normalize(global_embed, dim=-1))
        return spk_k, spk_v, spk_g

    # ------------------------------------------------------------------- step

    def forward(self, chunk: torch.Tensor, spk_k: torch.Tensor, spk_v: torch.Tensor, spk_g: torch.Tensor, *state: torch.Tensor):
        """Args:
            chunk: `[B, 1, chunk_samples]` audio at the model sample rate.
            spk_k, spk_v, spk_g: from `prepare_speaker`.
            *state: tensors ordered as `state_names`.
        Returns:
            (audio `[B, 1, chunk_samples]`, ready `[1]` bool, *next_state).
        """
        assert chunk.shape[-1] == self.chunk_samples, chunk.shape
        st: State = dict(zip(self.state_names, state))
        nx: State = {}
        m = self.model
        enc = m.encoder.net
        T = self.chunk_frames

        # 1. Causal CNN encoder; new latent frames go into the rolling history.
        frames = self._run(enc.encoder, chunk, st, nx)
        history = torch.cat([st["history"], frames], dim=-1)[..., -self.history_frames:]
        nx["history"] = history
        nx["encoded_seen"] = st["encoded_seen"] + T
        ready = nx["encoded_seen"] >= self.history_frames

        # 2. Encoder transformer on chunk + lookahead, then quantize the chunk.
        window = history[..., : T + self.lookahead]
        emb = self._projected_transformer(enc.encoder_transformer, window, st, nx, self.lookahead)
        z = enc.quantizer(emb[..., :T])[0]

        # 3. Time-varying timbre, mel transformer and PostNet.
        tvtimbre = self._tvtimbre(m.tvtimbre_encoder, z, spk_k, spk_v, spk_g, st, nx)
        mel = self._mel_synthesizer(m.decoder.decoder_transformer, z, tvtimbre, st, nx)
        mel = self._postnet(m.decoder.decoder, mel, st, nx)

        # 4. Vocoder: transformer + causal SEANet decoder.
        voc = m.vocoder.decoder
        audio = self._run(voc.decoder, self._projected_transformer(voc.decoder_transformer, mel, st, nx, 0), st, nx)

        # Warm-up gating: until `ready`, hold every cache past the CNN encoder.
        next_state = []
        for name in self.state_names:
            new = nx[name]
            if not name.startswith(self._always_advance):
                new = torch.where(ready, new, st[name])
            next_state.append(new)
        audio = audio * ready.to(audio.dtype)
        return (audio, ready, *next_state)

    # ------------------------------------------------------------ conv stacks

    def _conv(self, mod: StreamingConv1d, x: torch.Tensor, st: State, nx: State) -> torch.Tensor:
        name = self._names[mod] + ".tail"
        x = torch.cat([st[name], x], dim=-1)
        nx[name] = x[..., x.shape[-1] - st[name].shape[-1]:]
        # Calling the raw conv outside streaming mode is a plain conv without padding
        # (and still runs the weight-norm hook).
        return mod.conv.conv(x)

    def _convtr(self, mod: StreamingConvTranspose1d, x: torch.Tensor, st: State, nx: State) -> torch.Tensor:
        name = self._names[mod] + ".partial"
        raw = mod.convtr.convtr
        y = raw(x)  # [B, C, S * (T - 1) + K]
        partial = st[name]
        n = partial.shape[-1]  # K - S
        y = torch.cat([y[..., :n] + partial, y[..., n:]], dim=-1)
        keep = y.shape[-1] - n  # S * T samples receive no further contributions
        bias = raw.bias[:, None] if raw.bias is not None else 0.0
        nx[name] = y[..., keep:] - bias  # stored without bias, as it is added again next step
        return y[..., :keep]

    def _run(self, mod: nn.Module, x: torch.Tensor, st: State, nx: State) -> torch.Tensor:
        """Run a convolutional (sub)network, threading the conv caches through `st`/`nx`."""
        if isinstance(mod, StreamingConv1d):
            return self._conv(mod, x, st, nx)
        if isinstance(mod, StreamingConvTranspose1d):
            return self._convtr(mod, x, st, nx)
        if isinstance(mod, (SEANetEncoder, SEANetDecoder)):
            return self._run(mod.model, x, st, nx)
        if isinstance(mod, nn.Sequential):
            for child in mod:
                x = self._run(child, x, st, nx)
            return x
        if isinstance(mod, SEANetResnetBlock):
            # StreamingAdd only buffers when lengths differ, which never happens with fixed chunks.
            return self._run(mod.shortcut, x, st, nx) + self._run(mod.block, x, st, nx)
        if isinstance(mod, ConvNeXtBlock):
            return self._convnext(mod, x, st, nx)
        assert not any(isinstance(sub, _STATEFUL) for sub in mod.modules()), (
            f"{type(mod).__name__} has streaming state that VossaStreamingStep does not handle."
        )
        return mod(x)

    def _convnext(self, block: ConvNeXtBlock, x: torch.Tensor, st: State, nx: State) -> torch.Tensor:
        residual = x
        x = block.norm(self._conv(block.dwconv, x, st, nx))
        x = block.pwconv2(block.act(block.pwconv1(x.permute(0, 2, 1))))
        if block.gamma is not None:
            x = block.gamma * x
        return residual + block.drop_path(x.permute(0, 2, 1))

    def _postnet(self, post: PostNet, mel: torch.Tensor, st: State, nx: State) -> torch.Tensor:
        x = post.norm(self._conv(post.embed, mel, st, nx))
        for block in post.blocks:
            x = self._convnext(block, x, st, nx)
        out = mel + self._conv(post.output_proj, post.final_norm(x), st, nx)
        if post.smoothing_layer:
            out = self._conv(post.smoothing, out, st, nx)
        return out

    # ----------------------------------------------------------- transformers

    def _attention(
        self, layer: StreamingTransformerLayer, x: torch.Tensor, offset: torch.Tensor, lookahead: int, prefix: str, st: State, nx: State
    ) -> torch.Tensor:
        attn = layer.self_attn
        T = x.shape[1]
        Tq = T - lookahead
        k_cache, v_cache = st[prefix + ".k"], st[prefix + ".v"]
        capacity = k_cache.shape[2]

        q, k, v = rearrange(F.linear(x, attn.in_proj_weight), "b t (p h d) -> p b h t d", p=3, h=attn.num_heads)
        if attn.rope is not None:
            q, k = attn.rope(q, k, offset, time_before_heads=False)

        # Shift buffer holding the last `capacity` positions, oldest first. Lookahead frames
        # are attended to but not cached: they are recomputed once they become current.
        k_cache = torch.cat([k_cache, k[:, :, :Tq]], dim=2)[:, :, -capacity:]
        v_cache = torch.cat([v_cache, v[:, :, :Tq]], dim=2)[:, :, -capacity:]
        nx[prefix + ".k"], nx[prefix + ".v"] = k_cache, v_cache
        k_all = torch.cat([k_cache, k[:, :, Tq:]], dim=2)
        v_all = torch.cat([v_cache, v[:, :, Tq:]], dim=2)

        pos_k = offset + Tq - capacity + torch.arange(capacity + lookahead, device=x.device)
        pos_q = offset + torch.arange(T, device=x.device).view(-1, 1)
        delta = pos_q - pos_k.view(1, -1)
        mask = (pos_k >= 0) & (delta < attn.context) & (delta >= -(attn.right_context or 0))
        out = F.scaled_dot_product_attention(q, k_all, v_all, mask, dropout_p=0.0)
        return attn.out_proj(rearrange(out, "b h t d -> b t (h d)"))

    def _transformer(self, tr: StreamingTransformer, x: torch.Tensor, st: State, nx: State, lookahead: int) -> torch.Tensor:
        """`x`: `[B, T, C]`, the last `lookahead` frames being future context."""
        assert tr.positional_embedding in ("rope", "none"), tr.positional_embedding
        prefix = self._names[tr]
        offset = st[prefix + ".offset"]
        for i, layer in enumerate(tr.layers):
            assert not layer.skip_self_attn and not layer.weights_per_step
            update = self._attention(layer, layer.norm1(x), offset, lookahead, f"{prefix}.layers.{i}", st, nx)
            x = x + layer.layer_scale_1(update)
            x = layer._ff_block(x)
        nx[prefix + ".offset"] = offset + (x.shape[1] - lookahead)
        return x

    def _projected_transformer(self, proj: ProjectedTransformer, x: torch.Tensor, st: State, nx: State, lookahead: int) -> torch.Tensor:
        assert proj.conv_layout and len(proj.output_projs) == 1
        x = x.transpose(1, 2)
        if proj.input_proj is not None:
            x = proj.input_proj(x)
        x = self._transformer(proj.transformer, x, st, nx, lookahead)
        return proj.output_projs[0](x).transpose(1, 2)

    def _mel_synthesizer(self, mel: MelSynthesizer, z: torch.Tensor, timbre: torch.Tensor, st: State, nx: State) -> torch.Tensor:
        assert mel.conv_layout
        x = z.transpose(1, 2)
        if mel.input_proj is not None:
            x = mel.input_proj(x)
        x = mel.b1_ln(self._transformer(mel.b1_transformer, x, st, nx, 0).transpose(1, 2), timbre)
        x = mel.b2_ln(self._transformer(mel.b2_transformer, x.transpose(1, 2), st, nx, 0).transpose(1, 2), timbre)
        x = self._transformer(mel.b3_transformer, x.transpose(1, 2), st, nx, 0)
        return mel.output_proj(x).transpose(1, 2)

    # ----------------------------------------------------------------- timbre

    def _tvtimbre(
        self, tv: StreamingGlobalOnlyTVTimbre, x: torch.Tensor, spk_k: torch.Tensor, spk_v: torch.Tensor, spk_g: torch.Tensor, st: State, nx: State
    ) -> torch.Tensor:
        name = self._names[tv] + ".pos_offset"
        B, _, T = x.shape
        cross = tv.cross
        q = cross._shape(cross.q_proj(tv.q_proj(x).transpose(1, 2)), B)  # [B, H, T, d]
        alpha = tv.gate(x).transpose(1, 2)  # [B, T, 1]

        # Same rotation as `build_rope_cache_offset`, with the start offset kept as a tensor.
        half = tv.head_dim // 2
        inv_freq = 1.0 / (tv.rope_base ** (torch.arange(0, half, device=x.device).float() / half))
        t = st[name].float() + torch.arange(T, device=x.device).float()
        freqs = torch.einsum("l,d->ld", t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        q = rope_apply(q, emb.cos()[None, None], emb.sin()[None, None])

        w = F.softmax(torch.matmul(q, spk_k.transpose(-2, -1)) * cross.scale, dim=-1)
        attn = cross.out(torch.matmul(w, spk_v).transpose(1, 2).reshape(B, T, -1))  # [B, T, D]
        nx[name] = st[name] + T

        h = F.normalize(tv.post(attn.transpose(1, 2)), dim=1)  # [B, d_out, T]
        g = spk_g.unsqueeze(-1).expand(B, tv.d_out, T)
        return slerp(h.transpose(1, 2), g.transpose(1, 2), alpha).transpose(1, 2)
