from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from .streaming import StreamingModule


# ------------------------ #
# Rotary Positional Embeds #
# ------------------------ #

def build_rope_cache(seq_len: int, dim: int, device, base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    assert dim % 2 == 0
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("s,d->sd", t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

def build_rope_cache_offset(start: int, length: int, dim: int, device, base: float = 10000.0):
    assert dim % 2 == 0
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(start, start + length, device=device).float()
    freqs = torch.einsum("l,d->ld", t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

def rope_apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B,H,S,D], cos/sin: [1,1,S,D]; D even
    x1, x2 = x[..., ::2], x[..., 1::2]
    y1 = x1 * cos[..., ::2] - x2 * sin[..., ::2]
    y2 = x1 * sin[..., ::2] + x2 * cos[..., ::2]
    return torch.stack((y1, y2), dim=-1).flatten(-2)

# ---------------- #
# Token Dropout    #
# ---------------- #

def token_dropout_mask(T: int, p: float, device) -> torch.Tensor:
    if p <= 0:
        return torch.ones(T, device=device)
    m = (torch.rand(T, device=device) > p).float()
    if m.sum() == 0:
        m[torch.randint(0, T, (1,), device=device)] = 1.0
    return m

# ------------------------------ #
# Simple Cross-Attention (SDPA)  #
# ------------------------------ #

class CrossAttn(nn.Module):
    def __init__(self, dim_q: int, dim_kv: int, d_model: int, n_heads: int,
                 attn_dropout: float = 0.0, out_dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim_q, d_model, bias=False)
        self.k_proj = nn.Linear(dim_kv, d_model, bias=False)
        self.v_proj = nn.Linear(dim_kv, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = attn_dropout
        self.out_dropout = nn.Dropout(out_dropout)

    def _shape(self, x: torch.Tensor, B: int) -> torch.Tensor:
        return x.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self,
        Q_seq: torch.Tensor,     # [B, Sq, dim_q]
        K_seq: torch.Tensor,     # [B, Sk, dim_kv]
        V_seq: torch.Tensor,     # [B, Sk, dim_kv]
        rope_q: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rope_k: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        key_padding_mask: Optional[torch.Tensor] = None   # [B, Sk]
    ) -> torch.Tensor:
        B, Sq, _ = Q_seq.shape
        _,  Sk, _ = K_seq.shape
        q = self._shape(self.q_proj(Q_seq), B)  # [B,H,Sq,d]
        k = self._shape(self.k_proj(K_seq), B)  # [B,H,Sk,d]
        v = self._shape(self.v_proj(V_seq), B)  # [B,H,Sk,d]

        if rope_q is not None:
            q = rope_apply(q, rope_q[0], rope_q[1])
        if rope_k is not None:
            k = rope_apply(k, rope_k[0], rope_k[1])

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,Sq,Sk]
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :].bool(), float("-inf"))
        w = F.softmax(scores, dim=-1)
        if self.attn_dropout and self.training:
            w = F.dropout(w, p=self.attn_dropout)
        out = torch.matmul(w, v).transpose(1, 2).reshape(B, Sq, -1) # [B,Sq,d_model]
        return self.out_dropout(self.out(out))                       # [B,Sq,d_model]

# ------------------------------- #
# Global Timbre Memory (per-utt)  #
# ------------------------------- #

class GlobalTimbreMemory(nn.Module):
    """
    Expand a global timbre vector into per-utterance K/V token sets.
    """
    def __init__(self, g_dim: int = 192, d_kv: int = 128, num_tokens: int = 48, hidden: int = 256):
        super().__init__()
        self.T = num_tokens
        self.d_kv = d_kv
        # Simple MLP -> [B, T*(K+V)]
        self.net = nn.Sequential(
            nn.Linear(g_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 2 * num_tokens * d_kv)
        )
        # Optional per-token prior (learned prototypes) helps stabilize early training
        self.k_prior = nn.Parameter(torch.randn(num_tokens, d_kv) * 0.02)
        self.v_prior = nn.Parameter(torch.randn(num_tokens, d_kv) * 0.02)

    def forward(self, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        g: [B, g_dim]
        returns:
          K_seq, V_seq: [B, T, d_kv]
        """
        B = g.size(0)
        raw = self.net(g)                                 # [B, 2*T*d]
        k, v = torch.split(raw, self.T * self.d_kv, dim=-1)
        k = k.view(B, self.T, self.d_kv) + self.k_prior   # [B,T,d]
        v = v.view(B, self.T, self.d_kv) + self.v_prior   # [B,T,d]
        return k, v

class LocalTimbreMemory(nn.Module):
    """
    Streaming local timbre → (K_l, V_l)
    l_chunk: [B, d_l, T_chunk]
    returns: K_l, V_l ∈ [B, N, d_kv]
    """

    def __init__(
        self,
        d_l: int,
        d_kv: int = 128,
        num_tokens: int = 48,
        n_heads: int = 4,
        momentum: float = 0.9,
        hidden: int = 256,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.d_kv = d_kv
        self.momentum = momentum

        # learned slot queries (local timbre prototypes)
        self.q_tokens = nn.Parameter(torch.randn(1, num_tokens, d_kv) * 0.02)

        # local feature → key/value
        self.k_proj = nn.Conv1d(d_l, d_kv, 1, bias=False)
        self.v_proj = nn.Conv1d(d_l, d_kv, 1, bias=False)

        self.attn = nn.MultiheadAttention(
            d_kv, n_heads, batch_first=True
        )

        # lightweight post-projection
        self.post_k = nn.Sequential(
            nn.Linear(d_kv, hidden), nn.GELU(),
            nn.Linear(hidden, d_kv)
        )
        self.post_v = nn.Sequential(
            nn.Linear(d_kv, hidden), nn.GELU(),
            nn.Linear(hidden, d_kv)
        )

    def init_state(self, batch_size: int, device):
        return {
            "K_l": torch.zeros(batch_size, self.num_tokens, self.d_kv, device=device),
            "V_l": torch.zeros(batch_size, self.num_tokens, self.d_kv, device=device),
        }

    def forward(
        self,
        l_chunk: torch.Tensor,          # [B, d_l, T_chunk]
        state: dict | None = None,
    ):
        B, _, T = l_chunk.shape
        device = l_chunk.device

        if state is None:
            state = self.init_state(B, device)

        # project local chunk
        K_seq = self.k_proj(l_chunk).transpose(1, 2)  # [B, T, d_kv]
        V_seq = self.v_proj(l_chunk).transpose(1, 2)  # [B, T, d_kv]

        Q = self.q_tokens.expand(B, -1, -1)            # [B, N, d_kv]

        # attend slots to *current chunk only*
        dK, _ = self.attn(Q, K_seq, K_seq, need_weights=False)
        dV, _ = self.attn(Q, K_seq, V_seq, need_weights=False)

        dK = self.post_k(dK)
        dV = self.post_v(dV)

        # EMA update (streaming-safe)
        K_l = self.momentum * state["K_l"] + (1 - self.momentum) * dK
        V_l = self.momentum * state["V_l"] + (1 - self.momentum) * dV

        state["K_l"] = K_l
        state["V_l"] = V_l

        return K_l, V_l, state


def slerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a_n = F.normalize(a, dim=-1); b_n = F.normalize(b, dim=-1)
    dot = (a_n * b_n).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    omega = torch.acos(dot); sin_omega = torch.sin(omega).clamp_min(eps)
    return (torch.sin((1 - t) * omega) / sin_omega) * a_n + (torch.sin(t * omega) / sin_omega) * b_n


@dataclass
class _TVTimbreState:
    pos_offset: int

    def reset(self):
        self.pos_offset = 0

@dataclass
class _TVTimbreStatev2:
    pos_offset: int
    local_memory: dict | None = None

    def reset(self):
        self.pos_offset = 0
        self.local_memory = None

class StreamingGlobalOnlyTVTimbre(StreamingModule[_TVTimbreState]):
    def __init__(
            self, 
            content_dim: int, 
            g_dim: int = 704, 
            d_kv: int = 128, 
            d_out: int = 192, 
            num_tokens: int = 48,
            n_heads: int = 4, 
            attn_dropout: float = 0.0, 
            gate_hidden: int = 256,
            token_dropout_p: float = 0.1, 
            rope_base: float = 10000.0,
        ):
        super().__init__()
        assert d_kv % n_heads == 0
        self.d_kv = d_kv
        self.d_out = d_out
        self.head_dim = d_kv // n_heads
        self.num_tokens = num_tokens
        self.token_dropout_p = token_dropout_p
        self.rope_base = rope_base

        self.gtm = GlobalTimbreMemory(g_dim=g_dim, d_kv=d_kv, num_tokens=num_tokens)
        
        self.q_proj = nn.Conv1d(content_dim, d_kv, 1, bias=False)
        self.cross = CrossAttn(d_kv, d_kv, d_kv, n_heads, attn_dropout, 0.0)
        self.speaker_proj = nn.Linear(g_dim, d_out)
        self.post = nn.Conv1d(d_kv, d_out, 1)
        self.gate = nn.Sequential(nn.Conv1d(content_dim, gate_hidden, 1), nn.GELU(), nn.Conv1d(gate_hidden, 1, 1), nn.Sigmoid())

    def _init_streaming_state(self, batch_size: int) -> _TVTimbreState:
        return _TVTimbreState(pos_offset=0)

    def forward(self, x: torch.Tensor, global_timbre: torch.Tensor) -> torch.Tensor:
        B, _, T = x.shape
        state = self._streaming_state
        device = x.device

        K_seq, V_seq = self.gtm(global_timbre)  # [B,NT,D]
        if self.training and self.token_dropout_p > 0:
            keep = token_dropout_mask(K_seq.size(1), self.token_dropout_p, device)
            K_seq = K_seq * keep[None, :, None]
            V_seq = V_seq * keep[None, :, None]

        q_new = self.q_proj(x).transpose(1,2)                 # [B,T,D]
        alpha = self.gate(x).transpose(1, 2)                  # [B,T,1]

        if state is None:
            start = 0
        else:
            start = int(state.pos_offset)

        cos_q, sin_q = build_rope_cache_offset(start, T, self.head_dim, device, base=self.rope_base)
        attn = self.cross(q_new, K_seq, V_seq, rope_q=(cos_q, sin_q), rope_k=None) # [B,T,D]

        H = attn.transpose(1, 2)                                     # [B,D,T]
        H192 = F.normalize(self.post(H), dim=1)                      # [B,192,T]
        g = self.speaker_proj(F.normalize(global_timbre, dim=-1)).unsqueeze(-1).expand(B, self.d_out, T)  # [B,192,T]
        tv_embed = slerp(H192.transpose(1, 2), g.transpose(1, 2), alpha).transpose(1, 2)

        if state is not None:
            state.pos_offset += T

        return tv_embed


class StreamingTVTimbre(StreamingModule[_TVTimbreStatev2]):
    def __init__(
            self, 
            content_dim: int, 
            g_dim: int = 512, 
            l_dim: int = 128,
            d_kv: int = 128, 
            d_out: int = 192, 
            num_tokens: int = 48,
            n_heads: int = 4, 
            attn_dropout: float = 0.0, 
            gate_hidden: int = 256,
            token_dropout_p: float = 0.1, 
            rope_base: float = 10000.0,
            lambda_local: float = 1.0
        ):
        super().__init__()
        assert d_kv % n_heads == 0
        self.d_kv = d_kv
        self.d_out = d_out
        self.head_dim = d_kv // n_heads
        self.num_tokens = num_tokens
        self.token_dropout_p = token_dropout_p
        self.rope_base = rope_base
        self.lambda_local = lambda_local

        self.gtm = GlobalTimbreMemory(g_dim=g_dim, d_kv=d_kv, num_tokens=num_tokens)
        self.ltm = LocalTimbreMemory(d_l=l_dim, d_kv=d_kv, num_tokens=num_tokens)
        
        self.q_proj = nn.Conv1d(content_dim, d_kv, 1, bias=False)
        self.cross = CrossAttn(d_kv, d_kv, d_kv, n_heads, attn_dropout, 0.0)
        self.speaker_proj = nn.Linear(g_dim, d_out)
        self.post = nn.Conv1d(d_kv, d_out, 1)
        self.gate = nn.Sequential(nn.Conv1d(content_dim, gate_hidden, 1), nn.GELU(), nn.Conv1d(gate_hidden, 1, 1), nn.Sigmoid())

    def _init_streaming_state(self, batch_size: int) -> _TVTimbreStatev2:
        return _TVTimbreStatev2(pos_offset=0, local_memory=None)

    def set_lambda_local(self, lambda_local: float):
        self.lambda_local = lambda_local

    def forward(self, x: torch.Tensor, global_timbre: torch.Tensor, local_timbre: torch.Tensor, lambda_local: float = None) -> torch.Tensor:
        """
        x:            [B, content_dim, T]
        global_timbre:[B, g_dim]
        local_timbre: [B, l_dim, T']
        """

        B, _, T = x.shape
        state = self._streaming_state
        device = x.device

        lam = self.lambda_local if lambda_local is None else float(lambda_local)
        if state is None:
            start = 0
            ltm_state = None
        else:
            start = int(state.pos_offset)
            ltm_state = state.local_memory

        # --- Global tokens ---
        K_g, V_g = self.gtm(global_timbre)  # [B,NT,d_kv], [B,NT,d_kv]

        # --- Local tokens ---
        K_l, V_l, ltm_state = self.ltm(local_timbre, state=ltm_state)  # [B,NT,d_kv], [B,NT,d_kv]

        K_seq = K_g + lam * K_l  # [B,NT,d_kv]
        V_seq = V_g + lam * V_l  # [B,NT,d_kv]

        key_padding_mask = None  # no padding in streaming case
        if self.training and self.token_dropout_p > 0:
            keep = token_dropout_mask(K_seq.size(1), self.token_dropout_p, device)
            key_padding_mask = (keep == 0)[None, :].expand(B, -1)

        q_new = self.q_proj(x).transpose(1,2)                 # [B,T,D]
        alpha = self.gate(x).transpose(1, 2)                  # [B,T,1]

        cos_q, sin_q = build_rope_cache_offset(start, T, self.head_dim, device, base=self.rope_base)
        attn = self.cross(q_new, K_seq, V_seq, rope_q=(cos_q, sin_q), rope_k=None, key_padding_mask=key_padding_mask) # [B,T,D]

        H = attn.transpose(1, 2)                                     # [B,D,T]
        H192 = F.normalize(self.post(H), dim=1)                      # [B,192,T]
        g = self.speaker_proj(F.normalize(global_timbre, dim=-1)).unsqueeze(-1).expand(B, self.d_out, T)  # [B,192,T]
        tv_embed = slerp(H192.transpose(1, 2), g.transpose(1, 2), alpha).transpose(1, 2)

        if state is not None:
            state.pos_offset += T
            state.local_memory = ltm_state

        return tv_embed


class TimbreTokenBlock(nn.Module):
    def __init__(self, N_tok: int = 48, d_tok: int = 128, d_feat: int = 512):
        super().__init__()
        self.N_tok = N_tok
        self.d_tok = d_tok
        self.d_feat = d_feat

        # Learnable timbre tokens (queries)
        self.token_emb = nn.Parameter(torch.randn(1, d_tok, N_tok))

        # Cross-attention layers (queries = timbre tokens, keys/values = local features)
        self.cross_attn_1 = CrossAttn(dim_q=d_tok, dim_kv=d_feat, d_model=d_tok, n_heads=4)
        self.cross_attn_2 = CrossAttn(dim_q=d_tok, dim_kv=d_feat, d_model=d_tok, n_heads=4)

        self.norm = nn.LayerNorm(d_tok)

    def forward(self, spkr_embed: torch.Tensor) -> torch.Tensor:
        """
        spkr_embed: [B, d_feat, T]   # local feature sequence
        returns:    [B, d_tok, N_tok] # timbre tokens
        """
        B, _, T = spkr_embed.size()

        # Expand learnable tokens to batch
        token_emb = self.token_emb.expand(B, -1, -1).transpose(1, 2)  # [B, N_tok, d_tok]

        # Reorder local features
        local_feat = spkr_embed.transpose(1, 2)       # [B, T, d_feat]

        # Cross-attention with residual + norm
        out = self.cross_attn_1(token_emb, local_feat, local_feat) + token_emb
        out = self.norm(out)

        # Second cross-attention
        out = self.cross_attn_2(out, local_feat, local_feat)

        return out.transpose(1, 2)  # [B, d_tok, N_tok]


class StreamingTVTimbreModule(StreamingModule[_TVTimbreState]):
    def __init__(
            self, 
            content_dim: int, 
            d_kv: int = 128, 
            d_out: int = 192, 
            num_tokens: int = 48,
            n_heads: int = 4, 
            attn_dropout: float = 0.0, 
            gate_hidden: int = 256,
            token_dropout_p: float = 0.1, 
            rope_base: float = 10000.0,
        ):
        super().__init__()
        assert d_kv % n_heads == 0
        self.d_kv = d_kv
        self.d_out = d_out
        self.head_dim = d_kv // n_heads
        self.num_tokens = num_tokens
        self.token_dropout_p = token_dropout_p
        self.rope_base = rope_base

        self.K_seq = nn.Parameter(torch.randn(1, d_kv, num_tokens))
        self.ttb = TimbreTokenBlock(N_tok=num_tokens, d_tok=d_kv, d_feat=content_dim)
        
        self.q_proj = nn.Conv1d(content_dim, d_kv, 1, bias=False)
        self.cross = CrossAttn(d_kv, d_kv, d_kv, n_heads, attn_dropout, 0.0)
        self.speaker_proj = nn.Linear(content_dim, d_out)

        self.post = nn.Conv1d(d_kv, d_out, 1)
        self.gate = nn.Sequential(nn.Conv1d(content_dim, gate_hidden, 1), nn.GELU(), nn.Conv1d(gate_hidden, 1, 1), nn.Sigmoid())

    def _init_streaming_state(self, batch_size: int) -> _TVTimbreState:
        return _TVTimbreState(pos_offset=0)

    def forward(self, x: torch.Tensor, spkr_embed) -> torch.Tensor:
        B, _, T = x.shape
        state = self._streaming_state
        device = x.device

        global_spkr_embed, local_spkr_embed = spkr_embed 
        
        K_seq = self.K_seq.expand(B, -1, -1)  # [B, d_kv, N_tok]
        V_seq = self.ttb(local_spkr_embed) # timbre_tokens: [B, d_kv, N_tok]

        q_new = self.q_proj(x).transpose(1,2) # [B,T,D]
        alpha = self.gate(x).transpose(1, 2)  # [B,T,1]
        K_seq = K_seq.transpose(1, 2)         # [B, d_kv, N_tok] -> [B, N_tok, d_kv]
        V_seq = V_seq.transpose(1, 2)         # [B, d_kv, N_tok] -> [B, N_tok, d_kv]

        if self.training and self.token_dropout_p > 0:
            keep = token_dropout_mask(K_seq.size(1), self.token_dropout_p, device)
            K_seq = K_seq * keep[None, :, None]
            V_seq = V_seq * keep[None, :, None]


        if state is None:
            start = 0
        else:
            start = int(state.pos_offset)

        cos_q, sin_q = build_rope_cache_offset(start, T, self.head_dim, device, base=self.rope_base)
        attn = self.cross(q_new, K_seq, V_seq, rope_q=(cos_q, sin_q), rope_k=None) # [B, T, D]

        H = attn.transpose(1, 2)                   # [B, D, T]
        H192 = F.normalize(self.post(H), dim=1)    # [B, 192, T]
        g = self.speaker_proj(F.normalize(global_spkr_embed, dim=-1)).unsqueeze(-1).expand(B, self.d_out, T)  # [B, 192, T]
        tv_embed = slerp(H192.transpose(1, 2), g.transpose(1, 2), alpha).transpose(1, 2)

        if state is not None:
            state.pos_offset += T

        return tv_embed