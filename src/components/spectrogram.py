import typing as tp
from dataclasses import dataclass


import torch
import math
import numpy as np
import torch.nn as nn
from torch import Tensor
import torchaudio.functional as F
from torchaudio.transforms import MelScale

from .streaming import StreamingModule
from .conv import get_extra_padding_for_conv1d, pad1d


class LinearSpectrogram(nn.Module):
    def __init__(
        self, 
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 320,
        mode: str = "mag",        # "complex" | "mag" | "power" | "logmag"
        center: bool = False,     # keep False for streaming
        eps: float = 1e-8,
    ):
        """
        Initializes the streaming spectrogram module.
        
        Parameters:
            n_fft (int): Number of FFT points.
            win_length (int): Window length (in samples).
            hop_length (int): Hop length (in samples).
            mode (str): Calculation mode. "pow2_sqrt" computes magnitude as sqrt(sum(squared)).
        """
        super().__init__()
        assert mode in {"complex", "mag", "power", "logmag"}
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.mode = mode
        self.center = center
        self.eps = eps
        
        # Register the Hann window as a buffer.
        win = torch.hann_window(win_length)  # float32 CPU by default
        # Keep non-persistent to follow caller dtype/device at runtime
        self.register_buffer("window", win, persistent=False)

    @torch.no_grad()
    def _prep_window(self, y: torch.Tensor) -> torch.Tensor:
        # Ensure window matches y’s device/dtype and broadcast shape (B, F, win)
        if self.window.device != y.device or self.window.dtype != y.dtype:
            self.window = torch.hann_window(self.win_length, dtype=y.dtype, device=y.device)
        return self.window.view(1, 1, -1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the spectrogram from an input waveform chunk.
        
        Parameters:
            y (Tensor): Input waveform of shape (batch, time).
                                      
        Returns:
            spec: (B, n_fft//2+1, n_frames) if mode != "complex", else complex tensor
            tail: (B, T_tail)  # samples not consumed (useful for streaming); None if center=True
        """
        B, T = y.shape
        win = self._prep_window(y)  # (1,1,win)

        # Optional centering: pad so frames are centered like torch.stft(center=True)
        if self.center:
            pad = (self.win_length // 2, self.win_length // 2)
            y_pad = torch.nn.functional.pad(y, pad, mode="reflect")
        else:
            y_pad = y

        total = y_pad.size(1)
        if total < self.win_length:
            # Not enough for one frame
            empty = torch.empty(B, self.n_fft // 2 + 1, 0, device=y.device, dtype=y.dtype)
            return empty if self.mode != "complex" else empty.to(torch.complex64), y
        
        n_frames = 1 + (total - self.win_length) // self.hop_length
        used = (n_frames - 1) * self.hop_length + self.win_length
        # tail = y_pad[:, used:]  # return this so caller can prepend on next call


        # (B, n_frames, win)
        frames = y_pad[:, :used].unfold(dimension=1, size=self.win_length, step=self.hop_length)
        # window
        frames = frames * win  # broadcast to (B, n_frames, win)

        # rFFT over last dim
        spec_c = torch.fft.rfft(frames, n=self.n_fft, dim=-1)  # (B, n_frames, n_fft//2+1)

        if self.mode == "complex":
            spec = spec_c.transpose(1, 2).contiguous()  # (B, n_fft//2+1, n_frames)
            return spec #, tail if not self.center else None

        mag2 = (spec_c.real.pow(2) + spec_c.imag.pow(2))
        if self.mode == "mag":
            out = torch.sqrt(mag2.clamp_min(self.eps))
        elif self.mode == "power":
            out = mag2
        else:  # "logmag"
            out = 0.5 * torch.log(mag2.clamp_min(self.eps))

        spec = out.transpose(1, 2).contiguous()  # (B, n_fft//2+1, n_frames)
        return spec #, tail if not self.center else None



@dataclass
class _RawStreamingSTFTState:
    previous: torch.Tensor | None = None

    def reset(self):
        self.previous = None


class RawStreamingSTFT(LinearSpectrogram, StreamingModule[_RawStreamingSTFTState]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert (
            self.hop_length <= self.win_length
        ), "stride must be less than kernel_size."

    def _init_streaming_state(self, batch_size: int) -> _RawStreamingSTFTState:
        return _RawStreamingSTFTState()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        stride = self.hop_length
        kernel = self.win_length
        if self._streaming_state is None:
            return super().forward(input)
        else:
            # Due to the potential overlap, we might have some cache of the previous time steps.
            previous = self._streaming_state.previous
            if previous is not None:
                input = torch.cat([previous, input], dim=-1)
            B, C, T = input.shape
            # We now compute the number of full convolution frames, i.e. the frames
            # that are ready to be computed.
            num_frames = max(0, int(math.floor((T - kernel) / stride) + 1))
            offset = num_frames * stride
            # We will compute `num_frames` outputs, and we are advancing by `stride`
            # for each of the frame, so we know the data before `stride * num_frames`
            # will never be used again.
            self._streaming_state.previous = input[..., offset:]
            if num_frames > 0:
                input_length = (num_frames - 1) * stride + kernel
                out = super().forward(input[..., :input_length])
            else:
                # Not enough data as this point to output some new frames.
                out = torch.empty(
                    B, self.n_mels, 0, device=input.device, dtype=input.dtype
                )
            return out


@dataclass
class _StreamingSTFTState:
    padding_to_add: int
    original_padding_to_add: int

    def reset(self):
        self.padding_to_add = self.original_padding_to_add


class StreamingSTFT(StreamingModule[_StreamingSTFTState]):
    """LogMelSpectrogram with some builtin handling of asymmetric or causal padding
    """

    def __init__(
        self,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 320,
        mode: str = "complex",        # "complex" | "mag" | "power" | "logmag"
        causal: bool = False,
        pad_mode: str = "reflect",
    ):
        super().__init__()

        self.conv = LinearSpectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            mode=mode,
        )
        self.causal = causal
        self.pad_mode = pad_mode

    @property
    def _stride(self) -> int:
        return self.conv.hop_length

    @property
    def _kernel_size(self) -> int:
        return self.conv.win_length

    @property
    def _effective_kernel_size(self) -> int:
        return self._kernel_size

    @property
    def _padding_total(self) -> int:
        return self._effective_kernel_size - self._stride

    def _init_streaming_state(self, batch_size: int) -> _StreamingSTFTState:
        assert self.causal, "streaming is only supported for causal convs"
        return _StreamingSTFTState(self._padding_total, self._padding_total)

    def forward(self, x):
        B, C, T = x.shape
        padding_total = self._padding_total
        extra_padding = get_extra_padding_for_conv1d(
            x, self._effective_kernel_size, self._stride, padding_total
        )
        state = self._streaming_state
        if state is None:
            if self.causal:
                # Left padding for causal
                x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
            else:
                # Asymmetric padding required for odd strides
                padding_right = padding_total // 2
                padding_left = padding_total - padding_right
                x = pad1d(
                    x, (padding_left, padding_right + extra_padding), mode=self.pad_mode
                )
        else:
            if state.padding_to_add > 0 and x.shape[-1] > 0:
                x = pad1d(x, (state.padding_to_add, 0), mode=self.pad_mode)
                state.padding_to_add = 0
        return self.conv(x)



class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_fft=1024,
        win_length=1024,
        hop_length=320,
        n_mels=128,
        f_min=0.0,
        f_max=None,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or float(sample_rate // 2)

        self.spectrogram = LinearSpectrogram(n_fft, win_length, hop_length)

        fb = F.melscale_fbanks(
            n_freqs=self.n_fft // 2 + 1,
            f_min=self.f_min,
            f_max=self.f_max,
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        self.register_buffer(
            "fb",
            fb,
            persistent=False,
        )

    def compress(self, x: Tensor) -> Tensor:
        return torch.log(torch.clamp(x, min=1e-5))

    def decompress(self, x: Tensor) -> Tensor:
        return torch.exp(x)

    def apply_mel_scale(self, x: Tensor) -> Tensor:
        return torch.matmul(x.transpose(-1, -2), self.fb).transpose(-1, -2)

    def forward(
        self, x: Tensor, return_linear: bool = False, sample_rate: int = None
    ) -> Tensor:
        x = x.squeeze(1)
        if sample_rate is not None and sample_rate != self.sample_rate:
            x = F.resample(x, orig_freq=sample_rate, new_freq=self.sample_rate)

        linear = self.spectrogram(x)

        if linear.shape[-1] != 0:
            mel = self.apply_mel_scale(linear)
            mel = self.compress(mel)
        else:
            # Not enough samples to form one full frame: return an empty spectrogram.
            mel = torch.empty(
                linear.shape[0], self.n_mels, 0, device=linear.device, dtype=linear.dtype
            )

        compressed_linear = self.compress(linear) if linear.shape[-1] != 0 else linear
        if return_linear:
            return mel, compressed_linear

        return mel


@dataclass
class _StreamingSpecState:
    previous: torch.Tensor | None = None

    def reset(self):
        self.previous = None


class RawStreamingLogMelSpectrogram(LogMelSpectrogram, StreamingModule[_StreamingSpecState]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert (
            self.hop_length <= self.win_length
        ), "stride must be less than kernel_size."

    def _init_streaming_state(self, batch_size: int) -> _StreamingSpecState:
        return _StreamingSpecState()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        stride = self.hop_length
        kernel = self.win_length
        if self._streaming_state is None:
            return super().forward(input)
        else:
            # Due to the potential overlap, we might have some cache of the previous time steps.
            previous = self._streaming_state.previous
            if previous is not None:
                input = torch.cat([previous, input], dim=-1)
            B, C, T = input.shape
            # We now compute the number of full convolution frames, i.e. the frames
            # that are ready to be computed.
            num_frames = max(0, int(math.floor((T - kernel) / stride) + 1))
            offset = num_frames * stride
            # We will compute `num_frames` outputs, and we are advancing by `stride`
            # for each of the frame, so we know the data before `stride * num_frames`
            # will never be used again.
            self._streaming_state.previous = input[..., offset:]
            if num_frames > 0:
                input_length = (num_frames - 1) * stride + kernel
                out = super().forward(input[..., :input_length])
            else:
                # Not enough data as this point to output some new frames.
                out = torch.empty(
                    B, self.n_mels, 0, device=input.device, dtype=input.dtype
                )
            return out



@dataclass
class _StreamingLogMelSpecState:
    padding_to_add: int
    original_padding_to_add: int

    def reset(self):
        self.padding_to_add = self.original_padding_to_add


class StreamingLogMelSpectrogram(StreamingModule[_StreamingLogMelSpecState]):
    """LogMelSpectrogram with some builtin handling of asymmetric or causal padding
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 320,
        n_mels: int = 128,
        f_min: float = 0.0,
        f_max: float = None,
        causal: bool = False,
        pad_mode: str = "reflect",
    ):
        super().__init__()

        self.conv = RawStreamingLogMelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )
        self.causal = causal
        self.pad_mode = pad_mode

    @property
    def _stride(self) -> int:
        return self.conv.hop_length

    @property
    def _kernel_size(self) -> int:
        return self.conv.win_length

    @property
    def _effective_kernel_size(self) -> int:
        return self._kernel_size

    @property
    def _padding_total(self) -> int:
        return self._effective_kernel_size - self._stride

    def _init_streaming_state(self, batch_size: int) -> _StreamingLogMelSpecState:
        assert self.causal, "streaming is only supported for causal convs"
        return _StreamingLogMelSpecState(self._padding_total, self._padding_total)

    def forward(self, x):
        B, C, T = x.shape
        padding_total = self._padding_total
        extra_padding = get_extra_padding_for_conv1d(
            x, self._effective_kernel_size, self._stride, padding_total
        )
        state = self._streaming_state
        if state is None:
            if self.causal:
                # Left padding for causal
                x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
            else:
                # Asymmetric padding required for odd strides
                padding_right = padding_total // 2
                padding_left = padding_total - padding_right
                x = pad1d(
                    x, (padding_left, padding_right + extra_padding), mode=self.pad_mode
                )
        else:
            if state.padding_to_add > 0 and x.shape[-1] > 0:
                x = pad1d(x, (state.padding_to_add, 0), mode=self.pad_mode)
                state.padding_to_add = 0
        return self.conv(x)


class OverlapAdd1d(nn.Module):
    """
    Fixed ConvTranspose1d that performs overlap-add:
      in_channels = win_length, out_channels = 1, kernel=win_length, stride=hop.
    """
    def __init__(self, win_length: int, hop: int):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            in_channels=win_length,
            out_channels=1,
            kernel_size=win_length,
            stride=hop,
            bias=False,
        )
        # build identity‑impulse weights
        w = torch.zeros(win_length, 1, win_length)
        for c in range(win_length):
            w[c, 0, c] = 1.0
        self.deconv.weight.data.copy_(w)
        self.deconv.weight.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, win_length, F) → y: (B, 1, (F-1)*hop + win_length)
        y = self.deconv(x)
        return y.squeeze(1)  # → (B, T_out)


@dataclass
class _StreamingISTFTState:
    prev_buffer: torch.Tensor
    prev_norm: torch.Tensor
    held_tail: torch.Tensor  # for crossfade

    def reset(self):
        self.prev_buffer.zero_()
        self.prev_norm.zero_()
        self.held_tail.zero_()



class StreamingISTFT(StreamingModule[_StreamingISTFTState]):
    """
    Streaming ISTFT via overlap-add:
      - inverse FFT per frame
      - window multiplication
      - overlap-add using a fixed ConvTranspose1d
      - carry tail for next chunk
    """
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int | None = None,
        crossfade_len: int | None = None,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop_length
        self.win_length = win_length or n_fft

        self.crossfade_len = crossfade_len if crossfade_len is not None else min(self.hop // 2, 384)
        if self.crossfade_len < 0:
            self.crossfade_len = 0

        # raised-cosine (complementary) ramps that sum to 1
        if self.crossfade_len > 0:
            n = torch.arange(self.crossfade_len, dtype=torch.float32)
            w_prev = 0.5 * (1.0 + torch.cos(math.pi * (n / (self.crossfade_len - 1))))  # 1→0
            w_curr = 1.0 - w_prev  # 0→1
            self.register_buffer("xfade_prev", w_prev, persistent=False)
            self.register_buffer("xfade_curr", w_curr, persistent=False)

        # hann window
        win = torch.hann_window(self.win_length)
        self.register_buffer("window", win, persistent=False)
        self.register_buffer("eps", torch.tensor(1e-8), persistent=False)
        self.register_buffer("window_sq", (torch.hann_window(self.win_length)**2), persistent=False)

        # overlap‑add helper
        self.overlap_add = OverlapAdd1d(self.win_length, self.hop)

        # how many samples to carry
        self.tail = self.win_length - self.hop
        if self.tail < 0:
            raise ValueError("hop_length must be <= win_length")

    def _init_streaming_state(self, batch_size: int) -> _StreamingISTFTState:
        device = self.window.device
        buf = torch.zeros(batch_size, self.tail, device=device)
        held = torch.zeros(batch_size, self.crossfade_len, device=device) if self.crossfade_len > 0 else torch.zeros(batch_size, 0, device=device)
        return _StreamingISTFTState(prev_buffer=buf, prev_norm=buf.clone(), held_tail=held)

    def forward(self, S: Tensor) -> Tensor:
        """
        Args:
          S: complex STFT chunk, shape (B, n_fft//2+1, F_frames)
        Returns:
          time-domain chunk, shape (B, F_frames * hop_length)
        """
        B, n_freq, F = S.shape

        if F == 0:
            # Not enough samples to form one full frame: return an empty tensor.
            return torch.empty(B, 0, device=S.device, dtype=S.dtype)

        state = self._streaming_state
        # if state is None:
        #     # no streaming state, just do a regular ISTFT
        #     return torch.istft(
        #         S,
        #         n_fft=self.n_fft,
        #         hop_length=self.hop,
        #         win_length=self.win_length,
        #         window=self.window,
        #         center=False,
        #     )

        # (B, F, n_fft//2+1) -> real frames (B, F, n_fft)
        spec = S.transpose(1, 2)
        frames = torch.fft.irfft(spec, n=self.n_fft)
        # window + truncate
        frames = frames[..., : self.win_length] 
        
        # Typical torch.stft does, so keep this multiply + normalize by window^2 OLA:
        frames = frames * self.window  # (B, F, win)
        
        
        # Prepare for OLA
        x = frames.transpose(1, 2)  # (B, win, F)

        # Signal OLA
        out = self.overlap_add(x)  # (B, F*hop + tail)

        # Normalization OLA: overlap-add window^2 frames of ones
        # Build (B, win, F) where each frame is window^2
        norm_frames = self.window_sq.expand(F, self.win_length).T.unsqueeze(0).expand(B, -1, -1)
        norm = self.overlap_add(norm_frames)  # (B, F*hop + tail)

        if state is not None:
            # carry over both signal and norm tails
            out[:, : self.tail] += state.prev_buffer
            norm[:, : self.tail] += state.prev_norm

            ready = out[:, : F * self.hop]
            norm_ready = norm[:, : F * self.hop]

            # save new tails
            state.prev_buffer = out[:, F * self.hop :]
            state.prev_norm   = norm[:, F * self.hop :]
        else:
            ready = out[:, : F * self.hop]
            norm_ready = norm[:, : F * self.hop]

        # Final per-sample normalization (avoid divide-by-zero)
        ready = ready / (norm_ready + self.eps)

        # ready: (B, T_ready)
        L = self.crossfade_len
        if L > 0 and state is not None:
            B, T = ready.shape
            # If chunk too short, shrink crossfade safely
            L_eff = min(L, max(0, T // 2))
            if L_eff > 0:
                # Blend with previously held tail (same length)
                if state.held_tail.numel() > 0:
                    # match effective length if it shrank
                    prev_tail = state.held_tail[:, :L_eff]
                    w_prev = self.xfade_prev[:L_eff].to(ready.device).view(1, -1)
                    w_curr = self.xfade_curr[:L_eff].to(ready.device).view(1, -1)

                    blended = prev_tail * w_prev + ready[:, :L_eff] * w_curr
                    # Emit: blended + middle; hold back the new tail
                    middle_end = max(L_eff, T - L_eff)
                    emit_mid = ready[:, L_eff: middle_end]  # could be empty
                    out_chunk = torch.cat([blended, emit_mid], dim=1)

                    # Save new tail (last L_eff samples) for next crossfade
                    state.held_tail = ready[:, T - L_eff:].contiguous()
                else:
                    # First call: no previous tail to blend; emit everything except hold back the tail
                    out_chunk = ready[:, : max(0, T - L_eff)]
                    state.held_tail = ready[:, T - L_eff:].contiguous()
            else:
                # Chunk too small for crossfade—just emit it (and keep held_tail unchanged)
                out_chunk = ready
        else:
            # Crossfade disabled
            out_chunk = ready

        return out_chunk