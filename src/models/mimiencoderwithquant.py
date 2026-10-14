# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Part of this file is adapted from encodec.py in https://github.com/facebookresearch/audiocraft
# released under the following license.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Compression models or wrapper around existing models. In particular, provides the implementation
for Mimi. Also defines the main interface that a model must follow to be usable as an audio tokenizer.
"""

from abc import abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
import logging
import typing as tp

import torch
from torch import nn


from src.components.streaming import StreamingModule, State
from src.utils.compile import no_compile, CUDAGraphed


logger = logging.getLogger()


class EncoderAbsModel(StreamingModule[State]):
    """Base API for all compression model that aim at being used as audio tokenizers
    with a language model.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """See `MimiModel.encode`."""
        ...

    @abstractmethod
    def decode_latent(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode from the discrete codes to continuous latent space."""
        ...

    @property
    @abstractmethod
    def channels(self) -> int: ...

    @property
    @abstractmethod
    def frame_rate(self) -> float: ...

    @property
    @abstractmethod
    def sample_rate(self) -> int: ...



@dataclass
class _MimiEncoderState:
    graphed_tr_enc: CUDAGraphed | None

    def reset(self):
        pass


class MimiEncoderWithQuantModel(EncoderAbsModel[_MimiEncoderState]):
    """Mimi model operating on the raw waveform.

    Args:
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        quantizer (qt.BaseQuantizer): Quantizer network.
        frame_rate (float): Final frame rate of the quantized representatiopn.
        encoder_frame_rate (float): frame rate of the encoder model. Note that if `frame_rate != encopder_frame_rate`,
            the latent will be resampled linearly to match the desired `frame_rate` before and after quantization.
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.
        causal (bool): Whether to use a causal version of the model.
        encoder_transformer (nn.Module or None): optional transformer for the encoder.
        decoder_transformer (nn.Module or None): optional transformer for the decoder.
        resample_method (str): method to use for resampling the latent space before the quantizer.
        upsample_channel_wise_bug (bool): controls whether the upsampling is channel wise.
            Defaults to true to reproduce bug in original implementation.
        freeze_encoder: whether to freeze the encoder weights.
        freeze_quantizer: whether to freeze the quantizer weights.
        freeze_quantizer_level: If positive, freeze the quantizer up to this level.
        torch_compile_encoder_decoder (bool): if True, uses torch.compile on the encoder / decoder.
            Deactivated by default for training as this is incompatible at the moment with weight norm.
            See https://github.com/pytorch/pytorch/issues/121902
            Also this seems to work well with 2.2.0, but completely fail with 2.4.0.
    """

    def __init__(
        self,
        encoder: nn.Module,
        quantizer: nn.Module,
        frame_rate: float,
        sample_rate: int,
        channels: int,
        causal: bool = False,
        encoder_transformer: tp.Optional[nn.Module] = None,
        torch_compile_encoder_decoder: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_transformer = encoder_transformer
        self.quantizer = quantizer
        self._frame_rate = frame_rate
        self._sample_rate = sample_rate
        self._channels = channels
        self.torch_compile_encoder_decoder = torch_compile_encoder_decoder

        # We will need the dimension for the resampling. In general the encoder will be a SeanetEncoder
        # which exposes a `dimension` attribute.
        dimension = encoder.dimension
        assert isinstance(
            dimension, int
        ), f"Dimension should be int, got {dimension} of type {type(dimension)}."
        self.dimension = dimension

    def _init_streaming_state(self, batch_size: int) -> _MimiEncoderState:
        device = next(self.parameters()).device
        disable = device.type != 'cuda'
        graphed_tr_enc = None
        if self.encoder_transformer is not None:
            graphed_tr_enc = CUDAGraphed(self.encoder_transformer, disable=disable)
        
        return _MimiEncoderState(graphed_tr_enc)

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def frame_rate(self) -> float:
        return self._frame_rate

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def _context_for_encoder_decoder(self):
        if self.torch_compile_encoder_decoder:
            return nullcontext()
        else:
            return no_compile()

    def forward(self, x: torch.Tensor, return_intermediates=False) -> tuple[torch.Tensor, tp.Dict[str, torch.Tensor]]:
        assert x.dim() == 3
        length = x.shape[-1]
        extra_metrics: tp.Dict[str, torch.Tensor] = {}


        with self._context_for_encoder_decoder:
            cnn_emb = emb = self.encoder(x)
        if self.encoder_transformer is not None:
            if return_intermediates:
                (emb,), intermediates = self.encoder_transformer(emb, return_intermediates=True)
                extra_metrics["intermediates"] = intermediates
            else:
                (emb,) = self.encoder_transformer(emb)

        if emb.shape[-1] == 0:
            # Handle corner case where input is too short and the encoder returns an empty tensor
            z_q = emb
            commit_loss = torch.zeros(x.shape[0], device=x.device)
            codebook_loss = torch.zeros(x.shape[0], device=x.device)
            indices = torch.zeros((x.shape[0], 0), dtype=torch.long, device=x.device)
            z_e = emb
            extra_metrics["commit_loss"] = commit_loss
            extra_metrics["codebook_loss"] = codebook_loss
            return z_q, extra_metrics

        z_q, commit_loss, codebook_loss, indices, z_e = self.quantizer(emb)
        extra_metrics["commit_loss"] = commit_loss.mean()
        extra_metrics["codebook_loss"] = codebook_loss.mean()
        extra_metrics["indices"] = indices
        extra_metrics["z_e"] = z_e
        extra_metrics["emb"] = emb
        return z_q, extra_metrics
    
    def get_speaker_feature(self, x: torch.Tensor) -> torch.Tensor:
        with self._context_for_encoder_decoder:
            emb = self.encoder(x)
            _, intermediates = self.encoder_transformer(emb, return_intermediates=True)
        return intermediates
    
    def _encode_to_unquantized_latent(self, x: torch.Tensor, tr: int = 0) -> torch.Tensor:
        """Projects a batch of waveforms to unquantized latent space.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T].

        Returns:
            Unquantized embeddings.
        """
        assert (
            x.dim() == 3
        ), f"CompressionModel._encode_to_unquantized_latent expects audio of shape [B, C, T] but got {x.shape}"
        state = self._streaming_state
        with self._context_for_encoder_decoder:
            emb = self.encoder(x)
        if self.encoder_transformer is not None:
            if state is None:
                (emb,) = self.encoder_transformer(emb, tr)
            else:
                assert state.graphed_tr_enc is not None
                (emb,) = state.graphed_tr_enc(emb, tr)
        return emb

    def encode(self, x: torch.Tensor, tr: int = 0) -> torch.Tensor:
        """Encode the given input tensor to quantized representation.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T]

        Returns:
            codes (torch.Tensor): an int tensor of shape [B, T]
                with T the timestep.
        """
        emb = self._encode_to_unquantized_latent(x, tr=tr)
        z_q, commit_loss, codebook_loss, indices, z_e = self.quantizer(emb)
        return indices

    def encode_to_latent(self, x: torch.Tensor, tr: int = 0, quantize: bool = True) -> torch.Tensor:
        """Projects a batch of waveforms to latent space.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T].

        Returns:
            Embeddings, either quantized or not.
        """
        emb = self._encode_to_unquantized_latent(x, tr=tr)
        if not quantize:
            return emb

        if emb.shape[-1] == 0:
            # Handle corner case where input is too short and the encoder returns an empty tensor
            return emb
        z_q, commit_loss, codebook_loss, indices, z_e = self.quantizer(emb)
        return z_q

    def decode_latent(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode from the discrete codes to continuous latent space."""
        return self.quantizer.vq2emb(codes)