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


class DecoderAbsModel(StreamingModule[State]):
    """Base API for all compression model that aim at being used as audio tokenizers
    with a language model.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tp.Any: ...


    @abstractmethod
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """See `MimiModel.decode`."""
        ...


    @property
    @abstractmethod
    def frame_rate(self) -> float: ...

    @property
    @abstractmethod
    def sample_rate(self) -> int: ...


@dataclass
class _MimiVocoderState:
    graphed_tr_dec: CUDAGraphed | None

    def reset(self):
        pass


class MimiVocoderModel(DecoderAbsModel[_MimiVocoderState]):
    """Mimi model operating on the raw waveform.

    Args:
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        quantizer (qt.BaseQuantizer): Quantizer network.
        frame_rate (float): Final frame rate of the quantized representatiopn.
        encoder_frame_rate (float): frame rate of the encoder model. Note that if `frame_rate != encopder_frame_rate`,
            the latent will be resampled linearly to match the desired `frame_rate` before and after quantization.
        sample_rate (int): Audio sample rate.
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
        decoder: nn.Module,
        frame_rate: float,
        sample_rate: int,
        causal: bool = False,
        decoder_transformer: tp.Optional[nn.Module] = None,
        torch_compile_encoder_decoder: bool = False,
    ):
        super().__init__()
        self.decoder = decoder
        self.decoder_transformer = decoder_transformer
        self._frame_rate = frame_rate
        self._sample_rate = sample_rate
        self._hop_length = int(sample_rate / frame_rate)
        self.torch_compile_encoder_decoder = torch_compile_encoder_decoder

    def _init_streaming_state(self, batch_size: int) -> _MimiVocoderState:
        device = next(self.parameters()).device
        disable = device.type != 'cuda'
        graphed_tr_dec = None
        if self.decoder_transformer is not None:
            graphed_tr_dec = CUDAGraphed(self.decoder_transformer, disable=disable)
        return _MimiVocoderState(graphed_tr_dec)


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

    def forward(
            self, 
            x: torch.Tensor
        ) -> tp.Any:
        assert x.dim() == 3
        length = x.shape[-1] * self._hop_length
        extra_metrics: tp.Dict[str, torch.Tensor] = {}

        if self.decoder_transformer is not None:
            (emb,) = self.decoder_transformer(x)

        with self._context_for_encoder_decoder:
            out = self.decoder(emb)

        # remove extra padding added by the encoder and decoder
        assert out.shape[-1] >= length, (out.shape[-1], length)
        out = out[..., :length]

        return out

    def decode(
            self, 
            x: torch.Tensor
        ):
        """Decode the given codes to a reconstructed representation.

        Args:
            emb (torch.Tensor): Int tensor of shape [B, K, T]

        Returns:
            out (torch.Tensor): Float tensor of shape [B, C, T], the reconstructed audio.
        """
        state = self._streaming_state
        if self.decoder_transformer is not None:
            if state is None:
                (emb,) = self.decoder_transformer(x)
            else:
                assert state.graphed_tr_dec is not None
                (emb,) = state.graphed_tr_dec(x)
        with self._context_for_encoder_decoder:
            out = self.decoder(emb)
        # out contains extra padding added by the encoder and decoder
        return out