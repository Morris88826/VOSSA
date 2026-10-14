"""Streaming sessions on top of `VossaStreamingStep`.

    streamer = VossaStreamer(model)                    # once per model
    session = streamer.session(target_audio_16k)       # once per stream / speaker
    for block in microphone:                           # any block size, any sample rate
        play(session.push(block))
    play(session.flush())

Each session owns its own state tensors, so any number of sessions can share one model.
"""

import threading
import typing as tp

import numpy as np
import torch

from src.streaming.step import VossaStreamingStep
from src.utils.compile import CUDAGraphed, no_compile


class PyTorchBackend:
    """Runs the step in PyTorch, CUDA Graphed on GPU.

    The CUDA graph has a single set of static input/output buffers, so calls are
    serialized with a lock and outputs are cloned before the lock is released.
    """

    def __init__(self, step: VossaStreamingStep, use_cuda_graph: bool = True):
        self.step = step
        self.device = next(step.parameters()).device
        self._lock = threading.Lock()
        self._fn = CUDAGraphed(step, disable=not use_cuda_graph or self.device.type != "cuda")

    def initial_state(self) -> tp.Tuple[torch.Tensor, ...]:
        return self.step.initial_state(batch_size=1)

    @torch.no_grad()
    def __call__(self, chunk: torch.Tensor, speaker: tp.Sequence[torch.Tensor], state: tp.Sequence[torch.Tensor]):
        with self._lock, no_compile():
            out = tuple(t.clone() for t in self._fn(chunk, *speaker, *state))
        return out[0], out[1], out[2:]


class StreamingSession:
    """Converts one audio stream to the target speaker, chunk by chunk.

    Args:
        backend: shared backend running the streaming step.
        speaker: `(spk_k, spk_v, spk_g)` from `VossaStreamingStep.prepare_speaker`.
        sample_rate: sample rate of the audio given to `push`. Output is always at the
            model sample rate.
    """

    def __init__(self, backend: PyTorchBackend, speaker: tp.Sequence[torch.Tensor], sample_rate: int, model_sample_rate: int):
        self.backend = backend
        self.speaker = tuple(speaker)
        self.sample_rate = sample_rate
        self.model_sample_rate = model_sample_rate
        self.chunk_samples = backend.step.chunk_samples
        self.lag_chunks = backend.step.lag_chunks
        self.reset()

    def reset(self) -> None:
        """Start a new stream with the same speaker."""
        self._state = self.backend.initial_state()
        self._pending = np.zeros(0, dtype=np.float32)
        self._samples_in = 0
        self._samples_out = 0
        self._resampler = None
        if self.sample_rate != self.model_sample_rate:
            import soxr

            self._resampler = soxr.ResampleStream(self.sample_rate, self.model_sample_rate, 1, dtype="float32")

    def close(self) -> None:
        self._state = None
        self._pending = None

    def push(self, audio: tp.Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Feeds a block of mono audio of any length, returns the converted audio available so far."""
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if self._resampler is not None:
            audio = self._resampler.resample_chunk(audio)
        return self._consume(audio)

    def flush(self) -> np.ndarray:
        """Ends the stream: returns the remaining audio, so that the total output has the
        same length as the total input (at the model sample rate). The session is reset."""
        tail = np.zeros(0, dtype=np.float32)
        if self._resampler is not None:
            tail = self._resampler.resample_chunk(tail, last=True)
        self._samples_in += len(tail)
        pending = np.concatenate([self._pending, tail])
        padded = -(-len(pending) // self.chunk_samples) * self.chunk_samples
        # Extra zero chunks let the lookahead window drain the last real frames.
        padded += self.lag_chunks * self.chunk_samples
        self._pending = np.pad(pending, (0, padded - len(pending)))
        out = self._run_pending()
        out = out[: max(0, self._samples_in - self._samples_out)]
        self.reset()
        return out

    def _consume(self, audio: np.ndarray) -> np.ndarray:
        self._samples_in += len(audio)
        self._pending = np.concatenate([self._pending, audio])
        out = self._run_pending()
        self._samples_out += len(out)
        return out

    def _run_pending(self) -> np.ndarray:
        outs = []
        n_chunks = len(self._pending) // self.chunk_samples
        for i in range(n_chunks):
            block = self._pending[i * self.chunk_samples:(i + 1) * self.chunk_samples]
            chunk = torch.from_numpy(block).to(self.backend.device).view(1, 1, -1)
            audio, ready, self._state = self.backend(chunk, self.speaker, self._state)
            if ready.item():
                outs.append(audio.view(-1).cpu().numpy())
        self._pending = self._pending[n_chunks * self.chunk_samples:]
        return np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)


class VossaStreamer:
    """Entry point: wraps a trained `VOSSASynthesizer` for streaming conversion.

    Args:
        model: the synthesizer, already on its target device.
        chunk_frames: latent frames per step (5 = 100 ms at 16 kHz).
        use_cuda_graph: CUDA Graph the step when running on GPU.
    """

    def __init__(self, model, chunk_frames: int = 5, lookahead = None, use_cuda_graph: bool = True):
        self.model = model.eval()
        self.step = VossaStreamingStep(model, chunk_frames=chunk_frames, lookahead=lookahead)
        self.backend = PyTorchBackend(self.step, use_cuda_graph=use_cuda_graph)
        self.sample_rate = model.hparams.sample_rate

    @torch.no_grad()
    def speaker_embedding(self, target_audio: tp.Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Global speaker embedding `[1, g_dim]` from reference audio at the model sample rate."""
        audio = torch.as_tensor(np.asarray(target_audio, dtype=np.float32)).view(1, 1, -1)
        return self.model.get_speaker_embeddings(audio.to(self.backend.device))[0]

    def session(self, target: tp.Union[np.ndarray, torch.Tensor], sample_rate: tp.Optional[int] = None) -> StreamingSession:
        """Opens a session converting to the speaker of `target`.

        Args:
            target: reference audio of the target speaker (1D, model sample rate), or a
                precomputed global speaker embedding `[1, g_dim]`.
            sample_rate: sample rate of the audio that will be pushed (default: model rate).
        """
        if isinstance(target, torch.Tensor) and target.dim() == 2:
            embed = target.to(self.backend.device)
        else:
            embed = self.speaker_embedding(target)
        speaker = self.step.prepare_speaker(embed)
        return StreamingSession(self.backend, speaker, sample_rate or self.sample_rate, self.sample_rate)

    def warmup(self) -> None:
        """Runs a few silent steps so CUDA Graph capture does not hit the first real stream."""
        session = self.session(torch.zeros(1, self.model.tvtimbre_encoder.gtm.net[0].in_features, device=self.backend.device))
        session.push(np.zeros(3 * self.step.chunk_samples, dtype=np.float32))
        session.close()
