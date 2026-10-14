"""Checks the stateless streaming step against the module-state streaming reference.

    python -m src.streaming.check_parity [--ckpt ...] [--vocoder ...]
"""

import argparse
import contextlib
import time

import librosa
import numpy as np
import torch

from src.modules import VOSSASynthesizer
from src.streaming import VossaStreamer


@torch.no_grad()
def reference_streaming(model, src: torch.Tensor, global_embed: torch.Tensor, chunk: int = 5, lookahead: int = 4) -> np.ndarray:
    """Streaming with module-held state (`.streaming()` contexts), as in notebooks/inference_streaming.ipynb."""
    enc = model.encoder.net
    stride = chunk * 320
    n = -(-src.shape[-1] // stride) + -(-lookahead * 320 // stride)
    x = torch.nn.functional.pad(src.view(1, 1, -1), (0, n * stride - src.shape[-1]))
    outs, stack, ptr = [], torch.empty(1, enc.dimension, 0, device=src.device), 0
    with contextlib.ExitStack() as es:
        for mod in (enc, model.tvtimbre_encoder, model.vocoder.decoder, model.decoder):
            es.enter_context(mod.streaming(1))
        for i in range(n):
            with enc._context_for_encoder_decoder:
                stack = torch.cat([stack, enc.encoder(x[..., i * stride:(i + 1) * stride])], -1)
            while stack.shape[-1] - ptr >= lookahead + chunk:
                window = stack[..., ptr:ptr + lookahead + chunk]
                ptr += chunk
                (emb,) = enc._streaming_state.graphed_tr_enc(window, lookahead)
                z = enc.quantizer(emb)[0][..., :chunk]
                outs.append(model.vocoder(model.decode(z, global_embed, None)))
    return torch.cat(outs, -1).view(-1)[: src.shape[-1]].cpu().numpy()


def rel(a: np.ndarray, b: np.ndarray) -> float:
    assert a.shape == b.shape, (a.shape, b.shape)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def run(session, audio: np.ndarray, block_sizes=None) -> np.ndarray:
    if block_sizes is None:
        return np.concatenate([session.push(audio), session.flush()])
    outs, i, rng = [], 0, np.random.default_rng(0)
    while i < len(audio):
        n = int(rng.choice(block_sizes))
        outs.append(session.push(audio[i:i + n]))
        i += n
    outs.append(session.flush())
    return np.concatenate(outs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/vossa_epoch=52-val_loss=0.6670.ckpt")
    parser.add_argument("--vocoder", default="checkpoints/mimivocoder.ckpt")
    parser.add_argument("--src", default="samples/14_208_000001_000000.wav")
    parser.add_argument("--tgt", default="samples/arctic_a0002.wav")
    args = parser.parse_args()

    # Exact comparisons need full fp32 matmuls; TF32 alone gives ~1e-3 differences.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VOSSASynthesizer.load_from_checkpoint(
        args.ckpt, encoder_path=None, vocoder_path=args.vocoder, map_location=device, strict=False
    ).eval()

    src, _ = librosa.load(args.src, sr=16000)
    tgt, _ = librosa.load(args.tgt, sr=16000)
    streamer = VossaStreamer(model)
    embed = streamer.speaker_embedding(tgt)

    ref = reference_streaming(model, torch.from_numpy(src).to(device), embed)
    out = run(streamer.session(embed), src)
    print(f"[1] step vs module-state reference      : rel err {rel(out, ref):.2e}  (len {len(out)} / {len(src)})")

    blocks = run(streamer.session(embed), src, block_sizes=[1, 37, 320, 1599, 1601, 4000])
    print(f"[2] random block sizes vs one block      : max abs diff {np.abs(blocks - out).max():.2e}")

    eager = run(VossaStreamer(model, use_cuda_graph=False).session(embed), src)
    print(f"[3] CUDA graph vs eager                  : rel err {rel(out, eager):.2e}")

    # Two interleaved sessions with different speakers must not interfere.
    other = streamer.speaker_embedding(src)
    s1, s2 = streamer.session(embed), streamer.session(other)
    a, b = [], []
    for i in range(0, len(src), 1000):
        a.append(s1.push(src[i:i + 1000]))
        b.append(s2.push(src[i:i + 1000]))
    a, b = np.concatenate(a + [s1.flush()]), np.concatenate(b + [s2.flush()])
    alone = run(streamer.session(other), src)
    print(f"[4] interleaved sessions vs alone        : max abs diff {max(np.abs(a - out).max(), np.abs(b - alone).max()):.2e}")

    src44 = librosa.resample(src, orig_sr=16000, target_sr=44100)
    out44 = run(streamer.session(embed, sample_rate=44100), src44, block_sizes=[441, 1024])
    print(f"[5] 44.1 kHz input (resampled)           : len {len(out44)} (expected {round(len(src44) * 16000 / 44100)})")

    session = streamer.session(embed)
    chunk = src[:session.chunk_samples]
    for _ in range(5):
        session.push(chunk)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 50
    for _ in range(n):
        session.push(chunk)
    ms = (time.perf_counter() - t0) / n * 1000
    print(f"[6] time per 100 ms chunk (incl. CNN)    : {ms:.1f} ms  (RTF {ms / 100:.2f})")


if __name__ == "__main__":
    main()
