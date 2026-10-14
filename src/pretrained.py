"""Loads the pretrained VOSSA model from the Hugging Face Hub (or a local folder).

    from src.pretrained import load_vossa
    model = load_vossa()                            # downloads and caches the checkpoints
    model = load_vossa(local_dir="checkpoints")     # uses files already on disk

The Hub repo holds `config.yaml` plus the checkpoints it names. `config.yaml` is fetched on
every load: the Hub counts requests to it as model downloads.
"""

import typing as tp
from pathlib import Path

import torch
import yaml
from huggingface_hub import hf_hub_download

from src.modules import VOSSASynthesizer

HF_REPO_ID = "morris88826/VOSSA"
CONFIG_FILE = "config.yaml"

# Used when a local folder has no config.yaml.
DEFAULT_FILES = {
    "checkpoint": "vossa_epoch=52-val_loss=0.6670.ckpt",
    "vocoder": "mimivocoder.ckpt",
}


def _resolve_files(repo_id: str, local_dir: tp.Optional[str], revision: tp.Optional[str]) -> tp.Dict[str, str]:
    if local_dir is not None:
        root = Path(local_dir)
        config_path = root / CONFIG_FILE
        files = yaml.safe_load(config_path.read_text()) if config_path.exists() else DEFAULT_FILES
        return {key: str(root / files[key]) for key in DEFAULT_FILES}

    config_path = hf_hub_download(repo_id, CONFIG_FILE, revision=revision)
    files = yaml.safe_load(Path(config_path).read_text())
    return {key: hf_hub_download(repo_id, files[key], revision=revision) for key in DEFAULT_FILES}


def load_vossa(
    repo_id: str = HF_REPO_ID,
    local_dir: tp.Optional[str] = None,
    device: tp.Union[str, torch.device, None] = None,
    revision: tp.Optional[str] = None,
) -> VOSSASynthesizer:
    """Returns the pretrained `VOSSASynthesizer` in eval mode.

    Args:
        repo_id: Hugging Face Hub repo holding `config.yaml` and the checkpoints.
        local_dir: if set, load from this folder instead of the Hub.
        device: defaults to CUDA when available.
        revision: Hub branch, tag or commit to load.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    files = _resolve_files(repo_id, local_dir, revision)
    model = VOSSASynthesizer.load_from_checkpoint(
        files["checkpoint"],
        encoder_path=None,
        vocoder_path=files["vocoder"],
        map_location=device,
    )
    return model.eval()
