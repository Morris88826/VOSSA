# Adapted from: https://github.com/ashleve/lightning-hydra-template/blob/main/src/data/mnist_datamodule.py
from typing import Any
import random
from pathlib import Path
import librosa
import numpy as np
import torch
from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split


def load_audio(full_path, sampling_rate=16000):
    data, sampling_rate = librosa.load(full_path, sr = sampling_rate)
    return data, sampling_rate


class PasrDataset(Dataset):
    def __init__(
        self, 
        training_files,
        segment_size,
        code_hop_size, 
        sampling_rate, 
        split=True
    ):

        self.audio_files = training_files
        self.segment_size = segment_size
        self.code_hop_size = code_hop_size
        self.sampling_rate = sampling_rate
        self.split = split

        random.seed(1234)

    def _sample_interval(self, seqs, seq_len=None):
        N = max([v.shape[-1] for v in seqs])
        if seq_len is None:
            seq_len = self.segment_size if self.segment_size > 0 else N

        hops = [N // v.shape[-1] for v in seqs]
        lcm = np.lcm.reduce(hops)

        # Randomly pickup with the batch_max_steps length of the part
        interval_start = 0
        interval_end = N // lcm - seq_len // lcm

        start_step = random.randint(interval_start, interval_end)

        new_seqs = []
        for i, v in enumerate(seqs):
            start = start_step * (lcm // hops[i])
            end = (start_step + seq_len // lcm) * (lcm // hops[i])
            new_seqs += [v[..., start:end]]

        return new_seqs

    def __getitem__(self, index):
        wav_fpath = self.audio_files[index]
        tokens = wav_fpath.parent.parent / "code" / f"{wav_fpath.stem}.km"

        tokens = open(tokens, 'r').readlines()[0].strip()
        tokens = np.array([int(i) for i in tokens.split(" ")])

        audio, sampling_rate = load_audio(wav_fpath, self.sampling_rate)
        # if sampling_rate != self.sampling_rate:
        #     import resampy
        #     audio = resampy.resample(audio, sampling_rate, self.sampling_rate)

        # # audio = audio / MAX_WAV_VALUE
        # # audio = normalize(audio) * 0.95
        # audio = audio / (max(abs(audio)) + 0.001) * 0.95

        # Trim audio ending
        code_length = min(audio.shape[0] // self.code_hop_size, tokens.shape[-1])
        tokens = tokens[:code_length]
        
        audio = audio[:code_length * self.code_hop_size]
        
        assert audio.shape[0] // self.code_hop_size == tokens.shape[-1], "Code audio mismatch"

        while audio.shape[0] < self.segment_size:
            audio = np.hstack([audio, audio])
            tokens = np.hstack([tokens, tokens])

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        tokens = torch.from_numpy(tokens)

        assert audio.size(1) >= self.segment_size, "Padding not supported!!"
        
        audio, tokens = self._sample_interval([audio, tokens])

        return audio.squeeze(0), tokens, str(wav_fpath)

    def __len__(self):
        return len(self.audio_files)


class PasrDataModule(pl.LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.
    A DataModule implements 5 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test
    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        segment_size: int = 20480,
        code_hop_size: int = 320,
        sampling_rate: int = 16000,
        split: bool = True,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()

        # data transformations
        # self.transforms = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])

        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None

    @property
    def num_classes(self):
        return self.hparams.num_codes

    def prepare_data(self):
        """Download data if needed.
        Do not use it to assign state (self.x = y).
        """
        training_files = list(Path(self.hparams.data_dir).rglob("*.wav"))
        self.training_files = training_files
        # self.validation_files = training_files
        # self.test_files = training_files

        training_files, self.validation_files, _, _ = train_test_split(training_files, training_files, test_size=0.01, random_state=42)
        self.training_files, self.test_files, _, _ = train_test_split(training_files, training_files, test_size=0.001, random_state=42)

    def setup(self, stage: str = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = PasrDataset(
                training_files=self.training_files,
                segment_size=self.hparams.segment_size,
                code_hop_size=self.hparams.code_hop_size,
                sampling_rate=self.hparams.sampling_rate,
                split=True,
            )
            self.data_val = PasrDataset(
                training_files=self.validation_files,
                segment_size=self.hparams.segment_size,
                code_hop_size=self.hparams.code_hop_size,
                sampling_rate=self.hparams.sampling_rate,
                split=True,
            )
            self.data_test = PasrDataset(
                training_files=self.test_files,
                segment_size=self.hparams.segment_size,
                code_hop_size=self.hparams.code_hop_size,
                sampling_rate=self.hparams.sampling_rate,
                split=False,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: str = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    dm = PasrDataModule()
    dm.prepare_data()
    dm.setup()
    for batch in dm.train_dataloader():
        print(batch[0].shape)
        print(batch[1].shape)
        break
