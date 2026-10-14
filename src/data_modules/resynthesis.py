# Adapted from: https://github.com/ashleve/lightning-hydra-template/blob/main/src/data/mnist_datamodule.py
import glob
import torch
import random
import librosa
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from typing import Any
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import random, numpy as np, torch

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load_audio(full_path, sampling_rate=16000):
    data, sampling_rate = librosa.load(full_path, sr = sampling_rate)
    return data, sampling_rate


class ResynthesisDataset(Dataset):
    def __init__(self, source_file_list, target_file_list, segment_size, code_hop_size, sampling_rate, subj2id):
        super().__init__()
        self.source_file_list = source_file_list
        self.target_file_list = target_file_list

        self.segment_size = segment_size
        self.code_hop_size = code_hop_size
        self.sampling_rate = sampling_rate
        # random.seed(1234)
        
        self.src_subj_list = defaultdict(list)
        for file in self.source_file_list:
            subj_id = file.parent.parent.name
            self.src_subj_list[subj_id].append(file)

        self.tgt_subj_list = defaultdict(list)
        for file in self.target_file_list:
            subj_id = file.parent.name
            self.tgt_subj_list[subj_id].append(file)
        self.window_range = [2, 4]

        # build subject list
        self.subj2id = subj2id
        
    def __len__(self):
        return len(self.source_file_list)

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
        
    def _sample_audio(self, audio, seq_len=None, window_range = [2, 4], shuffle=False):
        N = audio.shape[-1]
        if seq_len is None:
            seq_len = self.segment_size if self.segment_size > 0 else N
        
        window_size = int((random.random() * (window_range[1] - window_range[0]) + window_range[0]) * seq_len)
        window_size = min(window_size, N)
        
        # # Randomly pickup with the batch_max_steps length of the part
        interval_start = 0
        interval_end = N - window_size

        start_step = random.randint(interval_start, interval_end)
        audio = audio[start_step:start_step + window_size]
        audio = self._pad_audio(audio)[0][:self.segment_size]
        
        # Shuffle by hop_size chunks
        if shuffle:
            hop_size = self.code_hop_size
            chunks = []
            for i in range(0, audio.shape[0], hop_size):
                chunks.append(audio[i:i+hop_size])
            random.shuffle(chunks)
            audio = np.hstack(chunks)
        return audio
        
    def _pad_audio(self, audio, **kwargs):
        while audio.shape[0] < self.segment_size:
            audio = np.hstack([audio, audio])
            for key in kwargs:
                kwargs[key] = np.hstack([kwargs[key], kwargs[key]])
        return audio, kwargs

    def __getitem__(self, index):
        src_wav_fpath = self.source_file_list[index]
        src_subj_id = src_wav_fpath.parent.parent.name

        tgt_subj_path = random.choice(self.target_file_list)
        tgt_subj_id = tgt_subj_path.parent.name
        tgt_wav_paths = list(glob.glob(str(Path(tgt_subj_path) / "*.wav")))
        random.shuffle(tgt_wav_paths)

        src_dataset_name = src_wav_fpath.parent.parent.parent.parent.name

        positive_wav_fpath = random.choice(self.src_subj_list[src_subj_id])
        while positive_wav_fpath == src_wav_fpath and len(self.src_subj_list[src_subj_id]) > 1:
            positive_wav_fpath = random.choice(self.src_subj_list[src_subj_id])
        
        if len(tgt_wav_paths) == 1:
            tgt_wav_fpath = Path(tgt_wav_paths[0])
            negative_wav_fpath = Path(tgt_wav_paths[0])
        else:
            tgt_wav_fpath = Path(tgt_wav_paths[0])
            negative_wav_fpath = Path(tgt_wav_paths[1])
        tgt_dataset_name = tgt_wav_fpath.parent.parent.parent.parent.name

        pitch = src_wav_fpath.parent.parent / "pitch" / f"{src_wav_fpath.stem}.pit.npy"
        energy = src_wav_fpath.parent.parent / "energy" / f"{src_wav_fpath.stem}.eng.npy"
        tokens = src_wav_fpath.parent.parent / "code" / f"{src_wav_fpath.stem}.km"
        tokens = open(tokens, 'r').readlines()[0].strip()
        tokens = np.array([int(i) for i in tokens.split(" ")])
        pitch = np.load(pitch)
        energy = np.load(energy)

        src_audio, _ = load_audio(src_wav_fpath, self.sampling_rate)
        src_wav_a = src_audio.copy()
        src_wav_b, _ = load_audio(positive_wav_fpath, self.sampling_rate)
        tgt_wav_a, _ = load_audio(tgt_wav_fpath, self.sampling_rate)
        tgt_wav_b, _ = load_audio(negative_wav_fpath, self.sampling_rate)
        

        # Trim audio ending
        code_length = min(src_audio.shape[0] // self.code_hop_size, tokens.shape[-1])
        tokens = tokens[:code_length]
        pitch = pitch[:code_length]
        energy = energy[:code_length]
        tokens = tokens[:code_length]        
        src_audio = src_audio[:code_length * self.code_hop_size]
        
        assert src_audio.shape[0] // self.code_hop_size == tokens.shape[-1], "Code audio mismatch"

        while src_audio.shape[0] < self.segment_size:
            src_audio = np.hstack([src_audio, src_audio])
            tokens = np.hstack([tokens, tokens])
            pitch = np.hstack([pitch, pitch])
            energy = np.hstack([energy, energy])
            
        src_audio, kwargs = self._pad_audio(src_audio, pitch=pitch, energy=energy, tokens=tokens)
        pitch, energy, tokens = kwargs['pitch'], kwargs['energy'], kwargs['tokens']

        src_audio = torch.FloatTensor(src_audio)
        src_audio = src_audio.unsqueeze(0)
        pitch = torch.FloatTensor(pitch)
        energy = torch.FloatTensor(energy)
        tokens = torch.tensor(tokens)

        assert src_audio.size(1) >= self.segment_size, "Padding not supported!!"
        
        src_audio, pitch, energy, tokens = self._sample_interval([src_audio, pitch, energy, tokens])
        
        # get target audio
        src_wav_a = self._sample_audio(src_wav_a, seq_len=self.segment_size, window_range=[2, 4])
        src_wav_b = self._sample_audio(src_wav_b, seq_len=self.segment_size, window_range=[1, 3]) # For contrastive learning
        tgt_wav_a = self._sample_audio(tgt_wav_a, seq_len=self.segment_size, window_range=[2, 4])
        tgt_wav_b = self._sample_audio(tgt_wav_b, seq_len=self.segment_size, window_range=[1, 3]) # For contrastive learning

        return src_audio.squeeze(0), (src_wav_a, src_wav_b, tgt_wav_a, tgt_wav_b), tokens, pitch, energy, (self.subj2id[src_dataset_name + "_" + str(src_subj_id)], self.subj2id[tgt_dataset_name + "_" + str(tgt_subj_id)], str(src_wav_fpath), str(tgt_wav_fpath))

    def __len__(self):
        return len(self.source_file_list)


class ResynthesisDataModule(pl.LightningDataModule):
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
        source_data_dir: str,
        target_data_dir, # list of str
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        segment_size: int = 48000,
        code_hop_size: int = 320,
        sampling_rate: int = 16000,
        n_speakers: int = 9701,
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

        self.training_files = None
        self.validation_files = None
        self.test_files = None
        self.target_training_subjects = None
        self.target_validation_subjects = None
        self.target_test_subjects = None

    def prepare_data(self):
        pass
            
    def setup(self, stage: str = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        if self.training_files is None:
            """Download data if needed.
            Do not use it to assign state (self.x = y).
            """
            subj_list = []
            src_dataset_name = Path(self.hparams.source_data_dir).parent.name
            src_metadata_df = pd.read_csv(Path(self.hparams.source_data_dir).parent / "metadata" / "speakers.csv")
            src_metadata_df = src_metadata_df[src_metadata_df['SUBSET'].isin(['train-clean-100', 'train-clean-360', 'train-other-500'])]

            training_files = []
            for _, row in src_metadata_df.iterrows():
                speaker_id = row['READER']
                speaker_dir = Path(self.hparams.source_data_dir) / str(speaker_id) / "wav"
                training_files += list(speaker_dir.rglob("*.wav"))
                subj_list.append(src_dataset_name + "_" + str(speaker_id))
            self.training_files = training_files
            training_files, self.validation_files = train_test_split(training_files, test_size=0.01, random_state=42)
            self.training_files, self.test_files = train_test_split(training_files, test_size=0.001, random_state=42)

            target_training_subjects = []
            self.target_test_subjects = []
            for target_data_dir in self.hparams.target_data_dir:
                target_dataset_name = Path(target_data_dir).parent.name
                tgt_metadata_df = pd.read_csv(Path(target_data_dir).parent / "metadata" / "speakers.csv")
                for _, row in tgt_metadata_df[tgt_metadata_df['SUBSET']=='train'].iterrows():
                    target_training_subjects.append(Path(target_data_dir) / row['READER'] / "wav")
                    subj_list.append(target_dataset_name + "_" + str(row['READER']))

                for _, row in tgt_metadata_df[tgt_metadata_df['SUBSET']=='test'].iterrows():
                    self.target_test_subjects.append(Path(target_data_dir) / row['READER'] / "wav") 
                    subj_list.append(target_dataset_name + "_" + str(row['READER']))

            self.target_training_subjects, self.target_validation_subjects = train_test_split(target_training_subjects, test_size=0.2, random_state=42)
            
            self.all_subj_list = subj_list
            assert len(self.all_subj_list) == self.hparams.n_speakers, "Number of speakers mismatch!"
            self.subj2id = {}
            for sid, subj_name in enumerate(self.all_subj_list):
                if subj_name in self.subj2id:
                    raise ValueError(f"Duplicate subject name: {subj_name}")
                self.subj2id[subj_name] = sid
            
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = ResynthesisDataset(
                source_file_list=self.training_files,
                target_file_list=self.target_training_subjects,
                segment_size=self.hparams.segment_size,
                code_hop_size=self.hparams.code_hop_size,
                sampling_rate=self.hparams.sampling_rate,
                subj2id=self.subj2id
            )
            self.data_val = ResynthesisDataset(
                source_file_list=self.validation_files,
                target_file_list=self.target_validation_subjects,
                segment_size=self.hparams.segment_size,
                code_hop_size=self.hparams.code_hop_size,
                sampling_rate=self.hparams.sampling_rate,
                subj2id=self.subj2id
            )
            self.data_test = ResynthesisDataset(
                source_file_list=self.test_files,
                target_file_list=self.target_test_subjects,
                segment_size=self.hparams.segment_size,
                code_hop_size=self.hparams.code_hop_size,
                sampling_rate=self.hparams.sampling_rate,
                subj2id=self.subj2id
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            worker_init_fn=seed_worker,
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
    dm = ResynthesisDataModule(source_data_dir="/data/mtseng/voice_datasets/LibriTTS/data", target_data_dir=["/data/mtseng/voice_datasets/voxceleb1/data"])
    dm.prepare_data()
    dm.setup()
    for batch in dm.train_dataloader():
        print(batch[0].shape)
        print(batch[1].shape)
        break
