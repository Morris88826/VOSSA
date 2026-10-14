# Data preparation

VOSSA trains on LibriTTS (source speakers) and VoxCeleb1/2 (target speakers). Inference only needs the pretrained checkpoints, not this data.

## 1. Preprocess the source corpus

The per-utterance features follow the same pipeline as [StreamVC](https://github.com/PSI-TAMU/streamVC) (also used by DarkStream). Follow the preprocessing instructions there to produce, for every LibriTTS utterance:

| File | Contents |
|---|---|
| `wav/<utt>.wav` | mono audio (resampled to 16 kHz when loaded) |
| `pitch/<utt>.pit.npy` | frame-level pitch, one value per 320-sample frame (50 Hz) |
| `energy/<utt>.eng.npy` | frame-level energy, one value per 320-sample frame (50 Hz) |
| `code/<utt>.km` | HuBERT k-means units (200 clusters, 50 Hz), one line of space-separated integers |

VoxCeleb only needs `wav/`.

## 2. Add speaker metadata

Each corpus needs `metadata/speakers.csv`, with one row per speaker:

| Corpus | Columns used | `SUBSET` values used |
|---|---|---|
| LibriTTS | `READER`, `SUBSET` | `train-clean-100`, `train-clean-360`, `train-other-500` (training speakers) |
| VoxCeleb1, VoxCeleb2 | `READER`, `SUBSET` | `train`, `test` |

Other columns (e.g. `GENDER`, `NAME`) are ignored.

## 3. Layout

```
<corpus>/
├── metadata/
│   └── speakers.csv
└── data/
    └── <speaker>/          # matches READER in speakers.csv
        ├── wav/
        ├── pitch/          # source corpus only
        ├── energy/         # source corpus only
        └── code/           # source corpus only
```

## 4. Point the config at the data

In `configs/data/resynthesis.yaml`:

```yaml
source_data_dir: <path>/LibriTTS/data
target_data_dir:
  - <path>/voxceleb1/data
  - <path>/voxceleb2/data
```

`n_speakers` (default 9701) must equal the number of LibriTTS training speakers plus all VoxCeleb speakers listed in the metadata. The released model used 2,338 + 1,251 + 6,112 = 9,701.
