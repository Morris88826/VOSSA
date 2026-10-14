# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import glob
from pathlib import Path
import numpy as np
from sklearn.cluster import MiniBatchKMeans

import joblib

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("learn_kmeans")


def get_km_model(
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
):
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=1,
        compute_labels=False,
        tol=tol,
        max_no_improvement=max_no_improvement,
        init_size=None,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
    )


def load_feature_shard(feat_dir, split, nshard, rank, percent):
    feat_path = f"{feat_dir}/{split}_{rank}_{nshard}.npy"
    leng_path = f"{feat_dir}/{split}_{rank}_{nshard}.len"
    with open(leng_path, "r") as f:
        lengs = [int(line.rstrip()) for line in f]
        offsets = [0] + np.cumsum(lengs[:-1]).tolist()

    if percent < 0:
        return np.load(feat_path, mmap_mode="r")
    else:
        nsample = int(np.ceil(len(lengs) * percent))
        indices = np.random.choice(len(lengs), nsample, replace=False)
        feat = np.load(feat_path, mmap_mode="r")
        sampled_feat = np.concatenate(
            [feat[offsets[i]: offsets[i] + lengs[i]] for i in indices], axis=0
        )
        logger.info(
            (
                f"sampled {nsample} utterances, {len(sampled_feat)} frames "
                f"from shard {rank}/{nshard}"
            )
        )
        return sampled_feat


def load_feature(feat_dir, split, nshard, seed, percent):
    assert percent <= 1.0
    feat = np.concatenate(
        [
            load_feature_shard(feat_dir, split, nshard, r, percent)
            for r in range(nshard)
        ],
        axis=0,
    )
    logging.info(f"loaded feature with dimension {feat.shape}")
    return feat


def learn_kmeans(
    feat_dir,
    split,
    nshard,
    km_path,
    n_clusters,
    seed,
    percent,
    init,
    max_iter,
    batch_size,
    tol,
    n_init,
    reassignment_ratio,
    max_no_improvement,
):
    np.random.seed(seed)
    feat = load_feature(feat_dir, split, nshard, seed, percent)
    km_model = get_km_model(
        n_clusters,
        init,
        max_iter,
        batch_size,
        tol,
        max_no_improvement,
        n_init,
        reassignment_ratio,
    )
    km_model.fit(feat)
    joblib.dump(km_model, km_path)

    inertia = -km_model.score(feat) / len(feat)
    logger.info("total intertia: %.5f", inertia)
    logger.info("finished successfully")


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("feat_dir", type=str)
    # parser.add_argument("split", type=str)
    # parser.add_argument("nshard", type=int)
    # parser.add_argument("km_path", type=str)
    # parser.add_argument("n_clusters", type=int)
    # parser.add_argument("--seed", default=0, type=int)
    # parser.add_argument(
    #     "--percent", default=-1, type=float, help="sample a subset; -1 for all"
    # )
    # parser.add_argument("--init", default="k-means++")
    # parser.add_argument("--max_iter", default=100, type=int)
    # parser.add_argument("--batch_size", default=10000, type=int)
    # parser.add_argument("--tol", default=0.0, type=float)
    # parser.add_argument("--max_no_improvement", default=100, type=int)
    # parser.add_argument("--n_init", default=20, type=int)
    # parser.add_argument("--reassignment_ratio", default=0.0, type=float)
    # args = parser.parse_args()
    # logging.info(str(args))

    # learn_kmeans(**vars(args))

    split = "train"
    nshard = 1
    # n_clusters = 256
    seed = 1234
    percent = -1
    init = "k-means++"
    max_iter = 100
    batch_size = 10000
    tol = 0.0
    max_no_improvement = 100
    n_init = 20
    reassignment_ratio = 0.0
    base_dir = Path("content_feats")
    
    n_clusters = 256
    files_npy = glob.glob(os.path.join(base_dir, "*_140ms_la_cl/train_*.npy"))
    error_files = []
    for file in files_npy:
        try:
            file = Path(file)
            logger.info(f"Processing {file.parent.name}, for n_clusters: {n_clusters}")
            feat_dir = file.parent
            km_path = (file.parent / f"km{n_clusters}_model.bin")
            if km_path.exists():
                logger.info(f"Model already exists: {km_path}")
                continue
            else:
                logger.info(f"Creating model: {km_path}")
            learn_kmeans(
                feat_dir=feat_dir,
                split=split,
                nshard=nshard,
                km_path=km_path,
                n_clusters=n_clusters,
                seed=seed,
                percent=percent,
                init=init,
                max_iter=max_iter,
                batch_size=batch_size,
                tol=tol,
                n_init=n_init,
                reassignment_ratio=reassignment_ratio,
                max_no_improvement=max_no_improvement,
            )
        except Exception as e:
            logger.error(f"Error processing {file.parent.name}: {e}")
            error_files.append(file.parent.name)
            continue

    logger.info("All files processed successfully for n_clusters: %d", n_clusters)
    logger.info(f"Error files: {error_files}")
    logger.info("All files processed successfully")

