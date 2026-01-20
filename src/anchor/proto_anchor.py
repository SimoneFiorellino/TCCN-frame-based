#!/usr/bin/env python3
"""
Proto Anchor generation script.

This script iterates over all datasets defined in the chosen configuration,
selects anchors by K‑means clustering and uniform sampling, and saves a
dictionary mapping each cluster to the chosen prototype indices and labels.

For reproducibility it cycles through the tuple SEEDS provided by the
configuration and writes one file per seed in the form
    <LATENTS_DIR>/<encoder>_frames_<n_cluster>_seed<seed>.pt
"""

import argparse
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

# -------------------------- CLI ---------------------------
parser = argparse.ArgumentParser(
    description="Run Proto Anchor selection with different config types"
)
parser.add_argument(
    "--config_type",
    choices=[
        "extension",
        "compression",
        "complete",
        "q",
        "model_types",
        "baselines",
        "proto",
    ],
    default="baselines",
    help="Which configuration to use",
)
args = parser.parse_args()
config_type = args.config_type

# ----------------------- Config import --------------------
if config_type == "compression":
    import configs.experiment_config_compression as cfg_module
elif config_type == "complete":
    import configs.experiment_config_complete as cfg_module
elif config_type == "q":
    import configs.experiment_config_q as cfg_module
elif config_type == "model_types":
    import configs.experiment_config_model_types as cfg_module
elif config_type == "baselines":
    import configs.experiment_config_baselines as cfg_module
elif config_type == "proto":
    import configs.experiment_config_proto as cfg_module
else:
    raise ValueError(
        "Invalid config type. Choose 'base', 'extension', or 'compression'."
    )

# --------------------- Main routine -----------------------

for dataset in cfg_module.DATASETS:
    cfg = cfg_module.get_config(dataset)
    LATENTS_DIR: Path = cfg["LATENTS_DIR"]

    for encoder in cfg_module.ENCODER_RX:
        # -------- Load latents --------
        dataset_path = LATENTS_DIR / "train" / f"{encoder}.pt"
        data = torch.load(dataset_path)
        anch_latents = F.normalize(data["anchors_latents"], p=2, dim=1)
        anch_labels = data["anchor_labels"]

        # -------- Cluster sizes --------
        for n_cluster in cfg["N_CLUSTERS"]:
            if n_cluster >= 800:
                continue

            for seed in cfg_module.SEEDS:
                # Fixed seeds for reproducibility
                np.random.seed(seed)
                torch.manual_seed(seed)

                out_file = (
                    LATENTS_DIR / f"{encoder}_{n_cluster}_{cfg_module.SxC}_{seed}.pt"
                )

                if out_file.exists() and cfg_module.check_anchors:
                    print(f"[{dataset}|{encoder}] Skipping existing {out_file.name}")
                    continue

                # Ensure directory exists
                out_file.parent.mkdir(parents=True, exist_ok=True)

                # ---------- K‑means ----------
                kmeans = KMeans(
                    n_clusters=n_cluster,
                    random_state=seed,
                    n_init="auto",
                ).fit(anch_latents)
                labels = kmeans.labels_

                # ---------- Sample prototypes ----------
                n_samples = cfg_module.SxC
                indices = {}
                for i in range(n_cluster):
                    cluster_idx = np.where(labels == i)[0]
                    n_available = len(cluster_idx)

                    n_select = min(n_samples, n_available)
                    indices[i] = np.random.choice(cluster_idx, n_select, replace=False)

                proto_anchors = {
                    i: {"idx": idxs, "labels": anch_labels[idxs]}
                    for i, idxs in indices.items()
                }

                # ---------- Save ----------
                torch.save(proto_anchors, out_file)
                print(f"[{dataset}|{encoder}] Saved anchors to {out_file.name}")
