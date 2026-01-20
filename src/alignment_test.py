from pathlib import Path
import argparse
import csv

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from src.modules.model import Decoder
from src.sim.similarities import relative_projection
from src.utils.pseudo_utils import (
    procrustes_alignment,
    linear_alignment,
    neural_alignment,
    uniform_quantize,
    white_noise,
    pseudo_inverse,
    frame_recon,
)
from src.utils.anchors_utils import get_anchors
from src.utils.func_utils import check_existing_result, update_results_csv

# ---------------------------------------------------------------------
# CLI & dynamic config import
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Run Proto Anchor experiments")
parser.add_argument(
    "--config_type",
    choices=[
        "base",
        "expansion",
        "compression",
        "complete",
        "q",
        "model_types",
        "baselines",
        "proto",
    ],
    default="base",
)
args = parser.parse_args()
config_type = args.config_type
save_individual = config_type == "complete"

if config_type == "baselines":
    import configs.experiment_config_baselines as cfg_module

    results_file = "results_baselines.csv"
elif config_type == "compression":
    import configs.experiment_config_compression as cfg_module

    results_file = "results_compression.csv"
elif config_type == "complete":
    import configs.experiment_config_complete as cfg_module

    results_file = "results_complete.csv"
elif config_type == "q":
    import configs.experiment_config_q as cfg_module

    results_file = "results_q.csv"
elif config_type == "model_types":
    import configs.experiment_config_model_types as cfg_module

    results_file = "results_model_types.csv"
elif config_type == "proto":
    import configs.experiment_config_proto as cfg_module

    results_file = "results_proto.csv"
else:
    import configs.experiment_config as cfg_module

    results_file = "results.csv"


# ---------------------------------------------------------------------
# Extract config variables
# ---------------------------------------------------------------------
DATASETS = cfg_module.DATASETS
ENCODERS_TX = cfg_module.ENCODERS_TX
N_CLUSTERS = cfg_module.N_CLUSTERS
ANCHORS_STRATEGIES = cfg_module.ANCHORS_STRATEGIES
SEEDS = cfg_module.SEEDS
SIMILS = cfg_module.SIMILS
SNR_DB = cfg_module.SNR_DB
QUANTIZATION = cfg_module.QUANTIZATION
SxC = cfg_module.SxC
X_RANGE = cfg_module.X_RANGE
check = cfg_module.check

# ---------------------------------------------------------------------
# CSV setup
# ---------------------------------------------------------------------
csv_path = Path(results_file)
mode = "a" if check else "w"
write_header = (mode == "w") or not csv_path.exists()
with csv_path.open(mode, newline="") as f:
    w = csv.writer(f)
    if write_header:
        w.writerow(
            [
                "DATASET",
                "ENCODER_TX",
                "DECODER_RX",
                "N_ANCHORS",
                "SIMILARITY",
                "SEED",
                "ACCURACY",
                "DECODER_TYPE",
                "SNR_DB",
                "RECON_LOSS",
                "Q",
                "SxC",
            ]
        )

# if not save_individual_results:
#     ENCODERS_TX.append("absolute")  # Ensure "absolute" is included

for DATASET in DATASETS:
    cfg = cfg_module.get_config(DATASET)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    REC_TYPES = (
        cfg["ANCHORS_STRATEGIES"] if ANCHORS_STRATEGIES is None else ANCHORS_STRATEGIES
    )
    DECODER_RX = cfg["ENCODER_E2E"]
    SEEDS = cfg["SEEDS"] if SEEDS is not None else SEEDS
    SIMILS = cfg["SIMILS"]
    N_ANCHORS = cfg["N_CLUSTERS"] if N_CLUSTERS is None else N_CLUSTERS
    EPOCHS = cfg["EPOCHS"]
    LATENTS_DIR = cfg["LATENTS_DIR"]
    Q_BITS = cfg["quantization"] if QUANTIZATION is None else QUANTIZATION
    SxC = cfg["N_SAMPLES"] if SxC is None else SxC

    # filter anchor counts
    if X_RANGE and DATASET in X_RANGE:
        lo, hi = X_RANGE[DATASET]
        N_ANCHORS = [n for n in N_ANCHORS if lo <= n <= hi]

    # -----------------------------
    # Pre-load test absolute and train anchors_latents to avoid repeated disk I/O.
    # -----------------------------

    loaded_test_latents_RX = {}
    loaded_anchors_RX = {}
    loaded_labels = {}

    for dec_y in DECODER_RX:
        # Load test absolute
        path_test = LATENTS_DIR / "test" / f"{dec_y}.pt"
        loaded_test_latents_RX[dec_y] = {}
        if path_test.exists():
            data_test = torch.load(path_test)
            absolute = data_test["absolute"] - data_test["absolute"].mean(
                dim=0, keepdim=True
            )
            loaded_test_latents_RX[dec_y] = F.normalize(absolute, p=2, dim=1)
            # breakpoint()
            # Determine label key based on dataset
            if DATASET == "cifar20":
                labels_key = "coarse_labels"
            else:
                labels_key = "labels"
            loaded_labels[dec_y] = data_test[labels_key]
            print(
                f"Loaded {dec_y} test absolute: {loaded_test_latents_RX[dec_y].shape}"
            )
            del data_test

        # Load train anchors_latents
        path_train = LATENTS_DIR / "train" / f"{dec_y}.pt"
        loaded_anchors_RX[dec_y] = {}
        if path_train.exists():
            data_train = torch.load(path_train)
            anchors = data_train["anchors_latents"] - data_train[
                "anchors_latents"
            ].mean(dim=0, keepdim=True)
            loaded_anchors_RX[dec_y] = F.normalize(anchors, p=2, dim=1)
            print(f"Loaded {dec_y} train anchors: {loaded_anchors_RX[dec_y].shape}")
            del data_train

    loaded_test_latents_TX = {}
    loaded_anchors_TX = {}

    for enc_x in ENCODERS_TX:
        if enc_x == "absolute":
            continue

        # Load test absolute
        path_test = LATENTS_DIR / "test" / f"{enc_x}.pt"
        loaded_test_latents_TX[enc_x] = {}
        if path_test.exists():
            data_test = torch.load(path_test)
            absolute = data_test["absolute"] - data_test["absolute"].mean(
                dim=0, keepdim=True
            )
            loaded_test_latents_TX[enc_x] = F.normalize(absolute, p=2, dim=1)
            print(
                f"Loaded {enc_x} test absolute: {loaded_test_latents_TX[enc_x].shape}"
            )
            del data_test

        # Load train anchors_latents
        path_train = LATENTS_DIR / "train" / f"{enc_x}.pt"
        loaded_anchors_TX[enc_x] = {}
        if path_train.exists():
            data_train = torch.load(path_train)
            anchors = data_train["anchors_latents"] - data_train[
                "anchors_latents"
            ].mean(dim=0, keepdim=True)
            loaded_anchors_TX[enc_x] = F.normalize(anchors, p=2, dim=1)
            print(f"Loaded {enc_x} train anchors: {loaded_anchors_TX[enc_x].shape}")
            del data_train

    # -----------------------------
    # Main nested loops
    # -----------------------------
    for q in Q_BITS:
        print(f"\n=== Running for {DATASET} with {q} bits quantization ===")

        for snr_db in SNR_DB:
            print(f"--- SNR = {snr_db} dB ---")

            for rec_type in REC_TYPES:
                print(f"Reconstruction Type: {rec_type}")

                for DEC in DECODER_RX:
                    print(f"  Using Decoder: {DEC}")

                    net = None

                    h_data = loaded_test_latents_RX[DEC].to(device)
                    y_labels = loaded_labels[DEC]

                    for SIMIL in SIMILS:
                        print(f"    Similarity: {SIMIL}")

                        for SEED in SEEDS:
                            print(f"      Seed: {SEED}")
                            # load decoder network once
                            if net is None:
                                net = Decoder(
                                    in_features=h_data.shape[1],
                                    out_features=h_data.shape[1] // 2,
                                    n_classes=cfg["N_CLASSES"],
                                ).to(device)
                                net.eval()
                            ckpt = torch.load(f"weights/{DATASET}_{DEC}_{SEED}.ckpt")
                            net.load_state_dict(
                                {
                                    k.replace("net.", ""): v
                                    for k, v in ckpt["state_dict"].items()
                                }
                            )

                            # --- absolute E2E baseline ---
                            if config_type != "complete":
                                x_abs = h_data.clone()
                                if q != 32:
                                    x_abs = uniform_quantize(x_abs, q)
                                if snr_db != 30:
                                    x_abs = x_abs + white_noise(x_abs, snr_db)
                                preds = net(x_abs).argmax(dim=1).cpu().numpy()
                                acc_abs = accuracy_score(y_labels, preds)
                                # record absolute
                                for anc in N_ANCHORS:
                                    if not check_existing_result(  # csv_file, dataset, encoder_tx, decoder_rx, n_anchors, similarity, seed, decoder_type, snr_db, q, SxC
                                        csv_path,
                                        DATASET,
                                        "absolute",
                                        DEC,
                                        anc,
                                        SIMIL,
                                        SEED,
                                        "absolute",
                                        snr_db,
                                        q,
                                        SxC,
                                    ):
                                        update_results_csv(
                                            csv_path,
                                            DATASET,
                                            "absolute",
                                            DEC,
                                            anc,
                                            SIMIL,
                                            SEED,
                                            acc_abs,
                                            "absolute",
                                            snr_db,
                                            0.0,
                                            q,
                                            SxC,
                                        )

                            for n_anchor in N_ANCHORS:
                                print(f"        #Anchors: {n_anchor}")

                                try:
                                    anchors_RX = get_anchors(
                                        rec_type,
                                        n_anchor,
                                        loaded_anchors_RX[DEC],
                                        LATENTS_DIR,
                                        MODEL_USED=DEC,
                                        SEED=SEED,
                                        SxC=SxC,
                                    ).to(device)
                                except Exception as e:
                                    print(
                                        f"  Error getting anchors for {DEC} with {n_anchor} anchors: {e}"
                                    )
                                    breakpoint()
                                    continue

                                if (
                                    "pseudo" in rec_type
                                    or rec_type == "proto"
                                    or rec_type == "combined"
                                ):
                                    anchs = anchors_RX
                                    reg_term = 1e-3 * torch.eye(
                                        anchs.size(1),
                                        dtype=anchs.dtype,
                                        device=anchs.device,
                                    )
                                    regularized_inv = torch.linalg.pinv(
                                        anchs.T @ anchs + reg_term
                                    )
                                else:
                                    regularized_inv = None

                                for encoder_x in ENCODERS_TX:
                                    if encoder_x == "absolute":
                                        continue
                                    # Optionally check if results already exist...
                                    if check:
                                        try:
                                            if check_existing_result(  # TODO: Implement this function
                                                csv_path,
                                                DATASET,
                                                encoder_x,
                                                DEC,
                                                anc,
                                                SIMIL,
                                                SEED,
                                                rec_type,
                                                snr_db,
                                                q,
                                                SxC,
                                            ):
                                                print(
                                                    f"Result for {DATASET}, {DEC}, {n_anchor}, {SEED}, {SIMIL} already exists. Skipping..."
                                                )
                                                continue
                                        except FileNotFoundError:
                                            print(
                                                f"Result for {DATASET}, {DEC}, {n_anchor}, {SEED}, {SIMIL} not found. Continuing..."
                                            )

                                    anchors_TX = get_anchors(
                                        rec_type,
                                        n_anchor,
                                        loaded_anchors_TX[encoder_x],
                                        LATENTS_DIR,
                                        MODEL_USED=DEC,
                                        SEED=SEED,
                                        SxC=SxC,
                                    ).to(device)

                                    dataset_TX_test = loaded_test_latents_TX.get(
                                        encoder_x, None
                                    )
                                    if dataset_TX_test is None:
                                        print(
                                            f"  No loaded test data for encoder {encoder_x}, skipping..."
                                        )
                                        continue

                                    x_data = loaded_test_latents_TX[encoder_x].to(
                                        device
                                    )
                                    local_ds = TensorDataset(x_data, y_labels, h_data)
                                    data_loader = DataLoader(
                                        local_ds,
                                        batch_size=128,
                                        num_workers=0,
                                        shuffle=False,
                                    )

                                    all_preds = []
                                    all_labels = []
                                    recon_losses = []
                                    mse_coef_val = 0.0

                                    if "procrustes" in rec_type:
                                        _, U, V = procrustes_alignment(
                                            anchors_RX,
                                            anchors_TX,
                                            k=n_anchor if n_anchor <= 768 else 768,
                                        )
                                    elif "linear" in rec_type:
                                        _, U, V = linear_alignment(
                                            anchors_RX,
                                            anchors_TX,
                                            k=n_anchor if n_anchor <= 768 else 768,
                                        )
                                    elif "neural" in rec_type:
                                        _, U, V = neural_alignment(
                                            anchors_RX,
                                            anchors_TX,
                                            k=n_anchor if n_anchor <= 768 else 768,
                                        )

                                    with torch.no_grad():
                                        for x_b, y_b, h_b in data_loader:
                                            x_b = x_b.to(device)
                                            y_b = y_b.to(device)
                                            h_b = h_b.to(device)

                                            if (
                                                "procrustes" in rec_type
                                                or "linear" in rec_type
                                                or "neural" in rec_type
                                            ):
                                                # Apply Procrustes alignment to x_b
                                                # breakpoint()
                                                x_b = x_b @ U
                                                if rec_type == "neural":
                                                    # apply sigmoid activation
                                                    x_b = torch.sigmoid(x_b)
                                                if q != 32:
                                                    x_b = uniform_quantize(x_b, q)
                                                if snr_db != 30:
                                                    x_b = x_b + white_noise(x_b, snr_db)
                                                x_b = x_b @ V.T
                                            else:
                                                relrep_TX = relative_projection(
                                                    x_b, anchors_TX, type=SIMIL
                                                )  # analysis operator

                                                if q != 32:
                                                    relrep_TX = uniform_quantize(
                                                        relrep_TX, q
                                                    )
                                                if snr_db is not None and snr_db != 30:
                                                    relrep_TX = relrep_TX + white_noise(
                                                        relrep_TX, snr_db
                                                    )
                                                if (
                                                    "pseudo" in rec_type
                                                    or rec_type == "proto"
                                                    or rec_type == "combined"
                                                ):
                                                    x_b = pseudo_inverse(
                                                        anchors_RX,
                                                        relrep_TX,
                                                        regularized_inv,
                                                    )
                                                elif "frames" in rec_type:
                                                    x_b = frame_recon(
                                                        anchors_RX, relrep_TX
                                                    )  # synthesis operator

                                            recon_losses.append(
                                                F.mse_loss(x_b, h_b).item()
                                            )
                                            outputs = net(x_b)
                                            _, preds_b = torch.max(outputs, 1)
                                            all_preds.extend(preds_b.cpu().numpy())
                                            all_labels.extend(y_b.cpu().numpy())

                                    accuracy = accuracy_score(all_labels, all_preds)
                                    mean_recon_loss = (
                                        np.mean(recon_losses) if recon_losses else 0.0
                                    )
                                    # mse_coef_val /= max(1, len(data_loader))

                                    print(
                                        f"        => rec_type={rec_type}, encoder_x={encoder_x}, SIMIL={SIMIL} "
                                        f"Acc={accuracy:.4f}, recon={mean_recon_loss:.4f}"
                                    )
                                    # Always save to main results file
                                    update_results_csv(
                                        csv_path,
                                        DATASET,
                                        encoder_x,
                                        DEC,
                                        n_anchor,
                                        SIMIL,
                                        SEED,
                                        accuracy,
                                        rec_type,
                                        snr_db,
                                        mean_recon_loss,
                                        q,
                                        SxC,
                                    )

                                    # Additionally save to individual results file if required
                                    if save_individual:
                                        indv_csv_file = Path(
                                            f"{DATASET}_{rec_type}_res.csv"
                                        )
                                        update_results_csv(
                                            indv_csv_file,
                                            DATASET,
                                            encoder_x,
                                            DEC,
                                            n_anchor,
                                            SIMIL,
                                            SEED,
                                            accuracy,
                                            rec_type,
                                            snr_db,
                                            mean_recon_loss,
                                            q,
                                            SxC,
                                        )

                                # End ENCODER_TX loop
                            # End n_anchor loop
                        # End SEED loop
                    # End SIMILS loop
                    del net  # free memory
                # End DECODER_RX loop
            # End REC_TYPE loop
        # End snr_db loop
    # End Q loop
    del loaded_test_latents_RX, loaded_anchors_RX, loaded_labels
# End DATASETS loop
