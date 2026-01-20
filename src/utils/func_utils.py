import csv
import pandas as pd
from pathlib import Path


def check_existing_result(
    csv_file,
    dataset,
    encoder_tx,
    decoder_rx,
    n_anchors,
    similarity,
    seed,
    decoder_type,
    snr_db,
    q,
    SxC,
):
    """Returns True if the result exists, otherwise False"""
    # csv_file = Path("results.csv")
    if not csv_file.exists():
        return False  # CSV file doesn't exist, so result certainly doesn't exist

    df = pd.read_csv(csv_file)
    result_exists = (
        (df["DATASET"] == dataset)
        & (df["ENCODER_TX"] == encoder_tx)
        & (df["DECODER_RX"] == decoder_rx)
        & (df["N_ANCHORS"] == n_anchors)
        & (df["SIMILARITY"] == similarity)
        & (df["SEED"] == seed)
        & (df["DECODER_TYPE"] == decoder_type)
        & (df["SNR_DB"] == snr_db)
        & (df["Q"] == q)
        & (df["SxC"] == SxC)
    ).any()
    return result_exists


def update_results_csv(
    name,
    dataset,
    encoder_tx,
    decoder_rx,
    n_anchors,
    similarity,
    seed,
    accuracy,
    decoder_type,
    snr_db,
    recon_loss,
    q,
    SxC,
):
    csv_file = Path(name)
    temp_file = Path("temp_results.csv")
    fieldnames = [
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

    # Read existing rows (if any)
    if csv_file.exists() and csv_file.stat().st_size > 0:
        with open(csv_file, "r", newline="") as f_in:
            reader = csv.DictReader(f_in)  # always infer header
            data = list(reader)
    else:
        data = []

    # Rewrite all rows + our update/insert
    with open(temp_file, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        updated = False
        for row in data:
            if (
                row["DATASET"] == dataset
                and row["ENCODER_TX"] == encoder_tx
                and row["DECODER_RX"] == decoder_rx
                and row["N_ANCHORS"] == str(n_anchors)
                and row["SIMILARITY"] == similarity
                and row["SEED"] == str(seed)
                and row["DECODER_TYPE"] == decoder_type
                and row["SNR_DB"] == str(snr_db)
                and row["Q"] == q
                and row["SxC"] == SxC
            ):
                row["ACCURACY"] = accuracy
                updated = True

            writer.writerow(row)

        if not updated:
            writer.writerow(
                {
                    "DATASET": dataset,
                    "ENCODER_TX": encoder_tx,
                    "DECODER_RX": decoder_rx,
                    "N_ANCHORS": n_anchors,
                    "SIMILARITY": similarity,
                    "SEED": seed,
                    "ACCURACY": accuracy,
                    "DECODER_TYPE": decoder_type,
                    "SNR_DB": snr_db,
                    "RECON_LOSS": recon_loss,
                    "Q": q,
                    "SxC": SxC,
                }
            )

    # Atomically replace original
    temp_file.replace(csv_file)
