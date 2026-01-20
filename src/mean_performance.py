import pandas as pd
import matplotlib.pyplot as plt


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    # Filter out entries where ENCODER_TX is 'absolute'
    df = df[df["ENCODER_TX"] != "absolute"]
    return df


def compute_statistics(df):
    results = []
    for n_anchors in df["N_ANCHORS"].unique():
        df_anchor = df[df["N_ANCHORS"] == n_anchors]
        for dataset in df_anchor["DATASET"].unique():
            df_dataset = df_anchor[df_anchor["DATASET"] == dataset]

            # Zero-shot calculations: ENCODER_TX != DECODER_RX
            zero_shot_df = df_dataset[
                df_dataset["ENCODER_TX"] != df_dataset["DECODER_RX"]
            ]
            zero_shot_mean = zero_shot_df["ACCURACY"].mean()
            zero_shot_std = zero_shot_df["ACCURACY"].std()

            # Absolute calculations: No additional filter for absolute here, assume separate handling or data source
            # For illustration, filtering for a specific DECODER_TYPE if needed
            absolute_df = df_dataset[
                (df_dataset["ENCODER_TX"] == df_dataset["DECODER_RX"])
                & (df_dataset["DECODER_TYPE"] == "absolute")
            ]
            absolute_mean = (
                absolute_df["ACCURACY"].mean() if not absolute_df.empty else None
            )
            absolute_std = (
                absolute_df["ACCURACY"].std() if not absolute_df.empty else None
            )

            results.append(
                {
                    "DATASET": dataset,
                    "ZERO-SHOT": f"{zero_shot_mean:.2f}±{zero_shot_std:.2f}"
                    if not zero_shot_df.empty
                    else "N/A",
                    "ABSOLUTE": f"{absolute_mean:.2f}±{absolute_std:.2f}"
                    if absolute_mean is not None
                    else "N/A",
                    "N_ANCHORS": n_anchors,
                }
            )
    return pd.DataFrame(results)


def plot_accuracy_vs_anchors(df):
    # Simple plot of the processed data
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, grp in df.groupby(["DATASET"]):
        ax.errorbar(
            grp["N_ANCHORS"],
            [float(x.split("±")[0]) for x in grp["ZERO-SHOT"]],
            yerr=[float(x.split("±")[1]) for x in grp["ZERO-SHOT"]],
            fmt="-o",
            label=f"Zero-Shot {key}",
        )
    ax.set_xlabel("Number of Anchors")
    ax.set_ylabel("Accuracy")
    ax.set_title("Zero-Shot Accuracy vs Number of Anchors")
    ax.legend()
    plt.show()


def main():
    filepath = "results.csv"
    df = load_and_preprocess_data(filepath)
    results_df = compute_statistics(df)
    print(results_df)
    plot_accuracy_vs_anchors(results_df)


if __name__ == "__main__":
    main()
