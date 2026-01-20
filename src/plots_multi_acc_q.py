import pandas as pd
import matplotlib.pyplot as plt
import configs.experiment_config_q as cfg_module
import matplotlib as mpl
import matplotlib.lines as mlines  # For custom legend handles

plt.style.use("./src/spaicom.mplstyle")

dataset_names = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "cifar20": "Coarse CIFAR-100",
    "tiny_imagenet": "Tiny ImageNet",
}


def plot_accuracy_vs_quantization(df, datasets, x_range_dict=None):
    """
    Plots accuracy vs compression factor (xi) while keeping datasets in the same row,
    with one unified legend for N anchors.
    """
    config0 = cfg_module.get_config(datasets[0])
    decoders = config0["ENCODER_E2E"]
    similarities = config0["SIMILS"]
    n_clusters = config0["N_CLUSTERS"]

    # Precompute a global color palette for the different N values
    all_colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    color_palette_global = all_colors[: len(n_clusters)]

    num_datasets = len(datasets)
    num_decoders = len(decoders)
    num_similarities = len(similarities)

    # Keep datasets in the same row
    total_rows = num_decoders
    total_cols = num_similarities * num_datasets

    fig, axes = plt.subplots(
        nrows=total_rows,
        ncols=total_cols,
        figsize=(8 * total_cols, 7 * total_rows),
        squeeze=False,
    )

    markers = ["o", "^", "s", "d", "v", "*", "P", "X"]

    for i, decoder in enumerate(decoders):
        for d_idx, dataset in enumerate(datasets):
            df_dataset = df[df["DATASET"] == dataset]
            if x_range_dict and dataset in x_range_dict:
                min_val, max_val = x_range_dict[dataset]
                df_dataset = df_dataset[
                    (df_dataset["N_ANCHORS"] >= min_val)
                    & (df_dataset["N_ANCHORS"] <= max_val)
                ]
            if df_dataset.empty:
                continue

            config = cfg_module.get_config(dataset)
            similarities = config["SIMILS"]

            for j, similarity in enumerate(similarities):
                ax = axes[i, d_idx * num_similarities + j]

                subset = df_dataset[
                    (df_dataset["DECODER_RX"] == decoder)
                    & (df_dataset["SIMILARITY"] == similarity)
                    & (df_dataset["SNR_DB"] == min(cfg_module.SNR_DB))
                    & (df_dataset["ENCODER_TX"].isin(cfg_module.ENCODERS_TX))
                    & (df_dataset["SEED"].isin(cfg_module.SEEDS))
                    & (df_dataset["Q"].isin(cfg_module.QUANTIZATION))
                ]

                for idx, n_anchor in enumerate(n_clusters):
                    subset_n = subset[subset["N_ANCHORS"] == n_anchor]
                    if subset_n.empty:
                        continue

                    summary = (
                        subset_n.groupby("Q")["ACCURACY"]
                        .agg(["mean", "std"])
                        .reset_index()
                        .sort_values("Q")
                    )
                    summary["std"] = summary["std"] / len(cfg_module.ENCODERS_TX)
                    summary["xi"] = summary["Q"] * n_anchor / (768 * 32)

                    ax.plot(
                        summary["xi"],
                        summary["mean"],
                        marker=markers[idx % len(markers)],
                        linestyle="-",
                        color=color_palette_global[idx],
                        label=f"N = {n_anchor}",
                    )
                    ax.fill_between(
                        summary["xi"],
                        summary["mean"] - summary["std"],
                        summary["mean"] + summary["std"],
                        color=color_palette_global[idx],
                        alpha=0.2,
                    )

                if d_idx == 0 and j == 0:
                    ax.set_ylabel(f"{decoder}\nAccuracy")
                if i == 0:
                    ax.set_title(f"{dataset_names.get(dataset, dataset)}")

                ax.set_xlabel(r"Compression Factor ($\xi$)")
                ax.set_yscale("linear")
                ax.set_xscale("log")
                ax.grid(True)

    # ------------------------------------------------------------------------
    # Unified legend for N anchors
    # ------------------------------------------------------------------------
    default_markersize = mpl.rcParams.get("lines.markersize", 6)
    legend_handles = []
    for idx, n_anchor in enumerate(n_clusters):
        handle = mlines.Line2D(
            [],
            [],
            color=color_palette_global[idx],
            marker=markers[idx % len(markers)],
            linestyle="-",
            markersize=default_markersize,
            label=f"N = {n_anchor}",
        )
        legend_handles.append(handle)

    fig.legend(
        handles=legend_handles,
        title="Number of Coefficients",
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        ncol=len(n_clusters),
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.9])

    pdf_file_path = "results/multiple_datasets_accuracy_vs_xi.pdf"
    png_file_path = "results/png/multiple_datasets_accuracy_vs_xi.png"
    fig.savefig(pdf_file_path, format="pdf")
    fig.savefig(png_file_path, format="png")
    plt.close(fig)


if __name__ == "__main__":
    df = pd.read_csv("results_q.csv")
    plot_accuracy_vs_quantization(df, cfg_module.DATASETS, cfg_module.X_RANGE)
