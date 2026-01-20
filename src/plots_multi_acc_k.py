import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
import matplotlib.lines as mlines

import argparse

parser = argparse.ArgumentParser(
    description="Run the Proto Anchor with different config types"
)
parser.add_argument(
    "--config_type",
    choices=[
        "base",
        "expansion",
        "compression",
        "complete",
        "model_types",
        "baselines",
        "proto",
    ],
    default="base",
    help="Which configuration to use",
)
args = parser.parse_args()

config_type = args.config_type

plt.style.use("./src/spaicom.mplstyle")

decoder_names = {
    "pseudo": "FE",
    "frames": "PFE",
    "proto_frames": "Proto-PFE",
    "proto": "Proto-FE",
    "combined": "PFE",
    "procrustes": "UPE",
    "proto_procrustes": "Proto-UPE",
    "linear": "LE",
    "neural": "NE",
    "proto_neural": "Proto-NE",
    "proto_linear": "Proto-LE",
}

# Pretty TX encoder names
encoder_pretty_names = {
    "rexnet_100": "rexnet_100",
    "mobilenetv3_large_100": "mobilenetv3_large_100",
    "timm-convit_base.fb_in1k": "convit_base",
    "timm-resmlp_12_224.fb_in1k": "resmlp_12_224",
    "timm-convnextv2_base.fcmae_ft_in22k_in1k": "convnextv2_base",
    "vit_small_patch16_224": "vit_small_patch16_224",
    "vit_base_patch16_224": "vit_base_patch16_224",
    "vit_base_patch32_clip_224": "vit_base_patch32_clip_224",
}


def plot_accuracy_vs_anchors_combined(
    df,
    datasets,
    x_range_dict=None,
    show_mean_only=False,
    config_type="base",
    log_scale=False,
    cfg=None,
):
    config0 = cfg.get_config(datasets[0])
    decoders = cfg.ENCODER_RX  # rows
    similarities = config0["SIMILS"]  # curves inside each subplot
    decoder_types = cfg.ANCHORS_STRATEGIES
    n_clusters = cfg.N_CLUSTERS

    # Layout: ROWS = decoders, COLS = datasets
    num_decoders = len(decoders)
    num_datasets = len(datasets)

    fig, axes = plt.subplots(
        nrows=num_decoders,
        ncols=num_datasets,
        figsize=(8 * num_datasets, 6.8 * num_decoders),
    )

    if num_decoders == 1:
        axes = axes[np.newaxis, :]
    if num_datasets == 1:
        axes = axes[:, np.newaxis]

    dataset_names = {
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
        "cifar20": "Coarse CIFAR-100",
        "tiny_imagenet": "Tiny ImageNet",
    }

    # Styles
    line_styles = {
        "procrustes": "solid",
        "proto_procrustes": "dashdot",
        "frames": "dashed",
        "proto_frames": "dotted",
        "pseudo": (0, (3, 1, 1, 1)),
        "proto": (0, (5, 2, 2, 2)),
        "combined": (0, (1, 1)),
        "linear": (0, (3, 5, 1, 5)),
        "neural": (0, (5, 1)),
        "proto_linear": (0, (3, 5, 1, 5)),
        "proto_neural": (0, (5, 1)),
    }

    mark_styles = {
        "proto_frames": "o",
        "frames": "o",
        "proto": "1",
        "pseudo": "1",
        "procrustes": "*",
        "proto_procrustes": "*",
        "linear": "1",
        "proto_linear": "1",
        "neural": "v",
        "proto_neural": "v",
    }

    # Colors
    unique_encoders = pd.unique(df[df["ENCODER_TX"] != "absolute"]["ENCODER_TX"])
    unique_encoders = [enc for enc in cfg.ENCODERS_TX if enc in unique_encoders]
    all_colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

    if show_mean_only:
        color_map_strategy = dict(zip(decoder_types, all_colors))
    else:
        colors = all_colors[: len(unique_encoders)]
        color_map = dict(zip(unique_encoders, colors))

    used_decoder_types = set()

    # -----------------------------------------------------
    # BUILD PLOT
    # -----------------------------------------------------
    for d_idx, dataset in enumerate(datasets):
        group = df[df["DATASET"] == dataset]
        if x_range_dict and dataset in x_range_dict:
            min_val, max_val = x_range_dict[dataset]
            group = group[
                (group["N_ANCHORS"] >= min_val) & (group["N_ANCHORS"] <= max_val)
            ]

        if group.empty:
            continue

        similarities = cfg.get_config(dataset)["SIMILS"]

        # --- for each DECODER (row)
        for i, decoder in enumerate(decoders):
            ax = axes[i, d_idx]

            # ---------- absolute baseline ----------
            abs_subset = group[
                (group["DECODER_RX"] == decoder)
                & (group["SIMILARITY"] == similarities[0])
                & (group["ENCODER_TX"] == "absolute")
                & (group["N_ANCHORS"].isin(n_clusters))
                & (group["Q"].isin(cfg.QUANTIZATION))
                & (group["SEED"].isin(cfg.SEEDS))
                & (group["SNR_DB"] == min(cfg.SNR_DB))
                & (group["SxC"] == cfg.SxC)
            ]

            if not abs_subset.empty:
                summary_abs = (
                    abs_subset.groupby("N_ANCHORS")["ACCURACY"]
                    .agg(["mean", "std"])
                    .reset_index()
                )
                ax.plot(
                    summary_abs["N_ANCHORS"],
                    summary_abs["mean"],
                    marker="p",
                    linestyle="-",
                    color="black",
                    label="absolute",
                )
                ax.fill_between(
                    summary_abs["N_ANCHORS"],
                    summary_abs["mean"] - summary_abs["std"],
                    summary_abs["mean"] + summary_abs["std"],
                    color="black",
                    alpha=0.1,
                )

            # ---------- similarity curves ----------
            for sim in similarities:
                for decoder_type in decoder_types:
                    subset = group[
                        (group["DECODER_RX"] == decoder)
                        & (group["SIMILARITY"] == sim)
                        & (group["DECODER_TYPE"] == decoder_type)
                        & (group["ENCODER_TX"] != "absolute")
                        & (group["ENCODER_TX"].isin(cfg.ENCODERS_TX))
                        & (group["ENCODER_TX"] != decoder)
                        & (group["N_ANCHORS"].isin(n_clusters))
                        & (group["Q"].isin(cfg.QUANTIZATION))
                        & (group["SEED"].isin(cfg.SEEDS))
                        & (group["SNR_DB"] == min(cfg.SNR_DB))
                        & (group["SxC"] == cfg.SxC)
                    ]

                    if subset.empty:
                        continue

                    used_decoder_types.add(decoder_type)

                    if show_mean_only:
                        summary = (
                            subset.groupby(["N_ANCHORS"])["ACCURACY"]
                            .agg(["mean", "std"])
                            .reset_index()
                        )
                        ax.plot(
                            summary["N_ANCHORS"],
                            summary["mean"],
                            marker=mark_styles.get(decoder_type, "o"),
                            linestyle=line_styles.get(decoder_type, "-"),
                            color=color_map_strategy[decoder_type],
                        )
                        ax.fill_between(
                            summary["N_ANCHORS"],
                            summary["mean"] - (0.75 * summary["std"]),
                            summary["mean"] + (0.75 * summary["std"]),
                            color=color_map_strategy[decoder_type],
                            alpha=0.15,
                        )
                    else:
                        summary = (
                            subset.groupby(["ENCODER_TX", "N_ANCHORS"])["ACCURACY"]
                            .agg(["mean", "std"])
                            .reset_index()
                        )
                        for enc, sub_enc in summary.groupby("ENCODER_TX"):
                            ax.plot(
                                sub_enc["N_ANCHORS"],
                                sub_enc["mean"],
                                marker=mark_styles.get(decoder_type, "o"),
                                linestyle=line_styles.get(decoder_type, "-"),
                                color=color_map[enc],
                            )
                            ax.fill_between(
                                sub_enc["N_ANCHORS"],
                                sub_enc["mean"] - sub_enc["std"],
                                sub_enc["mean"] + sub_enc["std"],
                                color=color_map[enc],
                                alpha=0.2,
                            )

            # --- Labels
            if i == 0:
                ax.set_title(dataset_names.get(dataset, dataset))
            if d_idx == 0:
                ax.set_ylabel(
                    f"RX Model: {encoder_pretty_names.get(decoder, decoder)}\nAccuracy"
                )

            ax.set_xlabel(r"$N$")
            if log_scale:
                ax.set_xscale("log")
            ax.grid(True)

    fig.tight_layout(rect=[0, 0, 1, 0.9])

    # -----------------------------------------------------
    # LEGENDS
    # -----------------------------------------------------
    absolute_handle = mlines.Line2D(
        [], [], color="black", linestyle="-", marker="p", label="Absolute"
    )

    if show_mean_only:
        strategy_handles = [absolute_handle] + [
            mlines.Line2D(
                [],
                [],
                color=color_map_strategy[dt],
                linestyle=line_styles.get(dt, "-"),
                marker=mark_styles.get(dt, "o"),
                label=decoder_names.get(dt, dt),
            )
            for dt in decoder_types
        ]

        fig.legend(
            handles=strategy_handles,
            title="Approaches",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(strategy_handles),
            frameon=False,
        )

    else:
        approach_handles = [absolute_handle] + [
            mlines.Line2D(
                [],
                [],
                color="black",
                linestyle=line_styles.get(dt, "-"),
                marker=mark_styles.get(dt, "o"),
                label=decoder_names.get(dt, dt),
            )
            for dt in decoder_types
        ]

        encoder_handles = [
            mlines.Line2D(
                [],
                [],
                color=color_map[enc],
                marker="o",
                linestyle="",
                label=encoder_pretty_names.get(enc, enc),
            )
            for enc in color_map.keys()
        ]

        fig.legend(
            handles=encoder_handles,
            title="Encoders (TX)",
            loc="upper left",
            bbox_to_anchor=(0.05, 1.0),  # 1.027
            ncol=4,
            frameon=False,
        )

        fig.legend(
            handles=approach_handles,
            title="Approaches",
            loc="upper right",
            bbox_to_anchor=(0.95, 1.0),
            ncol=4,
            frameon=False,
        )

    if config_type == "proto":
        pdf_file_path = (
            f"results/multiple_datasets_accuracy_vs_k_{config_type}_{cfg.SxC}.pdf"
        )
    else:
        pdf_file_path = f"results/multiple_datasets_accuracy_vs_k_{config_type}.pdf"
    png_file_path = (
        f"results/png/{cfg.SxC}_multiple_datasets_accuracy_vs_k_{config_type}.png"
    )

    fig.savefig(pdf_file_path, format="pdf")
    fig.savefig(png_file_path, format="png")
    plt.close(fig)


if __name__ == "__main__":
    # ---------------------------------------
    # Load data & invoke function
    # ---------------------------------------
    if config_type == "baselines":
        import configs.experiment_config_baselines as cfg
    elif config_type == "compression":
        import configs.experiment_config_compression as cfg
    elif config_type == "complete":
        import configs.experiment_config_complete as cfg
    elif config_type == "model_types":
        import configs.experiment_config_model_types as cfg
    elif config_type == "proto":
        import configs.experiment_config_proto as cfg
    else:
        raise ValueError("Invalid type.")

    df = pd.read_csv(f"results_{config_type}.csv")

    plot_accuracy_vs_anchors_combined(
        df,
        cfg.DATASETS,
        x_range_dict=cfg.X_RANGE,
        show_mean_only=cfg.show_mean_only_flag,
        config_type=config_type,
        log_scale=cfg.log_scale_flag,
    )
