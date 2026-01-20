import random
from functools import partial
from pathlib import Path
from typing import Mapping, Sequence, List, Dict

import timm
import torch
from datasets import load_dataset
from pytorch_lightning import seed_everything
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.sim.similarities import relative_projection


def get_dataset(
    split: str,
    perc: float = 1,
    val_perc: float = 0.15,
    dataset_name: str = "tiny_imagenet",
):
    seed_everything(42)
    assert 0 < perc <= 1
    assert 0 < val_perc < 1
    if dataset_name == "tiny_imagenet":
        dataset_name = "Maysee/tiny-imagenet"
    dataset = load_dataset(dataset_name)[split]

    # If the split is 'train', further split into train and validation sets
    if split == "train":
        # Shuffle and select a random subset for the entire training set
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        indices = indices[: int(len(indices) * perc)]

        # Calculate lengths and indices for train and validation subsets
        val_len = int(len(indices) * val_perc)
        train_len = len(indices) - val_len

        train_indices = indices[:train_len]
        val_indices = indices[train_len:]

        # Create train and validation subsets
        train_dataset = dataset.select(train_indices)
        val_dataset = dataset.select(val_indices)

        return train_dataset, val_dataset

    else:
        # For other splits, you can apply the original random subset logic
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        indices = indices[: int(len(indices) * perc)]
        dataset = dataset.select(indices)

        return dataset


def load_transformer(transformer_name):
    transformer = timm.create_model(transformer_name, pretrained=True, num_classes=0)
    return transformer.requires_grad_(False).eval()


@torch.no_grad()
def call_transformer(batch, transformer):
    #     batch["encoding"] = batch["encoding"].to(device)
    sample_encodings = transformer(batch["encoding"].to(device))
    #     hidden = sample_encodings["hidden_states"][-1]
    #     assert hidden.size(-1) == hidden.size(-2), hidden.size()
    #     print(sample_encodings.shape)
    return {"hidden": sample_encodings}


def get_latents(
    dataloader, anchors, split: str, transformer
) -> Dict[str, torch.Tensor]:
    absolute_latents: List = []
    relative_latents: List = []
    labels: List = []
    # coarse_labels: List = []
    #     logits_latents: List = []

    transformer = transformer.to(device)
    for batch in tqdm(dataloader, desc=f"[{split}] Computing latents"):
        with torch.no_grad():
            transformer_out = call_transformer(batch=batch, transformer=transformer)

            #             logits_latents.append(transformer_out["logits"].cpu())
            absolute_latents.append(transformer_out["hidden"].cpu())

            if anchors is not None:
                batch_rel_latents = relative_projection(
                    x=transformer_out["hidden"], anchors=anchors, type="cosine"
                )
                relative_latents.append(batch_rel_latents.cpu())

            labels.append(batch["labels"].cpu())
            # coarse_labels.append(batch["coarse_label"].cpu())

    absolute_latents: torch.Tensor = torch.cat(absolute_latents, dim=0).cpu()
    #     logits_latents: torch.Tensor = torch.cat(logits_latents, dim=0).cpu()
    relative_latents: torch.Tensor = (
        torch.cat(relative_latents, dim=0).cpu()
        if len(relative_latents) > 0
        else relative_latents
    )
    labels: torch.Tensor = torch.cat(labels, dim=0).cpu()
    # coarse_labels: torch.Tensor = torch.cat(coarse_labels, dim=0).cpu()

    transformer = transformer.cpu()
    return {
        "absolute": absolute_latents,
        "relative": relative_latents,
        "labels": labels,
        # "coarse_labels": coarse_labels,
        #         "logits": logits_latents
    }


def collate_fn(batch, feature_extractor, transform):
    #     encoding = feature_extractor(
    #         [sample[data_key] for sample in batch],
    #         return_tensors="pt",
    #     )
    #     encoding = {"pixel_values" : torch.stack([transform(sample['image'].convert("RGB")) for sample in batch], dim=0)}
    # mask = encoding["attention_mask"] * encoding["special_tokens_mask"].bool().logical_not()
    # return {"encoding": encoding, "mask": mask.bool()}
    return {
        "encoding": torch.stack(
            [transform(sample["image"].convert("RGB")) for sample in batch], dim=0
        ),
        "labels": torch.tensor([sample["label"] for sample in batch]),
    }


def encode_latents(
    transformer_names: Sequence[str], dataset, transformer_name2latents, split: str
):
    for transformer_name in transformer_names:
        # Load the transformer model
        transformer = load_transformer(transformer_name=transformer_name)

        # Create a transform for the data based on the transformer's requirements
        config = resolve_data_config({}, model=transformer)
        transform = create_transform(**config)

        # Process the anchor dataset
        anchor_latents_output = get_latents(
            dataloader=DataLoader(
                anchor_dataset,
                num_workers=0,
                pin_memory=True,
                collate_fn=partial(
                    collate_fn, feature_extractor=None, transform=transform
                ),
                batch_size=32,
            ),
            split=f"{transformer_name}, anchor, {split}",
            anchors=None,
            transformer=transformer,
        )
        anchors_latents = anchor_latents_output["absolute"]
        anchor_labels = anchor_latents_output["labels"]  # Capturing anchor labels
        # anchor_coarse_labels = anchor_latents_output["coarse_labels"]

        # Process the main dataset
        dataset_latents_output = get_latents(
            dataloader=DataLoader(
                dataset,
                num_workers=0,
                pin_memory=True,
                collate_fn=partial(
                    collate_fn, feature_extractor=None, transform=transform
                ),
                batch_size=32,
            ),
            split=f"{split}/{transformer_name}",
            anchors=anchors_latents.to(device),
            transformer=transformer,
        )

        # Store the latents and labels
        transformer_name2latents[transformer_name] = {
            "anchors_latents": anchors_latents,
            "anchor_labels": anchor_labels,
            # "anchor_coarse_labels": anchor_coarse_labels,
            **dataset_latents_output,
        }

        # Save latents and labels if caching is enabled
        if CACHE_LATENTS:
            print(f"Saving latents and labels for {transformer_name}...")
            transformer_path = (
                LATENTS_DIR / split / f"{transformer_name.replace('/', '-')}.pt"
            )
            transformer_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(transformer_name2latents[transformer_name], transformer_path)


def load_latents(split: str, transformer_names: Sequence[str]):
    transformer2latents = {}

    for transformer_name in transformer_names:
        transformer_path = (
            LATENTS_DIR / split / f"{transformer_name.replace('/', '-')}.pt"
        )
        if transformer_path.exists():
            transformer2latents[transformer_name] = torch.load(transformer_path)

    return transformer2latents


####################################################################################################
####################################################################################################


if __name__ == "__main__":
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    train_perc: float = 0.7
    num_anchors: int = 15000
    dataset_name: str = "tiny_imagenet"
    transformer_names = list(
        {
            # "mobilenetv3_small_100",
            # "mobilenetv3_large_100",
            "timm/convit_base.fb_in1k",
            "timm/resmlp_12_224.fb_in1k",
            "timm/convnextv2_base.fcmae_ft_in22k_in1k",
            # "efficientnet_b0.ra_in1k", # qui
            # "efficientnet_b3.ra2_in1k", # qui
            # "timm/efficientvit_m2.r224_in1k",
            # "timm/levit_128s.fb_dist_in1k",
            # "timm/eva_large_patch14_196.in22k_ft_in22k_in1k",
            # "timm/vit_mediumd_patch16_reg4_gap_256.sbb2_e200_in12k_ft_in1k",
            # "timm/vit_pwee_patch16_reg1_gap_256.sbb_in1k",
            # "rexnet_100",
            # "vit_small_patch16_224",
            # "vit_base_patch16_224",
            # "vit_base_patch32_clip_224",
            # "vit_base_resnet50_384",
        }
    )

    # "mobilenetv3_small_100", "mobilenetv3_large_100", "mobilevitv2_100", "rexnet_100", "vit_base_patch16_224", "vit_small_patch16_224", "vit_base_resnet50_384"

    CACHE_LATENTS: bool = True

    # Data

    LATENTS_DIR = Path(f"./data/latents/{dataset_name}")
    LATENTS_DIR.mkdir(exist_ok=True, parents=True)

    train_dataset, val_dataset = get_dataset(dataset_name=dataset_name, split="train")

    print(len(train_dataset), len(val_dataset))

    # assert num_anchors <= len(train_dataset)
    if "imagenet" in dataset_name:
        test_split = "valid"
    else:
        test_split = "test"
    test_dataset = get_dataset(split=test_split, dataset_name=dataset_name)
    print(len(test_dataset))

    seed_everything(42)
    anchor_idxs = list(range(len(train_dataset)))
    random.shuffle(anchor_idxs)
    anchor_idxs = anchor_idxs[:num_anchors]

    anchor_dataset = train_dataset.select(anchor_idxs)
    print(len(anchor_dataset))
    # anchor_dataset = torch.load(f"./data/anchors/{dataset_name}/anchors_{num_anchors}.pt")

    # Compute train latents

    FORCE_RECOMPUTE: bool = False
    CACHE_LATENTS: bool = True

    transformer2train_latents: Dict[str, Mapping[str, torch.Tensor]] = load_latents(
        split="train", transformer_names=transformer_names
    )
    missing_transformers = (
        transformer_names
        if FORCE_RECOMPUTE
        else [
            t_name
            for t_name in transformer_names
            if t_name not in transformer2train_latents
        ]
    )

    encode_latents(
        transformer_names=missing_transformers,
        dataset=train_dataset,
        transformer_name2latents=transformer2train_latents,
        split="train",
    )

    # Compute val latents

    FORCE_RECOMPUTE: bool = True
    CACHE_LATENTS: bool = True

    transformer2val_latents: Dict[str, Mapping[str, torch.Tensor]] = load_latents(
        split="val", transformer_names=transformer_names
    )

    missing_transformers = (
        transformer_names
        if FORCE_RECOMPUTE
        else [
            t_name
            for t_name in transformer_names
            if t_name not in transformer2val_latents
        ]
    )
    encode_latents(
        transformer_names=missing_transformers,
        dataset=val_dataset,
        transformer_name2latents=transformer2val_latents,
        split="val",
    )

    # Compute test latents

    FORCE_RECOMPUTE: bool = True
    CACHE_LATENTS: bool = True

    transformer2test_latents: Dict[str, Mapping[str, torch.Tensor]] = load_latents(
        split="test", transformer_names=transformer_names
    )
    missing_transformers = (
        transformer_names
        if FORCE_RECOMPUTE
        else [
            t_name
            for t_name in transformer_names
            if t_name not in transformer2test_latents
        ]
    )
    encode_latents(
        transformer_names=missing_transformers,
        dataset=test_dataset,
        transformer_name2latents=transformer2test_latents,
        split="test",
    )
