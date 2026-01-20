import os

import torch
from torch.utils.data import Dataset


def tight_frame_(A, trainset=None):
    # frame operator
    # S = trainset.T@trainset

    # if False:
    #     eigvals, eigvecs = torch.linalg.eigh(S)
    #     vals = eigvals.clamp(min=1e-6).rsqrt()
    #     S_inv_sqrt = eigvecs @ torch.diag(vals) @ eigvecs.T
    #     A_t = A @ S_inv_sqrt
    # else: # Lowdin-polar (SVD-based)
    U, _, Vt = torch.linalg.svd(A, full_matrices=False)
    A_t = U @ Vt

    # print("condition number: ", torch.linalg.cond(A_t))
    return A_t


def get_anchors(
    idx_flag,
    n_anchors,
    trainset,
    LATENTS_DIR,
    MODEL_USED,
    SEED=None,
    SxC=1,
    ratio=0.5,
    sub_frame=False,
):
    if (
        "pseudo" in idx_flag
        or idx_flag == "procrustes"
        or idx_flag == "frames"
        or idx_flag == "linear"
        or idx_flag == "neural"
    ):
        # select random indices
        idx = select_random_indices(n_anchors, seed=SEED)
        anchors_latents = trainset[idx]
    elif "proto" in idx_flag:
        if n_anchors < 800:
            selected_anchors = torch.load(
                LATENTS_DIR / f"{MODEL_USED}_{n_anchors}_{SxC}_{SEED}.pt"
            )
            S = {}
            # breakpoint()
            for i in range(n_anchors):
                S[i] = {"latents": trainset[selected_anchors[i]["idx"]]}
            # breakpoint()
            anchors_latents = torch.stack(
                [torch.mean(S[i]["latents"], dim=0) for i in range(n_anchors)], dim=0
            )
        else:
            idx = select_random_indices(n_anchors, seed=SEED)
            anchors_latents = trainset[idx]
    elif "all_procrustes" == idx_flag:
        selected_anchors = torch.load(
            LATENTS_DIR / f"{MODEL_USED}_{n_anchors}_{SxC}_{SEED}.pt"
        )
        S = {}
        # breakpoint()
        for i in range(n_anchors):
            S[i] = {"latents": trainset[selected_anchors[i]["idx"]]}
        # stack without computing the mean for cluster
        anchors_latents = torch.stack(
            [S[i]["latents"] for i in range(n_anchors)], dim=0
        )
        anchors_latents = anchors_latents.view(-1, anchors_latents.size(-1))
    if "frames" in idx_flag:
        anchors_latents = tight_frame_(anchors_latents, trainset)

    return anchors_latents


def return_all_anchors(n_anchors, trainset, LATENTS_DIR, ENCODER, SEED=None):
    """
    create a tensor of all the anchors, i.e. the idx of the selected anchors
    selected_anchors example: {0: {'idx': array([1251,  150,   26,  925,  653])}, 1: {'idx': array([297, 515, 142, 487, 215])}, 2: {'idx': array([ 214,  973, 1669,  301, 1100])}, 3: {'idx': array([1198, 1086, 1151, 1167, 1113])}, 4: {'idx': array([1647,  683,  638,  926, 1976])}, 5: {'idx': array([ 478, 1821, 1907, 1693, 1257])}, 6: {'idx': array([1277,  901, 1824, 1105,  312])}, 7: {'idx': array([ 977,  717,  465, 1441,  175])}, 8: {'idx': array([1585, 1164,   90,  861, 1849])}, 9: {'idx': array([1407,  691,  590,  207,  447])}}
    """
    selected_anchors = torch.load(LATENTS_DIR / f"{ENCODER}_{n_anchors}_{SEED}.pt")
    S = {}
    for i in range(n_anchors):
        S[i] = {"latents": trainset["anchors_latents"][selected_anchors[i]["idx"]]}
    S_tensor = torch.stack([S[i]["latents"] for i in range(n_anchors)])
    # flatten maintaing the last dimension
    S_tensor = S_tensor.view(-1, S_tensor.size(-1))
    return S_tensor


def select_random_indices(n, max=2000, seed=None):
    # Generate a tensor of random indices between 0 and 999
    if seed is not None:
        torch.manual_seed(seed)
    random_indices = torch.randperm(max)[:n]
    return random_indices


def save_anchors(
    train_dataset: Dataset,
    n_anchors: int = 100,
):
    """
    Save n_anchors as Tensors (torch.save(img)) in the anchors folder.
    if n_anchors is lesser or equal to 100, we choose 100 img with different labels. Otherwise, we choose n_anchors random image.
    """
    if not os.path.exists("data/anchors"):
        os.makedirs("data/anchors")

    if n_anchors <= 100:
        # select 10 images with different labels
        labels = []
        for i, (img, label) in enumerate(train_dataset):
            if label not in labels:
                labels.append(label)
                torch.save(img, f"data/anchors/anchor_{i}.pt")
            if len(labels) == n_anchors:
                break
    else:
        # select n_anchors random images
        for i, (img, label) in enumerate(train_dataset):
            if i == n_anchors:
                break
            torch.save(img, f"data/anchors/anchor_{i}.pt")


def load_anchors():
    """Load all the anchors, of the form anchor_n, from the anchors folder.
    The n number are note sequential, so we use a dictionary to store the anchors.
    return a Tensor of dimension (n_anchors, 3, 28, 28)."""
    anchors = []
    for img in os.listdir("data/anchors"):
        anchors.append(torch.load(f"data/anchors/{img}"))
    anchors = torch.stack(anchors)
    return anchors
