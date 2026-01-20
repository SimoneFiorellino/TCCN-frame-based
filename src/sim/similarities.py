import torch
import torch.nn.functional as F


def relative_projection(x, anchors, type: str = "cosine"):
    if type == "cosine" or type == "cosine_similarity":
        return cosine_similarity(x, anchors)
    else:
        raise ValueError(f"Invalid similarity type: {type}")


def cosine_similarity(x, anchors):
    """
    :param x: torch.Tensor of shape (b, m)
    :param y: torch.Tensor of shape (a, m)

    :return: torch.Tensor of shape (b, a)
    """
    x = F.normalize(x, p=2, dim=-1)
    anchors = F.normalize(anchors, p=2, dim=-1)
    return torch.einsum("bm, am -> ba", x, anchors)
