import torch.nn as nn

"""
this repo is based on the following script: https://github.com/lucmos/relreps/blob/main/experiments/sec%3Amodel-reusability-vision/vision_stitching_cifar100.ipynb
"""


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class Decoder(nn.Module):
    "feed-forward network - decoder part - classify the image into 100 classes"

    def __init__(self, in_features, out_features, n_classes):
        super().__init__()
        out_feat = out_features + n_classes
        self.model = nn.Sequential(
            nn.LayerNorm(normalized_shape=in_features),
            nn.Linear(in_features=in_features, out_features=out_feat),
            nn.ReLU(),
            Lambda(lambda x: x.permute(1, 0)),
            nn.InstanceNorm1d(num_features=out_feat),
            Lambda(lambda x: x.permute(1, 0)),
            nn.Linear(in_features=out_feat, out_features=n_classes),
        )

    def forward(self, x):
        return self.model(x)
