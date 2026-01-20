from pathlib import Path

# ----------------------------set up of the experiments----------------------------

DATASETS = ["cifar10", "cifar100", "tiny_imagenet"]  # "cifar20"

ENCODERS_TX = [
    # "mobilenetv3_small_100",
    # "mobilenetv3_large_100",
    # "rexnet_100", # qui
    # "efficientnet_b0.ra_in1k", # qui
    # "efficientnet_b3.ra2_in1k", # qui
    # "timm-efficientvit_m5.r224_in1k",
    # "timm-efficientvit_m2.r224_in1k",
    # "timm-levit_128s.fb_dist_in1k",
    "vit_small_patch16_224",  # qui
    "vit_base_patch16_224",
    # "vit_base_resnet50_384",
    "vit_base_patch32_clip_224",  # qui
]

ENCODER_RX = ["vit_base_resnet50_384", "timm-convnextv2_base.fcmae_ft_in22k_in1k"]

SIMILARITIES = ["cosine"]

SEEDS = [10, 11, 12, 13, 14]

# self.Nkt_options = [8,10,12,16,20,24,32,64,128] # [16,20,24,32,48,64,80,96,128,160,192,256]
# self.qk_options = [8,12,16,24,32]

SNR_DB = [30]  # 30 no AWGN
N_CLUSTERS = [
    256,
    384,
    512,
]  # [32,48,64,96,128,160,192,256,384,512,768,2048] # [53,64,96,128,160,192,256,384,512] # [768,890,1024,1280,1536,2048] # [8,10,12,16,20,24,32,43,53,64,96,128,192,256,384] # [128,384,768]
QUANTIZATION = [32]  # [4,6,8,12,16,24,32] # [32] #
ANCHORS_STRATEGIES = [
    "procrustes"
]  # ['combined'] # ['proto_frames'] # ['procrustes', 'frames', 'pseudo'] # ['frames','proto_frames'],
SxC = 3  # 5

check_anchors = True
check = False
# encoder_for_anchors = "vit_small_patch16_224"
selection_strategy = (
    "frames"  # "kmeans_random", "cosine_kmeans_random", "spectral_kmeans", "dbscan"
)
# encoder_for_anchors = "mobilenetv3_small_100"
X_RANGE = None
# ----------------------------specific set up for each dataset----------------------------


def get_config(dataset):
    if dataset == "tiny_imagenet":
        return {
            "LATENTS_DIR": Path(f"./data/latents/{dataset}"),
            # support set generation
            # "ENCODERS_FOR_ANCHORS": encoder_for_anchors,
            "SELECTION_STRAREGY": selection_strategy,
            "N_CLUSTERS": N_CLUSTERS,  # ,512,640,768], # [4,8,12,16,24,32,64,96,128,192,256,384,512,768,890,1024,1280,1536,2048,3072],# [5, 6, 7, 8, 9, 10, 12, 15, 18, 20], # [128,256,384,756,1024]
            "ANCHORS_STRATEGIES": ANCHORS_STRATEGIES,  # ['proto_frames'], # ['proto','frames','pseudo','proto_frames'], # ['frames', 'pseudo'], #
            "N_SAMPLES": SxC,
            # relative training
            "ENCODER_E2E": ENCODER_RX,
            "SEEDS": SEEDS,
            "SIMILS": SIMILARITIES,
            "EPOCHS": 25,
            "N_CLASSES": 200,
            # relative testing
            "ENCODERS_X": ENCODERS_TX,
            "quantization": QUANTIZATION,
        }
    elif dataset == "cifar10":
        return {
            "LATENTS_DIR": Path(f"./data/latents/{dataset}"),
            # support set generation
            # "ENCODERS_FOR_ANCHORS": encoder_for_anchors,
            "SELECTION_STRAREGY": selection_strategy,
            "N_CLUSTERS": N_CLUSTERS,  # [4,8,12,16,24,32,64,96,128,192,256,384,512,756,1024],# [5, 6, 7, 8, 9, 10, 12, 15, 18, 20], #  [2,4,6,8,10,12,16,20,24,32,43,53,64,96,128,160]
            "ANCHORS_STRATEGIES": ANCHORS_STRATEGIES,  # proto_frames
            "N_SAMPLES": SxC,
            # relative training
            "ENCODER_E2E": ENCODER_RX,
            "SEEDS": SEEDS,
            "SIMILS": SIMILARITIES,
            "EPOCHS": 20,
            "N_CLASSES": 10,
            # relative testing
            "ENCODERS_X": ENCODERS_TX,
            "quantization": QUANTIZATION,
        }
    elif dataset == "cifar20":
        return {
            "LATENTS_DIR": Path("./data/latents/cifar100"),
            # support set generation
            "SELECTION_STRAREGY": selection_strategy,
            "N_CLUSTERS": N_CLUSTERS,  # [16,20,24,32,48,64,80,96,128,160,192,256],
            "ANCHORS_STRATEGIES": ANCHORS_STRATEGIES,
            "N_SAMPLES": SxC,
            # relative training
            "ENCODER_E2E": ENCODER_RX,
            "SEEDS": SEEDS,
            "SIMILS": SIMILARITIES,
            "EPOCHS": 25,
            "N_CLASSES": 20,
            # relative testing
            "ENCODERS_X": ENCODERS_TX,
            "quantization": QUANTIZATION,
        }
    elif dataset == "cifar100":
        return {
            "LATENTS_DIR": Path(f"./data/latents/{dataset}"),
            # support set generation
            "SELECTION_STRAREGY": selection_strategy,
            "N_CLUSTERS": N_CLUSTERS,  # [64,96,128,160,192,256,320,384,512,640,768],
            "ANCHORS_STRATEGIES": ANCHORS_STRATEGIES,
            "N_SAMPLES": SxC,
            # relative training
            "ENCODER_E2E": ENCODER_RX,
            "SEEDS": SEEDS,
            "SIMILS": SIMILARITIES,
            "EPOCHS": 30,
            "N_CLASSES": 100,
            # relative testing
            "ENCODERS_X": ENCODERS_TX,
            "quantization": QUANTIZATION,
        }
