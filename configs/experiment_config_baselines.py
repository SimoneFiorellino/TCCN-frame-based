from pathlib import Path

DATASETS = [
    "cifar10",
    "cifar100",
    "tiny_imagenet",
]  # ["cifar10", "cifar100", "tiny_imagenet"] # "cifar20"

ENCODERS_TX = [
    # "rexnet_100",
    # "mobilenetv3_large_100",
    "vit_small_patch16_224",  # qui
    "vit_base_patch16_224",
    "vit_base_patch32_clip_224",  # qui
    "absolute",
]

ENCODER_RX = [
    "vit_base_resnet50_384",
]

ANCHORS_STRATEGIES = ["pseudo", "frames", "procrustes", "linear", "neural"]
N_CLUSTERS = [
    6,
    10,
    16,
    24,
    43,
    64,
    96,
    128,
    192,
    256,
    384,
    512,
    768,
    890,
    1024,
    1280,
    1536,
    2048,
    2560,
    3072,
]
QUANTIZATION = [32]
SNR_DB = [30]
SEEDS = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
SxC = 1

X_RANGE = {
    "cifar10": (5, 2048),
    "cifar100": (15, 3000),
    "tiny_imagenet": (64, 3000),
    # add more dataset ranges as needed...
}

check = True
check_anchors = True
show_mean_only_flag = True
log_scale_flag = True


def get_config(dataset):
    if dataset == "tiny_imagenet":
        return {
            "LATENTS_DIR": Path(f"./data/latents/{dataset}"),
            # support set generation
            # "ENCODERS_FOR_ANCHORS": encoder_for_anchors,
            "N_CLUSTERS": N_CLUSTERS,  # ,512,640,768], # [4,8,12,16,24,32,64,96,128,192,256,384,512,768,890,1024,1280,1536,2048,3072],# [5, 6, 7, 8, 9, 10, 12, 15, 18, 20], # [128,256,384,756,1024]
            "ANCHORS_STRATEGIES": ANCHORS_STRATEGIES,  # ['proto_frames'], # ['proto','frames','pseudo','proto_frames'], # ['frames', 'pseudo'], #
            "N_SAMPLES": SxC,
            # relative training
            "ENCODER_E2E": ENCODER_RX,
            "SEEDS": SEEDS,
            "SIMILS": ["cosine"],
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
            "N_CLUSTERS": N_CLUSTERS,  # [4,8,12,16,24,32,64,96,128,192,256,384,512,756,1024],# [5, 6, 7, 8, 9, 10, 12, 15, 18, 20], #  [2,4,6,8,10,12,16,20,24,32,43,53,64,96,128,160]
            "ANCHORS_STRATEGIES": ANCHORS_STRATEGIES,  # proto_frames
            "N_SAMPLES": SxC,
            # relative training
            "ENCODER_E2E": ENCODER_RX,
            "SEEDS": SEEDS,
            "SIMILS": ["cosine"],
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
            "N_CLUSTERS": N_CLUSTERS,  # [16,20,24,32,48,64,80,96,128,160,192,256],
            "ANCHORS_STRATEGIES": ANCHORS_STRATEGIES,
            "N_SAMPLES": SxC,
            # relative training
            "ENCODER_E2E": ENCODER_RX,
            "SEEDS": SEEDS,
            "SIMILS": ["cosine"],
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
            "N_CLUSTERS": N_CLUSTERS,  # [64,96,128,160,192,256,320,384,512,640,768],
            "ANCHORS_STRATEGIES": ANCHORS_STRATEGIES,
            "N_SAMPLES": SxC,
            # relative training
            "ENCODER_E2E": ENCODER_RX,
            "SEEDS": SEEDS,
            "SIMILS": ["cosine"],
            "EPOCHS": 30,
            "N_CLASSES": 100,
            # relative testing
            "ENCODERS_X": ENCODERS_TX,
            "quantization": QUANTIZATION,
        }
