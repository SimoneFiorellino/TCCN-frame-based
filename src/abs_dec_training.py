import itertools
import subprocess
from pathlib import Path

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

# ---------------------------------------
# Load data & invoke function
# ---------------------------------------
if config_type == "baselines":
    import configs.experiment_config_baselines as cfg_module
elif config_type == "compression":
    import configs.experiment_config_compression as cfg_module
elif config_type == "complete":
    import configs.experiment_config_complete as cfg_module
elif config_type == "model_types":
    import configs.experiment_config_model_types as cfg_module
elif config_type == "proto":
    import configs.experiment_config_proto as cfg_module
else:
    raise ValueError("Invalid type.")


# Function to construct the command from parameters
def construct_command(params):
    args = []
    for k, v in params.items():
        key = (
            k  # .replace('.', '_')  # Transform keys to match expected CLI args format
        )
        args.append(f"{key}={v}")
    return args


# Function to run the command
def run_simulation(params):
    args = construct_command(params)
    command = ["python", "-m", "src.train"] + args
    print(f"Running command: {' '.join(command)}")
    try:
        result = subprocess.run(command, capture_output=False, text=True, check=True)
        return result.stdout, None
    except subprocess.CalledProcessError as e:
        return e.stdout, e.stderr


# Main function to handle different datasets
def run_for_datasets(datasets):
    for dataset in datasets:
        cfg = cfg_module.get_config(dataset)
        params_grid = {
            "model": [dataset],
            "model.encoder": cfg["ENCODER_E2E"],
            "model.n_anchors": [cfg["N_CLUSTERS"][0]],
            "model.relative": [False],
            "seed": cfg["SEEDS"],
            "trainer.max_epochs": [cfg["EPOCHS"]],
        }

        # Create all combinations of parameters
        keys, values = zip(*params_grid.items())
        permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # Iterate over each parameter combination and execute the corresponding simulation
        results = []
        for params in permutations:
            # Construct weight path based on parameters
            weights_path = Path("weights/")
            if params["model.relative"]:
                weights_path = (
                    weights_path
                    / f"{params['model']}_{params['model.encoder']}_{params['model.n_anchors']}_{params['seed']}_{params['model.idx_flag']}_{params['model.sim_type']}.ckpt"
                )
            else:
                weights_path = (
                    weights_path
                    / f"{params['model']}_{params['model.encoder']}_{params['seed']}.ckpt"
                )

            # Check if checkpoint already exists
            if weights_path.exists():
                print(f"Trained model already exists: {weights_path}")
                continue  # Skip simulation if the checkpoint exists
            stdout, stderr = run_simulation(params)
            results.append((params, stdout, stderr))

        # Optional: Print results for this dataset
        for result in results:
            print(f"Dataset: {dataset}")
            print(f"Parameters: {result[0]}")
            print(f"Output: {result[1]}")
            if result[2]:
                print(f"Error: {result[2]}")


# Run the simulations for all datasets
run_for_datasets(cfg_module.DATASETS)
