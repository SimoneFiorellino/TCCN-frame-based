# #!/bin/bash

# # chmod +x name.sh

# echo "Have you checked the experiment.yaml and experiment_config.py setup? [y/n]"
# read answer

# if [ "$answer" = "y" ]; then

# Usage helper
usage() {
  echo "Usage: $0 [--config_type <base|expansion|compression|proto>]"
  exit 1
}

# Default
CONFIG_TYPE="baselines"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config_type)
      shift
      [[ "$1" =~ ^(base|expansion|compression|complete|baselines|proto)$ ]] || {
        echo "Error: config_type must be one of base, expansion, compression."
        usage
      }
      CONFIG_TYPE="$1"
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      ;;
  esac
done

python -m src.anchor.proto_anchor --config_type "$CONFIG_TYPE" # Create the anchor file

python -m src.abs_dec_training --config_type "$CONFIG_TYPE"

python -m src.alignment_test --config_type "$CONFIG_TYPE" # test equalizer

python -m src.plots_multi_acc_k --config_type "$CONFIG_TYPE"
python -m src.plots_multi_recon_k --config_type "$CONFIG_TYPE"

# else
#     echo "Please complete the necessary steps before running these scripts."
#     exit 1
# fi