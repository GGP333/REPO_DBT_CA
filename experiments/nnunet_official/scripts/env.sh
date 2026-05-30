# Source this file before invoking nnUNetv2 CLI commands.
# Usage:
#     source experiments/nnunet_official/scripts/env.sh
EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export nnUNet_raw="${EXP_ROOT}/nnunet_root/raw"
export nnUNet_preprocessed="${EXP_ROOT}/nnunet_root/preprocessed"
export nnUNet_results="${EXP_ROOT}/nnunet_root/results"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
echo "nnUNet_raw=$nnUNet_raw"
echo "nnUNet_preprocessed=$nnUNet_preprocessed"
echo "nnUNet_results=$nnUNet_results"
