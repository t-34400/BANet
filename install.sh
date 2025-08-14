#!/usr/bin/env bash
set -e

ENV_NAME="banet"
YML_FILE="environment.yml"

if ! command -v mamba &>/dev/null; then
    echo "[INFO] mamba not found. Using conda with libmamba solver."
    conda install -n base -c conda-forge conda-libmamba-solver -y
    conda config --set solver libmamba
    conda config --set channel_priority strict
fi

if command -v mamba &>/dev/null; then
    mamba env create -f "$YML_FILE"
else
    conda env create -f "$YML_FILE"
fi

source "$(conda info --base)/etc/profile.d/conda.sh" || { echo "[ERROR] Failed to source conda.sh"; exit 1; }
conda activate "$ENV_NAME" || { echo "[ERROR] Failed to activate environment '$ENV_NAME'"; exit 1; }

pip install torch==2.4.* torchvision==0.19.* --index-url https://download.pytorch.org/whl/cu124
pip install onnx "timm==0.5.*"

echo "[INFO] Environment '$ENV_NAME' setup completed."
