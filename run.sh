#!/bin/bash

# 新增3行，屏蔽损坏的用户库，关闭tf警告
#export PYTHONNOUSERSITE=1
#export TF_ENABLE_ONEDNN_OPTS=0
## 保证脚本内全程使用当前base conda python
#source /usr/local/iCompute/etc/profile.d/conda.sh
#conda activate base

# 下面你原来全部run.sh内容保持原样
#=============================================================================
# KFNpro - One-Click Run Script
# Robust Few-Shot Text Classification with Variational Prototype Learning
#=============================================================================
#
# Usage:
#   bash run.sh                     # Run with default settings (HuffPost, 1-shot, clean, BERT)
#   bash run.sh --help              # Show all options
#
# Before running:
#   1. Install dependencies: pip install -r requirements.txt
#   2. Prepare datasets (see --help)
#
#=============================================================================

set -e  # Exit on error

#=============================================================================
# ====== USER CONFIGURATION (Modify these) ======
#=============================================================================

# GPU device ID (0, 1, 2, ... or -1 for CPU)
GPU=0



# Few-shot setting: 1 or 5
KSHOT=5

# Experiment mode: clean | attack
MODE="clean"

# Model type: bert | custom | llm
#   bert:   standard BERT config (bert-base-uncased / deberta-v3-base)
#   custom: dataset-specific fine-tuned model (oos, banking, liu, clinic, hwu)
#   llm:    LLM config with LoRA (Qwen2.5-1.5B / Llama-3.2-1B / etc.)
MODEL_TYPE="custom"

# Random seed
SEED=101
# Dataset name: huffpost | 20news | amazon | banking | hwu | liu | oos | reuters
DATASET_NAME="liu"
# Dataset splits to run (space-separated): "01" "02" "03" "04" "05"
# Use "01" only for quick test
DATASET_NUMS=("01")

# Pre-trained model path (auto-selected by MODEL_TYPE, or override here)
# Examples:
#   BERT:    bert-base-uncased, microsoft/deberta-v3-base
#   LLM:     Qwen/Qwen2.5-1.5B, meta-llama/Llama-3.2-1B, mistralai/Mistral-7B-v0.3
MODEL_PATH=""

# Data root directory (where HuffPost/, attack_HuffPost/, etc. are located)
DATA_ROOT="../data"

# Output directory for results
OUTPUT_ROOT="../output"

#=============================================================================
# ====== ADVANCED CONFIGURATION (Usually no need to change) ======
#=============================================================================

# Training hyperparameters (will be loaded from JSON config, these are overrides)
EPOCHS=-1           # -1 = use config default
LEARNING_RATE=-1.0  # -1.0 = use config default
WARMUP_STEPS=-1     # -1 = use config default
WEIGHT_DECAY=0.00   # 0.00 = use config default
NUMFREEZE=0         # Number of BERT layers to freeze (0 = train all)
PROMPT_LEN=-1       # -1 = use config default
POOL_LEN=-1         # -1 = use config default

#=============================================================================
# ====== HELP ======
#=============================================================================

if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "=============================================="
    echo "KFNpro Run Script"
    echo "=============================================="
    echo ""
    echo "Configuration variables (edit in run.sh):"
    echo "  GPU           - GPU device ID (default: 0)"
    echo "  DATASET_NAME  - Dataset: huffpost|20news|amazon|banking|hwu|liu|oos|reuters"
    echo "  KSHOT         - Few-shot K: 1 or 5"
    echo "  MODE          - Experiment mode: clean or attack"
    echo "  SEED          - Random seed (default: 101)"
    echo "  DATASET_NUMS  - Which data splits to run (default: 01)"
    echo "  MODEL_PATH    - Path to BERT model (default: bert-base-uncased)"
    echo "  DATA_ROOT     - Path to data directory"
    echo ""
    echo "Setup steps:"
    echo "  1. pip install -r requirements.txt"
    echo "  2. Download BERT: python -c \"from transformers import AutoModel; AutoModel.from_pretrained('bert-base-uncased')\""
    echo "  3. Download data from: https://github.com/tttyyyzzz-zty/SELP/tree/master"
    echo "  4. Place data in: ${DATA_ROOT}/{DatasetName}/{01-05}/"
    echo "  5. (Optional) Generate attack data using BERT-Attack"
    echo ""
    exit 0
fi

#=============================================================================
# ====== SETUP ======
#=============================================================================

# Determine script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "KFNpro Experiment Runner"
echo "=============================================="
echo "  Dataset:    ${DATASET_NAME}"
echo "  K-shot:     ${KSHOT}"
echo "  Mode:       ${MODE}"
echo "  Model Type: ${MODEL_TYPE}"
echo "  GPU:        ${GPU}"
echo "  Seed:       ${SEED}"
echo "  Splits:     ${DATASET_NUMS[*]}"
echo "=============================================="

# Determine config file based on MODEL_TYPE
if [[ "$MODEL_TYPE" == "llm" ]]; then
    # LLM mode: use _llm config files
    if [[ "$MODE" == "clean" ]]; then
        CONFIG_FILE="./clean/${DATASET_NAME}_${KSHOT}shot_llm.json"
    else
        CONFIG_FILE="./attack/attack_${DATASET_NAME}_${KSHOT}shot_llm.json"
    fi
    # Fallback: if LLM config doesn't exist, use default + warn
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "WARNING: LLM config not found: $CONFIG_FILE"
        echo "Using default BERT config. Add use_lora/model settings via CLI."
        if [[ "$MODE" == "clean" ]]; then
            CONFIG_FILE="./clean/${DATASET_NAME}_${KSHOT}shot.json"
        else
            CONFIG_FILE="./attack/attack_${DATASET_NAME}_${KSHOT}shot.json"
        fi
    fi
elif [[ "$MODEL_TYPE" == "custom" ]]; then
    # Custom mode: dataset-specific fine-tuned model
    if [[ "$MODE" == "clean" ]]; then
        CONFIG_FILE="./clean/${DATASET_NAME}_${KSHOT}shot_custom.json"
    else
        CONFIG_FILE="./attack/attack_${DATASET_NAME}_${KSHOT}shot_custom.json"
    fi
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "WARNING: Custom config not found, fallback to BERT"
        if [[ "$MODE" == "clean" ]]; then
            CONFIG_FILE="./clean/${DATASET_NAME}_${KSHOT}shot.json"
        else
            CONFIG_FILE="./attack/attack_${DATASET_NAME}_${KSHOT}shot.json"
        fi
    fi
else
    # BERT mode: standard config files
    if [[ "$MODE" == "clean" ]]; then
        if [[ "$KSHOT" == "1" ]]; then
            CONFIG_FILE="./clean/${DATASET_NAME}_1shot.json"
        else
            CONFIG_FILE="./clean/${DATASET_NAME}_5shot.json"
        fi
    else
        if [[ "$KSHOT" == "1" ]]; then
            CONFIG_FILE="./attack/attack_${DATASET_NAME}_1shot.json"
        else
            CONFIG_FILE="./attack/attack_${DATASET_NAME}_5shot.json"
        fi
    fi
fi

TASK="5w${KSHOT}s"
OUTPUT_TAG="run"

# Check config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    echo "Available configs:"
    ls -1 ./clean/*.json ./attack/*.json 2>/dev/null
    exit 1
fi

echo "  Config:     ${CONFIG_FILE}"
echo ""

#=============================================================================
# ====== PRE-FLIGHT CHECKS ======
#=============================================================================

# Check Python
if ! command -v python &> /dev/null; then
    echo "ERROR: python not found. Please install Python 3.8+."
    exit 1
fi

# Check CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null || {
    echo "WARNING: CUDA not available. Training will be very slow on CPU."
    echo "Press Ctrl+C to abort, or wait 5 seconds to continue..."
    sleep 5
}

# Check dependencies
python -c "import transformers, torch, sklearn, ot, pandas, numpy" 2>/dev/null || {
    echo "ERROR: Missing dependencies. Run: pip install -r requirements.txt"
    exit 1
}

#=============================================================================
# ====== RUN EXPERIMENTS ======
#=============================================================================

echo "Starting experiments..."
echo ""

TOTAL=${#DATASET_NUMS[@]}
CURRENT=0

for k in "${DATASET_NUMS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo "----------------------------------------------"
    echo "  Running split ${k} (${CURRENT}/${TOTAL})"
    echo "----------------------------------------------"

    python main.py --config "$CONFIG_FILE" \
        --seed $SEED \
        --gpu $GPU \
        --kshot $KSHOT \
        --output $OUTPUT_TAG \
        --numFreeze $NUMFREEZE \
        --dataset_num "$k" \
        --dataset_name "$DATASET_NAME" \
        --task "$TASK" \
        --epochs $EPOCHS \
        --learning_rate $LEARNING_RATE \
        --warmup_steps $WARMUP_STEPS \
        --weight_decay $WEIGHT_DECAY \
        --prompt_len $PROMPT_LEN \
        --pool_len $POOL_LEN

    echo "  Split ${k} completed."
    echo ""
done

echo "=============================================="
echo "All experiments completed!"
echo "Results saved to: ${OUTPUT_ROOT}/${DATASET_NAME}/"
echo "=============================================="
