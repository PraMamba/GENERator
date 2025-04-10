#!/bin/bash

set -eu
export TOKENIZERS_PARALLELISM=false

# Load conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate OpenRLHF
cd ~/GENERator

# Static settings
MODEL_TYPE=GENERator-eukaryote-1.2b-base
MODEL_NAME=/data1/Mamba/Model/Genome/GENERator/GENERator-eukaryote-1.2b-base
DATASET_NAME=/data1/Mamba/Dataset/Genome/nucleotide_transformer_downstream_tasks_revised
BATCH_SIZE=256
MAX_LENGTH=1024
PROBLEM_TYPE="single_label_classification"
MAIN_METRICS="mcc"
CACHE_DIR=/pri_exthome/Mamba/HuggingFace_Cache
DISTRIBUTED_TYPE=deepspeed
GPU_DEVICES="1,3"
NUM_GPUS_PER_NODE=2

# Define subset names
SUBSET_LIST=(
    "H2AFZ"
    "H3K27ac"
    "H3K27me3"
    "H3K36me3"
    "H3K4me1"
    "H3K4me2"
    "H3K4me3"
    "H3K9ac"
    "H3K9me3"
    "H4K20me1"
    "enhancers"
    "enhancers_types"
    "promoter_all"
    "promoter_no_tata"
    "promoter_tata"
    "splice_sites_acceptors"
    "splice_sites_all"
    "splice_sites_donors"
)

# Define learning rates for each subset using associative array
declare -A LR_MAP=(
    [H2AFZ]=2e-5
    [H3K27ac]=2e-5
    [H3K27me3]=5e-5
    [H3K36me3]=1e-5
    [H3K4me1]=5e-5
    [H3K4me2]=2e-5
    [H3K4me3]=2e-5
    [H3K9ac]=2e-5
    [H3K9me3]=1e-4
    [H4K20me1]=5e-5
    [enhancers]=5e-5
    [enhancers_types]=2e-5
    [promoter_all]=1e-4
    [promoter_no_tata]=2e-5
    [promoter_tata]=5e-5
    [splice_sites_acceptors]=5e-5
    [splice_sites_all]=1e-4
    [splice_sites_donors]=5e-5
)

for SUBSET_NAME in "${SUBSET_LIST[@]}"; do
    # Lookup learning rate, default to 2e-5 if not found
    LEARNING_RATE=${LR_MAP[$SUBSET_NAME]:-2e-5}

    # Create experiment info
    DATE_SUFFIX=$(date +"%Y%m%d")
    DATASET_TYPE=NT_${SUBSET_NAME}
    EXPERIMENT_NAME=${MODEL_TYPE}_SFT_${DATASET_TYPE}_BS-${BATCH_SIZE}_LR-${LEARNING_RATE}_MAXLEN-${MAX_LENGTH}_${DATE_SUFFIX}
    OUTPUT_DIR=/data2/Mamba/Project/GENERator/${EXPERIMENT_NAME}

    mkdir -p "${OUTPUT_DIR}"
    LOG_FILE="${OUTPUT_DIR}/model_train.log"
    echo "Logging to $LOG_FILE"
    > "$LOG_FILE"

    echo "=============================================="
    echo "🚀 Training on subset: ${SUBSET_NAME}"
    echo "📈 Learning rate: ${LEARNING_RATE}"
    echo "📝 Real-Time Training Log:"
    echo "tail -f ${LOG_FILE}"
    echo "=============================================="

    CUDA_VISIBLE_DEVICES=${GPU_DEVICES} torchrun --nproc_per_node=${NUM_GPUS_PER_NODE} src/tasks/downstream/sequence_understanding.py \
        --distributed_type $DISTRIBUTED_TYPE \
        --model_name $MODEL_NAME \
        --dataset_name $DATASET_NAME \
        --subset_name ${SUBSET_NAME} \
        --batch_size $BATCH_SIZE \
        --problem_type ${PROBLEM_TYPE} \
        --main_metrics ${MAIN_METRICS} \
        --learning_rate $LEARNING_RATE \
        --max_length $MAX_LENGTH \
        --early_stopping_patience 5 \
        --output_dir $OUTPUT_DIR \
        --run_name $EXPERIMENT_NAME \
        >> "${LOG_FILE}" 2>&1
done
