#!/bin/bash

set -eu
export TOKENIZERS_PARALLELISM=false

source ~/anaconda3/etc/profile.d/conda.sh
conda activate OpenRLHF
cd ~/GENERator

MODEL_TYPE=GENERator-eukaryote-1.2b-base
MODEL_NAME=/data1/Mamba/Model/Genome/GENERator/GENERator-eukaryote-1.2b-base
DATASET_TYPE=NT_H2AFZ
DATASET_NAME=/data1/Mamba/Dataset/Genome/nucleotide_transformer_downstream_tasks_revised
SUBSET_NAME=H2AFZ
BATCH_SIZE=256
LEARNING_RATE=2e-5
MAX_LENGTH=1024
PROBLEM_TYPE="single_label_classification"
MAIN_METRICS="mcc"

CACHE_DIR=/pri_exthome/Mamba/HuggingFace_Cache
DISTRIBUTED_TYPE=deepspeed
DATE_SUFFIX=$(date +"%Y%m%d")
EXPERIMENT_NAME=${MODEL_TYPE}_SFT_${DATASET_TYPE}_BS-${BATCH_SIZE}_LR-${LEARNING_RATE}_MAXLEN-${MAX_LENGTH}_${DATE_SUFFIX}
OUTPUT_DIR=/data2/Mamba/Project/GENERator/${EXPERIMENT_NAME}

mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/model_train.log"
if [ -f "$LOG_FILE" ]; then
    echo "Overwrite Log: $LOG_FILE"
    > "$LOG_FILE"
else
    echo "Create Log: $LOG_FILE"
    touch "$LOG_FILE"
fi

echo "=============================================="
echo "Real-Time Training Log Monitoring"
echo "tail -f ${LOG_FILE}"
echo "=============================================="

GPU_DEVICES="1,3"
NUM_GPUS_PER_NODE=2
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
    --early_stopping_patience 10 \
    --output_dir $OUTPUT_DIR \
    --run_name $EXPERIMENT_NAME \
    >> "${LOG_FILE}" 2>&1
