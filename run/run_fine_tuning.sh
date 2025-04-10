#!/bin/bash

set -eu
export TOKENIZERS_PARALLELISM=false

source ~/anaconda3/etc/profile.d/conda.sh
conda activate OpenRLHF
cd ~/GENERator

MODEL_TYPE=GENERator-eukaryote-1.2b-base
MODEL_NAME=/data1/Mamba/Model/Genome/GENERator/GENERator-eukaryote-1.2b-base
DATASET_TYPE=Cytochrome-P450-CDS
DATASET_NAME=/data1/Mamba/Dataset/Genome/cytochrome-p450-cds
NUM_EPOCHS=10
BATCH_SIZE=64
LEARNING_RATE=1e-5
MAX_LENGTH=1024

CACHE_DIR=/pri_exthome/Mamba/HuggingFace_Cache
DISTRIBUTED_TYPE=deepspeed
DATE_SUFFIX=$(date +"%Y%m%d_%H%M")
EXPERIMENT_NAME=${MODEL_TYPE}_SFT_${DATASET_TYPE}_EPOCHS-${NUM_EPOCHS}_BS-${BATCH_SIZE}_LR-${LEARNING_RATE}_MAXLEN-${MAX_LENGTH}_${DATE_SUFFIX}
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

GPU_DEVICES="0,1,2,3"
NUM_GPUS_PER_NODE=4
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} torchrun --nproc_per_node=${NUM_GPUS_PER_NODE} src/tasks/downstream/fine_tuning.py \
    --distributed_type $DISTRIBUTED_TYPE \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_EPOCHS \
    --max_length $MAX_LENGTH \
    --pad_to_multiple_of_six \
    --output_dir $OUTPUT_DIR \
    --run_name $EXPERIMENT_NAME \
    >> "${LOG_FILE}" 2>&1