#!/bin/bash

set -eu
export TOKENIZERS_PARALLELISM=false

source ~/anaconda3/etc/profile.d/conda.sh
conda activate OpenRLHF
cd ~/GENERator

export CUDA_VISIBLE_DEVICES=2,3

# MODEL_NAME=Nucleotide-Transformer-V2-500M-Multi-Species
# MODEL_PATH=/data1/Mamba/Model/Genome/Nucleotide-Transformer/nucleotide-transformer-v2-500m-multi-species

# MODEL_NAME=HyenaDNA-Large-1M-Seqlen-hf
# MODEL_PATH=/data1/Mamba/Model/Genome/HyenaDNA/hyenadna-large-1m-seqlen-hf

# MODEL_NAME=GENERator-eukaryote-3b-base
# MODEL_PATH=/data1/Mamba/Model/Genome/GENERator/GENERator-eukaryote-3b-base

MODEL_NAME=GENERator-eukaryote-1.2b-base_GRPO_HISTONE-CDS_EPOCHS-5_BS-4_KL-0.0001_ROLLOUT-16-MAXLEN-512_20250406_Step-150
MODEL_PATH=/data2/Mamba/Project/GENERator/GENERator-eukaryote-1.2b-base_GRPO_HISTONE-CDS_EPOCHS-5_BS-4_KL-0.0001_ROLLOUT-16-MAXLEN-512_20250406_Step-150

python src/tasks/downstream/variant_effect_prediction.py \
    --hg38_path /data2/Mamba/Project/Evo-2/Benchmarks/BRCA1_classification/GRCh37_p13_chr17.parquet \
    --clinvar_path /data2/Mamba/Project/Evo-2/Benchmarks/BRCA1_classification/brca1.parquet \
    --model_name $MODEL_PATH \
    --batch_size 128 \
    --num_processes 8 \
    --dp_size 2 \
    --context_length 8192 \
    --output_path /data2/Mamba/Project/GENERator/${MODEL_NAME}/BRCA1_Variant_Effect_Prediction/variant_predictions.parquet 