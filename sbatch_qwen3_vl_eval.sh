#!/bin/bash

# Job Settings
#SBATCH -A hk-project-p0024638  # Project name
#SBATCH -J iTRAP_Qwen3_VL_eval  # Job name

# Cluster Settings
#SBATCH -p accelerated          # Partition name
#SBATCH -n 1                    # Number of tasks
#SBATCH --ntasks-per-node=1     # Number of tasks per node
#SBATCH --gres=gpu:1            # Number of GPUs
#SBATCH -c 4                    # Number of cores per task
#SBATCH -t 1:00:00              # Time limit

# Define the paths for storing output and error files
#SBATCH --output=/home/hk-project-p0024638/uruox/DIR/hkfswork/uruox-llama-factory/qwen3_vl/outputs/%x_%j.out
#SBATCH --error=/home/hk-project-p0024638/uruox/DIR/hkfswork/uruox-llama-factory/qwen3_vl/outputs/%x_%j.err

# -------------------------------

# Activate the virtualenv / conda environment
source /home/hk-project-p0024638/uruox/miniconda3/bin/activate lf

export TORCH_USE_CUDA_DSA=1

# Set the path to your fine-tuned model
# Update this to point to your trained checkpoint
MODEL_PATH="saves/Qwen3-VL-8B-Instruct/lora/train_2025-10-31_18-40-23"

# Set output directory for predictions
OUTPUT_DIR="${MODEL_PATH}/eval"
mkdir -p ${OUTPUT_DIR}

# Use train with do_predict to generate predictions on your custom dataset
llamafactory-cli train \
    --stage sft \
    --do_predict True \
    --predict_with_generate True \
    --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
    --adapter_name_or_path ${MODEL_PATH} \
    --finetuning_type lora \
    --template qwen3_vl_nothink \
    --flash_attn sdpa \
    --dataset_dir data \
    --eval_dataset iTRAP_qwen3_vl_val \
    --cutoff_len 2048 \
    --max_samples 100000 \
    --per_device_eval_batch_size 4 \
    --output_dir ${OUTPUT_DIR} \
    --bf16 True \
    --freeze_vision_tower True \
    --freeze_multi_modal_projector True \
    --image_max_pixels 589824 \
    --image_min_pixels 1024 \
    --video_max_pixels 65536 \
    --video_min_pixels 256
