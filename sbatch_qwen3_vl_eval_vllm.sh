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

# Set path to fine-tuned model checkpoint
MODEL_PATH="saves/Qwen3-VL-8B-Instruct/lora/train_3693802/merged_best"

# Set output directory for predictions
OUTPUT_DIR="${MODEL_PATH}/eval_vllm"
mkdir -p ${OUTPUT_DIR}

# Run vLLM inference - much faster than standard do_predict
python scripts/vllm_infer.py \
    --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
    --adapter_name_or_path ${MODEL_PATH} \
    --template qwen3_vl_nothink \
    --dataset iTRAP_qwen3_vl_both_cams_both_trajs_separate_queries_val \
    --dataset_dir data \
    --save_name ${OUTPUT_DIR}/generated_predictions.jsonl \
    --cutoff_len 2048 \
    --max_samples 100000 \
    --max_new_tokens 1024 \
    --temperature 0.7 \
    --top_p 0.9 \
    --top_k 50 \
    --repetition_penalty 1.0 \
    --batch_size 1024 \
    --image_max_pixels 589824 \
    --image_min_pixels 1024 \
    --video_fps 2.0 \
    --video_maxlen 128 \
    --vllm_config '{"disable_custom_all_reduce": true}'
