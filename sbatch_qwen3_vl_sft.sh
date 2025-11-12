#!/bin/bash

# Job Settings
#SBATCH -A hk-project-p0024638  # Project name
#SBATCH -J iTRAP_Qwen3_VL_SFT   # Job name

# Cluster Settings
#SBATCH -p accelerated          # Partition name
#SBATCH -n 1                    # Number of tasks
#SBATCH --ntasks-per-node=1     # Number of tasks per node
#SBATCH --gres=gpu:4            # Number of GPUs
#SBATCH -c 16                   # Number of cores per task
#SBATCH -t 10:00:00             # Time limit

# Define the paths for storing output and error files
#SBATCH --output=/home/hk-project-p0024638/uruox/DIR/hkfswork/uruox-llama-factory/qwen3_vl/logs/%x_%j.out
#SBATCH --error=/home/hk-project-p0024638/uruox/DIR/hkfswork/uruox-llama-factory/qwen3_vl/logs/%x_%j.err

# -------------------------------

# Activate the virtualenv / conda environment
source /home/hk-project-p0024638/uruox/miniconda3/bin/activate lf

export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=0

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
export OUTPUT_DIR="saves/Qwen3-VL-8B-Instruct/lora/train_${TIMESTAMP}_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template qwen3_vl_nothink \
    --flash_attn sdpa \
    --dataset_dir data \
    --dataset iTRAP_qwen3_vl \
    --eval_dataset iTRAP_qwen3_vl_val \
    --do_eval True \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --early_stopping_steps 3 \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 9 \
    --max_samples 100000 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --warmup_steps 20 \
    --packing False \
    --enable_thinking False \
    --report_to none \
    --output_dir $OUTPUT_DIR \
    --bf16 True \
    --save_safetensors True \
    --save_only_model True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch_fused \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all \
    --freeze_vision_tower True \
    --freeze_multi_modal_projector True \
    --image_max_pixels 589824 \
    --image_min_pixels 1024 \
    --video_max_pixels 65536 \
    --video_min_pixels 256

echo "Training completed."

# Find the best checkpoint from trainer_state.json
BEST_CHECKPOINT=$(python3 -c "
import json
try:
    with open('$OUTPUT_DIR/trainer_state.json', 'r') as f:
        state = json.load(f)
    print(state.get('best_model_checkpoint', ''))
except:
    print('')
")

if [ -z "$BEST_CHECKPOINT" ]; then
    echo "Warning: Could not find best_model_checkpoint. Using $OUTPUT_DIR."
    CKPT_PATH=$OUTPUT_DIR
else
    echo "Best checkpoint found: $BEST_CHECKPOINT"
    CKPT_PATH=$BEST_CHECKPOINT
fi

# Run predictions on best checkpoint
llamafactory-cli train \
    --stage sft \
    --do_predict True \
    --predict_with_generate True \
    --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
    --adapter_name_or_path $CKPT_PATH \
    --finetuning_type lora \
    --template qwen3_vl_nothink \
    --flash_attn sdpa \
    --dataset_dir data \
    --eval_dataset iTRAP_qwen3_vl_val \
    --cutoff_len 2048 \
    --max_samples 100000 \
    --per_device_eval_batch_size 4 \
    --output_dir $OUTPUT_DIR/merged_best \
    --bf16 True \
    --freeze_vision_tower True \
    --freeze_multi_modal_projector True \
    --image_max_pixels 589824 \
    --image_min_pixels 1024 \
    --video_max_pixels 65536 \
    --video_min_pixels 256

echo "Predictions saved to $OUTPUT_DIR/generated_predictions.jsonl"

llamafactory-cli export \
    --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
    --adapter_name_or_path $CKPT_PATH \
    --template qwen3_vl_nothink \
    --finetuning_type lora \
    --export_dir $OUTPUT_DIR/merged_best \
    --export_size 4 \
    --export_device cpu \
    --export_legacy_format False

echo "Best checkpoint ($CKPT_PATH) merged and saved to $OUTPUT_DIR/merged_best"
