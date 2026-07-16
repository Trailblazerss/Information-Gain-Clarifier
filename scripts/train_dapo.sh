#!/bin/bash
# DAPO Training with Flash Attention
#
# This script trains a language model using DAPO (Decoupled Alignment Policy Optimization)
# with Flash Attention 2 for memory efficiency.
#
# Key Features:
# - GRPO advantage estimation
# - PPO Dual-Clip policy loss
# - Entropy bonus for exploration
# - Log probability QA reward
#
# Configuration:
# - Learning rate: 5e-7
# - Clip ratio: [0.2, 0.28]
# - Entropy coefficient: 0.001

set -e

# ==================== Configuration ====================
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Model path - modify this to point to your model
MODEL_PATH="${MODEL_PATH:-/path/to/your/model}"

# Data paths (relative to project root)
TRAIN_FILE="${PROJECT_ROOT}/data/train.parquet"
VAL_FILE="${PROJECT_ROOT}/data/val.parquet"

# Verl installation path - defaults to the sibling verl checkout in this workspace
VERL_DIR="${VERL_DIR:-${PROJECT_ROOT}/../verl}"

# Sequence length configuration
# Optimized: prompt reduced to actual max, response increased to reduce truncation
max_prompt_length=1400
max_response_length=1760

# Batch configuration - gradient accumulation for large batch
# Generated samples = train_prompt_bsz * n_resp_per_prompt = 16 * 8 = 128
# Effective batch = train_prompt_bsz * SELECT_COUNT = 16 * 6 = 96
train_prompt_bsz=16          # prompts per step
gen_prompt_bsz=1             # reduce vLLM concurrency (1*8=8 concurrent)
train_prompt_mini_bsz=2      # PPO mini batch
n_resp_per_prompt=8          # responses per prompt
ppo_epochs=4
ppo_micro_bsz_per_gpu=1      # reduce to 1, use accumulation
ppo_max_token_len_per_gpu=16384
# Gradient accumulation steps = train_prompt_bsz / train_prompt_mini_bsz * n_resp_per_prompt / (ppo_micro_bsz * n_gpus)
# = 16 / 2 * 8 / (1 * 4) = 16 accumulation steps

# Rollout configuration
temperature=0.8
top_p=1.0
top_k=-1

# Fixed parameters
clip_ratio_low=0.2
loss_agg_mode="token-mean"
lr_warmup_steps=60

# GPU configuration - adjust based on your hardware
n_gpus_per_node=4
gpu_memory_utilization=0.60

# vLLM token budget
vllm_max_num_batched_tokens=$((max_prompt_length + max_response_length))

# Output configuration
PROJECT_NAME="dapo-training"

# Training steps
# Adjust based on your dataset size
total_steps=2010
save_steps=500

# ==================== Environment Variables ====================
export PYTHONPATH="${VERL_DIR}:${PROJECT_ROOT}/reward:$PYTHONPATH"
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# ==================== Cleanup Function ====================
cleanup() {
    echo ""
    echo "Cleaning up resources..."
    ray stop --force 2>/dev/null || true
    echo "Resources released"
}
trap cleanup EXIT INT TERM

# ==================== Activate Environment ====================
echo "Activating environment..."
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ig_pipeline}"
if [ "${CONDA_DEFAULT_ENV:-}" = "${CONDA_ENV_NAME}" ]; then
    echo "Using existing conda environment: ${CONDA_DEFAULT_ENV}"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
else
    echo "WARNING: conda.sh not found; continuing without environment activation"
fi

# ==================== Training Function ====================
run_training() {
    local config_name=$1
    local learning_rate=$2
    local clip_ratio_high=$3
    local entropy_coeff=$4
    
    local EXPERIMENT_NAME="${config_name}"
    local OUTPUT_DIR="${PROJECT_ROOT}/outputs/${EXPERIMENT_NAME}"
    local LOG_FILE="${OUTPUT_DIR}/train.log"
    local METRICS_FILE="${OUTPUT_DIR}/metrics.log"
    
    mkdir -p "${OUTPUT_DIR}"
    
    echo ""
    echo "========================================"
    echo "Training Config: ${config_name}"
    echo "========================================"
    echo "Model: $MODEL_PATH"
    echo "Attention: flash_attention_2"
    echo "Sequence: ${max_prompt_length} + ${max_response_length} = $((max_prompt_length + max_response_length)) tokens"
    echo ""
    echo "Parameters:"
    echo "   Learning rate: ${learning_rate}"
    echo "   Clip ratio: [${clip_ratio_low}, ${clip_ratio_high}]"
    echo "   Entropy coeff: ${entropy_coeff}"
    echo ""
    echo "Training:"
    echo "   Total steps: ${total_steps}"
    echo "   Save every: ${save_steps} steps"
    echo ""
    echo "Log files:"
    echo "   Full log: ${LOG_FILE}"
    echo "   Metrics: ${METRICS_FILE}"
    echo "========================================"
    
    # Stop old Ray cluster
    echo ""
    echo "Stopping existing Ray cluster..."
    ray stop --force 2>/dev/null || true
    sleep 2
    
    # Start Ray
    echo "Starting Ray cluster..."
    ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265
    sleep 5
    echo "Ray Cluster started"
    
    # Training
    echo ""
    echo "Starting training..."
    
    cd "${VERL_DIR}"
    
    python3 "${PROJECT_ROOT}/reward/main_trainer.py" \
        data.train_files="${TRAIN_FILE}" \
        data.val_files="${VAL_FILE}" \
        data.prompt_key=prompt \
        data.truncation=left \
        data.max_prompt_length=${max_prompt_length} \
        data.max_response_length=${max_response_length} \
        data.gen_batch_size=${gen_prompt_bsz} \
        data.train_batch_size=${train_prompt_bsz} \
        data.shuffle=True \
        '+data.apply_chat_template_kwargs.enable_thinking=true' \
        actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.agent.num_workers=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.max_num_batched_tokens=${vllm_max_num_batched_tokens} \
        actor_rollout_ref.rollout.temperature=${temperature} \
        actor_rollout_ref.rollout.top_p=${top_p} \
        actor_rollout_ref.rollout.top_k=${top_k} \
        actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
        actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
        actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.rollout.disable_log_stats=True \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=True \
        algorithm.kl_ctrl.kl_coef=0.0 \
        algorithm.filter_groups.enable=False \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
        actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
        actor_rollout_ref.actor.clip_ratio_c=10.0 \
        actor_rollout_ref.actor.ppo_epochs=${ppo_epochs} \
        actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_bsz_per_gpu} \
        actor_rollout_ref.actor.optim.lr=${learning_rate} \
        actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
        actor_rollout_ref.actor.optim.weight_decay=0.01 \
        actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
        actor_rollout_ref.actor.grad_clip=1.0 \
        actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.actor.use_torch_compile=False \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.model.use_remove_padding=False \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.ref.fsdp_config.param_offload=False \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.ref.use_torch_compile=False \
        custom_reward_function.path="${PROJECT_ROOT}/reward/simple_reward.py" \
        custom_reward_function.name=compute_score \
        reward_model.reward_manager=dapo \
        reward_model.enable=False \
        reward_model.enable_resource_pool=False \
        trainer.logger='["console"]' \
        trainer.project_name="${PROJECT_NAME}" \
        trainer.experiment_name="${EXPERIMENT_NAME}" \
        trainer.n_gpus_per_node=${n_gpus_per_node} \
        trainer.nnodes=1 \
        trainer.val_before_train=False \
        trainer.test_freq=-1 \
        trainer.save_freq=${save_steps} \
        trainer.total_epochs=20 \
        trainer.total_training_steps=${total_steps} \
        trainer.default_local_dir="${OUTPUT_DIR}" \
        trainer.resume_mode=disable \
        2>&1 | tee "${LOG_FILE}"
    
    # Extract key metrics
    echo ""
    echo "Extracting key metrics..."
    grep -E "step:|LOG PROB WITH QA|LOG PROB WITHOUT QA|REWARD|log_prob_reward|actor/entropy|actor/pg_loss|actor/grad_norm|advantages|perf/throughput" "${LOG_FILE}" > "${METRICS_FILE}" 2>/dev/null || true
    
    echo ""
    echo "========================================"
    echo "Training completed: ${config_name}"
    echo "   Output directory: ${OUTPUT_DIR}"
    echo "   Full log: ${LOG_FILE}"
    echo "   Key metrics: ${METRICS_FILE}"
    echo "========================================"
    
    # Stop Ray for next config
    ray stop --force 2>/dev/null || true
    sleep 3
}

# ==================== Select Configuration ====================
# Usage:
#   bash train_dapo.sh A      # Run config A (conservative)
#   bash train_dapo.sh B      # Run config B (balanced)
#   bash train_dapo.sh C      # Run config C (aggressive)
#   bash train_dapo.sh all    # Run config A

CONFIG="${1:-all}"

echo ""
echo "========================================"
echo "DAPO Training with Flash Attention"
echo "========================================"
echo "Model: ${MODEL_PATH}"
echo "Attention: flash_attention_2"
echo "Sequence: ${max_prompt_length} + ${max_response_length} = $((max_prompt_length + max_response_length)) tokens"
echo ""
echo "Available Configurations:"
echo "  A: lr=5e-7, clip_high=0.28, entropy=0.001 (conservative)"
echo "  B: lr=1e-6, clip_high=0.30, entropy=0.01  (balanced)"
echo "  C: lr=2e-6, clip_high=0.35, entropy=0.02  (aggressive)"
echo ""
echo "Metrics logged:"
echo "   - log_prob with QA / without QA"
echo "   - reward mean/std/min/max"
echo "   - actor/entropy, actor/pg_loss, actor/grad_norm"
echo "   - advantages mean/max/min"
echo "   - throughput, timing"
echo "========================================"

case "${CONFIG}" in
    A|a)
        echo "Running Config A (conservative)..."
        run_training "config-A-lr5e7-clip28-ent001" "5e-7" "0.28" "0.001"
        ;;
    B|b)
        echo "Running Config B (balanced)..."
        run_training "config-B-lr1e6-clip30-ent01" "1e-6" "0.30" "0.01"
        ;;
    C|c)
        echo "Running Config C (aggressive)..."
        run_training "config-C-lr2e6-clip35-ent02" "2e-6" "0.35" "0.02"
        ;;
    all|ALL)
        echo "Running Config A..."
        run_training "config-A-lr5e7-clip28-ent001" "5e-7" "0.28" "0.001"
        ;;
    *)
        echo "Invalid config: ${CONFIG}"
        echo "Usage: bash $0 [A|B|C|all]"
        echo ""
        echo "Options:"
        echo "  A   - Conservative: lr=5e-7, clip_high=0.28, entropy=0.001"
        echo "  B   - Balanced: lr=1e-6, clip_high=0.30, entropy=0.01"
        echo "  C   - Aggressive: lr=2e-6, clip_high=0.35, entropy=0.02"
        echo "  all - Run config A"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "All requested training completed!"
echo ""
echo "Output directories:"
echo "   ${PROJECT_ROOT}/outputs/"
echo ""
echo "View key metrics:"
echo "   cat ${PROJECT_ROOT}/outputs/config-*/metrics.log"
echo "========================================"
