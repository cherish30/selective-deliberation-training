#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VERL_TRAIN_DIR=${VERL_TRAIN_DIR:-${PROJECT_ROOT}}
LOG_DIR=${VERL_TRAIN_DIR}/logs/grpo
mkdir -p ${LOG_DIR}

LOG_FILE="${LOG_DIR}/grpo_selective_action_$(date +%Y%m%d_%H%M%S).log"

# Mode configuration
export THINKING_MODE=dt_action
export EXPERIMENT_NAME=grpo_qwen3_8b_selective_action
export PROJECT_NAME=verl_agent_alfworld_selective

# Hardware
export CUDA_VISIBLE_DEVICES=0,1
export NUM_GPUS=2

# Hyperparameters
export TRAIN_BATCH_SIZE=8
export VAL_BATCH_SIZE=16
export GROUP_SIZE=8
export PPO_MINI_BATCH_SIZE=32
export PPO_MICRO_BATCH=16
export MAX_STEPS_PER_EP=30
export MAX_PROMPT_LEN=2048
export MAX_RESPONSE_LEN=512

# Memory
export VLLM_GPU_MEM_UTIL=0.6
export ACTOR_PARAM_OFFLOAD=true
export CRITIC_PARAM_OFFLOAD=true
export TORCH_NCCL_TIMEOUT=1800

echo "Starting Sel.-Action GRPO training (background)..."
echo "Log: ${LOG_FILE}"
echo "To follow: tail -f ${LOG_FILE}"

nohup bash examples/grpo_trainer/run_alfworld_grpo_selective.sh > "${LOG_FILE}" 2>&1 &
PID=$!
echo "${PID}" > "${LOG_DIR}/grpo_selective_action_latest.pid"
echo "PID: ${PID}"
