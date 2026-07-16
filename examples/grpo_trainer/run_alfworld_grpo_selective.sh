#!/bin/bash

set -x
set -e

ENGINE=${ENGINE:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

# ---- selectable knobs ----
THINKING_MODE=${THINKING_MODE:-dt_action}   # dt_action | selective_gate
MODEL_PATH=${MODEL_PATH:?'Error: MODEL_PATH must be set to the base model checkpoint path'}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_qwen3_8b_${THINKING_MODE}}
PROJECT_NAME=${PROJECT_NAME:-verl_agent_alfworld_selective}

ACTOR_STRATEGY=${ACTOR_STRATEGY:-fsdp}
CRITIC_STRATEGY=${CRITIC_STRATEGY:-fsdp}

NUM_GPUS=${NUM_GPUS:-2}
NUM_NODES=${NUM_NODES:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-32}
GROUP_SIZE=${GROUP_SIZE:-8}
# Adaptive deep-think call-count reference N_bar_dt (paper Eq. 10 / Appendix B: "initialised at 5,
# the empirical sweet spot for ALFWorld"). Configurable here for N-ablation runs.
DT_REFERENCE_INIT_VAL=${DT_REFERENCE_INIT_VAL:-5.0}
# Success-rate hard-switch threshold for the token-efficiency penalty (paper: lambda_tok
# turns on once SR-bar >= this value). lo==hi by default (hard switch, no ramp);
# configurable here for SR-threshold ablation runs.
SR_THRESHOLD_LO=${SR_THRESHOLD_LO:-0.60}
SR_THRESHOLD_HI=${SR_THRESHOLD_HI:-0.60}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
PPO_MICRO_BATCH=${PPO_MICRO_BATCH:-2}
# log_prob passes (rollout old_log_prob + ref) are forward-only (no grad/optimizer
# state), so they can safely use a larger micro batch than the actor's training
# step even when GPU headroom is tight; defaults to PPO_MICRO_BATCH for backward
# compatibility with existing launch scripts that don't set it.
LOG_PROB_MICRO_BATCH=${LOG_PROB_MICRO_BATCH:-$PPO_MICRO_BATCH}
MAX_STEPS_PER_EP=${MAX_STEPS_PER_EP:-24}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-2048}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-512}

NUM_CPUS_PER_ENV=${NUM_CPUS_PER_ENV:-0.1}

# ---- paths ----
# VERL_TRAIN_DIR: base directory for checkpoints and logs (default: project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VERL_TRAIN_DIR=${VERL_TRAIN_DIR:-${PROJECT_ROOT}}
CHECKPOINT_DIR=${VERL_TRAIN_DIR}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}
LOG_DIR=${VERL_TRAIN_DIR}/logs/${PROJECT_NAME}/${EXPERIMENT_NAME}

# lora_only saves only the LoRA adapter (~1% of full checkpoint size)
CHECKPOINT_CONTENTS=${CHECKPOINT_CONTENTS:-lora_only}

mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${LOG_DIR}

echo "========================================="
echo "Checkpoint dir : ${CHECKPOINT_DIR}"
echo "Log dir        : ${LOG_DIR}"
echo "Checkpoint type: ${CHECKPOINT_CONTENTS}"
echo "========================================="

# ---- env data prep (offline placeholder parquet, no HuggingFace download needed) ----
PARQUET_DIR=${VERL_TRAIN_DIR}/data/verl-agent/text
mkdir -p $PARQUET_DIR
python3 -c "
import os, pandas as pd
local_dir = '${PARQUET_DIR}'
os.makedirs(local_dir, exist_ok=True)
def make_placeholder(n, split):
    return [{'data_source':'text','prompt':[{'role':'user','content':''}],'ability':'agent','extra_info':{'split':split,'index':i}} for i in range(n)]
pd.DataFrame(make_placeholder(${TRAIN_BATCH_SIZE},'train')).to_parquet(os.path.join(local_dir,'train.parquet'))
pd.DataFrame(make_placeholder(${VAL_BATCH_SIZE},'test')).to_parquet(os.path.join(local_dir,'test.parquet'))
print(f'Placeholder parquet generated at {local_dir}')
"

# ---- launch GRPO ----
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${PARQUET_DIR}/train.parquet \
    data.val_files=${PARQUET_DIR}/test.parquet \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=$MAX_RESPONSE_LEN \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_torch_compile=${ACTOR_TORCH_COMPILE:-True} \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.2 \
    actor_rollout_ref.actor.token_efficiency_penalty.base_coef=0.05 \
    actor_rollout_ref.actor.token_efficiency_penalty.sr_threshold_lo=$SR_THRESHOLD_LO \
    actor_rollout_ref.actor.token_efficiency_penalty.sr_threshold_hi=$SR_THRESHOLD_HI \
    actor_rollout_ref.actor.token_efficiency_penalty.window_size=128 \
    actor_rollout_ref.actor.token_efficiency_penalty.token_budget=4096.0 \
    actor_rollout_ref.actor.dt_quality_reward.excess_coef_won=0.01 \
    actor_rollout_ref.actor.dt_quality_reward.adaptive_threshold.init_val=$DT_REFERENCE_INIT_VAL \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD:-false} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$NUM_GPUS \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_GPU_MEM_UTIL:-0.5} \
    actor_rollout_ref.rollout.enable_chunked_prefill=${VLLM_CHUNKED_PREFILL:-False} \
    actor_rollout_ref.rollout.enforce_eager=${VLLM_ENFORCE_EAGER:-false} \
    actor_rollout_ref.rollout.free_cache_engine=${VLLM_FREE_CACHE_ENGINE:-false} \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH \
    actor_rollout_ref.ref.fsdp_config.param_offload=${CRITIC_PARAM_OFFLOAD:-true} \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=$MAX_STEPS_PER_EP \
    env.history_length=5 \
    env.thinking_mode=$THINKING_MODE \
    env.rollout.n=$GROUP_SIZE \
    env.resources_per_worker.num_cpus=$NUM_CPUS_PER_ENV \
    trainer.critic_warmup=0 \
    trainer.logger=${TRAINER_LOGGER:-['console']} \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=${SAVE_FREQ:-10} \
    trainer.test_freq=${TEST_FREQ:-5} \
    trainer.total_epochs=${TOTAL_EPOCHS:-150} \
    trainer.val_before_train=False \
    trainer.default_local_dir=${CHECKPOINT_DIR} \
    trainer.resume_mode=auto \
    actor_rollout_ref.actor.checkpoint.contents=[${CHECKPOINT_CONTENTS}] \
    critic.checkpoint.contents=[${CHECKPOINT_CONTENTS}] \
    actor_rollout_ref.actor.strategy=${ACTOR_STRATEGY} \
    critic.strategy=${CRITIC_STRATEGY}