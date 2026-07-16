# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import warnings
from typing import Optional, Union

import torch
import torch.distributed
from accelerate import init_empty_weights
from torch.distributed.fsdp import FullStateDictConfig, ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig, PreTrainedTokenizer, ProcessorMixin

from verl.utils.device import is_cuda_available
from verl.utils.fs import copy_to_local, is_non_local
from verl.utils.fsdp_utils import fsdp_version, get_fsdp_state_ctx

from .checkpoint_manager import BaseCheckpointManager


def extract_lora_state_dict(model):
    """
    Extract only LoRA parameters from the model state dict.
    LoRA parameters typically have 'lora_' in their names.
    """
    state_dict = {}
    # Handle both FSDP wrapped and unwrapped models
    if hasattr(model, 'module'):
        # For DataParallel or similar wrappers
        model_to_check = model.module
    else:
        model_to_check = model
    
    # Get all parameters
    for name, param in model_to_check.named_parameters():
        if 'lora_' in name:
            state_dict[name] = param.data.cpu() if param.is_cuda else param.data
    
    # Also check for LoRA in state dict directly if available.
    # This model.state_dict() call is a collective FSDP operation and MUST be
    # wrapped in the correct state_dict_type context, or under real multi-GPU
    # FSDP sharding it desyncs flat_param.data from flat_param._local_shard and
    # crashes later in offload_fsdp_model_to_cpu's consistency assertion.
    # (Single-GPU/world_size=1 runs use FSDP's NO_SHARD path and never hit this,
    # which is why this bug can silently pass single-GPU smoke tests.)
    try:
        state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with get_fsdp_state_ctx(model, StateDictType.FULL_STATE_DICT, state_dict_config, None):
            full_state_dict = model.state_dict()
        for name, param in full_state_dict.items():
            if 'lora_' in name:
                state_dict[name] = param.cpu() if param.is_cuda else param
    except:
        pass
        
    return state_dict


def _json_sanitize(value):
    """Recursively convert values not natively JSON-serializable (e.g. PEFT's
    target_modules, which is a set at runtime, or nested dataclass config
    objects like LoraRuntimeConfig) into JSON-safe equivalents."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (set, frozenset)):
        return sorted(value) if all(isinstance(v, str) for v in value) else list(value)
    if isinstance(value, dict):
        return {k: _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    if hasattr(value, "value"):
        # enum.Enum members (e.g. peft.TaskType, peft.PeftType)
        return value.value
    if hasattr(value, "__dict__"):
        # Arbitrary nested config objects (e.g. peft's LoraRuntimeConfig)
        return _json_sanitize(vars(value))
    return value


def save_lora_config(save_directory, lora_config):
    """
    Save LoRA configuration to adapter_config.json
    """
    os.makedirs(save_directory, exist_ok=True)
    config_file = os.path.join(save_directory, "adapter_config.json")

    # Convert LoraConfig to dict if it's an object
    if hasattr(lora_config, '__dict__'):
        config_dict = lora_config.__dict__
    elif isinstance(lora_config, dict):
        config_dict = lora_config
    else:
        # Fallback: create basic config
        config_dict = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "inference_mode": False
        }

    config_dict = _json_sanitize(config_dict)
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2)


def get_lora_config_from_model(model):
    """
    Try to extract LoRA configuration from the model
    """
    # Check if model has PEFT attributes
    if hasattr(model, 'peft_config'):
        if model.peft_config:
            # Return the first peft config (usually there's only one)
            return list(model.peft_config.values())[0]
    
    # Check if model module has peft_config
    if hasattr(model, 'module') and hasattr(model.module, 'peft_config'):
        if model.module.peft_config:
            return list(model.module.peft_config.values())[0]
    
    # Fallback to basic config
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM", 
        "inference_mode": False,
        "r": 32,  # default rank
        "lora_alpha": 64,  # default alpha
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": "all-linear"
    }


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    Manage FSDP checkpointing in SPMD training.

    - Saves/loads per-rank sharded model & optimizer states
    - Persists full lr_scheduler and RNG state
    - Stores HF tokenizer/processor and model/config for unified restore

    Args:
        model (FSDP): Wrapped model instance.
        optimizer (Optimizer): Training optimizer.
        lr_scheduler (LRScheduler): Learning-rate scheduler.
        processing_class (PreTrainedTokenizer or ProcessorMixin, optional):
            Pre-/post-processing artifact handler.
        checkpoint_contents (list[str], optional):
            Components to include; must contain 'model', 'optimizer', 'extra'.
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin] = None,
        checkpoint_contents: Optional[list] = None,
        **kwargs,
    ):
        if checkpoint_contents is None:
            checkpoint_contents = ["model", "optimizer", "extra"]
        if processing_class is None:
            assert "tokenizer" in kwargs, "tokenizer or processor must be provided"
            warnings.warn("`tokenizer` is deprecated. use `processing_class` instead.", DeprecationWarning, stacklevel=2)
            processing_class = kwargs.pop("tokenizer")
        # 支持lora_only模式，或者传统的model/optimizer/extra模式
        if "lora_only" not in checkpoint_contents:
            assert "model" in checkpoint_contents and "optimizer" in checkpoint_contents and "extra" in checkpoint_contents, f"FSDPCheckpointManager must include ['model', 'optimizer', 'extra'] or ['lora_only'], got {checkpoint_contents}"

        super().__init__(
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            processing_class=processing_class,
            checkpoint_contents=checkpoint_contents,
        )

    def _load_lora_checkpoint(self, local_path: str) -> bool:
        """Load LoRA-only checkpoint. Returns True on success."""
        lora_candidates = [
            os.path.join(local_path, "lora_adapter", "adapter_model.safetensors"),
            os.path.join(local_path, "lora_adapter", "adapter_model.bin"),
            os.path.join(local_path, "adapter_model.safetensors"),
            os.path.join(local_path, "adapter_model.bin"),
        ]
        lora_file = next((p for p in lora_candidates if os.path.exists(p)), None)
        if lora_file is None:
            print(f"[rank-{self.rank}]: No LoRA adapter file found in {local_path}, skipping load")
            return False

        print(f"[rank-{self.rank}]: Loading LoRA adapter from {lora_file}")
        if lora_file.endswith(".safetensors"):
            from safetensors.torch import load_file as safetensors_load
            lora_sd = safetensors_load(lora_file, device="cpu")
        else:
            lora_sd = torch.load(lora_file, weights_only=False, map_location="cpu")

        # Gather the current sharded state dict and patch LoRA keys into it.
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
            current_sd = self.model.state_dict()
            matched: dict = {}
            for k, v in lora_sd.items():
                if k in current_sd:
                    matched[k] = v
                else:
                    # PEFT saves keys with "base_model.model." prefix; strip it.
                    stripped = k.replace("base_model.model.", "", 1)
                    if stripped in current_sd:
                        matched[stripped] = v
            if not matched:
                print(f"[rank-{self.rank}]: Warning: no matching LoRA keys found – skipping load")
                return False
            current_sd.update(matched)
            self.model.load_state_dict(current_sd)
        print(f"[rank-{self.rank}]: Loaded {len(matched)} LoRA parameters (optimizer state not restored)")
        return True

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load=False):
        """
        Load an FSDP checkpoint for this rank.

        Downloads and loads:
          - model and optimizer shards
          - extra state dict (scheduler + RNG)
        For lora_only checkpoints, loads only the LoRA adapter weights
        (optimizer/scheduler state is not restored in this mode).

        Args:
            local_path: Directory with per-rank checkpoint files.
            hdfs_path: Unused (for API compatibility).
            del_local_after_load: Remove local files after loading.
        """
        if local_path is None:
            return

        # lora_only checkpoint: no full FSDP shards were saved.
        is_lora_only = (
            "lora_only" in self.checkpoint_contents
            and "model" not in self.checkpoint_contents
        )
        if is_lora_only:
            self._load_lora_checkpoint(local_path)
            return

        # every rank download its own checkpoint
        remote_model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
        remote_optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
        remote_extra_state_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")
        print(f"[rank-{self.rank}]: Loading from {remote_model_path} and {remote_optim_path} and {remote_extra_state_path}")
        local_model_path = copy_to_local(remote_model_path)
        local_optim_path = copy_to_local(remote_optim_path)
        local_extra_state_path = copy_to_local(remote_extra_state_path)

        model_state_dict = torch.load(local_model_path, weights_only=False)
        optimizer_state_dict = torch.load(local_optim_path, weights_only=False)
        extra_state_dict = torch.load(local_extra_state_path, weights_only=False)

        if del_local_after_load:
            try:
                os.remove(local_model_path) if is_non_local(local_model_path) else None
                os.remove(local_optim_path) if is_non_local(local_optim_path) else None
                os.remove(local_extra_state_path) if is_non_local(local_extra_state_path) else None
            except Exception as e:
                print(f"[rank-{self.rank}]: remove local resume ckpt file after loading failed, exception {e} will be ignored")

        lr_scheduler_state_dict = extra_state_dict["lr_scheduler"]

        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
            self.model.load_state_dict(model_state_dict)
            if self.optimizer is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)
        # recover random state
        if "rng" in extra_state_dict:
            # 'rng' may not exist for backward compatibility
            self.load_rng_state(extra_state_dict["rng"])

        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):
        """
        Save an FSDP checkpoint for this rank.

        Writes:
          - model & optimizer shard files
          - extra state dict (scheduler + RNG)
          - HF tokenizer/processor and model/config on rank 0
          - optional full HF model under 'huggingface/' if requested
          - optional LoRA adapter only under 'lora_adapter/' if requested

        Rotates old checkpoints, keeping at most `max_ckpt_to_keep`.

        Args:
            local_path: Target directory for checkpoint files.
            hdfs_path: Unused (for API compatibility).
            global_step: Current training step (used for bookkeeping).
            max_ckpt_to_keep: Number of recent checkpoints to retain.
        """
        if local_path is None:
            return

        # record the previous global step
        self.previous_global_step = global_step

        # remove previous local_path
        if max_ckpt_to_keep and isinstance(max_ckpt_to_keep, int) and max_ckpt_to_keep > 0 and len(self.previous_saved_paths) >= max_ckpt_to_keep:
            keep_start = len(self.previous_saved_paths) - max_ckpt_to_keep + 1
            self.remove_previous_save_local_path(self.previous_saved_paths[:keep_start])
            self.previous_saved_paths = self.previous_saved_paths[keep_start:]

        local_path = self.local_mkdir(local_path)
        torch.distributed.barrier()

        # Check if we should save only LoRA adapter
        save_lora_only = "lora_only" in self.checkpoint_contents and "model" not in self.checkpoint_contents
        
        if save_lora_only:
            # Only save LoRA adapter (much smaller)
            # extract_lora_state_dict() calls model.state_dict() on the FSDP-wrapped
            # module, which is a collective operation -- every rank must call it,
            # otherwise rank 0 blocks forever waiting for the other ranks' shards
            # while they sit idle at the barrier below (observed as a 30-minute
            # torch.distributed.barrier() timeout with world_size > 1).
            try:
                lora_state_dict = extract_lora_state_dict(self.model)
            except Exception as e:
                lora_state_dict = None
                if self.rank == 0:
                    print(f"[rank-{self.rank}]: Error extracting LoRA adapter: {e}")
            if self.rank == 0:
                print(f"[rank-{self.rank}]: Saving LoRA adapter only to {os.path.abspath(local_path)}")
                try:
                    if lora_state_dict:
                        lora_path = os.path.join(local_path, "adapter_model.bin")
                        torch.save(lora_state_dict, lora_path)
                        print(f"[rank-{self.rank}]: Saved LoRA adapter with {len(lora_state_dict)} parameters")

                        # Save adapter config
                        lora_config = get_lora_config_from_model(self.model)
                        save_lora_config(local_path, lora_config)
                        print(f"[rank-{self.rank}]: Saved adapter_config.json")
                    else:
                        print(f"[rank-{self.rank}]: Warning: No LoRA parameters found in model!")
                except Exception as e:
                    print(f"[rank-{self.rank}]: Error saving LoRA adapter: {e}")
                    # Fall back to normal checkpoint saving if LoRA fails
                    save_lora_only = False

                # The lora_only path used to `return` here before ever reaching the
                # tokenizer/model-config save below (which only runs in the full
                # checkpoint branch) -- so every lora_only checkpoint was missing the
                # HF tokenizer/processor and base model config.json entirely, making
                # the saved adapter directory unusable on its own for eval/resume
                # without manually copying those files from the base model path.
                # Save them here too, same as the full-checkpoint branch below.
                try:
                    if fsdp_version(self.model) == 1:
                        unwrap_model = self.model._fsdp_wrapped_module
                    else:
                        unwrap_model = self.model
                    model_config = unwrap_model.config
                    if unwrap_model.can_generate() and hasattr(model_config, "name_or_path") and model_config.name_or_path:
                        generation_config = GenerationConfig.from_pretrained(model_config.name_or_path)
                        generation_config.save_pretrained(local_path)
                    model_config.save_pretrained(local_path)
                    self.processing_class.save_pretrained(local_path)
                    print(f"[rank-{self.rank}]: Saved tokenizer/processor and base model config alongside LoRA adapter")
                except Exception as e:
                    print(f"[rank-{self.rank}]: Error saving tokenizer/model config alongside LoRA adapter: {e}")
            torch.distributed.barrier()
            self.previous_saved_paths.append(local_path)
            return

        # every rank will save its own model and optim shard
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                model_state_dict = self.model.state_dict()
                optimizer_state_dict = self.optimizer.state_dict() if self.optimizer is not None else None
                lr_scheduler_state_dict = self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None

                extra_state_dict = {
                    "lr_scheduler": lr_scheduler_state_dict,
                    "rng": self.get_rng_state(),
                }
                model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                extra_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

                print(f"[rank-{self.rank}]: Saving model to {os.path.abspath(model_path)}")
                print(f"[rank-{self.rank}]: Saving optim to {os.path.abspath(optim_path)}")
                print(f"[rank-{self.rank}]: Saving extra_state to {os.path.abspath(extra_path)}")
                torch.save(model_state_dict, model_path)
                torch.save(optimizer_state_dict, optim_path)  # TODO: address optimizer is None
                torch.save(extra_state_dict, extra_path)

        if self.rank == 0:
            if fsdp_version(self.model) == 1:
                unwrap_model = self.model._fsdp_wrapped_module
            else:
                unwrap_model = self.model

            model_config = unwrap_model.config
            if unwrap_model.can_generate() and hasattr(model_config, "name_or_path") and model_config.name_or_path:
                # Some model's name_or_path is empty if not initialized from pretrained,
                # in this cases, we don't save generation config.
                generation_config = GenerationConfig.from_pretrained(model_config.name_or_path)
                generation_config.save_pretrained(local_path)
            else:
                generation_config = None

            model_config.save_pretrained(local_path)
            self.processing_class.save_pretrained(local_path)

        # wait for everyone to dump to local
        torch.distributed.barrier()

        if "hf_model" in self.checkpoint_contents:
            hf_local_path = os.path.join(local_path, "huggingface")
            os.makedirs(hf_local_path, exist_ok=True)

            # Only rank 0 will save hf model and,
            # offload to cpu to save LLMs which may be too large to fit in one GPU
            state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with get_fsdp_state_ctx(self.model, StateDictType.FULL_STATE_DICT, state_dict_config, None):
                state_dict = self.model.state_dict()

            if self.rank == 0:
                if "ForTokenClassification" in model_config.architectures[0]:
                    from transformers import AutoModelForTokenClassification

                    auto_model_cls = AutoModelForTokenClassification
                elif "ForCausalLM" in model_config.architectures[0]:
                    from transformers import AutoModelForCausalLM

                    auto_model_cls = AutoModelForCausalLM
                elif "ForConditionalGeneration" in model_config.architectures[0]:
                    from transformers import AutoModelForVision2Seq

                    auto_model_cls = AutoModelForVision2Seq
                else:
                    raise NotImplementedError(f"Unknown architecture {model_config['architectures']}")

                with init_empty_weights():
                    save_model = auto_model_cls.from_config(model_config, torch_dtype=torch.bfloat16)
                save_model.to_empty(device="cpu")

                if save_model.can_generate():
                    if generation_config is not None:
                        save_model.generation_config = generation_config
                    else:
                        print(f"Warning: {self.__class__.__name__}.save_checkpoint: Generation config file not found in, using a generation config created from the model config when saving hf_model.")

                save_model.save_pretrained(hf_local_path, state_dict=state_dict)
                self.processing_class.save_pretrained(hf_local_path)
                del state_dict
                del save_model

            # wait for rank0 to dump hf_model to local
            torch.distributed.barrier()

        # Also save LoRA adapter alongside full checkpoint if requested
        if "lora_only" in self.checkpoint_contents and self.rank == 0:
            try:
                lora_state_dict = extract_lora_state_dict(self.model)
                if lora_state_dict:
                    lora_path = os.path.join(local_path, "adapter_model.bin")
                    torch.save(lora_state_dict, lora_path)
                    print(f"[rank-{self.rank}]: Also saved LoRA adapter to {os.path.abspath(lora_path)}")
                    
                    # Save adapter config
                    lora_config = get_lora_config_from_model(self.model)
                    save_lora_config(local_path, lora_config)
            except Exception as e:
                print(f"[rank-{self.rank}]: Warning: Could not save additional LoRA adapter: {e}")

        self.previous_saved_paths.append(local_path)
