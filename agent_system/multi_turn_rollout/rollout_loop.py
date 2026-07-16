# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
# Modified by Anonymous Authors, 2026: added selective_thinking_multi_turn_loop for dt_action and selective_gate modes.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# and the verl-agent (GiGPO) team.
#     http://www.apache.org/licenses/LICENSE-2.0
# and the verl-agent (GiGPO) team.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

# Gate decisions need only "Deep_think: true/false" (~5 tokens).
# Cap generation to avoid burning the full action token budget on a trivial binary choice.
_GATE_MAX_RESPONSE_TOKENS = 64
# Outer Thought:/Action: turn is a short two-line reply; native thinking is
# disabled for this call (see extra_template_kwargs below) so 256 is ample.
_ACTION_MAX_RESPONSE_TOKENS = 256
# DT phase 1 (Deep think: reasoning) is the one call allowed to run long.
_DT_REASON_MAX_RESPONSE_TOKENS = 512
# DT phase 2 (Response: memo) only needs to restate the conclusion in 2-4
# sentences given the phase-1 reasoning as context.
_DT_MEMO_MAX_RESPONSE_TOKENS = 256

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
        extra_template_kwargs: dict | None = None,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        apply_chat_template_kwargs = dict(self.config.data.get("apply_chat_template_kwargs", {}))
        if extra_template_kwargs:
            apply_chat_template_kwargs.update(extra_template_kwargs)

        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_systems = obs.get('system', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        obs_system = obs_systems[item] if obs_systems is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content:
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")

        chat_messages: list[dict] = []
        if obs_system is not None and str(obs_system).strip():
            chat_messages.append({"content": str(obs_system), "role": "system"})
        chat_messages.append({"content": obs_content, "role": "user"})
        chat = np.array(chat_messages)
        
        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs
        )
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)
        
        

        if is_multi_modal:

            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto,
        obs: Dict,
        extra_template_kwargs: dict | None = None,
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
                extra_template_kwargs=extra_template_kwargs,
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            total_output_tokens: np.ndarray | None = None,
            action_output_tokens: np.ndarray | None = None,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    # tool_callings
                    data['tool_callings'] = tool_callings[bs]
                    # total_output_tokens
                    if total_output_tokens is not None:
                        data['total_output_tokens'] = total_output_tokens[bs]
                    # action_output_tokens (excludes DT/gate tokens)
                    if action_output_tokens is not None:
                        data['action_output_tokens'] = action_output_tokens[bs]
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value

                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """

        batch_size = len(gen_batch.batch)

        # Initial observations from the environment
        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"
        
        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info

            # pad to be divisible by dp_size
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            # # unpad
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid

            batch = batch.union(batch_output)
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            
            next_obs, rewards, dones, infos = envs.step(text_actions)

            
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            # Create reward tensor, only assign rewards for active environments
            # episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            
            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)
                
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break
        
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings

    # ------------------------------------------------------------------
    # selective-thinking rollout loop (dt_action / selective_gate)
    # ------------------------------------------------------------------
    def _generate_with_obs(
        self,
        *,
        obs_systems: list[str | None],
        obs_texts: list[str],
        gen_batch: DataProto,
        actor_rollout_wg,
        active_indices: list[int] | None = None,
        return_batch: bool = False,
        extra_template_kwargs: dict | None = None,
    ) -> list[str]:
        """One-shot helper that runs `actor_rollout_wg.generate_sequences`
        with custom (system, user) prompt strings per env and returns the
        decoded text outputs.

        When return_batch=True, also returns the combined DataProto (prompt +
        response tensors) so the caller can insert the step into the training
        trajectory (used for selective_gate: gate tokens enter GRPO batch).
        """
        sub_obs = {
            "text": obs_texts,
            "image": None,
            "anchor": list(obs_texts),
            "system": obs_systems,
        }
        if active_indices is None:
            active_indices = list(range(len(obs_texts)))
        sub_gen_batch = gen_batch.select_idxs(active_indices) if hasattr(gen_batch, "select_idxs") else gen_batch
        batch = self.preprocess_batch(gen_batch=sub_gen_batch, obs=sub_obs, extra_template_kwargs=extra_template_kwargs)

        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        batch_input = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        batch_input.meta_info = gen_batch.meta_info

        batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
        batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
        batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)
        decoded = self.tokenizer.batch_decode(batch_output.batch["responses"], skip_special_tokens=True)
        # Count response token lengths from attention_mask for accurate token accounting.
        resp_masks = batch_output.batch["attention_mask"][:, -batch_output.batch["responses"].shape[1]:]
        token_counts = resp_masks.sum(dim=-1).cpu().tolist()
        if return_batch:
            combined = batch.union(batch_output)
            return decoded, token_counts, combined
        return decoded, token_counts

    def selective_thinking_multi_turn_loop(
            self,
            gen_batch: DataProto,
            actor_rollout_wg,
            envs,  # AlfWorldSelectiveThinkingEnvironmentManager
            ) -> DataProto:
        """Rollout loop for the selective-thinking modes.

        Per env step the loop performs (dt_action mode):
          1. action generation with deep_think available (native thinking off,
             256-token cap)
          2. if response is `Action: deep_think`:
                - mark the row is_dt_step=True, active_masks=True (its tokens
                  drive GRPO loss for the gate decision), DO NOT step env
                - run a deep_think generation in TWO separate calls (native
                  thinking off for both): phase 1 with
                  INNER_DEEP_THINK_SYSTEM_PROMPT_REASON (512-token cap) produces
                  the "Deep think:" reasoning; phase 2 with
                  INNER_DEEP_THINK_SYSTEM_PROMPT_MEMO (128-token cap), given
                  phase 1's reasoning as context, produces the "Response:" memo.
                  Splitting into two calls guarantees the memo is never
                  truncated mid-reasoning the way a single combined call could be.
                  (their tokens do NOT enter the trajectory)
                - state.record_deep_think(memo)
                - re-prompt with deep_think disabled and run another action
                  generation; that response drives env.step
             else: step env directly
        For selective_gate mode:
          0. gate generation with TWO_PHASE_GATE_SYSTEM_PROMPT (tokens DO NOT
             enter trajectory; we want gate behaviour to stay close to SFT)
          1. if gate=true: two-phase deep-think generation (see above) -> memo
          2. action generation (system prompt depends on whether DT just ran)
             -> response drives env.step
        """
        from agent_system.environments.env_package.alfworld.selective_thinking import (
            DT_ACTION_SYSTEM_PROMPT_DT_AVAIL,
            DT_ACTION_SYSTEM_PROMPT_DT_USED,
            INNER_DEEP_THINK_SYSTEM_PROMPT_REASON,
            INNER_DEEP_THINK_SYSTEM_PROMPT_MEMO,
            TWO_PHASE_GATE_SYSTEM_PROMPT,
            TWO_PHASE_ACTION_SYSTEM_PROMPT_POST_DT,
            TWO_PHASE_ACTION_SYSTEM_PROMPT_NO_DT,
            ACTION_GUIDANCE_BASE,
            extract_action_text,
            extract_gate_decision,
            extract_memo_text,
            extract_reason_text,
            is_deep_think_call,
            build_gate_user,
        )

        mode = envs.thinking_mode
        batch_size = len(gen_batch.batch)

        obs, _ = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))
        envs.reset_dt_states()

        if self.config.env.rollout.n > 0:
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else:
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(batch_size)], dtype=object)

        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list: list[list[dict]] = [[] for _ in range(batch_size)]
        total_infos: list[list[dict]] = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        # Track total output tokens per episode (action + gate + DT).
        total_output_tokens = np.zeros(batch_size, dtype=np.float32)
        # Track action-only output tokens (excludes DT and gate tokens).
        action_output_tokens = np.zeros(batch_size, dtype=np.float32)
        # tool_callings doubles as the deep_think counter for the reward path.
        tool_callings = np.zeros(batch_size, dtype=np.float32)

        max_steps = self.config.env.max_steps
        for env_step in range(max_steps):
            active_masks = np.logical_not(is_done)
            states = envs.dt_states

            # ---------- (selective_gate only) phase-1 gate ----------
            gate_decisions = [True] * batch_size  # default for dt_action: action generates everything
            if mode == "selective_gate":
                gate_active_indices = [i for i in range(batch_size) if active_masks[i]]
                gate_systems_active = [TWO_PHASE_GATE_SYSTEM_PROMPT] * len(gate_active_indices)
                gate_users_active = []
                for i in gate_active_indices:
                    # Strip the action_guidance suffix that build_text_obs appended;
                    # the gate sees task context but not the "take an action" directive.
                    user_ctx = obs["text"][i]
                    if user_ctx.endswith(ACTION_GUIDANCE_BASE):
                        user_ctx = user_ctx[: -len(ACTION_GUIDANCE_BASE)].rstrip()
                    facts_text = build_gate_user(state=states[i], current_step=env_step + 1)
                    gate_users_active.append(f"{user_ctx}\n\n{facts_text}")

                gate_gen_batch = copy.copy(gen_batch)
                gate_gen_batch.meta_info = {**gen_batch.meta_info, 'response_length': _GATE_MAX_RESPONSE_TOKENS}
                gate_outs, gate_token_counts, gate_combined_batch = self._generate_with_obs(
                    obs_systems=gate_systems_active,
                    obs_texts=gate_users_active,
                    gen_batch=gate_gen_batch,
                    actor_rollout_wg=actor_rollout_wg,
                    active_indices=gate_active_indices,
                    return_batch=True,
                    extra_template_kwargs={'enable_thinking': False},
                )
                gate_batch_list = to_list_of_dict(gate_combined_batch)

                for j, i in enumerate(gate_active_indices):
                    total_output_tokens[i] += gate_token_counts[j]
                    decision = extract_gate_decision(gate_outs[j])
                    # Default to false if unparseable, keeping gate sparse.
                    gate_decisions[i] = bool(decision) and not states[i].just_called_dt

                    # Gate steps enter training batch so GRPO can directly
                    # optimize gate=true vs gate=false decisions.
                    gate_entry = gate_batch_list[j]
                    gate_entry['uid'] = uid_batch[i]
                    gate_entry['traj_uid'] = traj_uid[i]
                    gate_entry['is_gate_step'] = True
                    gate_entry['gate_decision'] = gate_decisions[i]  # True/False; used by penalty
                    gate_entry['is_action_valid'] = True
                    gate_entry['rewards'] = 0.0
                    gate_entry['active_masks'] = True
                    gate_entry['is_dt_step'] = False
                    total_batch_list[i].append(gate_entry)

                for i in range(batch_size):
                    if not active_masks[i]:
                        gate_decisions[i] = False

            # ---------- (selective_gate only) phase-1.5 DT generation (two calls) ----------
            if mode == "selective_gate":
                dt_indices = [i for i in range(batch_size) if active_masks[i] and gate_decisions[i]]
                if dt_indices:
                    # Only generate for the envs that actually need DT (not all batch_size).
                    dt_reason_systems_active = [INNER_DEEP_THINK_SYSTEM_PROMPT_REASON] * len(dt_indices)
                    dt_users_active = [obs["text"][i] for i in dt_indices]
                    dt_reason_gen_batch = copy.copy(gen_batch)
                    dt_reason_gen_batch.meta_info = {
                        **gen_batch.meta_info, 'response_length': _DT_REASON_MAX_RESPONSE_TOKENS,
                    }
                    reason_outs, reason_token_counts = self._generate_with_obs(
                        obs_systems=dt_reason_systems_active,
                        obs_texts=dt_users_active,
                        gen_batch=dt_reason_gen_batch,
                        actor_rollout_wg=actor_rollout_wg,
                        active_indices=dt_indices,
                        extra_template_kwargs={'enable_thinking': False},
                    )

                    dt_memo_systems_active = [INNER_DEEP_THINK_SYSTEM_PROMPT_MEMO] * len(dt_indices)
                    dt_memo_users_active = [
                        f"{dt_users_active[j]}\n\nYour deliberation:\n{extract_reason_text(reason_outs[j])}"
                        for j in range(len(dt_indices))
                    ]
                    dt_memo_gen_batch = copy.copy(gen_batch)
                    dt_memo_gen_batch.meta_info = {
                        **gen_batch.meta_info, 'response_length': _DT_MEMO_MAX_RESPONSE_TOKENS,
                    }
                    memo_outs, memo_token_counts = self._generate_with_obs(
                        obs_systems=dt_memo_systems_active,
                        obs_texts=dt_memo_users_active,
                        gen_batch=dt_memo_gen_batch,
                        actor_rollout_wg=actor_rollout_wg,
                        active_indices=dt_indices,
                        extra_template_kwargs={'enable_thinking': False},
                    )

                    for j, i in enumerate(dt_indices):
                        total_output_tokens[i] += reason_token_counts[j] + memo_token_counts[j]
                        memo = extract_memo_text(memo_outs[j])
                        states[i].record_deep_think(memo, env_step + 1)
                        tool_callings[i] += 1.0

            # ---------- action / dt_action outer generation ----------
            # Build per-env system prompts for the action call. obs["text"]
            # already has the action_guidance that matches state.just_called_dt.
            outer_systems: list[str | None] = []
            for i in range(batch_size):
                if mode == "dt_action":
                    outer_systems.append(
                        DT_ACTION_SYSTEM_PROMPT_DT_USED
                        if states[i].just_called_dt
                        else DT_ACTION_SYSTEM_PROMPT_DT_AVAIL
                    )
                else:  # selective_gate
                    outer_systems.append(
                        TWO_PHASE_ACTION_SYSTEM_PROMPT_POST_DT
                        if gate_decisions[i] and active_masks[i]
                        else TWO_PHASE_ACTION_SYSTEM_PROMPT_NO_DT
                    )
            obs_with_system = {**obs, "system": outer_systems}

            # Outer Thought:/Action: turn: native thinking off, 256-token cap
            # (see _ACTION_MAX_RESPONSE_TOKENS comment near the top of this file).
            action_gen_batch = copy.copy(gen_batch)
            action_gen_batch.meta_info = {
                **gen_batch.meta_info, 'response_length': _ACTION_MAX_RESPONSE_TOKENS,
            }
            batch = self.preprocess_batch(
                gen_batch=action_gen_batch,
                obs=obs_with_system,
                extra_template_kwargs={'enable_thinking': False},
            )

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )
            batch_input.meta_info = action_gen_batch.meta_info
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid
            batch = batch.union(batch_output)

            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)

            # Accumulate action response tokens into total_output_tokens and action_output_tokens.
            resp_masks = batch_output.batch['attention_mask'][:, -batch_output.batch['responses'].shape[1]:]
            action_token_counts = resp_masks.sum(dim=-1).cpu().tolist()
            for i in range(batch_size):
                if active_masks[i]:
                    total_output_tokens[i] += action_token_counts[i]
                    action_output_tokens[i] += action_token_counts[i]

            # --- dt_action: detect deep_think calls before stepping env ---
            is_dt_step_arr = np.zeros(batch_size, dtype=bool)
            inplace_actions = list(text_actions)  # we mutate per env
            if mode == "dt_action":
                dt_indices = []
                for i in range(batch_size):
                    if not active_masks[i]:
                        continue
                    if states[i].just_called_dt:
                        # DT was already used last call; cannot call again.
                        continue
                    parsed = extract_action_text(text_actions[i])
                    if is_deep_think_call(parsed):
                        is_dt_step_arr[i] = True
                        dt_indices.append(i)
                if dt_indices:
                    dt_reason_systems_active = [INNER_DEEP_THINK_SYSTEM_PROMPT_REASON] * len(dt_indices)
                    dt_users_active = [obs["text"][i] for i in dt_indices]
                    dt_reason_gen_batch = copy.copy(gen_batch)
                    dt_reason_gen_batch.meta_info = {
                        **gen_batch.meta_info, 'response_length': _DT_REASON_MAX_RESPONSE_TOKENS,
                    }
                    reason_outs, reason_token_counts = self._generate_with_obs(
                        obs_systems=dt_reason_systems_active,
                        obs_texts=dt_users_active,
                        gen_batch=dt_reason_gen_batch,
                        actor_rollout_wg=actor_rollout_wg,
                        active_indices=dt_indices,
                        extra_template_kwargs={'enable_thinking': False},
                    )

                    dt_memo_systems_active = [INNER_DEEP_THINK_SYSTEM_PROMPT_MEMO] * len(dt_indices)
                    dt_memo_users_active = [
                        f"{dt_users_active[j]}\n\nYour deliberation:\n{extract_reason_text(reason_outs[j])}"
                        for j in range(len(dt_indices))
                    ]
                    dt_memo_gen_batch = copy.copy(gen_batch)
                    dt_memo_gen_batch.meta_info = {
                        **gen_batch.meta_info, 'response_length': _DT_MEMO_MAX_RESPONSE_TOKENS,
                    }
                    memo_outs, memo_token_counts = self._generate_with_obs(
                        obs_systems=dt_memo_systems_active,
                        obs_texts=dt_memo_users_active,
                        gen_batch=dt_memo_gen_batch,
                        actor_rollout_wg=actor_rollout_wg,
                        active_indices=dt_indices,
                        extra_template_kwargs={'enable_thinking': False},
                    )

                    for j, i in enumerate(dt_indices):
                        total_output_tokens[i] += reason_token_counts[j] + memo_token_counts[j]
                        memo = extract_memo_text(memo_outs[j])
                        states[i].record_deep_think(memo, env_step + 1)
                        tool_callings[i] += 1.0
                    # For dt_action steps we replace the projection input with
                    # an empty string; projection_f will mark it invalid but
                    # we override is_action_valid below.
                    for i in dt_indices:
                        inplace_actions[i] = "Action: <action>noop</action>"  # parser will fail -> invalid

            # ---- step the env (for non-DT rows) ----
            next_obs, rewards, dones, infos = envs.step(inplace_actions)

            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                dones = dones.squeeze(1)

            # Override: a DT-call row did NOT execute a real env action; we
            # neither reward, mark done, nor mark invalid.
            for i in range(batch_size):
                if is_dt_step_arr[i]:
                    rewards[i] = 0.0
                    dones[i] = False
                    infos[i]['is_action_valid'] = True  # the deep_think call itself is "valid"

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array(
                    [info['is_action_valid'] for info in infos], dtype=bool
                )
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            batch.non_tensor_batch['is_dt_step'] = is_dt_step_arr.copy()
            batch.non_tensor_batch['is_gate_step'] = np.zeros(batch_size, dtype=bool)
            batch.non_tensor_batch['gate_decision'] = np.zeros(batch_size, dtype=bool)
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)

            # episode bookkeeping (DT rows count toward DT calls but not steps/rewards)
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            real_action = active_masks & (~is_dt_step_arr)
            episode_lengths[real_action] += 1

            batch_list: list[dict] = to_list_of_dict(batch)
            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # cooldown bookkeeping for dt_action: a DT call sets just_called_dt;
            # a real action clears it.
            for i in range(batch_size):
                if is_dt_step_arr[i]:
                    pass  # already set inside record_deep_think
                elif active_masks[i]:
                    states[i].consume_dt_block()

            is_done = np.logical_or(is_done, dones)
            obs = next_obs
            if is_done.all():
                break

        success: Dict[str, np.ndarray] = envs.success_evaluator(
            total_infos=total_infos,
            total_batch_list=total_batch_list,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
        )
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings, total_output_tokens, action_output_tokens

    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(batch_list=batch_list, 
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                tool_callings=tool_callings, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)

        thinking_mode = getattr(self.config.env, "thinking_mode", None)
        use_selective = thinking_mode in ("dt_action", "selective_gate") and isinstance(
            envs,
            __import__("agent_system.environments.env_manager", fromlist=["AlfWorldSelectiveThinkingEnvironmentManager"]).AlfWorldSelectiveThinkingEnvironmentManager,
        )

        # Initial observations from the environment
        if self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO) — falls back to vanilla
            # for selective thinking modes (filtering hooks expect a single per-step
            # response per env which matches our last-action row by construction).
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        elif use_selective:
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings, total_output_tokens, total_action_output_tokens = \
                self.selective_thinking_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        else:
            # Vanilla Sampling
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            # Vanilla rollouts have no gate/DT tokens; use zeros as placeholder.
            total_output_tokens = np.zeros(len(total_episode_rewards), dtype=np.float32)
            total_action_output_tokens = np.zeros(len(total_episode_rewards), dtype=np.float32)
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)


        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
            total_output_tokens=total_output_tokens,
            action_output_tokens=total_action_output_tokens,
        )
        
        return gen_batch_output
