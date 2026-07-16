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
"""
Naive rollout using huggingface's generate function, mainly for correctness testing.
"""

import torch

from tensordict import TensorDict

from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length

from ..base import BaseRollout

__all__ = ["NaiveRollout"]


class NaiveRollout(BaseRollout):
    def __init__(self, module: torch.nn.Module, config):
        """A naive rollout. It requires the module to be compatible with huggingface APIs. That is:
        The module should define __call__ to receive input_ids, attention_mask and position_ids.
        It outputs a structure that contains logits and past_key_values.
        We use generate() to generate the sequence.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
        """
        super().__init__()
        self.config = config
        self.module = module

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = prompts.batch["attention_mask"]  # left-padded attention_mask
        position_ids = prompts.batch["position_ids"]

        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]

        batch_size = idx.size(0)
        prompt_length = idx.size(1)

        self.module.eval()
        do_sample = prompts.meta_info.get("do_sample", True)

        response = self.module.generate(
            input_ids=idx,
            attention_mask=attention_mask,
            do_sample=do_sample,
            max_new_tokens=self.config.response_length,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )

        response = response[:, prompt_length:]

        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, pad_token_id)

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        return DataProto(batch=batch)
