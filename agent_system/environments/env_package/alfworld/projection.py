# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
# Modified by Anonymous Authors, 2026: rewrote action parsing to support 'Action:' format and deep_think token dispatch.
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

from typing import List


_DEEP_THINK_TOKEN = "deep_think"


def _normalize_action_text(text: str) -> str:
    return " ".join((text or "").strip().split()).lower()


def _build_action_lookup(action_pool: List[str] | None) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for action in action_pool or []:
        normalized = _normalize_action_text(action)
        if normalized and normalized not in lookup:
            lookup[normalized] = normalized
    return lookup

def alfworld_projection(actions: List[str], action_pools: List[List[str]]):
    """
    Extract the action text and mark it valid only when it belongs to the
    current admissible command list for that environment step.

    The deep_think tool token is recognized as a separate, always-valid action
    (the rollout loop intercepts it before the env step). It is returned as
    the literal string "deep_think" so callers can dispatch on it.
    """

    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i] or ""
        lowered_str = original_str.lower()
        action_lookup = _build_action_lookup(action_pools[i] if i < len(action_pools) else None)

        # Try `Action: ...` line format first (dt_action / selective_gate);
        # fall back to legacy <action>...</action> tag format.
        extracted_action = ""
        for line in lowered_str.splitlines():
            stripped = line.strip()
            if stripped.startswith("action:"):
                extracted_action = _normalize_action_text(stripped[len("action:"):])
                break
            if stripped.startswith("action="):
                extracted_action = _normalize_action_text(stripped[len("action="):])
                break

        if not extracted_action:
            start_tag = "<action>"
            end_tag = "</action>"
            start_idx = lowered_str.find(start_tag)
            end_idx = lowered_str.find(end_tag)
            try:
                if start_idx == -1 or end_idx == -1:
                    actions[i] = lowered_str[-30:]
                    continue
                extracted_action = _normalize_action_text(
                    lowered_str[start_idx + len(start_tag):end_idx]
                )
            except Exception:
                actions[i] = lowered_str[-30:]
                continue
        if not extracted_action:
            actions[i] = lowered_str[-30:]
            continue

        if extracted_action == _DEEP_THINK_TOKEN:
            actions[i] = _DEEP_THINK_TOKEN
            valids[i] = 1
            continue

        actions[i] = action_lookup.get(extracted_action, extracted_action)
        valids[i] = int(extracted_action in action_lookup)

    return actions, valids
