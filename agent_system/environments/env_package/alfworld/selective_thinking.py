# Copyright 2026 Anonymous Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Selective-thinking helpers shared by:
#   - the offline ALFWorld evaluation script
#     (scripts/eval_alfworld_native_think_modes.py)
#   - the online RL rollout loop (agent_system/multi_turn_rollout/rollout_loop.py)
#
# Two execution patterns are supported:
#
# - dt_action       : the outer model itself emits `Action: deep_think` to
#                     trigger an internal deep-thinking call. After the DT
#                     call, the next outer step disables `deep_think` so the
#                     agent must execute a real environment action.
#
# - selective_gate  : a separate gate call decides true/false; on true the DT
#                     tool runs and produces a memo, then the action call
#                     uses the memo as guidance.
#
# All prompt strings here MUST stay byte-for-byte aligned with the SFT
# templates in scripts/build_alfworld_sft_dt_action.py and the eval prompts
# in scripts/eval_alfworld_native_think_modes.py, otherwise SFT->RL transfer
# breaks.

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable


OUTER_SYSTEM_PROMPT_PREFIX = (
    "You are an expert ALFWorld agent.\n"
    "Follow the required response format exactly.\n"
    "Do not output extra text outside the required response format.\n"
)

DT_ACTION_SYSTEM_PROMPT_DT_AVAIL = (
    OUTER_SYSTEM_PROMPT_PREFIX
    + "Reply in exactly two lines.\n"
    + "Line 1 must start with `Thought:` and contain a brief task-relevant local judgment.\n"
    + "Line 2 must start with `Action:` and contain exactly one admissible action from the prompt "
    + "or the exact token deep_think.\n"
)

DT_ACTION_SYSTEM_PROMPT_DT_USED = (
    OUTER_SYSTEM_PROMPT_PREFIX
    + "Reply in exactly two lines.\n"
    + "Line 1 must start with `Thought:` and contain a brief task-relevant local judgment.\n"
    + "Line 2 must start with `Action:` and contain exactly one admissible action from the prompt.\n"
)

ACTION_GUIDANCE_BASE = (
    "Now it's your turn to take an action.\n"
    "Output one admissible environment action."
)

ACTION_GUIDANCE_DT_AVAIL = (
    "Now it's your turn to take an action.\n"
    "Use deep_think as the main place for task-level thinking and planning.\n"
    "Use Thought only for a brief current-state judgment about the immediate next move.\n"
    "If the task needs search, planning, uncertainty resolution, recovery, or you cannot bind "
    "the plan to one admissible action yet, output `Action: deep_think` instead.\n"
    "Otherwise output one admissible environment action."
)

ACTION_GUIDANCE_DT_USED = (
    "Now it's your turn to take an action.\n"
    "A deep-think memo was just produced. Use it as high-level guidance to choose the best "
    "admissible action for this step.\n"
    "Treat the memo as guidance, not a literal script; pick the admissible action that best "
    "advances its plan.\n"
    "deep_think is not available on this step."
)

INNER_DEEP_THINK_SYSTEM_PROMPT = (
    "You are an internal deep-thinking tool for an ALFWorld agent.\n"
    "You receive the same ALFWorld task context, recent trajectory context, current "
    "observation, and current admissible actions.\n"
    "Reply with exactly two labeled sections in this order:\n\n"
    "Deep think: [your internal reasoning and deliberation]\n"
    "Response: [your concise action memo]\n\n"
    "Use \"Deep think:\" for internal reasoning and deliberation: analyze the current state, "
    "what has been tried, what is blocking progress, and what the best forward plan is. "
    "Reason toward a conclusion rather than listing possibilities.\n"
    "Use \"Response:\" for one compact memo (2-4 direct sentences) that will be written "
    "back into later context as the outer agent's guide. Put the key conclusion and the "
    "concrete forward plan here.\n"
    "Do not output an environment action."
)

TWO_PHASE_GATE_SYSTEM_PROMPT = (
    OUTER_SYSTEM_PROMPT_PREFIX
    + "Reply with exactly one short line and nothing else.\n"
    + "The line must be exactly `Deep_think: true` or `Deep_think: false`.\n"
    + "Default to `Deep_think: false`. Only choose `true` when a specific gate signal applies.\n"
    + "Do not output Thought, Action, explanations, or any other content.\n"
)

TWO_PHASE_ACTION_SYSTEM_PROMPT_POST_DT = (
    OUTER_SYSTEM_PROMPT_PREFIX
    + "Reply in exactly two lines.\n"
    + "Line 1 must start with `Thought:` and contain a brief task-relevant local judgment.\n"
    + "Line 2 must start with `Action:` and contain exactly one admissible action from the prompt.\n"
    + "A fresh deep-think memo was just produced; use it as high-level guidance.\n"
    + "Do not output deep_think; the gate already decided this step.\n"
)

TWO_PHASE_ACTION_SYSTEM_PROMPT_NO_DT = (
    OUTER_SYSTEM_PROMPT_PREFIX
    + "Reply in exactly two lines.\n"
    + "Line 1 must start with `Thought:` and contain a brief task-relevant local judgment.\n"
    + "Line 2 must start with `Action:` and contain exactly one admissible action from the prompt.\n"
    + "Do not output deep_think; the gate already decided this step.\n"
)


# ---------------------------------------------------------------------------
# user-side templates (replace the legacy <think>/<action> alfworld templates)
# ---------------------------------------------------------------------------

ALFWORLD_DT_USER_TEMPLATE_NO_HIS = (
    "You are an expert agent operating in the ALFRED Embodied Environment.\n"
    "Your current observation is: {current_observation}\n"
    "Your admissible actions of the current situation are: [{admissible_actions}].\n\n"
    "{action_guidance}"
)

ALFWORLD_DT_USER_TEMPLATE = (
    "You are an expert agent operating in the ALFRED Embodied Environment. "
    "Your task is to: {task_description}\n"
    "Prior to this step, you have already taken {step_count} step(s). Below are the most "
    "recent {history_length} observations and the corresponding actions you took: "
    "{action_history}\n"
    "{memo_block}"
    "You are now at step {current_step} and your current observation is: "
    "{current_observation}\n"
    "Your admissible actions of the current situation are: [{admissible_actions}].\n\n"
    "{action_guidance}"
)


def render_memo_block(memo_text: str, memo_step: int | None, current_step: int | None) -> str:
    """Render the latest deep-think memo as a separate context block.

    Returns an empty string when there is no memo so the prompt stays compact.
    """
    if not memo_text:
        return ""
    age_clause = ""
    if memo_step is not None and current_step is not None:
        age = max(0, current_step - memo_step)
        age_clause = f" (from step {memo_step}, age {age})"
    return f"Latest deep-think memo{age_clause}: {memo_text}\n"


# ---------------------------------------------------------------------------
# parsing helpers
# ---------------------------------------------------------------------------


_DEEP_THINK_DECISION_LINE_RE = re.compile(
    r"(?im)^\s*deep[_\s-]*think\s*[:=]\s*(true|false|yes|no|y|n|1|0|call|use|skip|act)\b",
)
_DEEP_THINK_DECISION_TAG_RE = re.compile(
    r"<deep[_\s-]*think>\s*(true|false|yes|no|y|n|1|0)\s*</deep[_\s-]*think>",
    flags=re.IGNORECASE,
)


def _strip_think_block(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    if re.search(r"</think>", text, flags=re.IGNORECASE):
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    return text


def extract_gate_decision(raw_output: str) -> bool | None:
    """Parse a `Deep_think: true|false` line. Tolerant of <think> leakage."""
    text = _strip_think_block(raw_output)
    line_match = _DEEP_THINK_DECISION_LINE_RE.search(text)
    if line_match:
        token = line_match.group(1).lower()
        if token in {"true", "yes", "y", "1", "call", "use"}:
            return True
        if token in {"false", "no", "n", "0", "skip", "act"}:
            return False
    tag_match = _DEEP_THINK_DECISION_TAG_RE.search(text)
    if tag_match:
        token = tag_match.group(1).lower()
        if token in {"true", "yes", "y", "1"}:
            return True
        if token in {"false", "no", "n", "0"}:
            return False
    return None


_ACTION_LINE_RE = re.compile(r"(?im)^\s*action\s*[:=]\s*(.+?)\s*$")
_ACTION_TAG_RE = re.compile(
    r"<action>\s*(.+?)\s*</action>", flags=re.IGNORECASE | re.DOTALL
)


def extract_action_text(raw_output: str) -> str:
    """Pull the action substring out of a model output.

    Looks first for `Action: ...`, then for `<action>...</action>`. Returns the
    raw extracted string (lower-case stripped) or an empty string if no match.
    """
    text = _strip_think_block(raw_output)
    line_match = _ACTION_LINE_RE.search(text)
    if line_match:
        return line_match.group(1).strip().lower()
    tag_match = _ACTION_TAG_RE.search(text)
    if tag_match:
        return tag_match.group(1).strip().lower()
    return ""


def is_deep_think_call(action_text: str) -> bool:
    """True if the parsed action text equals the deep_think tool token."""
    return (action_text or "").strip().lower() == "deep_think"


_RESPONSE_BLOCK_RE = re.compile(
    r"(?is)\bresponse\s*[:=]\s*(.+?)(?:\n\s*deep\s*think\s*[:=]|\Z)",
)
_RESPONSE_TAG_RE = re.compile(
    r"<response>\s*(.+?)\s*</response>", flags=re.IGNORECASE | re.DOTALL
)


def extract_memo_text(raw_output: str) -> str:
    """Extract the `Response: ...` memo from a deep-think output.

    Falls back to the whole text if no marker is present so a malformed DT
    call still produces *some* guidance text rather than an empty memo.
    """
    if not raw_output:
        return ""
    text = raw_output
    block_match = _RESPONSE_BLOCK_RE.search(text)
    if block_match:
        return _normalize_whitespace(block_match.group(1))
    tag_match = _RESPONSE_TAG_RE.search(text)
    if tag_match:
        return _normalize_whitespace(tag_match.group(1))
    return _normalize_whitespace(text)


def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def summarize_memo(memo: str, max_chars: int = 320) -> str:
    """Trim a memo to ~max_chars without breaking sentence flow."""
    flat = _normalize_whitespace(memo)
    if not flat or max_chars <= 0 or len(flat) <= max_chars:
        return flat
    return flat[:max_chars].rstrip(" ,;:") + " ..."


# ---------------------------------------------------------------------------
# per-trajectory bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryDTState:
    """State that the rollout loop must keep alive across a single episode."""

    # latest deep-think memo, summarized
    latest_memo: str = ""
    # the env step at which `latest_memo` was produced
    memo_step: int | None = None
    # number of completed deep_think calls in this episode
    dt_count: int = 0
    # was the immediately preceding generation a deep_think call?
    # if so the next step must NOT have deep_think available (dt_action rule:
    # "two consecutive DT calls are forbidden").
    just_called_dt: bool = False
    # most recent gate decisions (selective_gate only) for diagnostics
    gate_history: list[bool] = field(default_factory=list)
    # last invalid-action flag (for gate facts)
    last_action_invalid: bool = False
    # last action text (for gate facts and repeat detection)
    last_action: str = ""
    # consecutive repeat count of the last action (for gate facts)
    last_action_repeat: int = 0

    def record_action(self, action_text: str, invalid: bool) -> None:
        if action_text and action_text == self.last_action:
            self.last_action_repeat += 1
        else:
            self.last_action_repeat = 0
        self.last_action = action_text or ""
        self.last_action_invalid = bool(invalid)

    def record_deep_think(self, memo_text: str, env_step: int) -> None:
        self.dt_count += 1
        self.just_called_dt = True
        self.latest_memo = summarize_memo(memo_text)
        self.memo_step = env_step

    def consume_dt_block(self) -> None:
        """Called after a real env action is executed; clears the one-step
        cooldown so DT becomes available again the next step."""
        self.just_called_dt = False


# ---------------------------------------------------------------------------
# prompt construction
# ---------------------------------------------------------------------------


def build_dt_action_user(
    *,
    init: bool,
    task_description: str,
    current_observation: str,
    admissible_actions_str: str,
    step_count: int,
    history_length: int,
    action_history: str,
    current_step: int,
    state: TrajectoryDTState,
    suppress_dt_guidance: bool = False,
) -> str:
    """Build the user-side prompt for the dt_action / selective_gate action step.

    Pass suppress_dt_guidance=True for selective_gate mode so the action-step
    user prompt does not mention deep_think (matching SFT data format and
    eliminating the dt-avail bias that inflates gate=true decisions).
    """
    if suppress_dt_guidance:
        action_guidance = ACTION_GUIDANCE_BASE
    else:
        dt_avail = not state.just_called_dt
        action_guidance = ACTION_GUIDANCE_DT_AVAIL if dt_avail else ACTION_GUIDANCE_DT_USED
    memo_block = render_memo_block(state.latest_memo, state.memo_step, current_step)
    if init:
        return ALFWORLD_DT_USER_TEMPLATE_NO_HIS.format(
            current_observation=current_observation,
            admissible_actions=admissible_actions_str,
            action_guidance=action_guidance,
        )
    return ALFWORLD_DT_USER_TEMPLATE.format(
        task_description=task_description,
        step_count=step_count,
        history_length=history_length,
        action_history=action_history,
        memo_block=memo_block,
        current_step=current_step,
        current_observation=current_observation,
        admissible_actions=admissible_actions_str,
        action_guidance=action_guidance,
    )


def pick_dt_action_system_prompt(state: TrajectoryDTState) -> str:
    """Pick the dt_action system prompt for this step (deep_think available or not)."""
    return (
        DT_ACTION_SYSTEM_PROMPT_DT_USED
        if state.just_called_dt
        else DT_ACTION_SYSTEM_PROMPT_DT_AVAIL
    )


def build_gate_user(
    *,
    state: TrajectoryDTState,
    current_step: int,
) -> str:
    """Phase-1 gate user prompt for selective_gate mode.

    The gate sees the same task context as the action call (the rollout loop
    builds it once and passes it in via `gate_facts`); this string is just the
    instruction block appended to that context.
    """
    facts: dict[str, object] = {
        "memo_present": bool(state.latest_memo),
        "memo_step": state.memo_step,
        "current_step": current_step,
        "memo_age": (None if state.memo_step is None else max(0, current_step - state.memo_step)),
        "last_action": state.last_action or None,
        "last_action_invalid": state.last_action_invalid,
        "last_action_repeat_count": state.last_action_repeat,
    }
    return _build_gate_instruction(deep_think_available=not state.just_called_dt, gate_facts=facts)


def _build_gate_instruction(
    *,
    deep_think_available: bool,
    gate_facts: dict[str, object],
) -> str:
    if not deep_think_available:
        return (
            "deep_think is not available on this step.\n"
            "Reply with exactly one line: `Deep_think: false`\n"
            "Do not output anything else."
        )
    facts_block = _format_gate_facts(gate_facts)
    facts_section = f"{facts_block}\n\n" if facts_block else ""
    freshness_clause = ""
    memo_age = gate_facts.get("memo_age")
    if isinstance(memo_age, int) and memo_age <= 2:
        freshness_clause = (
            f"A deep-think note from {memo_age} step(s) ago is in context. "
            "Strongly prefer `Deep_think: false` and follow that note unless its plan is clearly broken.\n"
        )
    return (
        "Phase 1 of 2 — gate decision only.\n"
        + facts_section
        + "Reply with EXACTLY one line: `Deep_think: true` or `Deep_think: false`.\n"
        "Default: `Deep_think: false`. Calling deep_think is expensive; use it sparingly "
        "(target ~1 in every 4–6 steps).\n"
        + freshness_clause
        + "Choose `Deep_think: true` ONLY when a concrete signal applies:\n"
        "  - the same action has repeated 2+ times consecutively with no progress,\n"
        "  - the previous action was invalid AND the current observation gives no obvious recovery,\n"
        "  - the latest deep-think note directly contradicts the current observation,\n"
        "  - the task phase just changed and no note covers the new phase.\n"
        "Otherwise choose `Deep_think: false`.\n"
        "Do not output Thought, Action, <think>, <deep_think>, explanations, or any other content."
    )


def _format_gate_facts(facts: dict[str, object]) -> str:
    lines: list[str] = []
    memo_step = facts.get("memo_step")
    current_step = facts.get("current_step")
    memo_age = facts.get("memo_age")
    if memo_step is not None and current_step is not None and memo_age is not None:
        lines.append(
            f"- Latest memo: from step {memo_step} (current step {current_step}, age {memo_age})"
        )
    elif facts.get("memo_present"):
        lines.append("- Latest memo: present")
    else:
        lines.append("- Latest memo: none yet")
    last_action = facts.get("last_action")
    if last_action:
        lines.append(f"- Last action: {last_action!r}")
    else:
        lines.append("- Last action: none (this is the first step)")
    last_invalid = facts.get("last_action_invalid")
    if last_invalid is not None:
        lines.append(f"- Last action invalid: {str(bool(last_invalid)).lower()}")
    repeat = facts.get("last_action_repeat_count")
    if repeat is not None:
        lines.append(f"- Last action repeated consecutively: {repeat} time(s)")
    return "\n".join(lines)


# convenient enumeration of every supported selective-thinking mode
SELECTIVE_THINKING_MODES = ("dt_action", "selective_gate")


def is_selective_thinking_mode(mode: str | None) -> bool:
    return (mode or "").lower() in SELECTIVE_THINKING_MODES


__all__ = [
    "OUTER_SYSTEM_PROMPT_PREFIX",
    "DT_ACTION_SYSTEM_PROMPT_DT_AVAIL",
    "DT_ACTION_SYSTEM_PROMPT_DT_USED",
    "ACTION_GUIDANCE_DT_AVAIL",
    "ACTION_GUIDANCE_DT_USED",
    "INNER_DEEP_THINK_SYSTEM_PROMPT",
    "TWO_PHASE_GATE_SYSTEM_PROMPT",
    "TWO_PHASE_ACTION_SYSTEM_PROMPT_POST_DT",
    "TWO_PHASE_ACTION_SYSTEM_PROMPT_NO_DT",
    "TrajectoryDTState",
    "build_dt_action_user",
    "pick_dt_action_system_prompt",
    "build_gate_user",
    "extract_gate_decision",
    "extract_action_text",
    "extract_memo_text",
    "is_deep_think_call",
    "summarize_memo",
    "render_memo_block",
    "SELECTIVE_THINKING_MODES",
    "is_selective_thinking_mode",
]
