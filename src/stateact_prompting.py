from __future__ import annotations

import json
from pathlib import Path

from stateact_reference_helpers import build_stateact_aligned_messages, load_demonstrations


class StateActPromptAdapter:
    def __init__(self, *, evaluator, config, demonstrations, log_path, generation_tokens):
        self.evaluator = evaluator
        self.config = config
        self.demonstrations = load_demonstrations(Path(demonstrations))
        self.log_path = Path(log_path)
        self.generation_tokens = generation_tokens
        self.base_builder = evaluator.build_next_action_messages
        self.original_select = evaluator.select_action
        self.original_episode = evaluator.run_episode
        self.resources = None
        self.previous_note = ""
        self.task_id = ""
        self.step_index = 0

    def install(self):
        self.evaluator.build_next_action_messages = self.build
        self.evaluator.select_action = self.select
        self.evaluator.run_episode = self.episode

    def episode(self, **kwargs):
        self.task_id = kwargs["task"]["task_id"]
        self.previous_note = ""
        return self.original_episode(**kwargs)

    def select(self, **kwargs):
        self.resources = (kwargs["model"], kwargs["tokenizer"])
        self.step_index = len(kwargs["previous_actions"])
        action, raw = self.original_select(**kwargs)
        try:
            trace = json.loads(raw)
        except json.JSONDecodeError:
            trace = {"proposal_output": raw, "selected_action": action}
        trace["stateact_style"] = {"config": self.config, "reasoning_tokens_limit":
                                   self.generation_tokens if self.config.startswith("generated") else 0,
                                   "extra_generation_calls": 1 if self.config.startswith("generated") else 0}
        return action, json.dumps(trace, ensure_ascii=False)

    def build(self, *, instruction, previous_actions, observation):
        from workflow_state import observable_state_summary
        messages = build_stateact_aligned_messages(
            base_builder=self.base_builder, state_summarizer=observable_state_summary,
            demonstrations=self.demonstrations,
            example_count=1 if self.config == "tracked_one" else 2,
            instruction=instruction, previous_actions=previous_actions,
            observation=observation, history_window=0)
        if not self.config.startswith("generated"):
            return messages
        from run_hf_next_action_eval import generate_action
        model, tokenizer = self.resources
        context = messages[-1]["content"].removesuffix(
            "Use the examples and current state to choose exactly one legal next action.")
        context = context.replace("Return the next action only.", "")
        reasoning_messages = [
            {"role": "system", "content": "Track the task goal and observable state before selecting a skill. "
             "Write concise plain-text fields: Goal, Locations visited, Current location, "
             "Current inventory, Thought. Use the observations and executed actions. "
             "Do not issue an action in this response."},
            {"role": "user", "content": context + "\nPrevious state/thought:\n" +
             (self.previous_note or "(initial decision)") +
             "\nUpdate the state and give a short thought about the next task objective."}
        ]
        note, _ = generate_action(model=model, tokenizer=tokenizer, messages=reasoning_messages,
                                  max_new_tokens=self.generation_tokens)
        self.previous_note = note
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"task_id": self.task_id, "step_index": self.step_index,
                                     "state_thought": note, "config": self.config},
                                    ensure_ascii=False) + "\n")
        messages[-1]["content"] += "\nStateAct state/thought:\n" + note + \
            "\nSelect exactly one legal next action. Return the action string only."
        return messages
