"""text_paths: the vanilla reasoner.

Feeds the explorer's top-k diverse paths to the LLM as plain text and trains it
to emit the answer. Stays a standard HF causal LM, so it is vLLM-servable and
needs no custom collator -- the reference implementation the spine is tested on.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import torch

from ...base.config import BaseReasonerConfig
from ...base.reasoner import BaseReasoner
from ...base.registry import register_reasoner
from ...base.schema import ReasonerExample, ReasonerOutput
from ...common.eval import parse_model_output
from ...common.paths import select_paths, verbalize_path

SYSTEM = (
    "You are a reasoning assistant. You are given a question and a set of candidate "
    "reasoning paths extracted from a knowledge graph (each path is a chain of "
    "(subject)-[relation]->(object) triples). Use the paths to determine the answer. "
    "Give the final answer inside <answer> and </answer> as a JSON list of entities, "
    'e.g. <answer>["Beijing"]</answer>.'
)

USER_TEMPLATE = """Question: {question}

Starting entities: {q_entity}

Candidate reasoning paths:
{paths}

Give the final answer inside <answer></answer>."""


@dataclass
class TextPathsConfig(BaseReasonerConfig):
    max_new_tokens: int = 256


@register_reasoner("text_paths")
class TextPathsReasoner(BaseReasoner):
    config_cls = TextPathsConfig
    config: TextPathsConfig

    def build_model(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = self.config
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.llm_path, trust_remote_code=cfg.trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.llm_path,
            trust_remote_code=cfg.trust_remote_code,
            torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        )
        if cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
        if cfg.lora_rank > 0:
            from peft import LoraConfig, get_peft_model

            self.model = get_peft_model(
                self.model,
                LoraConfig(
                    r=cfg.lora_rank,
                    lora_alpha=cfg.lora_alpha,
                    target_modules="all-linear",
                    task_type="CAUSAL_LM",
                ),
            )

    # --- prompt construction ---------------------------------------------
    def _messages(self, ex: ReasonerExample) -> list[dict[str, str]]:
        paths = select_paths(ex.paths, self.config.num_paths)
        path_text = "\n".join(f"{i+1}. {verbalize_path(p)}" for i, p in enumerate(paths)) or "(none)"
        user = USER_TEMPLATE.format(
            question=ex.question,
            q_entity=", ".join(ex.q_entity) or "Unknown",
            paths=path_text,
        )
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]

    def _target_text(self, ex: ReasonerExample) -> str:
        return f"<answer>{json.dumps(ex.answers, ensure_ascii=False)}</answer>"

    # --- required hooks ---------------------------------------------------
    def build_inputs(self, ex: ReasonerExample) -> dict[str, Any]:
        tok = self.tokenizer
        prompt = tok.apply_chat_template(
            self._messages(ex), tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tok(prompt, add_special_tokens=False).input_ids
        target_ids = tok(self._target_text(ex), add_special_tokens=False).input_ids
        if tok.eos_token_id is not None:
            target_ids = target_ids + [tok.eos_token_id]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        # left-truncate the prompt if over budget, always keeping the target.
        if len(input_ids) > self.config.max_len:
            overflow = len(input_ids) - self.config.max_len
            input_ids = input_ids[overflow:]
            labels = labels[overflow:]
        return {"input_ids": input_ids, "labels": labels}

    def compute_loss(self, batch: dict[str, Any]):
        return self.model(**batch).loss

    @torch.no_grad()
    def generate(self, ex: ReasonerExample) -> ReasonerOutput:
        tok = self.tokenizer
        prompt = tok.apply_chat_template(
            self._messages(ex), tokenize=False, add_generation_prompt=True
        )
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        gen = self.model.generate(
            **enc,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        completion = tok.decode(gen[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
        parsed = parse_model_output(completion)
        answers = parsed["final_answer"]
        if not isinstance(answers, list):
            answers = [answers] if answers else []
        return ReasonerOutput(
            pred_answers=[str(a) for a in answers], reasoning=parsed["reasoning"], raw=completion
        )
