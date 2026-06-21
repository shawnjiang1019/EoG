"""Merge a LoRA adapter into the base model -> a standalone HF model.

The merged dir loads like any HF model, so it plugs straight into test_sft.py,
the GRPO stage, and the reasoner's llm_path.
"""
import argparse
import glob
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_adapter(path: str) -> str:
    """Return the dir containing adapter_config.json (search recursively, latest)."""
    if os.path.isfile(os.path.join(path, "adapter_config.json")):
        return path
    hits = sorted(glob.glob(os.path.join(path, "**", "adapter_config.json"), recursive=True))
    if not hits:
        raise FileNotFoundError(
            f"No adapter_config.json found under {path}. If the verl checkpoint is "
            f"sharded (not a PEFT adapter), run `python -m verl.model_merger merge "
            f"--backend fsdp --local_dir <ckpt> --target_dir <dir>` first."
        )
    return os.path.dirname(hits[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base model dir (e.g. Qwen2.5-7B-Instruct)")
    ap.add_argument("--adapter", required=True, help="adapter dir, or SAVE root to search")
    ap.add_argument("--out", required=True, help="output dir for the merged HF model")
    args = ap.parse_args()

    adapter_dir = find_adapter(args.adapter)
    print(f"base    = {args.base}\nadapter = {adapter_dir}\nout     = {args.out}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.merge_and_unload()  # fold adapter weights into the base

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"merged model saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
