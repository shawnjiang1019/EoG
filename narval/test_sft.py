"""Quick sanity test of an SFT'd EoG checkpoint.

Loads the merged HF model, rebuilds the exact training prompt from the SFT
parquet (system+user, dropping the gold assistant turn), generates, and checks
the output contains <think>...</think><answer>...</answer>.
"""
import argparse

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged HF model dir")
    ap.add_argument("--data", required=True, help="SFT parquet (has `messages` column)")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda().eval()

    df = pd.read_parquet(args.data).head(args.n)
    for i, row in df.iterrows():
        msgs = [dict(m) for m in row["messages"]]
        prompt_msgs = [m for m in msgs if m["role"] != "assistant"]   # prompt only
        gold = next((m["content"] for m in msgs if m["role"] == "assistant"), "")

        prompt = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=pad_id
            )
        completion = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)

        has_think = "<think>" in completion and "</think>" in completion
        has_answer = "<answer>" in completion and "</answer>" in completion
        print("=" * 80)
        print(f"SAMPLE {i}  | format: think={has_think} answer={has_answer}")
        print("--- model output ---")
        print(completion[:2000])
        print("--- gold answer span ---")
        print(gold[-300:])
    print("DONE")


if __name__ == "__main__":
    main()
