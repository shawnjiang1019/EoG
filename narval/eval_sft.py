"""Batch-evaluate an SFT'd checkpoint: Hit@1, F1, and format-rate.

Generates over N samples of the SFT parquet (rebuilding the exact training
prompt, dropping the gold assistant turn) and scores predictions against the
gold answer parsed from each sample's <answer> block.

NOTE: the parquet is training data (the only data with graphs), so absolute
Hit@1/F1 are in-distribution (inflated). Use this for the full-vs-LoRA relative
comparison and the format-rate; a held-out score needs test-set graphs.
"""
import argparse
import ast
import json
import re

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_answers(x) -> list[str]:
    if not x:
        return []
    if isinstance(x, list):
        return [str(a).strip() for a in x if str(a).strip()]
    t = str(x).strip().replace("<answer>", "").replace("</answer>", "").strip()
    if not t:
        return []
    if t.startswith("[") and t.endswith("]"):
        try:
            v = json.loads(t)
            if isinstance(v, list):
                return [str(a).strip() for a in v if str(a).strip()]
        except json.JSONDecodeError:
            inner = t[1:-1].strip()
            if inner:
                return [p.strip().strip('"').strip("'") for p in inner.split(",") if p.strip()]
    if "," in t:
        return [p.strip() for p in t.split(",") if p.strip()]
    return [t]


def answer_block(text: str) -> list[str]:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    return extract_answers(m.group(1).strip()) if m else []


def hit1(pred, gold) -> float:
    if not pred or not gold:
        return 0.0
    for p in pred:
        pl = p.lower().strip()
        for c in gold:
            cl = c.lower().strip()
            if pl == cl or cl in pl or pl in cl:
                return 1.0
    return 0.0


def f1(pred, gold) -> float:
    ps, gs = {p.lower().strip() for p in pred}, {c.lower().strip() for c in gold}
    if not ps and not gs:
        return 1.0
    if not ps or not gs:
        return 0.0
    inter = ps & gs
    if not inter:
        return 0.0
    prec, rec = len(inter) / len(ps), len(inter) / len(gs)
    return 2 * prec * rec / (prec + rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged HF model dir")
    ap.add_argument("--data", required=True, help="SFT parquet (messages column)")
    ap.add_argument("--n", type=int, default=300, help="random sample size (0 = all)")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda().eval()

    df = pd.read_parquet(args.data)
    if args.n and args.n < len(df):
        df = df.sample(args.n, random_state=0)

    n = hit = fpts = fmt_full = fmt_ans = 0
    for _, row in df.iterrows():
        msgs = [dict(m) for m in row["messages"]]
        gold = answer_block(next((m["content"] for m in msgs if m["role"] == "assistant"), ""))
        prompt_msgs = [m for m in msgs if m["role"] != "assistant"]
        prompt = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=pad
            )
        comp = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
        pred = answer_block(comp)

        hit += hit1(pred, gold)
        fpts += f1(pred, gold)
        fmt_full += int(all(t in comp for t in ("<think>", "</think>", "<answer>", "</answer>")))
        fmt_ans += int("<answer>" in comp and "</answer>" in comp)
        n += 1
        if n % 25 == 0:
            print(f"  {n}/{len(df)}  hit@1={hit/n:.3f} f1={fpts/n:.3f}", flush=True)

    print(f"\n=== {args.model} (n={n}) ===")
    print(f"Hit@1                          : {hit / n:.4f}")
    print(f"F1                             : {fpts / n:.4f}")
    print(f"full-format rate (think+answer): {fmt_full / n:.4f}")
    print(f"answer-tag rate (<answer>)     : {fmt_ans / n:.4f}")


if __name__ == "__main__":
    main()
