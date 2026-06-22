"""Build per-question KG graphs from 2Wiki context passages (paper Appendix F).

For each passage, prompt an instruction LLM to emit (subject, relation, object)
triples; union all triples per question into `graph` (no ranking/filtering).
Writes the committed test-split schema with `graph` populated. GPU job.

Paper uses Gemma-2-9b-it. Default here is whatever --model you pass (Qwen2.5-7B
works and needs no extra/gated download); see the fidelity note in the sbatch.
"""
import argparse
import ast
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = "You extract knowledge-graph triples from text."
USER = (
    "Extract all factual relationships in the passage below as a JSON list of "
    "[subject, relation, object] triples. Use concise entity and relation names. "
    "Only include facts explicitly stated. Output ONLY the JSON list.\n\nPassage:\n{passage}"
)


def passages_of(context) -> list[str]:
    """context = [[title, [sentence, ...]], ...] -> ['title: joined sentences', ...]"""
    out = []
    for item in context or []:
        if isinstance(item, list) and len(item) >= 2:
            title = item[0]
            sents = item[1] if isinstance(item[1], list) else [str(item[1])]
            out.append(f"{title}: {' '.join(str(s) for s in sents)}")
    return out


def parse_triples(text: str) -> list[list[str]]:
    triples = []
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        for parser in (json.loads, ast.literal_eval):
            try:
                v = parser(m.group(0))
                if isinstance(v, list):
                    for t in v:
                        if isinstance(t, (list, tuple)) and len(t) >= 3:
                            triples.append([str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()])
                    if triples:
                        return triples
            except Exception:
                continue
    # fallback: bare [a, b, c] patterns
    for mm in re.finditer(r"\[([^\[\]]+?),([^\[\]]+?),([^\[\]]+?)\]", text):
        triples.append([mm.group(i).strip().strip("\"'") for i in (1, 2, 3)])
    return triples


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="raw jsonl from prep_2wiki_dev.py")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=0, help="cap #questions (0 = all)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda().eval()

    rows = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.n:
        rows = rows[: args.n]

    # one extraction job per (question, passage)
    jobs = []  # (row_idx, prompt)
    for i, r in enumerate(rows):
        for p in passages_of(r.get("context")):
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER.format(passage=p[:4000])},
            ]
            jobs.append((i, tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)))

    print(f"{len(rows)} questions, {len(jobs)} passages to extract", flush=True)
    graphs: list[list] = [[] for _ in rows]
    for b in range(0, len(jobs), args.batch_size):
        batch = jobs[b : b + args.batch_size]
        enc = tok([p for _, p in batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        out = model.generate(
            **enc, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id
        )
        gen = out[:, enc.input_ids.shape[1]:]
        for (ri, _), seq in zip(batch, gen):
            graphs[ri].extend(parse_triples(tok.decode(seq, skip_special_tokens=True)))
        if (b // args.batch_size) % 20 == 0:
            print(f"  {min(b + args.batch_size, len(jobs))}/{len(jobs)} passages", flush=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for r, g in zip(rows, graphs):
            seen, ded = set(), []
            for t in g:
                k = tuple(s.lower() for s in t)
                if k not in seen:
                    seen.add(k)
                    ded.append(t)
            f.write(
                json.dumps(
                    {
                        "id": r["id"],
                        "question": r["question"],
                        "q_entity": r.get("q_entity", []),
                        "reasoning_path": r.get("reasoning_path", []),
                        "graph": ded,
                        "a_entity": r.get("a_entity", []),
                        "answer": r.get("answer", []),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {len(rows)} rows -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
