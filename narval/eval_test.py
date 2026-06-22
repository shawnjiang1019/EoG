"""Held-out eval: score a checkpoint on a graph-jsonl test set (Hit@1/F1/format).

Unlike eval_sft.py (which reads the training messages parquet), this builds the
EoG prompt from question + graph + starting entity -- the SAME system/user
templates as kg_qa_sft_process.py, so the prompt distribution matches training --
generates, and scores the answer against the gold `answer` field.

Use on data/extract_graphs.py output (2wiki_test_graphs.jsonl): a real held-out
test set with extracted graphs.
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_sft import answer_block, f1, hit1  # reuse the metric helpers

# --- prompt templates copied verbatim from data/kg_qa_sft_process.py ----------
SYSTEM = """You are a helpful assistant that answers questions based on knowledge graphs.
1. First, reason through the problem inside <think> and </think> tags. Here you can planning, memory, check for mistakes to reflect，prune the entities and relations.
2. When confident, output the final answer inside <answer> and </answer> tags. Your answer must strictly follow the rules provided by the user.

You will receive:
- question: A natural language question that needs to be answered
- Knowledge Graph: Knowledge graph information in the form of triples (subject, relation, object)
- Starting Entity: The starting entity in the knowledge graph that can be used as a reference point

You must generate a response that includes:

1. **Reasoning Chain** (<think> section):
   - Step-by-step logical reasoning process
   - Analysis of the question and relevant graph information
   - Identification of key entities and relationships, starting from the begin_entity
   - Show how you traverse from the begin_entity to reach the answer
   - Use of graph entities and relations in your reasoning

2. **Final Answer** (<answer> section):
   - Clear, concise answer to the question
   - Must match the provided correct answer
   - The answer MUST be one or more entities that exist in the provided graph
   - CRITICAL: Only use entities that are explicitly present in the graph triples

- Always start with the <think> section to show your reasoning
- Use the begin_entity as your starting point for reasoning
- Be explicit about which entities and relations you're considering
- Show logical connections between different pieces of information
- Ensure your answer only contains entities that exist in the provided graph triples
- Use clear, natural language that demonstrates good reasoning skills"""

USER = """Answer the given question based on the knowledge graph. You must first conduct reasoning step by step, and put your final answer inside <answer> and </answer>.

Rules:
1. When you have the final answer, you can directly provide the answer inside <answer></answer>, without detailed illustrations. For example, <answer> ["Washington, D.C."] </answer>
2. CRITICAL: Your final answer must only contain entities that are explicitly present in the provided graph triples
3. If there are multiple possible answers, present them in list format using square brackets []. For example, <answer>["Beijing", "Washington DC"]</answer>
4. Because questions usually have multiple answers, you should consider all possible answers and provide them in the list format
5. If multiple starting entities are provided, you should analyze each one systematically，such as Consider how different starting points might lead to different answers ,combine insights from all starting entities to form answers and if starting entities are related, explore their connections in the knowledge graph.

You must use this format:
<think>...</think>
<answer>...</answer>

Question: {question}

Knowledge Graph:
{graph_text}

Starting Entity: {entity_text}"""


def format_entities(ents) -> str:
    if not ents:
        return "Unknown"
    if len(ents) == 1:
        return ents[0]
    return "\n".join(f"{i}. {e}" for i, e in enumerate(ents, 1))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged HF model dir")
    ap.add_argument("--data", required=True, help="graph jsonl (from extract_graphs.py)")
    ap.add_argument("--n", type=int, default=0, help="cap #questions (0 = all)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda().eval()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    if args.n:
        rows = rows[: args.n]

    prompts, golds = [], []
    for r in rows:
        user = USER.format(
            question=r["question"],
            graph_text=str(r.get("graph", [])),
            entity_text=format_entities(r.get("q_entity", [])),
        )
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        golds.append([str(a) for a in (r.get("answer") or [])])

    n = hit = fpts = fmt_full = fmt_ans = 0
    for b in range(0, len(prompts), args.batch_size):
        pb, gb = prompts[b : b + args.batch_size], golds[b : b + args.batch_size]
        enc = tok(pb, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        out = model.generate(
            **enc, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id
        )
        for seq, gold in zip(out[:, enc.input_ids.shape[1]:], gb):
            comp = tok.decode(seq, skip_special_tokens=True)
            pred = answer_block(comp)
            hit += hit1(pred, gold)
            fpts += f1(pred, gold)
            fmt_full += int(all(t in comp for t in ("<think>", "</think>", "<answer>", "</answer>")))
            fmt_ans += int("<answer>" in comp and "</answer>" in comp)
            n += 1
        if (b // args.batch_size) % 10 == 0:
            print(f"  {n}/{len(prompts)} hit@1={hit/n:.3f} f1={fpts/n:.3f}", flush=True)

    print(f"\n=== {args.model}  (HELD-OUT, n={n}) ===")
    print(f"Hit@1                          : {hit / n:.4f}")
    print(f"F1                             : {fpts / n:.4f}")
    print(f"full-format rate (think+answer): {fmt_full / n:.4f}")
    print(f"answer-tag rate (<answer>)     : {fmt_ans / n:.4f}")


if __name__ == "__main__":
    main()
