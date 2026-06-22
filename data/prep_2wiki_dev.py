"""Download 2WikiMultihopQA dev (with context passages) -> raw jsonl for graph
extraction. Run on a LOGIN node (needs internet).

Output rows carry the committed test-split fields PLUS `context` (the Wikipedia
passages) that extract_graphs.py turns into the `graph`.
"""
import argparse
import json
import os

import pandas as pd


def parse_field(x):
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0, help="0 = all dev rows")
    ap.add_argument("--repo", default="xanhho/2WikiMultihopQA")
    ap.add_argument("--file", default="dev.parquet")
    args = ap.parse_args()

    # Download via huggingface_hub (already installed) then read the local file,
    # so we don't need fsspec/aiohttp to read a parquet over http.
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(repo_id=args.repo, filename=args.file, repo_type="dataset")
    df = pd.read_parquet(local)
    if args.n:
        df = df.head(args.n)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            ans = parse_field(row["answer"])
            answer = ans if isinstance(ans, list) else [ans]
            ev = parse_field(row["evidences"])
            ev = ev if isinstance(ev, list) else []
            context = parse_field(row["context"])
            context = context if isinstance(context, list) else []

            q = str(row["question"]).lower()
            ents = []
            for t in ev:
                if isinstance(t, list) and len(t) >= 3:
                    for e in (t[0], t[2]):
                        if isinstance(e, str) and e.lower() in q and e not in ents:
                            ents.append(e)
            if not ents and ev and isinstance(ev[0], list) and ev[0]:
                ents = [ev[0][0]]

            rec = {
                "id": row["_id"],
                "question": row["question"],
                "q_entity": ents,
                "reasoning_path": ev,        # gold path (= evidences)
                "a_entity": answer,
                "answer": answer,
                "context": context,          # [[title, [sentences...]], ...]
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
