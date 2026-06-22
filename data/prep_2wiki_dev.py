"""Download 2WikiMultihopQA dev (with context passages) -> raw jsonl for graph
extraction. Run on a LOGIN node (needs internet).

Output rows carry the committed test-split fields PLUS `context` (the Wikipedia
passages) that extract_graphs.py turns into the `graph`.
"""
import argparse
import json

import pandas as pd

DEV_URL = "https://huggingface.co/datasets/xanhho/2WikiMultihopQA/resolve/main/dev.parquet"


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
    ap.add_argument("--url", default=DEV_URL)
    args = ap.parse_args()

    df = pd.read_parquet(args.url)
    if args.n:
        df = df.head(args.n)

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
