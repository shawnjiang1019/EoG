"""Path utilities every variant needs: verbalize, dedup, diversity-select."""
from __future__ import annotations

from ..base.schema import Path


def verbalize_path(path: Path) -> str:
    """Render a path as readable text, e.g. '(A) -[r1]-> (B) -[r2]-> (C)'."""
    if not path.triples:
        return ""
    parts = [f"({path.triples[0][0]})"]
    for s, r, o in path.triples:
        parts.append(f" -[{r}]-> ({o})")
    return "".join(parts)


def _key(path: Path) -> tuple:
    return tuple(tuple(t) for t in path.triples)


def select_paths(paths: list[Path], k: int, diversity: bool = True) -> list[Path]:
    """Dedup then take up to k paths.

    Highest-score first when scores exist. With diversity=True, greedily skip a
    path whose triple-set is a subset of an already-selected one, so the k slots
    cover distinct reasoning rather than near-duplicates.
    """
    seen: set[tuple] = set()
    deduped: list[Path] = []
    for p in paths:
        key = _key(p)
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    if any(p.score is not None for p in deduped):
        deduped.sort(key=lambda p: (p.score is not None, p.score), reverse=True)

    if not diversity:
        return deduped[:k]

    chosen: list[Path] = []
    chosen_sets: list[set] = []
    for p in deduped:
        s = {tuple(t) for t in p.triples}
        if any(s <= cs for cs in chosen_sets):  # subset of an existing pick
            continue
        chosen.append(p)
        chosen_sets.append(s)
        if len(chosen) >= k:
            break
    if len(chosen) < k:  # backfill if diversity pruned too aggressively
        for p in deduped:
            if p not in chosen:
                chosen.append(p)
            if len(chosen) >= k:
                break
    return chosen[:k]
