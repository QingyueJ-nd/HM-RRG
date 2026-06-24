from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class EmbeddingRecord:
    uid: str
    subject_id: str
    split: str
    embedding: np.ndarray


def load_embedding_jsonl(path: str) -> List[EmbeddingRecord]:
    records: List[EmbeddingRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            records.append(
                EmbeddingRecord(
                    uid=str(row["id"]),
                    subject_id=str(row.get("subject_id", "")),
                    split=str(row.get("split", "")),
                    embedding=np.asarray(row["embedding"], dtype=np.float32),
                )
            )
    return records


def cosine_neighbors(
    queries: Sequence[EmbeddingRecord],
    candidates: Sequence[EmbeddingRecord],
    *,
    top_k: int,
    search_k: Optional[int] = None,
    exclude_same_subject: bool = True,
) -> Dict[str, List[Tuple[str, float]]]:
    """Small numpy cosine retriever used for reproducible public preprocessing."""

    if not candidates:
        return {q.uid: [] for q in queries}

    cand_ids = [c.uid for c in candidates]
    cand_subjects = [c.subject_id for c in candidates]
    cand = np.stack([c.embedding for c in candidates]).astype(np.float32)
    cand = cand / np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-12)

    limit = int(search_k or max(top_k * 5, top_k))
    out: Dict[str, List[Tuple[str, float]]] = {}
    for query in queries:
        q = query.embedding.astype(np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        scores = cand @ q
        order = np.argsort(-scores)[:limit]

        pairs: List[Tuple[str, float]] = []
        for idx in order:
            if cand_ids[idx] == query.uid:
                continue
            if exclude_same_subject and cand_subjects[idx] == query.subject_id:
                continue
            pairs.append((cand_ids[idx], float(scores[idx])))
            if len(pairs) >= top_k:
                break
        out[query.uid] = pairs
    return out


def write_neighbors_jsonl(neighbors: Mapping[str, Sequence[Tuple[str, float]]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for qid, pairs in neighbors.items():
            row = {"id": qid, "neighbors": [[nid, score] for nid, score in pairs]}
            f.write(json.dumps(row) + "\n")
