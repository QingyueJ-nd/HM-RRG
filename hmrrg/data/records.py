from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class StudyRecord:
    """One radiology-report generation example."""

    uid: str
    subject_id: str
    study_id: int
    study_date: str
    split: str
    image_relpath: str
    report: str
    prior_reports: Tuple[str, ...] = field(default_factory=tuple)
    retrieved_reports: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def sort_key(self) -> Tuple[str, int, str]:
        return (self.study_date, self.study_id, self.uid)

    def image_path(self, train_root: str, valtest_root: str) -> str:
        root = train_root if self.split == "train" else valtest_root
        return os.path.join(root, self.image_relpath)


def _clean_report(text: Any) -> str:
    lines = [ln.strip() for ln in str(text or "").replace("\r", "\n").split("\n")]
    return "\n".join(ln for ln in lines if ln).strip()


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _first_image_path(obj: Mapping[str, Any]) -> Optional[str]:
    image_path = obj.get("image_path") or obj.get("image_relpaths") or obj.get("image_relpath")
    if isinstance(image_path, str):
        return image_path
    if isinstance(image_path, Sequence) and image_path:
        return str(image_path[0])
    return None


def load_annotation_records(annotation_json: str) -> Dict[str, List[StudyRecord]]:
    """Load train/val/test records from an annotation JSON.

    Invalid rows without a report or image path are skipped.
    """

    with open(annotation_json, "r", encoding="utf-8") as f:
        raw = json.load(f)

    out: Dict[str, List[StudyRecord]] = {"train": [], "val": [], "test": []}
    for split in out:
        for obj in raw.get(split, []) or []:
            image_relpath = _first_image_path(obj)
            report = _clean_report(obj.get("report", ""))
            uid = str(obj.get("id", "")).strip()
            if not uid or not image_relpath or not report:
                continue
            out[split].append(
                StudyRecord(
                    uid=uid,
                    subject_id=str(obj.get("subject_id", "")).strip(),
                    study_id=_as_int(obj.get("study_id")),
                    study_date=str(obj.get("study_date", "")).strip(),
                    split=split,
                    image_relpath=image_relpath,
                    report=report,
                )
            )
    return out


def build_id_to_report(records_by_split: Mapping[str, Sequence[StudyRecord]], splits: Iterable[str]) -> Dict[str, str]:
    id_to_report: Dict[str, str] = {}
    for split in splits:
        for rec in records_by_split.get(split, []):
            id_to_report[rec.uid] = rec.report
    return id_to_report


def _subject_index(records: Iterable[StudyRecord]) -> Dict[str, List[StudyRecord]]:
    index: Dict[str, List[StudyRecord]] = {}
    for rec in records:
        index.setdefault(rec.subject_id, []).append(rec)
    for values in index.values():
        values.sort(key=lambda r: r.sort_key)
    return index


def attach_prior_reports(
    records_by_split: Mapping[str, Sequence[StudyRecord]],
    *,
    max_prior_reports: Optional[int] = None,
    split_local: bool = True,
) -> Dict[str, List[StudyRecord]]:
    """Attach chronological same-subject prior reports to each record.

    `split_local=True` avoids accidental leakage if a custom split is not
    patient-disjoint. Set it to False only when the split protocol guarantees
    patient-level separation.
    """

    result: Dict[str, List[StudyRecord]] = {}
    global_records = [rec for records in records_by_split.values() for rec in records]

    for split, records in records_by_split.items():
        index_source = records if split_local else global_records
        by_subject = _subject_index(index_source)
        split_out: List[StudyRecord] = []
        for rec in records:
            priors = [
                prev.report
                for prev in by_subject.get(rec.subject_id, [])
                if prev.sort_key < rec.sort_key and prev.report
            ]
            if max_prior_reports is not None:
                priors = priors[-int(max_prior_reports) :]
            split_out.append(replace(rec, prior_reports=tuple(priors)))
        result[split] = split_out
    return result


def load_retrieval_neighbors(path: str) -> Dict[str, List[Tuple[str, float]]]:
    """Load query -> [(neighbor_id, score)] from JSON/JSONL/JSONL.GZ."""

    opener = gzip.open if path.endswith(".gz") else open
    neighbors: Dict[str, List[Tuple[str, float]]] = {}

    with opener(path, "rt", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "{":
            obj = json.load(f)
            if "neighbors" in obj and "id" in obj:
                rows = [obj]
            else:
                rows = [{"id": k, "neighbors": v} for k, v in obj.items()]
        else:
            rows = [json.loads(line) for line in f if line.strip()]

    for row in rows:
        qid = str(row.get("id", "")).strip()
        pairs = []
        for item in row.get("neighbors", []) or []:
            if isinstance(item, Mapping):
                nid = str(item.get("id", "")).strip()
                score = float(item.get("score", 0.0))
            else:
                nid = str(item[0]).strip()
                score = float(item[1]) if len(item) > 1 else 0.0
            if nid:
                pairs.append((nid, score))
        if qid:
            neighbors[qid] = pairs
    return neighbors


def attach_retrieved_reports(
    records_by_split: Mapping[str, Sequence[StudyRecord]],
    retrieval_neighbors: Mapping[str, Sequence[Tuple[str, float]]],
    id_to_report: Mapping[str, str],
    *,
    top_k: int = 5,
    dedupe_against_priors: bool = True,
) -> Dict[str, List[StudyRecord]]:
    """Attach retrieved cross-patient report texts by neighbor IDs."""

    out: Dict[str, List[StudyRecord]] = {}
    for split, records in records_by_split.items():
        split_out: List[StudyRecord] = []
        for rec in records:
            reports: List[str] = []
            seen = set()
            prior_set = set(rec.prior_reports) if dedupe_against_priors else set()
            for neighbor_id, _score in retrieval_neighbors.get(rec.uid, []):
                text = (id_to_report.get(str(neighbor_id), "") or "").strip()
                if not text or text in seen or text in prior_set:
                    continue
                seen.add(text)
                reports.append(text)
                if len(reports) >= int(top_k):
                    break
            split_out.append(replace(rec, retrieved_reports=tuple(reports)))
        out[split] = split_out
    return out
