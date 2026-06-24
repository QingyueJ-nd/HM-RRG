from __future__ import annotations

from typing import List, Sequence


def format_report_block(reports: Sequence[str], label: str, empty: str = "None.") -> str:
    chunks: List[str] = []
    for idx, report in enumerate(reports, 1):
        text = (report or "").strip()
        if text:
            chunks.append(f"[{label} {idx}]\n{text}")
    return "\n\n".join(chunks) if chunks else empty


def build_hmrrg_prompt(
    *,
    instruction: str,
    prior_reports: Sequence[str],
    image_placeholder: str = "",
) -> str:
    image_part = f"<Img> {image_placeholder} </Img> " if image_placeholder else ""
    history = format_report_block(prior_reports, "Prior")
    return (
        f"### User: {image_part}{instruction.strip()}\n"
        "Here are the patient's historical diagnosis reports in chronological order <Reports>\n"
        f"{history}\n"
        "</Reports>\n"
        "### Assistant:"
    )
