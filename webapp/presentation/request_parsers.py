from __future__ import annotations

from webapp.domain.entities import SubjectInfo


def capture_payload_from_form(form) -> dict:
    subject = SubjectInfo(
        name=form.get("name", "").strip() or "subject",
        height_cm=int(form.get("height_cm", "170")),
        weight_kg=int(form.get("weight_kg", "70")),
        hand=safe_hand(form.get("hand", "right")),
    )
    return {"subject": subject, "camera_ids": form.getlist("camera_ids")}


def capture_subject_from_json(data: dict) -> SubjectInfo:
    return SubjectInfo(
        name=str(data.get("name", "")).strip(),
        height_cm=int(data.get("height_cm")),
        weight_kg=int(data.get("weight_kg")),
        hand=safe_hand(str(data.get("hand", "right"))),
    )


def payload_camera_label(payload) -> str:
    if isinstance(payload, dict):
        return str(payload.get("camera_label") or "").strip()
    return ""


def safe_hand(hand: str) -> str:
    value = str(hand or "right").strip().lower()
    return value if value in {"right", "left"} else "right"


def calibration_mode(value: str) -> str:
    normalized = str(value or "intrinsic").strip().lower()
    if normalized in {"intrinsic", "intr", "in"}:
        return "INTR"
    if normalized in {"extrinsic", "extr", "ex"}:
        return "EXTR"
    raise ValueError("unknown_calibration_mode")
