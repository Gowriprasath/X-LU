import json
import os
from datetime import datetime, timezone

from paths import SHADOW_JOURNAL_PATH
from file_lock_registry import read_json, write_json


GATE_REVIEW_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'Data', 'Memory', 'claude_gate_review.json'
)
GATE_REVIEW_PATH = os.path.normpath(GATE_REVIEW_PATH)

MIN_BARS_FOR_OUTCOME = 48


def _empty_review_data() -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": "1.0",
        "created_date": now,
        "description": "Claude Gate Review - confidence gate audit log",
        "audit_records": [],
        "DECISIONS": [],
        "stats": {
            "total_records": 0,
            "good_blocks": 0,
            "missed_opportunities": 0,
            "neutral": 0,
            "last_updated": None,
            "last_monthly_review": None,
        },
    }


def _shadow_entries() -> list[dict]:
    shadow_data = read_json(SHADOW_JOURNAL_PATH)
    if isinstance(shadow_data, list):
        return shadow_data
    if isinstance(shadow_data, dict):
        entries = shadow_data.get("entries", [])
        if isinstance(entries, list):
            return entries
    return []


def get_unprocessed_gate_blocks() -> list[dict]:
    entries = _shadow_entries()
    return [
        entry for entry in entries
        if isinstance(entry, dict)
        and entry.get("gate", "") == "confidence_gate"
        and int(entry.get("bars_forward", 0) or 0) >= MIN_BARS_FOR_OUTCOME
        and entry.get("outcome", "") != ""
        and entry.get("reviewed_by_gate_builder", False) is not True
    ]


def classify_outcome(entry: dict) -> str:
    outcome = entry.get("outcome", "")
    if outcome == "HIT_SL":
        return "GOOD_BLOCK"
    if outcome == "HIT_TP":
        return "MISSED_OPPORTUNITY"
    if outcome == "NEUTRAL":
        return "NEUTRAL"
    return "NEUTRAL"


def build_review_entry(entry: dict) -> dict:
    return {
        "timestamp": entry.get("timestamp", ""),
        "regime": entry.get("regime", ""),
        "direction": entry.get("direction", ""),
        "confidence": entry.get("confidence", 0.0),
        "gate_limit": 0.40,
        "actual_outcome": entry.get("outcome", ""),
        "classification": classify_outcome(entry),
        "analysis": "",
        "session": entry.get("session", ""),
        "bars_held": entry.get("bars_forward", 0),
        "hypothetical_rr": entry.get("rr", 0.0),
        "reviewed": False,
    }


def process_new_blocks() -> int:
    unprocessed = get_unprocessed_gate_blocks()
    if not unprocessed:
        return 0

    data = read_json(GATE_REVIEW_PATH)
    if not data:
        data = _empty_review_data()

    data.setdefault("audit_records", [])
    data.setdefault("DECISIONS", [])
    stats = data.setdefault("stats", {})
    stats.setdefault("total_records", len(data["audit_records"]))
    stats.setdefault("good_blocks", 0)
    stats.setdefault("missed_opportunities", 0)
    stats.setdefault("neutral", 0)
    stats.setdefault("last_updated", None)
    stats.setdefault("last_monthly_review", None)

    for entry in unprocessed:
        record = build_review_entry(entry)
        data["audit_records"].append(record)

        classification = record["classification"]
        stats["total_records"] += 1
        if classification == "GOOD_BLOCK":
            stats["good_blocks"] += 1
        if classification == "MISSED_OPPORTUNITY":
            stats["missed_opportunities"] += 1
        if classification == "NEUTRAL":
            stats["neutral"] += 1

    stats["last_updated"] = datetime.now(timezone.utc).isoformat()
    write_json(GATE_REVIEW_PATH, data)

    processed_timestamps = {
        entry.get("timestamp", "") for entry in unprocessed
        if entry.get("timestamp", "")
    }
    shadow_data = read_json(SHADOW_JOURNAL_PATH)
    if isinstance(shadow_data, list):
        shadow_entries = shadow_data
    elif isinstance(shadow_data, dict):
        shadow_entries = shadow_data.get("entries", [])
    else:
        shadow_entries = []

    for entry in shadow_entries:
        if isinstance(entry, dict) and entry.get("timestamp", "") in processed_timestamps:
            entry["reviewed_by_gate_builder"] = True

    if shadow_data is not None:
        write_json(SHADOW_JOURNAL_PATH, shadow_data)

    return len(unprocessed)


def should_trigger_monthly_review() -> bool:
    data = read_json(GATE_REVIEW_PATH)
    if not data:
        return False

    audit_records = data.get("audit_records", [])
    unreviewed = [r for r in audit_records if not r.get("reviewed", False)]
    if len(unreviewed) >= 100:
        return True

    stats = data.get("stats", {})
    last_review = stats.get("last_monthly_review")
    if last_review is None:
        return False

    days_since = (
        datetime.now(timezone.utc) - datetime.fromisoformat(last_review)
    ).days
    if days_since >= 30:
        return True

    return False
