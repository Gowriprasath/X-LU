import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import paths
from file_lock_registry import read_json, write_json
from ai_client import call_ai
from datetime import datetime, timezone


REGIMES = [
    "BULL_TREND",
    "BEAR_TREND",
    "RANGE",
    "REVERSAL",
    "BREAKOUT",
]


def build_meta_review_prompt(data: dict) -> str:
    stats = data.get("stats", {})
    audit_records = data.get("audit_records", [])
    decisions = data.get("DECISIONS", [])

    regime_counts = {}
    for record in audit_records:
        regime = record.get("regime", "UNKNOWN") or "UNKNOWN"
        classification = record.get("classification", "NEUTRAL")
        regime_counts.setdefault(regime, {
            "GOOD_BLOCK": 0,
            "MISSED_OPPORTUNITY": 0,
            "NEUTRAL": 0,
        })
        if classification not in regime_counts[regime]:
            classification = "NEUTRAL"
        regime_counts[regime][classification] += 1

    for regime in REGIMES:
        regime_counts.setdefault(regime, {
            "GOOD_BLOCK": 0,
            "MISSED_OPPORTUNITY": 0,
            "NEUTRAL": 0,
        })

    regime_summary_lines = []
    for regime in sorted(regime_counts):
        counts = regime_counts[regime]
        total = sum(counts.values())
        missed = counts["MISSED_OPPORTUNITY"]
        miss_rate = (missed / total * 100) if total else 0.0
        regime_summary_lines.append(
            f"{regime}:\n"
            f"  GOOD_BLOCKS: {counts['GOOD_BLOCK']}\n"
            f"  MISSED_OPPORTUNITIES: {missed}\n"
            f"  NEUTRAL: {counts['NEUTRAL']}\n"
            f"  MISS_RATE: {miss_rate:.1f}%"
        )

    unreviewed = [
        record for record in audit_records
        if not record.get("reviewed", False)
    ]
    sample_records = unreviewed[-10:]

    return f"""
ROLE
You are a quantitative trading system auditor.
You are reviewing the confidence gate performance
of a XAUUSD trading bot over the past 30 days.
Your job is to identify if the confidence gate
thresholds are correctly calibrated per regime.

GATE REVIEW DATA
Stats:
{json.dumps(stats, indent=2)}

Counts per regime and classification:
{chr(10).join(regime_summary_lines)}

SAMPLE RECORDS
{json.dumps(sample_records, indent=2)}

EXISTING DECISIONS
{json.dumps(decisions, indent=2)}

TASK
Analyze the gate performance by regime.
For each regime, determine if the confidence
gate threshold of 0.40 is:
  - TOO_STRICT: miss rate > 60% (blocking profitable trades)
  - CALIBRATED: miss rate 40-60% (filtering correctly)
  - TOO_LOOSE:  miss rate < 40% (letting through losses)

Output your analysis as a single JSON object with
this exact structure. Output ONLY the JSON, no other text:

{{
  "verdict_by_regime": {{
    "BULL_TREND": {{
      "current_threshold": 0.40,
      "miss_rate": 0.0,
      "sample_size": 0,
      "assessment": "TOO_STRICT | CALIBRATED | TOO_LOOSE",
      "recommended_threshold": 0.40,
      "reasoning": "string",
      "confidence_in_recommendation": "LOW | MEDIUM | HIGH"
    }}
  }},
  "overall_assessment": "string",
  "operator_recommendation": "string - plain English instructions for the operator",
  "operator_review_required": true,
  "minimum_data_for_reliability": "string - note if more data needed before acting"
}}
"""


def run_monthly_review() -> bool:
    data = read_json(paths.GATE_REVIEW_PATH)
    if not data or not data.get("audit_records"):
        print("[GateReview] No audit records available - skipping")
        return False

    total = data.get("stats", {}).get("total_records", 0)
    if total < 20:
        print(f"[GateReview] Only {total} records - "
              f"need 20+ for meaningful analysis. Skipping.")
        return False

    prompt = build_meta_review_prompt(data)

    print("[GateReview] Running monthly meta-review...")
    print(f"[GateReview] {total} audit records | sending to Claude")

    response = call_ai(prompt, max_tokens=2000)
    if response is None:
        print("[GateReview] Claude unavailable - skipping")
        return False

    try:
        clean = response.strip()
        if clean.startswith("```"):
            lines = clean.split('\n')
            clean = '\n'.join(lines[1:-1])
        analysis = json.loads(clean)
    except json.JSONDecodeError:
        print("[GateReview] Could not parse Claude response as JSON")
        print(f"[GateReview] Raw response: {response[:500]}")
        return False

    now = datetime.now(timezone.utc).isoformat()
    decision = {
        "timestamp": now,
        "verdict_by_regime": analysis.get("verdict_by_regime", {}),
        "overall_assessment": analysis.get("overall_assessment", ""),
        "operator_recommendation": analysis.get("operator_recommendation", ""),
        "operator_review_required": True,
        "minimum_data_note": analysis.get("minimum_data_for_reliability", ""),
        "records_reviewed": total,
        "applied": False,
        "applied_date": None,
        "applied_notes": "",
    }

    data.setdefault("DECISIONS", []).append(decision)

    for entry in data.get("audit_records", []):
        entry["reviewed"] = True

    data.setdefault("stats", {})["last_monthly_review"] = now
    write_json(paths.GATE_REVIEW_PATH, data)

    print("=" * 60)
    print("  CLAUDE GATE REVIEW - MONTHLY DECISION")
    print("=" * 60)
    print(f"  Records reviewed: {total}")
    print(f"  Overall: {decision['overall_assessment']}")
    print()
    print("  Per-regime verdicts:")
    for regime, verdict in decision["verdict_by_regime"].items():
        print(f"  {regime}:")
        print(f"    Assessment: {verdict.get('assessment')}")
        print(f"    Miss rate:  {verdict.get('miss_rate', 0) * 100:.1f}%")
        print(f"    Sample:     {verdict.get('sample_size')} records")
        print(f"    Current:    {verdict.get('current_threshold')}")
        print(f"    Suggested:  {verdict.get('recommended_threshold')}")
        print(f"    Confidence: {verdict.get('confidence_in_recommendation')}")
        print()
    print("  OPERATOR RECOMMENDATION:")
    print(f"  {decision['operator_recommendation']}")
    print()
    print("  operator_review_required: TRUE")
    print("  No changes applied automatically.")
    print("  Review claude_gate_review.json DECISIONS block")
    print("  then update master_controls.py manually.")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = run_monthly_review()
    if success:
        print("[GateReview] Monthly review complete.")
    else:
        print("[GateReview] Review not completed - see messages above.")
