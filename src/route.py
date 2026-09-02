"""
Appetite routing engine.

Consumes the eight standardized fields and emits one of:
    BOP | GL | PL | REFERRAL | DECLINE

Every decision returns an audit record: the rule that fired, the fields it
relied on, the raw text those fields came from, and the confidence of any
inferred value. Nothing routes on a value whose provenance is not recorded.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RULES = yaml.safe_load(open(ROOT / "config" / "appetite_rules.yaml"))

REQUIRED_FOR_QUOTE = ["naics", "revenue", "limit_occurrence", "location", "loss_history"]


def _matches(rule, f):
    """A rule matches only if every condition it declares is satisfied."""
    naics = f.get("naics")
    if "naics_prefix" in rule:
        if not naics or not any(str(naics).startswith(p) for p in rule["naics_prefix"]):
            return False
    if "states" in rule:
        if f.get("state") not in rule["states"]:
            return False
    if "revenue_min" in rule:
        v = f.get("revenue")
        if v is None or v < rule["revenue_min"]:
            return False
    if "revenue_max" in rule:
        v = f.get("revenue")
        if v is None or v > rule["revenue_max"]:
            return False
    if "payroll_min" in rule:
        v = f.get("payroll")
        if v is None or v < rule["payroll_min"]:
            return False
    if "payroll_max" in rule:
        v = f.get("payroll")
        # unknown payroll should not silently qualify a risk for BOP
        if v is None or v > rule["payroll_max"]:
            return False
    if "limit_occurrence_min" in rule:
        v = f.get("limit_occurrence")
        if v is None or v < rule["limit_occurrence_min"]:
            return False
    if "deductible_max" in rule:
        v = f.get("deductible")
        if v is None or v > rule["deductible_max"]:
            return False
    if "loss_count_min" in rule:
        v = f.get("loss_count_5yr")
        if v is None or v < rule["loss_count_min"]:
            return False
    if "loss_incurred_min" in rule:
        v = f.get("loss_incurred_5yr")
        if v is None or v < rule["loss_incurred_min"]:
            return False
    return True


def route(fields, evidence=None, naics_confidence=None):
    """Return an audit record describing the routing decision."""
    evidence = evidence or {}
    trail = []

    missing = [k for k in REQUIRED_FOR_QUOTE if fields.get(k) is None]

    # --- gate 0: cannot classify -> referral, no appetite rule can be trusted
    if fields.get("naics") is None:
        trail.append({"stage": "classification", "rule": "CLS-00",
                      "result": "no NAICS stated and classifier below confidence floor"})
        return _record("REFERRAL", None, "CLS-00",
                       "Business class could not be determined with confidence",
                       trail, missing, fields, evidence, naics_confidence)

    # --- stage 1: declines
    for r in RULES["declines"]:
        if _matches(r, fields):
            trail.append({"stage": "decline", "rule": r["id"], "result": "matched"})
            return _record("DECLINE", None, r["id"], r["reason"], trail,
                           missing, fields, evidence, naics_confidence)
    trail.append({"stage": "decline", "rule": "-", "result": "no decline rule matched"})

    # --- stage 2: mandatory referrals
    for r in RULES["referrals"]:
        if _matches(r, fields):
            trail.append({"stage": "referral", "rule": r["id"], "result": "matched"})
            return _record("REFERRAL", None, r["id"], r["reason"], trail,
                           missing, fields, evidence, naics_confidence)
    trail.append({"stage": "referral", "rule": "-", "result": "no referral rule matched"})

    # --- gate: missing data blocks straight-through quoting
    if missing:
        trail.append({"stage": "completeness", "rule": "INC-01",
                      "result": f"missing {', '.join(missing)}"})
        return _record("REFERRAL", None, "INC-01",
                       f"Submission incomplete - missing {', '.join(missing)}",
                       trail, missing, fields, evidence, naics_confidence)

    # --- stage 3: product assignment
    for r in RULES["products"]:
        if _matches(r, fields):
            trail.append({"stage": "product", "rule": r["id"], "result": "matched"})
            return _record(r["product"], r["product"], r["id"], r["reason"], trail,
                           missing, fields, evidence, naics_confidence)

    fb = RULES["fallback"]
    trail.append({"stage": "product", "rule": fb["id"], "result": "fallback"})
    return _record(fb["route"], None, fb["id"], fb["reason"], trail,
                   missing, fields, evidence, naics_confidence)


def _record(route_, product, rule_id, reason, trail, missing, fields, evidence, conf):
    return {
        "route": route_,
        "product": product,
        "rule_id": rule_id,
        "reason": reason,
        "quote_ready": route_ in ("BOP", "GL", "PL"),
        "missing_fields": missing,
        "naics_confidence": None if conf is None else round(conf, 3),
        "naics_source": fields.get("naics_source"),
        "decision_trail": trail,
        "fields_used": {k: fields.get(k) for k in
                        ["naics", "class_code", "revenue", "payroll", "limit_occurrence",
                         "deductible", "loss_count_5yr", "loss_incurred_5yr", "state"]},
        "evidence": evidence,
    }


def route_from_truth(rec):
    """Ground-truth routing: the same rules applied to the true field values.

    This is the label the pipeline is scored against, so any routing error is
    attributable to extraction or classification, never to rule disagreement.
    """
    f = {
        "naics": rec["naics"], "class_code": rec["class_code"], "revenue": rec["revenue"],
        "payroll": rec["payroll"], "limit_occurrence": rec["limit_occurrence"],
        "deductible": rec["deductible"], "loss_count_5yr": rec["loss_count_5yr"],
        "loss_incurred_5yr": rec["loss_incurred_5yr"], "state": rec["state"],
        "location": rec["state"], "loss_history": rec["loss_count_5yr"],
        "naics_source": "ground_truth",
    }
    return route(f)["route"]
