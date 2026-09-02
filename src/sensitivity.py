"""
Missing-data sensitivity analysis.

The pipeline refuses to quote on incomplete submissions, so the quote-ready
rate is capped by what brokers actually send. This backfills one field at a
time from ground truth and re-runs routing, isolating how much of the quote
gap each field is responsible for.

The output is the operational answer to "what should we chase brokers for?"
"""

import json
from pathlib import Path

import classify
import extract
import route as router

ROOT = Path(__file__).resolve().parents[1]

BACKFILLABLE = {
    "payroll": ["payroll"],
    "deductible": ["deductible"],
    "loss_history": ["loss_count_5yr", "loss_incurred_5yr", "loss_history"],
    "limit_occurrence": ["limit_occurrence"],
    "revenue": ["revenue"],
    "naics": ["naics", "class_code"],
    "location": ["state", "location"],
}


def pipeline_fields(rec):
    fields, ev = extract.extract(rec["email_text"])
    conf = None
    if fields.get("naics") is None and fields.get("description"):
        naics, code, conf = classify.predict(fields["description"])
        if naics:
            fields["naics"], fields["class_code"] = naics, code
            fields["naics_source"] = "inferred"
    elif fields.get("naics"):
        fields["class_code"] = classify.naics_to_class_code().get(fields["naics"])
    return fields, ev, conf


def truth_value(rec, key):
    if key == "loss_history":
        return rec["loss_count_5yr"]
    if key in ("state", "location"):
        return rec["state"]
    return rec.get(key)


def main():
    rows = [json.loads(l) for l in open(ROOT / "data" / "submissions.jsonl")]
    classify.get_model()

    cache = [(rec, *pipeline_fields(rec)) for rec in rows]
    n = len(rows)

    def quote_rate(mutate=None):
        q = 0
        for rec, fields, ev, conf in cache:
            f = dict(fields)
            if mutate:
                for key in mutate:
                    if f.get(key) is None:
                        f[key] = truth_value(rec, key)
            if router.route(f, ev, conf)["quote_ready"]:
                q += 1
        return q / n

    base = quote_rate()
    ceiling = sum(router.route_from_truth(r) in ("BOP", "GL", "PL") for r in rows) / n

    results = {
        "baseline_quote_ready_rate": round(base, 4),
        "ceiling_with_complete_data": round(ceiling, 4),
        "gap": round(ceiling - base, 4),
        "per_field_lift": {},
    }
    for name, keys in BACKFILLABLE.items():
        lifted = quote_rate(keys)
        results["per_field_lift"][name] = {
            "quote_ready_rate_if_always_supplied": round(lifted, 4),
            "absolute_lift_pp": round((lifted - base) * 100, 2),
        }

    all_keys = [k for keys in BACKFILLABLE.values() for k in keys]
    results["all_fields_supplied"] = round(quote_rate(all_keys), 4)

    results["per_field_lift"] = dict(sorted(
        results["per_field_lift"].items(),
        key=lambda kv: -kv[1]["absolute_lift_pp"]))

    with open(ROOT / "outputs" / "sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
