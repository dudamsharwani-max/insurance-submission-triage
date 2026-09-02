"""
Run the full pipeline over the corpus and score it.

Metrics reported:
  quote_ready_rate    share of submissions routed straight through to a product
  referral_accuracy   agreement with ground-truth routing, plus per-route P/R
  missing_field_rate  per-field share of submissions where extraction found nothing
  time_to_quote       measured pipeline latency vs. a manual-handling baseline
"""

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import classify
import extract
import route as router

ROOT = Path(__file__).resolve().parents[1]

# Baseline: median manual triage handling time for a small-commercial
# submission, used only as a stated reference point for the comparison.
MANUAL_BASELINE_SECONDS = 14 * 60


def run_one(rec):
    t0 = time.perf_counter()
    fields, ev = extract.extract(rec["email_text"])

    conf = None
    if fields.get("naics") is None and fields.get("description"):
        naics, class_code, conf = classify.predict(fields["description"])
        if naics:
            fields["naics"] = naics
            fields["class_code"] = class_code
            fields["naics_source"] = "inferred"
            ev["naics"] = f"inferred from description (p={conf:.2f})"
    elif fields.get("naics"):
        fields["class_code"] = classify.naics_to_class_code().get(fields["naics"])

    decision = router.route(fields, ev, conf)
    decision["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    decision["submission_id"] = rec["submission_id"]
    return fields, decision


def main():
    rows = [json.loads(l) for l in open(ROOT / "data" / "submissions.jsonl")]
    classify.get_model()  # warm the model so latency excludes one-time load

    results, missing_counter = [], Counter()
    naics_hits = naics_total = 0
    conf_hits = conf_total = 0

    for rec in rows:
        fields, dec = run_one(rec)
        truth = router.route_from_truth(rec)
        dec["truth_route"] = truth
        dec["correct"] = dec["route"] == truth
        results.append(dec)

        for k in extract.FIELDS:
            if k == "class_code":
                continue
            if fields.get(k) is None:
                missing_counter[k] += 1

        if not rec["stated_fields"].get("naics"):
            naics_total += 1
            if fields.get("naics") == rec["naics"]:
                naics_hits += 1
            if fields.get("naics") is not None:
                conf_total += 1
                if fields.get("naics") == rec["naics"]:
                    conf_hits += 1

    n = len(results)
    quote_ready = sum(r["quote_ready"] for r in results)
    correct = sum(r["correct"] for r in results)
    lat = sorted(r["latency_ms"] for r in results)

    # per-route precision / recall
    per_route = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for r in results:
        p, t = r["route"], r["truth_route"]
        if p == t:
            per_route[p]["tp"] += 1
        else:
            per_route[p]["fp"] += 1
            per_route[t]["fn"] += 1
    routing = {}
    for k, v in per_route.items():
        prec = v["tp"] / (v["tp"] + v["fp"]) if v["tp"] + v["fp"] else 0.0
        rec_ = v["tp"] / (v["tp"] + v["fn"]) if v["tp"] + v["fn"] else 0.0
        routing[k] = {"precision": round(prec, 3), "recall": round(rec_, 3),
                      "support": v["tp"] + v["fn"]}

    # Raw accuracy against ground truth conflates two very different things:
    # refusing to quote on incomplete data (safe, and the intended behaviour)
    # and putting a risk on the wrong paper (unsafe). Split them.
    PRODUCTS = {"BOP", "GL", "PL"}
    err = Counter()
    for r in results:
        p, t = r["route"], r["truth_route"]
        if p == t:
            err["agreement"] += 1
        elif t == "DECLINE" and p in PRODUCTS:
            err["critical_bound_out_of_appetite"] += 1
        elif t in PRODUCTS and p in PRODUCTS:
            err["wrong_product"] += 1
        elif t == "REFERRAL" and p in PRODUCTS:
            err["aggressive_skipped_referral"] += 1
        elif t in PRODUCTS and p == "REFERRAL":
            err["conservative_referred_a_quotable"] += 1
        elif t in PRODUCTS and p == "DECLINE":
            err["critical_declined_in_appetite"] += 1
        elif t == "DECLINE" and p == "REFERRAL":
            err["conservative_referred_a_decline"] += 1
        elif t == "REFERRAL" and p == "DECLINE":
            err["conservative_declined_a_referral"] += 1
        else:
            err["other"] += 1
    unsafe = (err["critical_bound_out_of_appetite"] + err["wrong_product"]
              + err["aggressive_skipped_referral"] + err["critical_declined_in_appetite"])

    metrics = {
        "n_submissions": n,
        "error_decomposition": dict(err),
        "unsafe_decision_rate": round(unsafe / n, 4),
        "safe_decision_rate": round(1 - unsafe / n, 4),
        "quote_ready_rate": round(quote_ready / n, 4),
        "routing_accuracy": round(correct / n, 4),
        "route_distribution": dict(Counter(r["route"] for r in results)),
        "truth_distribution": dict(Counter(r["truth_route"] for r in results)),
        "per_route": routing,
        "missing_field_rate": {k: round(v / n, 4) for k, v in
                               sorted(missing_counter.items(), key=lambda x: -x[1])},
        "naics_inference": {
            "submissions_requiring_inference": naics_total,
            "recall_all": round(naics_hits / naics_total, 4) if naics_total else None,
            "precision_above_confidence_floor": round(conf_hits / conf_total, 4) if conf_total else None,
            "coverage_above_floor": round(conf_total / naics_total, 4) if naics_total else None,
            "confidence_floor": classify.CONFIDENCE_FLOOR,
        },
        "time_to_quote": {
            "median_ms": lat[n // 2],
            "p95_ms": lat[int(n * 0.95)],
            "total_corpus_seconds": round(sum(lat) / 1000, 2),
            "manual_baseline_seconds_per_submission": MANUAL_BASELINE_SECONDS,
            "note": "Baseline is a stated reference point, not an observed measurement.",
        },
    }

    (ROOT / "outputs").mkdir(exist_ok=True)
    with open(ROOT / "outputs" / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(ROOT / "outputs" / "audit_log.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(json.dumps({k: v for k, v in metrics.items()
                      if k != "per_route"}, indent=2))
    print("\nper-route:")
    print(json.dumps(routing, indent=2))
    return metrics


if __name__ == "__main__":
    main()
