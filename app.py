"""
Commercial Submission Triage - Streamlit front end.

Run:  streamlit run app.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import classify  # noqa: E402
import extract  # noqa: E402
import route as router  # noqa: E402

st.set_page_config(page_title="Submission Triage", page_icon="📋", layout="wide")

ROUTE_STYLE = {
    "BOP": ("#1b7f4d", "Businessowners Policy"),
    "GL": ("#1b5f9f", "General Liability"),
    "PL": ("#6b3fa0", "Professional Liability"),
    "REFERRAL": ("#b3701a", "Underwriter referral"),
    "DECLINE": ("#a12d2d", "Out of appetite"),
}

SAMPLE = """Hi team,

New submission for your review.

Insured: Cedar Park Salon LLC
Business description: full service salon and day spa
Address: 4120 Commerce St, Naperville, IL 60540
Annual revenue: ~1.2M
Annual payroll: $410,000
Requested limits: $1M per occurrence / $2M aggregate
Deductible: $2,500
Loss history: none in the past 5 years, loss runs attached.

Let me know what you need. Thanks,
Dana Whitfield
Halstead Risk Partners"""


@st.cache_resource
def _model():
    return classify.get_model()


@st.cache_data
def _load(name):
    p = ROOT / "outputs" / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def _audit():
    p = ROOT / "outputs" / "audit_log.jsonl"
    if not p.exists():
        return None
    return pd.DataFrame([json.loads(l) for l in p.open()])


def run_pipeline(text):
    fields, ev = extract.extract(text)
    conf = None
    if fields.get("naics") is None and fields.get("description"):
        _model()
        naics, code, conf = classify.predict(fields["description"])
        if naics:
            fields["naics"], fields["class_code"] = naics, code
            fields["naics_source"] = "inferred"
            ev["naics"] = f"inferred from description (p={conf:.2f})"
    elif fields.get("naics"):
        fields["class_code"] = classify.naics_to_class_code().get(fields["naics"])
    return fields, router.route(fields, ev, conf)


st.title("Commercial Submission Triage")
st.caption("Synthetic data. Rules-based appetite engine with an NLP classification "
           "assist — no LLM in the decision path.")

tab1, tab2, tab3 = st.tabs(["Triage a submission", "Pipeline performance",
                            "Missing-data economics"])

# ---------------------------------------------------------------- tab 1
with tab1:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Broker submission")
        text = st.text_area("Paste the submission email", SAMPLE, height=380,
                            label_visibility="collapsed")
        go = st.button("Run triage", type="primary")

    with right:
        if go or text:
            fields, dec = run_pipeline(text)
            color, label = ROUTE_STYLE.get(dec["route"], ("#555", ""))
            st.markdown(
                f"<div style='background:{color};color:white;padding:14px 18px;"
                f"border-radius:8px;'><span style='font-size:26px;font-weight:700'>"
                f"{dec['route']}</span><br><span style='opacity:.85'>{label}</span>"
                f"</div>", unsafe_allow_html=True)
            st.markdown(f"**Rule fired:** `{dec['rule_id']}` — {dec['reason']}")

            c1, c2, c3 = st.columns(3)
            c1.metric("Quote ready", "Yes" if dec["quote_ready"] else "No")
            c2.metric("Missing fields", len(dec["missing_fields"]))
            c3.metric("Class confidence",
                      "stated" if dec["naics_source"] == "stated"
                      else (f"{dec['naics_confidence']:.2f}"
                            if dec["naics_confidence"] else "—"))

            st.subheader("Standardized fields")
            rows = []
            for k, v in dec["fields_used"].items():
                rows.append({"Field": k, "Value": "—" if v is None else v,
                             "Source text": dec["evidence"].get(
                                 {"loss_count_5yr": "loss_history",
                                  "loss_incurred_5yr": "loss_history",
                                  "state": "location"}.get(k, k)) or "—"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

            st.subheader("Audit trail")
            for step in dec["decision_trail"]:
                st.markdown(f"- **{step['stage']}** · `{step['rule']}` — {step['result']}")
            if dec["missing_fields"]:
                st.warning("Blocked from straight-through quoting by missing: "
                           + ", ".join(dec["missing_fields"]))

# ---------------------------------------------------------------- tab 2
with tab2:
    m = _load("metrics.json")
    if not m:
        st.info("Run `python src/evaluate.py` to generate metrics.")
    else:
        c = st.columns(4)
        c[0].metric("Submissions", f"{m['n_submissions']:,}")
        c[1].metric("Quote-ready rate", f"{m['quote_ready_rate']:.1%}")
        c[2].metric("Safe decisions", f"{m['safe_decision_rate']:.1%}")
        c[3].metric("Median latency", f"{m['time_to_quote']['median_ms']:.0f} ms")

        st.subheader("Where decisions disagree with ground truth")
        st.caption("Raw accuracy hides the distinction that matters: refusing to "
                   "quote on thin data is safe, putting a risk on the wrong paper "
                   "is not.")
        ed = pd.DataFrame(
            [{"Outcome": k.replace("_", " "), "Count": v}
             for k, v in m["error_decomposition"].items()]).sort_values(
            "Count", ascending=False)
        st.dataframe(ed, hide_index=True, width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Route distribution")
            dist = pd.DataFrame({
                "Pipeline": m["route_distribution"],
                "Ground truth": m["truth_distribution"]}).fillna(0)
            st.bar_chart(dist)
        with c2:
            st.subheader("Missing-field rate")
            st.bar_chart(pd.Series(m["missing_field_rate"], name="rate"))

        st.subheader("Per-route precision and recall")
        st.dataframe(pd.DataFrame(m["per_route"]).T, width="stretch")

        ni = m["naics_inference"]
        st.subheader("Class inference")
        st.markdown(
            f"NAICS was absent on **{ni['submissions_requiring_inference']}** "
            f"submissions. The classifier answered on "
            f"**{ni['coverage_above_floor']:.1%}** of those at a "
            f"{ni['confidence_floor']} confidence floor, and was correct "
            f"**{ni['precision_above_confidence_floor']:.1%}** of the time when it "
            f"did answer. The rest abstain to an underwriter rather than guess.")

        df = _audit()
        if df is not None:
            st.subheader("Audit log")
            st.dataframe(df[["submission_id", "route", "truth_route", "rule_id",
                             "reason", "quote_ready", "latency_ms"]].head(200),
                         hide_index=True, width="stretch")

# ---------------------------------------------------------------- tab 3
with tab3:
    s = _load("sensitivity.json")
    if not s:
        st.info("Run `python src/sensitivity.py` to generate this analysis.")
    else:
        c = st.columns(3)
        c[0].metric("Current quote-ready", f"{s['baseline_quote_ready_rate']:.1%}")
        c[1].metric("Ceiling with complete data", f"{s['ceiling_with_complete_data']:.1%}")
        c[2].metric("Gap", f"{s['gap']*100:.1f} pp")

        st.subheader("Quote-rate lift if brokers always supplied the field")
        lift = pd.DataFrame([
            {"Field": k, "Lift (pp)": v["absolute_lift_pp"]}
            for k, v in s["per_field_lift"].items()])
        st.bar_chart(lift.set_index("Field"))
        st.dataframe(lift, hide_index=True, width="stretch")
        st.info("Payroll and deductible carry **negative** lift. Supplying them "
                "surfaces referral triggers — payroll above $5M, deductibles below "
                "$1,000 — that were invisible while the field was blank. Chasing "
                "brokers for them improves pricing accuracy, not throughput.")
