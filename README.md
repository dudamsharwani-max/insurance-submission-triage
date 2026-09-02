# AI-Assisted Commercial Insurance Submission Triage

A submission intake pipeline that takes an unstructured broker email, standardizes it
into eight underwriting fields, and routes it to **BOP**, **General Liability**,
**Professional Liability**, **underwriter referral**, or **decline** against a
configurable appetite guide — with a full audit trail behind every decision.

**Live demo:** _(add your Streamlit Cloud URL here)_

---

## The problem

Small-commercial submissions arrive as prose. A broker writes "gross sales ~2.6M,
1 claim last 5 years, $37K incurred" and an underwriting assistant spends ten to
fifteen minutes rekeying it, looking up a class code, and deciding whether it can
be quoted straight through. Most of that work is deterministic. The part that
isn't — deciding what kind of business this actually *is* from a free-text
description — is a classification problem.

This project separates those two jobs and measures each one.

## Architecture

```
broker email
     │
     ▼
┌─────────────────┐   regex + unit parsing, every value keeps the raw span
│  extract.py     │   it came from
└────────┬────────┘
         │  NAICS missing on 62% of submissions
         ▼
┌─────────────────┐   TF-IDF (word 1-2gram + char_wb 3-5gram) → calibrated
│  classify.py    │   logistic regression. Abstains below a 0.55 confidence floor.
└────────┬────────┘
         ▼
┌─────────────────┐   YAML appetite table: declines → mandatory referrals →
│  route.py       │   completeness gate → product assignment
└────────┬────────┘
         ▼
   routing decision + audit record
```

The decision path contains **no LLM**. Everything that determines a route is either
a regex with recorded provenance or a linear model with a calibrated probability.
That is a deliberate choice: an underwriter has to be able to ask "why did this
decline?" and get a rule ID, not a paraphrase.

## The eight standardized fields

NAICS · class code · revenue · payroll · limits · deductibles · loss history · location

## Data

**The corpus is synthetic — 1,000 generated submissions.** No public dataset of real
commercial submissions exists, and real ones contain named insureds and loss detail
that cannot be published.

The generator (`src/generate_submissions.py`) builds a clean ground-truth record
first, then renders a messy broker email that only partially reveals it:

- Field-specific drop rates modeled on what actually goes missing (payroll 34%,
  deductible 28%, loss history 22%, NAICS 62%)
- Money written six different ways — `~2.6M`, `$1.2M`, `approximately $410,000`,
  `1,796,000`
- Broker shorthand and typos injected into descriptions (`comml janitorial svcs`,
  `restaraunt`)
- 30 business classes across retail, professional services, healthcare, contracting,
  and excluded classes

The pipeline never sees the ground-truth record. That separation is what makes the
metrics below meaningful.

## Results (n = 1,000)

| Metric | Value |
|---|---|
| Quote-ready rate | 14.8% |
| Routing agreement with ground truth | 76.0% |
| **Safe-decision rate** | **96.4%** |
| Risks bound out of appetite | **0** |
| In-appetite risks declined | **0** |
| Median time-to-quote | 4 ms |

### Accuracy alone is the wrong metric

Raw agreement was 76.0%, but that number treats two very different failures as
equivalent. Decomposing it:

| Outcome | n |
|---|---|
| Agreement | 760 |
| Conservative — referred something quotable | 150 |
| Conservative — referred something declinable | 54 |
| **Wrong product** | **25** |
| **Skipped a required referral** | **11** |

Nearly all the disagreement is the pipeline declining to quote on thin data, which
is the intended behavior. Only 3.6% of decisions are unsafe, and the two worst
categories — binding a risk that should have been declined, declining a risk that
was in appetite — are zero.

Per-route precision: DECLINE 1.00, BOP 0.90, PL 0.88, REFERRAL 0.70, GL 0.63. BOP
recall is low (0.22) precisely because BOP eligibility depends on payroll, the field
brokers omit most often — an unknown payroll cannot silently qualify a risk for a
package policy, so it routes to GL or referral instead.

### Extraction

Missing-field rates converge on the generator's true drop rates within ~1pp, meaning
extraction recall is effectively complete and the missing-field metric is measuring
broker behavior rather than parser failure.

| Field | Missing | True drop rate |
|---|---|---|
| Payroll | 35.3% | 34% |
| Deductible | 28.6% | 28% |
| Loss history | 21.2% | 22% |
| Limits | 12.8% | 14% |
| Revenue | 11.4% | 10% |
| Location | 4.9% | 5% |

Getting there required fixing a bug worth naming: the loss-history extractor was
matching the deductible line, because `Deductible: $500 per claim` contains the word
"claim". Anchoring on loss-labeled line starts and excluding deductible lines
recovered 9pp of loss history.

### Classification

Evaluation splits by **phrasing template, not by row** — one description variant per
class is held out of training entirely, so the model is tested on wording it has
never seen. Splitting by row would leak templates across the split.

| Split | Accuracy |
|---|---|
| Seen wording | 100% |
| **Unseen wording** | **54.2%** |

That gap is the honest result, and it is why the confidence floor exists. On the 615
submissions with no stated NAICS, the classifier answers on 71.7% of them and is
correct **95.2%** of the time when it answers. The remaining 28.3% abstain to an
underwriter rather than guess. Routing tolerates near-misses better than
classification does, because appetite rules key on the 4-digit NAICS prefix — confusing
a law firm with an accounting firm still routes correctly to PL.

### Missing-data economics

`src/sensitivity.py` backfills one field at a time from ground truth and re-runs
routing, isolating what each missing field costs in quote throughput.

| Field always supplied | Quote-rate lift |
|---|---|
| Loss history | +4.5 pp |
| NAICS | +3.1 pp |
| Revenue | +2.0 pp |
| Limits | +1.5 pp |
| Location | +0.8 pp |
| Payroll | **−0.5 pp** |
| Deductible | **−0.5 pp** |

Loss history is the highest-value thing to chase a broker for. Payroll and deductible
have *negative* lift — supplying them surfaces referral triggers (payroll above $5M,
deductible below $1,000) that were invisible while the field was blank. Chasing them
improves pricing accuracy, not throughput. Ceiling with fully complete data is 28.7%,
so roughly half the current quote gap is recoverable through better intake.

## Audit trail

Every decision writes a record naming the rule that fired, the fields it relied on,
the raw text each field came from, and the confidence of any inferred value:

```json
{
  "submission_id": "SUB-00042",
  "route": "REFERRAL",
  "rule_id": "REF-02",
  "reason": "Loss history present - any incurred loss over $75k in 5 years",
  "naics_source": "inferred",
  "naics_confidence": 0.83,
  "missing_fields": ["payroll"],
  "decision_trail": [
    {"stage": "decline",  "rule": "-",      "result": "no decline rule matched"},
    {"stage": "referral", "rule": "REF-02", "result": "matched"}
  ],
  "evidence": {
    "revenue": "Gross sales: ~1.5M",
    "loss_history": "Losses: 1 reported claim past 5 yrs totaling $85,300 incurred."
  }
}
```

## Running it

```bash
pip install -r requirements.txt
python src/generate_submissions.py   # build the corpus
python src/classify.py               # train + evaluate the classifier
python src/evaluate.py               # end-to-end metrics + audit log
python src/sensitivity.py            # missing-data analysis
streamlit run app.py                 # interactive triage + dashboards
```

## Limitations

- **Synthetic data.** Generated from a hand-built taxonomy of 30 classes. Real
  submissions have more class variety, worse formatting, attachments, and ACORD
  forms. Extraction performance here is an upper bound.
- **The appetite guide is invented**, modeled on typical small-commercial appetite
  rather than any carrier's actual filed program.
- **The manual baseline (14 min/submission) is a stated reference point**, not an
  observed measurement, so the time-to-quote comparison is illustrative.
- **Unseen-wording classification at 54.2% is not production-ready.** The confidence
  floor makes it safe, not good. More training phrasings, an embedding-based
  retrieval step against the real NAICS index, or a human-in-the-loop correction
  feed would all move it.

## Stack

Python · scikit-learn · pandas · PyYAML · Streamlit
