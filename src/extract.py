"""
Field extraction: messy broker email -> the eight standardized fields.

Deterministic rules only. Every extracted value carries the raw span it came
from, which is what the audit trail surfaces when an underwriter asks why a
submission was routed the way it was.
"""

import re

FIELDS = ["naics", "class_code", "revenue", "payroll", "limit_occurrence",
          "deductible", "loss_history", "location"]

# --- money parsing ---------------------------------------------------------
_MONEY = r"(?:~|approximately\s+|approx\.?\s+|about\s+)?\$?\s?([\d,]+(?:\.\d+)?)\s?([KkMm])?"


def parse_money(num, suffix):
    v = float(num.replace(",", ""))
    if suffix:
        v *= 1_000 if suffix.lower() == "k" else 1_000_000
    return int(round(v))


def _find(patterns, text):
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m
    return None


REVENUE_PAT = [
    rf"(?:annual\s+revenue|gross\s+sales|annual\s+sales|revenue|receipts|sales)\s*[:\-]?\s*{_MONEY}",
]
PAYROLL_PAT = [
    rf"(?:annual\s+payroll|total\s+payroll|payroll)\s*[:\-]?\s*{_MONEY}",
]
LIMIT_PAT = [
    rf"{_MONEY}\s*(?:per\s+occurrence|/\s*occ|occurrence)",
    rf"(?:requested\s+limits?|limits?)\s*[:\-]?\s*{_MONEY}",
]
DED_PAT = [
    rf"(?:deductible|ded\.?|retention|SIR)\s*[:\-]?\s*{_MONEY}",
]
NAICS_PAT = [r"naics\s*(?:code)?\s*[:\-]?\s*(\d{6})"]
STATE_PAT = [r",\s*([A-Z]{2})\s+\d{5}", r",\s*([A-Z]{2})\s*\(", r",\s*([A-Z]{2})\b"]

CLEAN_LOSS = [
    r"no\s+claims", r"clean[,\s]", r"none\s+in\s+the\s+past", r"no\s+(?:prior\s+)?losses",
    r"loss\s+free", r"no\s+claims\s+reported",
]
LOSS_COUNT_PAT = [
    r"(\d+)\s+(?:reported\s+)?claims?",
    r"(\d+)\s+(?:reported\s+)?losses",
    # "Prior claims - 2 in 5 years": the count trails the noun
    r"(?:claims?|losses)\s*[-–:]\s*(\d+)\s+in\s+\d+",
    r"(\d+)\s+in\s+(?:the\s+)?(?:last\s+|past\s+)?\d+\s*(?:yrs?|years)",
]
LOSS_AMT_PAT = [
    rf"(?:totaling|total\s+incurred|incurred)\s*[:\-]?\s*{_MONEY}",
    rf"{_MONEY}\s+total\s+incurred",
    rf"{_MONEY}\s+incurred",
]

DESC_PAT = [
    r"business\s+description\s*[:\-]\s*(.+)",
    r"operations\s*[:\-]\s*(?:they\s+run\s+an?\s+)?(.+?)\.?$",
]


def extract(email_text):
    """Return (fields dict, evidence dict). Missing fields are None."""
    t = email_text
    f, ev = {}, {}

    def money_field(key, pats):
        m = _find(pats, t)
        if m:
            f[key] = parse_money(m.group(1), m.group(2))
            ev[key] = m.group(0).strip()
        else:
            f[key] = None
            ev[key] = None

    money_field("revenue", REVENUE_PAT)
    money_field("payroll", PAYROLL_PAT)
    money_field("limit_occurrence", LIMIT_PAT)
    money_field("deductible", DED_PAT)

    m = _find(NAICS_PAT, t)
    f["naics"] = m.group(1) if m else None
    ev["naics"] = m.group(0).strip() if m else None
    f["naics_source"] = "stated" if m else None

    # location / state
    m = _find(STATE_PAT, t)
    f["state"] = m.group(1) if m else None
    ev["location"] = m.group(0).strip() if m else None
    f["location"] = f["state"]

    # description (drives classification when NAICS is absent)
    d = None
    for p in DESC_PAT:
        mm = re.search(p, t, re.IGNORECASE | re.MULTILINE)
        if mm:
            d = mm.group(1).strip().rstrip(".")
            break
    f["description"] = d
    ev["description"] = d

    # loss history
    # A deductible line reads "$500 per claim", so keyword presence alone is not
    # enough - the line must be anchored on a loss-history label.
    LOSS_ANCHOR = r"^\s*(loss\s+history|loss\s+runs|losses|prior\s+(?:losses|claims)|claims)\b"
    EXCLUDE = r"\b(deductible|retention|SIR|per\s+claim\s*$)\b"
    loss_line = None
    for line in t.splitlines():
        if re.search(EXCLUDE, line, re.IGNORECASE):
            continue
        if re.search(LOSS_ANCHOR, line, re.IGNORECASE):
            loss_line = line
            break
    if loss_line is None:  # fall back to any non-deductible line mentioning claims
        for line in t.splitlines():
            if re.search(EXCLUDE, line, re.IGNORECASE):
                continue
            if re.search(r"\b(loss(es)?|claims?)\b", line, re.IGNORECASE):
                loss_line = line
                break
    if loss_line is None:
        f["loss_count_5yr"], f["loss_incurred_5yr"] = None, None
        ev["loss_history"] = None
    elif _find(CLEAN_LOSS, loss_line):
        f["loss_count_5yr"], f["loss_incurred_5yr"] = 0, 0
        ev["loss_history"] = loss_line.strip()
    else:
        mc = _find(LOSS_COUNT_PAT, loss_line)
        ma = _find(LOSS_AMT_PAT, loss_line)
        f["loss_count_5yr"] = int(mc.group(1)) if mc else None
        f["loss_incurred_5yr"] = parse_money(ma.group(1), ma.group(2)) if ma else None
        ev["loss_history"] = loss_line.strip()

    f["loss_history"] = None if f["loss_count_5yr"] is None else f["loss_count_5yr"]
    f["class_code"] = None  # filled by the classifier stage
    return f, ev


def missing_fields(f):
    out = []
    for k in FIELDS:
        v = f.get(k)
        if v is None:
            out.append(k)
    return out
