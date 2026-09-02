"""
Generate a synthetic corpus of commercial insurance submissions.

Every submission has two halves:
  1. a clean ground-truth record (the eight standardized fields), and
  2. a messy free-text broker email that only partially reveals it.

The pipeline never sees half 1. That separation is what makes the
extraction and routing metrics in evaluate.py meaningful.
"""

import json
import random
from pathlib import Path

SEED = 20260902
ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Class taxonomy. class_code is an ISO-style GL code (synthetic but plausible).
# ---------------------------------------------------------------------------
CLASSES = [
    # (label, naics, class_code, rev_range, payroll_ratio, description phrasings)
    ("restaurant", "722511", "16901", (400_000, 4_500_000), 0.32, [
        "full service restaurant with a small bar", "sit-down Italian restaurant",
        "family owned diner serving breakfast and lunch", "casual dining establishment, 60 seats",
        "neighborhood restaurant, table service, beer and wine only",
    ]),
    ("coffee_shop", "722515", "16916", (180_000, 1_800_000), 0.30, [
        "coffee shop and bakery", "espresso bar with light pastry service",
        "quick service cafe, counter only", "third wave coffee roaster and retail cafe",
    ]),
    ("grocery", "445110", "18501", (900_000, 12_000_000), 0.18, [
        "independent grocery store", "neighborhood supermarket with deli counter",
        "specialty food market and butcher", "small format grocer, 8 employees",
    ]),
    ("liquor_store", "445320", "12356", (500_000, 5_000_000), 0.14, [
        "retail liquor store", "wine and spirits shop, off premises only",
        "package liquor store with beer cave",
    ]),
    ("clothing_retail", "448110", "18206", (250_000, 6_000_000), 0.16, [
        "mens clothing boutique", "womens apparel retail store",
        "small chain of clothing stores, 3 locations", "streetwear retail shop",
    ]),
    ("gift_shop", "453220", "18435", (120_000, 1_500_000), 0.19, [
        "gift shop and card store", "home goods and gift retailer",
        "souvenir shop near the waterfront",
    ]),
    ("hardware", "444140", "18206", (600_000, 9_000_000), 0.17, [
        "hardware store", "paint and hardware retailer",
        "independent hardware and garden supply",
    ]),
    ("office_lessor", "531210", "61225", (300_000, 20_000_000), 0.06, [
        "commercial real estate office, we lease office suites",
        "property management firm handling office buildings",
        "real estate brokerage and building lessor",
    ]),
    ("salon", "812112", "10111", (150_000, 1_400_000), 0.38, [
        "hair salon", "beauty salon offering cuts and color",
        "full service salon and day spa", "barbershop, four chairs",
    ]),
    ("dry_cleaner", "812320", "11126", (200_000, 2_200_000), 0.29, [
        "dry cleaning and laundry", "dry cleaner with pickup and delivery",
        "commercial laundry service",
    ]),
    ("law_firm", "541110", "66524", (400_000, 18_000_000), 0.45, [
        "law firm, general practice", "boutique litigation firm, 6 attorneys",
        "immigration law practice", "estate planning and probate attorneys",
    ]),
    ("accounting", "541211", "66512", (300_000, 14_000_000), 0.42, [
        "CPA firm doing tax and audit work", "accounting and bookkeeping services",
        "tax preparation practice, seasonal staff",
    ]),
    ("architecture", "541310", "66629", (500_000, 16_000_000), 0.44, [
        "architecture firm", "architectural design studio, commercial projects",
        "landscape architecture practice",
    ]),
    ("engineering", "541330", "66630", (700_000, 30_000_000), 0.46, [
        "civil engineering consultancy", "structural engineering firm",
        "MEP engineering consultants",
    ]),
    ("it_consulting", "541512", "66590", (350_000, 22_000_000), 0.48, [
        "IT consulting and systems integration", "software development consultancy",
        "managed IT services provider for small business", "cybersecurity advisory firm",
    ]),
    ("mgmt_consulting", "541611", "66591", (250_000, 25_000_000), 0.47, [
        "management consulting firm", "operations consulting practice",
        "HR consulting and staffing advisory",
    ]),
    ("marketing_agency", "541810", "66593", (300_000, 11_000_000), 0.43, [
        "digital marketing agency", "advertising agency, creative and media buying",
        "brand strategy and design studio",
    ]),
    ("medical_office", "621111", "80001", (400_000, 13_000_000), 0.40, [
        "family medicine practice", "physicians office, two doctors",
        "outpatient primary care clinic",
    ]),
    ("dental", "621210", "80003", (350_000, 6_000_000), 0.39, [
        "dental practice", "orthodontics office", "general dentistry, one operatory",
    ]),
    ("janitorial", "561720", "91555", (250_000, 8_000_000), 0.52, [
        "commercial janitorial services", "office cleaning contractor",
        "building maintenance and custodial services",
    ]),
    ("trucking", "484110", "99999", (800_000, 14_000_000), 0.35, [
        "local trucking and delivery, 12 power units",
        "short haul freight carrier", "last mile delivery operation",
    ]),
    ("printing", "323111", "56913", (400_000, 7_000_000), 0.31, [
        "commercial printing shop", "digital printing and signage",
        "print and mailing services",
    ]),
    ("warehouse", "493110", "99999", (600_000, 18_000_000), 0.22, [
        "general warehousing and storage", "third party logistics warehouse",
        "cold storage warehouse operator",
    ]),
    ("fitness", "713940", "18435", (200_000, 4_000_000), 0.34, [
        "fitness center and gym", "boutique fitness studio, yoga and pilates",
        "24 hour gym with free weights",
    ]),
    ("landscaping", "561730", "97047", (180_000, 5_000_000), 0.44, [
        "landscaping and lawn maintenance", "commercial grounds keeping",
        "landscape design and installation",
    ]),
    # --- classes that should hit hard declines ---
    ("residential_contractor", "236220", "91340", (900_000, 30_000_000), 0.40, [
        "commercial building general contractor", "GC doing ground up commercial construction",
        "construction firm, mixed commercial projects",
    ]),
    ("roofing", "238160", "91746", (500_000, 12_000_000), 0.42, [
        "roofing contractor, residential and light commercial",
        "roofing and gutter installation",
    ]),
    ("electrical_contractor", "238210", "92478", (600_000, 15_000_000), 0.41, [
        "electrical contractor", "commercial electrical installation and service",
    ]),
    ("gun_shop", "459310", "12356", (300_000, 3_000_000), 0.17, [
        "sporting goods and firearms retailer", "gun shop and indoor range",
    ]),
    ("cannabis", "111419", "01476", (700_000, 9_000_000), 0.30, [
        "licensed cannabis cultivation facility", "indoor cannabis grow operation",
    ]),
]

STATES = ["IL", "WI", "IN", "MI", "OH", "TX", "CA", "NY", "GA", "NC",
          "AZ", "CO", "MN", "MO", "PA", "FL", "LA", "AK"]
STATE_W = [14, 8, 7, 7, 7, 9, 9, 7, 5, 5, 4, 4, 4, 3, 5, 3, 2, 1]

CITIES = {
    "IL": ["Chicago", "Naperville", "Rockford", "Evanston"],
    "WI": ["Milwaukee", "Madison", "Green Bay"], "IN": ["Indianapolis", "Fort Wayne"],
    "MI": ["Detroit", "Grand Rapids", "Ann Arbor"], "OH": ["Columbus", "Cleveland"],
    "TX": ["Austin", "Dallas", "Houston"], "CA": ["San Diego", "Fresno", "Oakland"],
    "NY": ["Buffalo", "Rochester", "Brooklyn"], "GA": ["Atlanta", "Savannah"],
    "NC": ["Charlotte", "Raleigh"], "AZ": ["Phoenix", "Tempe"], "CO": ["Denver", "Boulder"],
    "MN": ["Minneapolis", "St Paul"], "MO": ["St Louis", "Kansas City"],
    "PA": ["Philadelphia", "Pittsburgh"], "FL": ["Tampa", "Orlando"],
    "LA": ["New Orleans", "Baton Rouge"], "AK": ["Anchorage"],
}

LIMITS = [500_000, 1_000_000, 1_000_000, 1_000_000, 2_000_000, 2_000_000, 3_000_000, 5_000_000]
DEDUCTIBLES = [500, 1_000, 1_000, 2_500, 2_500, 5_000, 10_000, 25_000]

BROKERS = ["Halstead Risk Partners", "Kenmore & Doyle Insurance", "Ridgeline Brokerage",
           "Anchor Commercial Group", "Trillium Insurance Services", "Bayard Agency"]
AGENTS = ["Dana Whitfield", "Marcus Oyelaran", "Priya Raghunathan", "Tom Brennan",
          "Elena Vasquez", "Jordan Kimball", "Renee Okafor", "Sam Petrosyan"]
BIZ_TOKENS = ["Summit", "Cedar", "Northgate", "Blue Harbor", "Pioneer", "Elmwood",
              "Redstone", "Vantage", "Copperfield", "Larkspur", "Ironbridge", "Marlowe"]
BIZ_SUFFIX = ["LLC", "Inc.", "Group", "Partners", "Co.", "Holdings LLC"]


# ---------------------------------------------------------------------------
# Number formatting: the same value written the way brokers actually write it.
# ---------------------------------------------------------------------------
def fmt_money(v, rng, allow_fuzzy=True):
    style = rng.random()
    if allow_fuzzy and style < 0.18:
        m = v / 1_000_000
        return f"~{m:.1f}M" if m >= 1 else f"~{v/1000:.0f}k"
    if style < 0.34:
        m = v / 1_000_000
        return f"${m:.1f}M" if m >= 1 else f"${v/1000:.0f}K"
    if style < 0.50:
        return f"{v:,}"
    if style < 0.62:
        return f"approximately ${v:,}"
    return f"${v:,}"


def fmt_limit(v, rng):
    if rng.random() < 0.45:
        return f"${v//1_000_000}M" if v >= 1_000_000 else f"${v//1000}K"
    return f"${v:,}"


def make_record(rng, idx):
    label, naics, class_code, rev_range, pr_ratio, phrasings = rng.choice(CLASSES)
    revenue = int(rng.uniform(*rev_range))
    revenue = round(revenue, -3)
    payroll = int(revenue * pr_ratio * rng.uniform(0.7, 1.3))
    payroll = round(payroll, -3)
    state = rng.choices(STATES, weights=STATE_W, k=1)[0]
    city = rng.choice(CITIES[state])
    limit = rng.choice(LIMITS)
    ded = rng.choice(DEDUCTIBLES)

    # loss history: most risks are clean
    r = rng.random()
    if r < 0.58:
        loss_count, loss_incurred = 0, 0
    elif r < 0.82:
        loss_count = 1
        loss_incurred = round(int(rng.uniform(2_000, 90_000)), -2)
    elif r < 0.94:
        loss_count = 2
        loss_incurred = round(int(rng.uniform(15_000, 220_000)), -2)
    else:
        loss_count = rng.randint(3, 6)
        loss_incurred = round(int(rng.uniform(90_000, 900_000)), -2)

    name = f"{rng.choice(BIZ_TOKENS)} {rng.choice(['', '', 'Street ', 'Park '])}" \
           f"{label.split('_')[0].title()} {rng.choice(BIZ_SUFFIX)}".replace("  ", " ")

    return {
        "submission_id": f"SUB-{idx:05d}",
        "class_label": label,
        "business_name": name,
        "description": rng.choice(phrasings),
        "naics": naics,
        "class_code": class_code,
        "revenue": revenue,
        "payroll": payroll,
        "limit_occurrence": limit,
        "deductible": ded,
        "loss_count_5yr": loss_count,
        "loss_incurred_5yr": loss_incurred,
        "state": state,
        "city": city,
    }


# ---------------------------------------------------------------------------
# Email rendering. Fields are dropped at field-specific rates that mirror what
# actually goes missing on real small-commercial submissions.
# ---------------------------------------------------------------------------
DROP_RATES = {
    "revenue": 0.10, "payroll": 0.34, "limit_occurrence": 0.14,
    "deductible": 0.28, "loss_history": 0.22, "location": 0.05,
    "naics": 0.62,  # brokers rarely supply NAICS; it must be inferred
}

OPENERS = [
    "Hi team,\n\nNew submission for your review.",
    "Good morning,\n\nPlease see the below new business opportunity.",
    "Hello,\n\nSubmitting the following account for a quote.",
    "Team,\n\nFresh submission attached below - client is looking to bind by month end.",
    "Hi,\n\nQuoting request, effective date is flexible.",
]
CLOSERS = [
    "Let me know what you need. Thanks,",
    "Appreciate a quick turnaround on this one. Best,",
    "Happy to chase anything missing. Regards,",
    "Please advise on appetite. Thank you,",
    "Client is shopping this so timing matters. Thanks,",
]


ABBREV = {
    "and": "&", "services": "svcs", "commercial": "comml", "restaurant": "restaraunt",
    "consulting": "consultng", "management": "mgmt", "contractor": "contr",
    "professional": "prof", "maintenance": "maint", "company": "co",
    "installation": "install", "residential": "resi", "insurance": "ins",
}


def noisify(text, rng):
    """Broker shorthand and typos - the classifier has to survive this."""
    words = text.split()
    out = []
    for w in words:
        low = w.lower().strip(",.")
        if low in ABBREV and rng.random() < 0.5:
            out.append(ABBREV[low])
            continue
        if len(w) > 6 and rng.random() < 0.08:  # transpose two chars
            i = rng.randint(1, len(w) - 2)
            w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        out.append(w)
    s = " ".join(out)
    if rng.random() < 0.25:
        s = s.upper() if rng.random() < 0.3 else s.lower()
    return s


def render_email(rec, rng):
    """Return (email_text, dict of which fields were actually stated)."""
    stated = {}
    lines = [rng.choice(OPENERS), ""]
    lines.append(f"Insured: {rec['business_name']}")

    # description is always present, sometimes buried in prose, often messy
    desc = rec["description"]
    if rng.random() < 0.40:
        desc = noisify(desc, rng)
    rec["description_as_written"] = desc
    if rng.random() < 0.35:
        lines.append(f"Operations: they run a {desc}.")
    else:
        lines.append(f"Business description: {desc}")
    stated["description"] = True

    if rng.random() > DROP_RATES["naics"]:
        lines.append(f"NAICS: {rec['naics']}")
        stated["naics"] = True

    if rng.random() > DROP_RATES["location"]:
        if rng.random() < 0.3:
            lines.append(f"Location: {rec['city']}, {rec['state']} (1 location)")
        else:
            lines.append(f"Address: {rng.randint(100, 8999)} "
                         f"{rng.choice(['Main', 'Oak', 'Lake', 'Third', 'Commerce'])} St, "
                         f"{rec['city']}, {rec['state']} {rng.randint(10000, 99999)}")
        stated["location"] = True

    if rng.random() > DROP_RATES["revenue"]:
        lbl = rng.choice(["Annual revenue", "Gross sales", "Revenue", "Annual sales", "Receipts"])
        lines.append(f"{lbl}: {fmt_money(rec['revenue'], rng)}")
        stated["revenue"] = True

    if rng.random() > DROP_RATES["payroll"]:
        lbl = rng.choice(["Annual payroll", "Payroll", "Total payroll"])
        lines.append(f"{lbl}: {fmt_money(rec['payroll'], rng)}")
        stated["payroll"] = True

    if rng.random() > DROP_RATES["limit_occurrence"]:
        lines.append(f"Requested limits: {fmt_limit(rec['limit_occurrence'], rng)} per occurrence"
                     f" / {fmt_limit(rec['limit_occurrence'] * 2, rng)} aggregate")
        stated["limit_occurrence"] = True

    if rng.random() > DROP_RATES["deductible"]:
        lines.append(f"Deductible: {fmt_limit(rec['deductible'], rng)}"
                     f"{rng.choice(['', ' per claim', ' property'])}")
        stated["deductible"] = True

    if rng.random() > DROP_RATES["loss_history"]:
        if rec["loss_count_5yr"] == 0:
            lines.append(rng.choice([
                "Loss history: none in the past 5 years, loss runs attached.",
                "Loss runs: clean, no claims 5 years.",
                "Prior losses: no claims reported.",
            ]))
        else:
            n = rec["loss_count_5yr"]
            lines.append(rng.choice([
                f"Loss history: {n} claim{'s' if n > 1 else ''} in the last 5 years, "
                f"{fmt_money(rec['loss_incurred_5yr'], rng, allow_fuzzy=False)} total incurred.",
                f"Losses: {n} reported claim{'s' if n > 1 else ''} past 5 yrs "
                f"totaling {fmt_money(rec['loss_incurred_5yr'], rng, allow_fuzzy=False)} incurred.",
                f"Prior claims - {n} in 5 years, incurred "
                f"{fmt_money(rec['loss_incurred_5yr'], rng, allow_fuzzy=False)}.",
            ]))
        stated["loss_history"] = True

    lines += ["", rng.choice(CLOSERS),
              f"{rng.choice(AGENTS)}", f"{rng.choice(BROKERS)}"]
    return "\n".join(lines), stated


def main(n=1000, out=None):
    rng = random.Random(SEED)
    out = Path(out or ROOT / "data")
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(1, n + 1):
        rec = make_record(rng, i)
        email, stated = render_email(rec, rng)
        rec["email_text"] = email
        rec["stated_fields"] = stated
        rows.append(rec)

    with open(out / "submissions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"wrote {len(rows)} submissions -> {out/'submissions.jsonl'}")
    return rows


if __name__ == "__main__":
    main()
