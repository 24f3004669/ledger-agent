import os, re, json, calendar, urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

HERE = os.path.dirname(os.path.abspath(__file__))
IST = timezone(timedelta(hours=5, minutes=30))
TS = re.compile(r"(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d)(?:\.(\d+))?(Z|[+-]\d\d:\d\d)$")


def parse_ts(s):
    y, mo, d, h, mi, sec, frac, tz = TS.match(s.strip()).groups()
    if tz == "Z":
        off = timezone.utc
    else:
        sign = 1 if tz[0] == "+" else -1
        off = timezone(sign * timedelta(hours=int(tz[1:3]), minutes=int(tz[4:6])))
    micro = int((frac or "0")[:6].ljust(6, "0"))
    return datetime(int(y), int(mo), int(d), int(h), int(mi), int(sec), micro, tzinfo=off)


# ---------- load and clean the data once ----------
RATES = json.load(open(os.path.join(HERE, "raw_rates.txt"), encoding="utf-8"))["usd_per_unit"]
RAW = [json.loads(l) for l in open(os.path.join(HERE, "raw_export.txt"), encoding="utf-8") if l.strip()]

latest = {}
for r in RAW:
    t = parse_ts(r["updated_at"])
    cur = latest.get(r["id"])
    if cur is None or t >= cur[0]:
        latest[r["id"]] = (t, r)

ORDERS = []
for _, r in latest.values():
    created = parse_ts(r["created_at"]).astimezone(IST)
    ORDERS.append({
        "id": r["id"], "customer": r["customer"], "region": r["region"],
        "product": r["product"], "qty": int(r["qty"]), "currency": r["currency"],
        "status": r["status"],
        "usd": Decimal(str(r["amount"])) * Decimal(str(RATES[r["currency"]])),
        "day": created.strftime("%Y-%m-%d"), "month": created.strftime("%Y-%m"),
    })

REGIONS = sorted({o["region"] for o in ORDERS})
PRODUCTS = sorted({o["product"] for o in ORDERS})
CANON = {v.lower(): v for v in REGIONS + PRODUCTS}
MONTHS = sorted({o["month"] for o in ORDERS})

# ---------- LLM: question -> plan ----------
SYSTEM = f"""You convert a question about an order ledger into a JSON query plan. Reply with JSON only.

Data: one row per order with fields region, product, customer (like C1044), qty, currency, status, USD value, and an IST business date.
Valid regions: {REGIONS}
Valid products: {PRODUCTS}
Data covers {MONTHS[0]} to {MONTHS[-1]}. Dates are in India time.
Statuses: "paid" = revenue/sales; "refunded" = refunds; "void" = ignore.

Plan format:
{{
 "metric": "sum_usd" | "count" | "sum_qty" | "avg_usd" | "distinct_customers",
 "statuses": ["paid"],
 "filters": {{"region": [], "product": [], "customer": []}},
 "date_from": "YYYY-MM-DD" or null,
 "date_to": "YYYY-MM-DD" or null,
 "group_by": null | "region" | "product" | "customer" | "month",
 "rank": null | {{"order": "desc" | "asc", "n": 1, "return": "key" | "value"}}
}}

Rules:
- Filters are always lists, even for one value.
- "revenue", "sales", "orders", "units sold" -> statuses ["paid"]. "refunds"/"refunded" -> statuses ["refunded"]. Only use several statuses if the question says so.
- Money questions use metric sum_usd. "How many orders" -> count. "How many units" -> sum_qty. "Average order value" -> avg_usd. "How many customers" or "distinct customers" -> distinct_customers.
- date_from and date_to are INCLUSIVE. A month like March 2026 -> 2026-03-01 to 2026-03-31. A quarter or half-year is a range of whole months.
- "Which X had the highest/most/top ..." -> group_by X, rank {{"order":"desc","n":1,"return":"key"}}. "Top 3 ..." -> n 3. "Lowest" -> order "asc". If the question asks for the amount of the top item, use "return":"value".
- Leave rank null for a plain total, or when all groups should be listed.
- Use filters only for things the question names. Use null/empty for the rest."""


def llm_plan(question):
    token = os.environ.get("AIPIPE_TOKEN", "")
    base = os.environ.get("LLM_URL", "https://aipipe.org/openai/v1/chat/completions")
    model = os.environ.get("LLM_MODEL", "gpt-4.1-mini")
    body = json.dumps({
        "model": model, "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": question}],
    }).encode()
    req = urllib.request.Request(base, body, {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "curl/8.5.0",
        "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=6) as r:
        data = json.loads(r.read().decode())
    return json.loads(data["choices"][0]["message"]["content"])


# ---------- backup planner (works without the LLM) ----------
MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july",
               "august", "september", "october", "november", "december"]


def rule_plan(q):
    ql = q.lower()
    plan = {"metric": "sum_usd", "statuses": ["paid"],
            "filters": {"region": [], "product": [], "customer": []},
            "date_from": None, "date_to": None, "group_by": None, "rank": None}
    for p in PRODUCTS:
        if p.lower() in ql:
            plan["filters"]["product"].append(p)
    for r in REGIONS:
        if r.lower() in ql:
            plan["filters"]["region"].append(r)
    if "refund" in ql:
        plan["statuses"] = ["refunded"]
    if "distinct customer" in ql or "unique customer" in ql or "how many customers" in ql:
        plan["metric"] = "distinct_customers"
    elif "how many orders" in ql or "number of orders" in ql:
        plan["metric"] = "count"
    elif "units" in ql:
        plan["metric"] = "sum_qty"
    elif "average" in ql:
        plan["metric"] = "avg_usd"
    m = re.search("(" + "|".join(MONTH_NAMES) + r")\s+(\d{4})", ql)
    if m:
        mo, yr = MONTH_NAMES.index(m.group(1)) + 1, int(m.group(2))
        plan["date_from"] = f"{yr}-{mo:02d}-01"
        plan["date_to"] = f"{yr}-{mo:02d}-{calendar.monthrange(yr, mo)[1]:02d}"
    return plan


# ---------- run the plan exactly ----------
def money(x):
    return float(Decimal(x).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def clean_list(vals, kind):
    if isinstance(vals, str):
        vals = [vals]
    out = []
    for v in vals or []:
        v = str(v).strip()
        out.append(CANON.get(v.lower(), v) if kind in ("region", "product") else v.upper())
    return out


def measure(rows, metric):
    if metric == "count":
        return len(rows)
    if metric == "sum_qty":
        return sum(o["qty"] for o in rows)
    if metric == "distinct_customers":
        return len({o["customer"] for o in rows})
    total = sum((o["usd"] for o in rows), Decimal(0))
    if metric == "avg_usd":
        return money(total / len(rows)) if rows else 0.0
    return money(total)


def run(plan):
    metric = plan.get("metric") or "sum_usd"
    statuses = plan.get("statuses") or ["paid"]
    if isinstance(statuses, str):
        statuses = [statuses]
    f = plan.get("filters") or {}
    flt = {k: clean_list(f.get(k), k) for k in ("region", "product", "customer")}
    d0, d1 = plan.get("date_from"), plan.get("date_to")

    rows = []
    for o in ORDERS:
        if o["status"] not in statuses:
            continue
        if any(flt[k] and o[k] not in flt[k] for k in flt):
            continue
        if d0 and o["day"] < d0:
            continue
        if d1 and o["day"] > d1:
            continue
        rows.append(o)

    gb = plan.get("group_by")
    if not gb:
        return measure(rows, metric)

    groups = defaultdict(list)
    for o in rows:
        groups[o[gb]].append(o)
    table = {k: measure(v, metric) for k, v in groups.items()}

    rank = plan.get("rank")
    if not rank:
        return dict(sorted(table.items()))
    desc = (rank.get("order") or "desc") == "desc"
    n = int(rank.get("n") or 1)
    items = sorted(table.items(), key=lambda kv: ((-kv[1] if desc else kv[1]), kv[0]))[:n]
    if rank.get("return") == "value":
        vals = [v for _, v in items]
        return vals[0] if n == 1 else vals
    keys = [k for k, _ in items]
    return keys[0] if n == 1 and keys else keys


# ---------- web service ----------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def home():
    return {"status": "ok", "service": "ledger-agent", "orders": len(ORDERS),
            "usage": "POST {\"question\": \"...\"}"}


@app.post("/")
@app.post("/{full_path:path}")
async def ask(request: Request, full_path: str = "", debug: int = 0):
    try:
        question = (await request.json()).get("question", "")
    except Exception:
        return {"answer": None, "error": "send JSON like {\"question\": \"...\"}"}

    plan, answer, err = None, None, ""
    for _ in range(2):
        try:
            plan = llm_plan(question)
            answer = run(plan)
            err = ""
            break
        except Exception as e:
            err = str(e)[:200]
            answer = None
    if answer is None:
        try:
            plan = rule_plan(question)
            answer = run(plan)
        except Exception as e:
            return {"answer": None, "error": err + " | " + str(e)[:150]}

    out = {"answer": answer}
    if debug:
        out["plan"] = plan
        out["llm_error"] = err
    return out