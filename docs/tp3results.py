#!/usr/bin/env python3
"""Tables of the TP=3 epic's legs from the files docs/tp3leg.sh writes (epic task 03's measurement set).

  python3 -I docs/tp3results.py [leg ...]          (default: every leg under docs/speed/tp3, then T2 from docs/speed)

Markdown on stdout: decode (docs/dec.py, greedy) with each leg's median and spread; sparkDash Structured and prose
(aggregate tok/s) and prefill; the contract against docs/speed/tp3/contract-baseline.json; quality and the seeded
probe. Deltas are against T3 (TP=3, every switch off). A delta counts only above 2x the larger trial spread of the
two legs (the noise rule of epic/README.md).
"""
import glob, json, os, statistics, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speed")
TP3 = os.path.join(ROOT, "tp3")
REF = "T3"


def where(leg):
    return ROOT if leg == "T2" or leg.startswith("T2-") else TP3


def load(leg, kind, ext="json"):
    p = os.path.join(where(leg), f"{kind}-{leg}.{ext}")
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def dec(leg):
    d = load(leg, "dec") or {}
    out = {}
    for n, v in d.items():
        t = [x["total_tps"] for x in v.get("trials", []) if x.get("total_tps")]
        if t:
            med = statistics.median(t)
            out[int(n)] = (med, (max(t) - min(t)) / med * 100 if len(t) > 1 else 0.0)
    return out


def sd(leg, kind):
    p = os.path.join(where(leg), f"sd-{leg}-{kind}.json")
    try:
        j = json.load(open(p))
    except (OSError, ValueError):
        return {}
    if kind == "prefill":
        return {r["targetTokens"]: r.get("prefillTps") for r in j.get("results") or []}
    return {r["concurrency"]: r.get("aggregateDecodeTps") for r in j.get("results") or []}


def contract(leg):
    base = load("baseline", "contract") if where(leg) == TP3 else None
    cur = load(leg, "contract")
    if not cur:
        return "-"
    if any(v.get("token_sha") is None for v in cur.values()):
        return "errors"
    if base is None:
        return "-"
    bad = [k for k in base if base[k].get("token_sha") != cur.get(k, {}).get("token_sha")]
    return "identical" if not bad else "differs: " + ",".join(bad)


def pct(a, b):
    return f"{(a / b - 1) * 100:+.1f}%" if a and b else ""


def main():
    legs = sys.argv[1:] or (["T2"] + sorted({os.path.basename(p)[4:-5] for p in glob.glob(os.path.join(TP3, "dec-*.json"))
                                             if "-prof" not in p}, key=lambda s: (s != REF, s)))
    ref_dec = dec(REF)
    print("### Decode, docs/dec.py (greedy, thinking off): median tok/s of 3 trials, spread, delta vs T3\n")
    print("| Leg | 1 | 2 | 4 | 8 |\n| --- | ---: | ---: | ---: | ---: |")
    for leg in legs:
        d = dec(leg)
        cells = []
        for n in (1, 2, 4, 8):
            if n not in d:
                cells.append("-"); continue
            med, spread = d[n]
            delta = ""
            if leg != REF and n in ref_dec:
                rmed, rsp = ref_dec[n]
                floor = 2 * max(spread, rsp)
                delta = f" ({pct(med, rmed)}{'' if abs(med / rmed - 1) * 100 > floor else ', noise'})"
            cells.append(f"{med:.1f} ±{spread:.1f}%{delta}")
        print(f"| {leg} | " + " | ".join(cells) + " |")
    for kind, title in (("structured", "sparkDash Structured (count 1..200)"), ("prose", "sparkDash prose")):
        print(f"\n### {title}: aggregate tok/s, T=0.7, thinking on, top_k 20, 400 tokens\n")
        print("| Leg | 1 | 4 | 6 | 8 |\n| --- | ---: | ---: | ---: | ---: |")
        ref = sd(REF, kind)
        for leg in legs + ["P3"]:
            s = sd(leg, kind)  # P3 (the Python engine as it serves) is measured the same way, into the same directory
            if not s:
                continue
            print(f"| {leg} | " + " | ".join(
                f"{s.get(n) or '-'}" + (f" ({pct(s.get(n), ref.get(n))})" if leg not in (REF,) and s.get(n) and ref.get(n) else "")
                for n in (1, 4, 6, 8)) + " |")
    print("\n### sparkDash prefill: tok/s, fresh content (cached 0)\n")
    legs = legs + ["P3"]
    sizes = (2048, 4096, 8192, 16384, 32768, 131072)
    print("| Leg | " + " | ".join(f"{s // 1024}K" for s in sizes) + " |\n| --- |" + " ---: |" * len(sizes))
    ref = sd(REF, "prefill")
    for leg in legs:
        s = sd(leg, "prefill")
        if s:
            print(f"| {leg} | " + " | ".join(f"{s.get(n):.0f}" + (f" ({pct(s.get(n), ref.get(n))})" if leg != REF and ref.get(n) else "")
                                             if s.get(n) else "-" for n in sizes) + " |")
    print("\n### Contract, quality, seeded\n")
    print("| Leg | Contract vs T3 | NLL | Answers sha | Seeded repeatable | Seeded concurrent = solo | Seeded solo sha |")
    print("| --- | --- | ---: | --- | --- | --- | --- |")
    for leg in [l for l in legs if l != "P3"]:
        q = load(leg, "q") or {}
        s = load(leg, "seeded") or {}
        solo = ",".join(s.get("solo") or [])[:35]
        print(f"| {leg} | {contract(leg)} | {q.get('nll', '-') if not q else round(q['nll'], 4)} | {q.get('answers_sha', '-')} | "
              f"{s.get('repeatable', '-')} | {s.get('concurrent_equals_solo', '-')} | {solo or '-'} |")


main()
