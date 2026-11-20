#!/usr/bin/env python3
"""sparkDash's measurement set for one leg, through sparkDash's own API: decode and prefill.

  python3 -I docs/sd.py <label> <outdir> [decode|prefill|all]

Decode: Structured and prose, T=0.7, thinking on, top_k 20 (sparkDash's default), 400 tokens, at 1/4/6/8 streams.
Prefill: 2K, 4K, 8K, 16K, 32K and 128K of fresh content. SD (default http://127.0.0.1:5555) is the sparkDash
server; SD_UNIT is the sparkDash unit that probes this recipe's PORT (scripts/local.sh). Writes
<outdir>/sd-<label>-{structured,prose,prefill}.json (the whole job) and prints a short table.
"""
import json, os, sys, time, urllib.request

SD = os.environ.get("SD", "http://127.0.0.1:5555")
UNIT = os.environ.get("SD_UNIT") or sys.exit("SD_UNIT is not set (scripts/local.sh): the sparkDash unit to measure")
BASE = f"{SD}/api/sparks/{UNIT}/llm"
LEVELS = [1, 4, 6, 8]
PREFILL = [2048, 4096, 8192, 16384, 32768, 131072]


def call(method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def run(path, body, timeout_s):
    job = call("POST", path, body)
    jid = job.get("benchId") or job.get("id")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        time.sleep(3)
        j = call("GET", f"{path}/{jid}")
        if j.get("status") not in ("running", "queued", "pending", "starting"):
            return j
    sys.exit(f"timeout waiting for {path} {jid}")


def main():
    label, out = sys.argv[1], sys.argv[2]
    what = sys.argv[3] if len(sys.argv) > 3 else "all"
    if what in ("decode", "all"):
        for pt in ("structured", "prose"):
            j = run("/bench", {"concurrencies": LEVELS, "maxTokens": 400, "promptType": pt,
                               "temperature": 0.7, "thinking": True, "topK": 20}, 1800)
            json.dump(j, open(f"{out}/sd-{label}-{pt}.json", "w"), indent=1)
            print(f"{label} {pt} status={j.get('status')} topK={j.get('config', {}).get('topK')}")
            for r in j.get("results") or []:
                print(f"  x{r['concurrency']}: aggregate {r.get('aggregateDecodeTps')} tok/s  "
                      f"mean {r.get('meanDecodeTps')}  ttft {r.get('meanTtftMs')} ms  failed {r.get('streamsFailed')}")
    if what in ("prefill", "all"):
        j = run("/prefill-bench", {"contextSizes": PREFILL}, 3600)
        json.dump(j, open(f"{out}/sd-{label}-prefill.json", "w"), indent=1)
        print(f"{label} prefill status={j.get('status')}")
        for r in j.get("results") or []:
            print(f"  {r.get('targetTokens')}: {r.get('prefillTps')} tok/s  cached {r.get('cachedTokens')}  "
                  f"lowConf {r.get('lowConfidence')}  err {r.get('error')}")


main()
