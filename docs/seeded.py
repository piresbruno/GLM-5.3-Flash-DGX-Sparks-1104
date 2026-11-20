#!/usr/bin/env python3
"""Fixed-seed sampled replies, for a sampler change's gate (as TF_GLM_SAMPLE_ROOT's, epic task 08): equal off and on.

  python3 -I docs/seeded.py <label> <outdir>

Four prompts at T=0.7, top_k 20, seed 1234, thinking off, 160 tokens: once one at a time, then all four at once and
eight at once (each prompt twice), so the sampler's rows go through lone and shared verify windows. Every reply's
token sha is written to <outdir>/seeded-<label>.json; a second call on the same boot should match itself (the
repeatability check) before an off/on comparison means anything. U is the server (scripts/local.sh's PORT).
"""
import hashlib, json, os, sys, threading, urllib.request

U = os.environ.get("U", "http://127.0.0.1:8888")
M = os.environ.get("M", "GLM-5.3-Flash")
PROMPTS = [
    "Write four lines about a lighthouse keeper's last night on duty.",
    "List six unusual uses for a paperclip, one per line.",
    "Describe the smell of rain on hot asphalt in two sentences.",
    "Invent a name and a one-line motto for a bakery on the moon.",
]


def ask(p):
    body = {"model": M, "messages": [{"role": "user", "content": p}], "max_tokens": 160, "temperature": 0.7,
            "top_k": 20, "seed": 1234, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(U + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    text = r["choices"][0]["message"].get("content") or ""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def at_once(ps):
    out = [None] * len(ps)
    def go(i):
        out[i] = ask(ps[i])
    ts = [threading.Thread(target=go, args=(i,)) for i in range(len(ps))]
    [t.start() for t in ts]; [t.join() for t in ts]
    return out


def main():
    label, outdir = sys.argv[1], sys.argv[2]
    res = {"solo": [ask(p) for p in PROMPTS], "x4": at_once(PROMPTS), "x8": at_once(PROMPTS + PROMPTS)}
    res["solo_again"] = [ask(p) for p in PROMPTS]
    res["repeatable"] = res["solo"] == res["solo_again"]
    res["concurrent_equals_solo"] = res["x4"] == res["solo"] and res["x8"] == res["solo"] + res["solo"]
    json.dump(res, open(os.path.join(outdir, f"seeded-{label}.json"), "w"), indent=1)
    print(f"seeded {label}: repeatable {res['repeatable']}, concurrent = solo {res['concurrent_equals_solo']}")
    print(f"  solo {res['solo']}")


main()
