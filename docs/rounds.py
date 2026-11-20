#!/usr/bin/env python3
"""A round's anatomy under sparkDash's decode protocol, for either engine (this recipe's or the Python-engine recipe's).

  python3 -I docs/rounds.py <label> <outdir>

sparkDash's two decode prompts (count 1..200, the hash-map prose), T=0.7, top_p 1, top_k 20, thinking on, 400 tokens,
at 1, 4 and 8 streams (LOADS), TRIALS waves each after one warm-up. Not streamed, so each reply carries the engine's
own counters: rounds, drafted and accepted tokens, decode seconds. Per load it prints tokens a round, the round's
time, drafts a round, acceptance and the total tok/s, and writes every reply's counters to <outdir>/rounds-<label>.json.
U is the server (scripts/local.sh's PORT). Both engines answer the same request, so their rows compare directly.
"""
import json, os, statistics, sys, threading, time, urllib.request

U = os.environ.get("U", "http://127.0.0.1:8888")
LOADS = [int(x) for x in os.environ.get("LOADS", "1,4,8").split(",")]
TRIALS = int(os.environ.get("TRIALS", "2"))
MAXTOK = int(os.environ.get("MAXTOK", "400"))
PROMPTS = {
    "structured": "Count from 1 to 200. Output only the numbers, separated by spaces. No other text.",
    "prose": "Write a detailed step-by-step explanation of how a hash map works, "
             "including collision handling, resizing, and time complexity. Be thorough.",
}


def model():
    return os.environ.get("M") or json.load(urllib.request.urlopen(U + "/v1/models", timeout=30))["data"][0]["id"]


def ask(m, prompt):
    body = {"model": m, "messages": [{"role": "user", "content": prompt}], "max_tokens": MAXTOK, "temperature": 0.7,
            "top_p": 1, "top_k": 20, "chat_template_kwargs": {"enable_thinking": True}}
    req = urllib.request.Request(U + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    tf, sp = r.get("tensorfold") or {}, r.get("speculative") or {}
    n = (r.get("usage") or {}).get("completion_tokens")
    rounds = sp.get("rounds") or tf.get("rounds")
    drafted = sp.get("drafted") if sp.get("drafted") is not None else tf.get("drafted")
    accepted = sp.get("accepted") if sp.get("accepted") is not None else tf.get("accepted")
    decode_s = tf.get("decode_s")
    if decode_s is None and tf.get("seconds") is not None:
        decode_s = tf["seconds"] - (tf.get("prefill_seconds") or 0)
    return {"wall": time.time() - t0, "completion": n, "rounds": rounds, "drafted": drafted, "accepted": accepted,
            "decode_s": decode_s, "tensorfold": {k: v for k, v in tf.items() if k != "token_sha"}, "speculative": sp}


def wave(m, prompt, n):
    out = [None] * n
    def go(i):
        try:
            out[i] = ask(m, prompt)
        except Exception as e:  # a failed stream is reported, not hidden
            out[i] = {"error": str(e)}
    ts = [threading.Thread(target=go, args=(i,)) for i in range(n)]
    [t.start() for t in ts]; [t.join() for t in ts]
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else float("nan")


def main():
    label, outdir = sys.argv[1], sys.argv[2]
    m = model()
    ask(m, PROMPTS["structured"])  # warm-up
    res = {}
    for kind, prompt in PROMPTS.items():
        for n in LOADS:
            waves = [wave(m, prompt, n) for _ in range(TRIALS)]
            rows = []
            for w in waves:
                ok = [o for o in w if o and "error" not in o and o.get("rounds")]
                if len(ok) < n:
                    print(f"{label} {kind} x{n}: {n - len(ok)} of {n} streams failed or carry no counters")
                if not ok:
                    continue
                rows.append({
                    "tps": sum(o["completion"] / (o["decode_s"] or o["wall"]) for o in ok),
                    "tok_round": mean([o["completion"] / o["rounds"] for o in ok]),
                    "round_ms": mean([(o["decode_s"] or o["wall"]) / o["rounds"] * 1000 for o in ok]),
                    "drafted_round": mean([o["drafted"] / o["rounds"] for o in ok if o["drafted"] is not None]),
                    "accept": mean([o["accepted"] / o["drafted"] for o in ok if o["drafted"]]),
                })
            res[f"{kind}-x{n}"] = {"summary": rows, "streams": waves}
            if rows:
                print(f"{label} {kind:10s} x{n}: {mean([r['tps'] for r in rows]):6.1f} tok/s  "
                      f"tok/round {mean([r['tok_round'] for r in rows]):.2f}  "
                      f"round {mean([r['round_ms'] for r in rows]):6.1f} ms  "
                      f"drafted/round {mean([r['drafted_round'] for r in rows]):.2f}  "
                      f"accept {mean([r['accept'] for r in rows]):.2f}", flush=True)
    json.dump(res, open(os.path.join(outdir, f"rounds-{label}.json"), "w"), indent=1)


main()
