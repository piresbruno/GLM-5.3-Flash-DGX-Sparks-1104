#!/usr/bin/env python3
"""Remove a kit's private details from measurement files before they are committed (the fork is public).

  python3 -I docs/scrub.py <file|dir> ...        (rewrites in place; prints what it changed; --check only reports)

Replaced: ssh targets (user@address) by <worker>, private IPv4 addresses (10/8, 172.16/12, 192.168/16) by <addr>, this
login's home directory by ~, sparkDash's "sparkId" by <unit>, host names given in SCRUB_HOSTS (comma-separated, e.g.
the kit's three) by <host>, and this login's user name by <user>. Run it over docs/speed/ after a window's legs.
"""
import getpass, os, re, sys

HOME = os.path.expanduser("~")
HOSTS = [h for h in os.environ.get("SCRUB_HOSTS", "").split(",") if h]
PATTERNS = [
    (re.compile(r"[A-Za-z0-9_.-]+@(?:\d{1,3}\.){3}\d{1,3}"), "<worker>"),
    (re.compile(r"\b(?:10|192\.168|172\.(?:1[6-9]|2\d|3[01]))(?:\.\d{1,3}){2,3}\b"), "<addr>"),
    (re.compile(re.escape(HOME) + r"(?=/|\b)"), "~"),
    (re.compile(r'"sparkId": "[^"]*"'), '"sparkId": "<unit>"'),
] + [(re.compile(r"\b" + re.escape(h) + r"\b"), "<host>") for h in HOSTS] + [
    (re.compile(r"\b" + re.escape(getpass.getuser()) + r"\b"), "<user>"),
]


def files(args):
    for a in args:
        if os.path.isdir(a):
            for root, _, names in os.walk(a):
                for n in names:
                    yield os.path.join(root, n)
        elif os.path.isfile(a):
            yield a


def main():
    check = "--check" in sys.argv
    hits = 0
    for f in files([a for a in sys.argv[1:] if a != "--check"]):
        s = open(f, errors="surrogateescape").read()
        t = s
        for p, r in PATTERNS:
            t = p.sub(r, t)
        if t != s:
            hits += 1
            print(("would scrub " if check else "scrubbed ") + f)
            if not check:
                open(f, "w", errors="surrogateescape").write(t)
    print(f"{hits} file(s) {'to scrub' if check else 'scrubbed'}")
    return 1 if check and hits else 0


sys.exit(main())
