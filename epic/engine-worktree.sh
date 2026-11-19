#!/usr/bin/env bash
# Build the engine tree the epic's patches are written in: TensorFold at the recipe's pin plus every
# patches/engine/*.patch, one commit each, tagged recipe-<last patch number>. Task 04.
#
# Usage: epic/engine-worktree.sh <dir>                 # clones TF_REPO into <dir> (full history: the backports need main)
#        TF_REPO=/path/to/clone epic/engine-worktree.sh <dir>
# Mail-format patches go in with git am (their own author and message); plain diffs with git apply, as the Dockerfile
# does, committed under their file name.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
dir=${1:?usage: epic/engine-worktree.sh <dir>}
repo=${TF_REPO:-https://github.com/ashhart/TensorFold.git}
pin=$(sed -n 's/^TF_REF="\${TF_REF:-\([0-9a-f]\{40\}\)}".*/\1/p' "$here/scripts/config.sh")
[[ -n "$pin" ]] || { echo "no TF_REF pin found in scripts/config.sh" >&2; exit 1; }
[[ ! -e "$dir" ]] || { echo "$dir exists" >&2; exit 1; }

git clone -q "$repo" "$dir"
cd "$dir"
git -c advice.detachedHead=false checkout -q -b tp3-perf "$pin"
n=0 plain=0 last=
for p in "$here"/patches/engine/*.patch; do
  b=$(basename "$p" .patch)
  if head -1 "$p" | grep -q '^From [0-9a-f]\{40\} '; then
    git am -q "$p" || { git am --abort; echo "FAIL $b" >&2; exit 1; }
  else
    git apply --index "$p" || { echo "FAIL $b" >&2; exit 1; }
    git -c user.name=recipe -c user.email=recipe@localhost commit -q -m "recipe patch $b"
    plain=$((plain + 1))
  fi
  n=$((n + 1)) last=${b%%-*}
done
git tag -f "recipe-$last" >/dev/null
echo "$dir: $pin + $n patches ($plain plain diffs), branch tp3-perf, tag recipe-$last"
