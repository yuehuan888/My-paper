#!/usr/bin/env bash
# Fetch the curated subset of Yuan1z0825/nature-skills (Apache-2.0).
#
# Why not `git clone` / tarball:
#   * the accelerator drops large transfers (git: "early EOF"; codeload:
#     truncated gzip). codeload serves no Content-Length, so `curl -C -`
#     cannot resume -- it silently produced a 9.5 MB truncated file.
#   * raw.githubusercontent.com works fine through the same accelerator.
# So: list the tree once via the API, then pull only the files we want.
#
# Why a CURATED subset rather than all 20 skills:
#   the repo targets Nature-family journals. Their narrative structure
#   (broad-audience framing, Results/Discussion split) does not map onto a
#   4-6 page ICIP/EUSIPCO/ICME paper. Only the venue-agnostic skills are
#   installed: figure, statistics, reference verification, polishing,
#   reviewer. See SKILLS below.

set -u
REPO="Yuan1z0825/nature-skills"
BRANCH="main"
DEST="${1:-$HOME/.claude/skills}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Venue-agnostic subset
SKILLS=(nature-figure nature-statistics nature-ref-verifier nature-polishing nature-reviewer)

echo "listing tree for $REPO ..."
curl -sL --ssl-revoke-best-effort --max-time 120 \
  "https://api.github.com/repos/$REPO/git/trees/$BRANCH?recursive=1" -o "$TMP/tree.json"

python - "$TMP/tree.json" "$TMP/paths.txt" "${SKILLS[@]}" <<'PY'
import json, sys
tree, out, wanted = sys.argv[1], sys.argv[2], sys.argv[3:]
d = json.load(open(tree, encoding="utf-8"))
if "tree" not in d:
    sys.exit("tree API returned: %s" % str(d)[:300])
keep = []
for e in d["tree"]:
    if e.get("type") != "blob":
        continue
    p = e["path"]
    if not p.startswith("skills/"):
        continue
    parts = p.split("/")
    if len(parts) >= 3 and parts[1] in wanted:
        keep.append(p)
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(keep))
print("matched %d files across %d skills" % (len(keep), len(wanted)))
PY

[ -s "$TMP/paths.txt" ] || { echo "FATAL: no files matched -- aborting"; exit 1; }

n=0; fail=0
while IFS= read -r p; do
    rel="${p#skills/}"                 # e.g. nature-figure/SKILL.md
    out="$DEST/$rel"
    mkdir -p "$(dirname "$out")"
    if [ -s "$out" ]; then              # resumable: skip already-fetched
        n=$((n+1)); continue
    fi
    ok=0
    for try in 1 2 3; do
        if curl -sL --ssl-revoke-best-effort --max-time 90 \
             "https://raw.githubusercontent.com/$REPO/$BRANCH/$p" -o "$out" \
           && [ -s "$out" ]; then
            ok=1; break
        fi
        sleep 2
    done
    if [ "$ok" -eq 1 ]; then n=$((n+1)); else fail=$((fail+1)); echo "  FAILED: $p"; fi
done < "$TMP/paths.txt"

echo "fetched $n files, $fail failures -> $DEST"
echo "installed skills:"
for s in "${SKILLS[@]}"; do
    if [ -f "$DEST/$s/SKILL.md" ]; then
        printf "  OK   %-24s %s files\n" "$s" "$(find "$DEST/$s" -type f | wc -l)"
    else
        printf "  MISS %s (SKILL.md absent)\n" "$s"
    fi
done
