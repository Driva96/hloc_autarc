#!/usr/bin/env bash
set -euo pipefail

# Targets: COLMAP + common OpenMVS tools if present
TARGETS=()
for f in \
  /usr/local/bin/colmap \
  /usr/local/bin/InterfaceCOLMAP \
  /usr/local/bin/InterfaceVisualSFM \
  /usr/local/bin/DensifyPointCloud \
  /usr/local/bin/ReconstructMesh \
  /usr/local/bin/RefineMesh \
  /usr/local/bin/TextureMesh \
  /usr/local/bin/TransformScene \
  /usr/local/bin/ExportData \
  /usr/local/bin/Viewer
do
  [[ -x "$f" ]] && TARGETS+=("$f")
done

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  echo "No COLMAP/OpenMVS targets found in /usr/local/bin"
  exit 1
fi

echo "== Targets =="
printf ' - %s\n' "${TARGETS[@]}"
echo

echo "== Missing libraries (ldd: not found) =="
missing=0
for t in "${TARGETS[@]}"; do
  out="$(ldd "$t" 2>/dev/null | grep 'not found' || true)"
  if [[ -n "$out" ]]; then
    missing=1
    echo "## $t"
    echo "$out"
    echo
  fi
done
[[ $missing -eq 0 ]] && echo "None"
echo

echo "== Resolved shared objects used by targets =="
: > /tmp/needed-so.txt
for t in "${TARGETS[@]}"; do
  ldd "$t" 2>/dev/null | awk '/=> \//{print $3}' >> /tmp/needed-so.txt
done
sort -u /tmp/needed-so.txt | tee /tmp/needed-so-uniq.txt
echo

echo "== Debian packages providing those .so files =="
xargs -a /tmp/needed-so-uniq.txt -r dpkg -S 2>/dev/null \
  | cut -d: -f1 | sort -u | tee /tmp/needed-pkgs.txt
echo

if [[ -n "${CANDIDATES:-}" ]]; then
  echo "== Candidate packages not referenced by these binaries (possible removals) =="
  for p in $CANDIDATES; do
    grep -qx "$p" /tmp/needed-pkgs.txt || echo "possibly unused: $p"
  done
fi