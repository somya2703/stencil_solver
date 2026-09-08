#!/usr/bin/env bash
# scripts/format.sh — run clang-format in-place on all C++/CUDA sources.
# Usage: ./scripts/format.sh [--check]
#   --check  dry-run; exits 1 if any file would be modified (used by CI)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLANG_FORMAT="${CLANG_FORMAT:-clang-format-15}"

if ! command -v "$CLANG_FORMAT" &>/dev/null; then
    echo "ERROR: $CLANG_FORMAT not found. Install with: apt install clang-format-15"
    exit 1
fi

FILES=$(find "$ROOT/src" "$ROOT/include" "$ROOT/tests" \
    \( -name "*.cpp" -o -name "*.cu" -o -name "*.hpp" -o -name "*.cuh" \) \
    | sort)

if [[ "${1:-}" == "--check" ]]; then
    echo "Checking format..."
    FAIL=0
    while IFS= read -r f; do
        if ! "$CLANG_FORMAT" --dry-run --Werror "$f" 2>/dev/null; then
            echo "  NEEDS FORMAT: ${f#"$ROOT/"}"
            FAIL=1
        fi
    done <<< "$FILES"
    if [[ $FAIL -eq 0 ]]; then
        echo "All files are correctly formatted."
    fi
    exit $FAIL
else
    echo "Formatting..."
    while IFS= read -r f; do
        "$CLANG_FORMAT" -i "$f"
        echo "  formatted: ${f#"$ROOT/"}"
    done <<< "$FILES"
    echo "Done."
fi
