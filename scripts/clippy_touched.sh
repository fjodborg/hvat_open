#!/usr/bin/env bash
set -euo pipefail

BASE_SHA="${1:-}"

fallback_base_sha() {
    if git rev-parse --verify HEAD~1 >/dev/null 2>&1; then
        git rev-parse HEAD~1
    else
        git rev-parse HEAD
    fi
}

if [[ -n "$BASE_SHA" && "$BASE_SHA" =~ ^0+$ ]]; then
    BASE_SHA="$(git rev-list --max-parents=0 HEAD | tail -n 1)"
fi

if [[ -z "$BASE_SHA" ]]; then
    BASE_SHA="$(fallback_base_sha)"
fi

if ! git rev-parse --verify "${BASE_SHA}^{commit}" >/dev/null 2>&1; then
    echo "Base SHA '$BASE_SHA' is not available locally; falling back to HEAD~1/HEAD."
    BASE_SHA="$(fallback_base_sha)"
fi

mapfile -t changed_rust_files < <(git diff --name-only "$BASE_SHA"...HEAD -- '*.rs' | sed '/^$/d')

if [[ ${#changed_rust_files[@]} -eq 0 ]]; then
    echo "No changed Rust files between $BASE_SHA and HEAD."
    exit 0
fi

echo "Running clippy and checking touched Rust files:"
printf '  - %s\n' "${changed_rust_files[@]}"

changed_lines_out="$(mktemp)"
clippy_out="$(mktemp)"
trap 'rm -f "$clippy_out" "$changed_lines_out"' EXIT

git diff -U0 "$BASE_SHA"...HEAD -- '*.rs' | awk '
    /^\+\+\+ b\// {
        file = substr($0, 7);
        next;
    }
    /^@@/ {
        if (file == "") {
            next;
        }
        if (match($0, /\+([0-9]+)(,([0-9]+))?/, m)) {
            start = m[1] + 0;
            len = (m[3] == "" ? 1 : m[3] + 0);
            if (len == 0) {
                next;
            }
            for (i = 0; i < len; i++) {
                print file ":" (start + i);
            }
        }
    }
' >"$changed_lines_out"

if [[ ! -s "$changed_lines_out" ]]; then
    echo "No changed Rust line ranges detected."
    exit 0
fi

if ! cargo clippy --workspace --all-targets --all-features --message-format=short >"$clippy_out" 2>&1; then
    cat "$clippy_out"
    exit 1
fi

violations="$(awk -F: '
    NR == FNR {
        changed[$0] = 1;
        next;
    }
    {
        file = $1;
        line = $2;
        key = file ":" line;
        if (changed[key] && $0 ~ /: (warning|error):/) {
            print $0;
        }
    }
' "$changed_lines_out" "$clippy_out")"

if [[ -n "$violations" ]]; then
    echo "Clippy diagnostics found on changed Rust lines:"
    echo "$violations"
    exit 1
fi

echo "No clippy diagnostics on changed Rust lines."
