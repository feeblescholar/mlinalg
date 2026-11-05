#!/usr/bin/env bash
# Lints the library through the test translation units, so its templates are instantiated.
# Only findings in include/ count; a translation unit that fails to parse is always fatal, because
# clang-tidy would otherwise report a clean result for code it never analysed.
#
# Usage: clang-tidy.sh <build-dir> [--report-only]
# CLANG_TIDY_EXTRA_ARG, if set, is passed to the compiler as one extra argument.
set -uo pipefail

build_dir=$1
report_only=${2:-}

root=$(pwd -P)
# The tests include "../include/...", so headers can be reported as test/../include/.
header_filter="^$root/(test/\.\./)?include/"

if [[ $(uname -s) == Darwin ]]; then
    SDKROOT=$(xcrun --show-sdk-path)
    export SDKROOT
fi

if [[ ! -f $build_dir/compile_commands.json ]]; then
    echo "No compile_commands.json in $build_dir; configure a preset first." >&2
    exit 2
fi

log=$(mktemp)
trap 'rm -f "$log"' EXIT

# xargs exits 123 when clang-tidy merely reports errors, and above that when it could not run at all.
find test -maxdepth 1 -name '*.cpp' -print0 |
    xargs -0 -n1 -P"$(getconf _NPROCESSORS_ONLN)" clang-tidy -p "$build_dir" --quiet \
        --header-filter="$header_filter" ${CLANG_TIDY_EXTRA_ARG:+"--extra-arg=$CLANG_TIDY_EXTRA_ARG"} \
        >"$log" 2>/dev/null
if ((PIPESTATUS[1] > 123)); then
    echo "clang-tidy could not be run." >&2
    exit 2
fi

if grep -E 'error: .*\[clang-diagnostic-error\]' "$log"; then
    echo "clang-tidy could not parse the code, see the errors above." >&2
    exit 2
fi

findings=$(grep -E "${header_filter}[^:]+:[0-9]+:[0-9]+: (warning|error):" "$log" |
    sed -E "s|^$root/(test/\.\./)?||; s|[^/]+/\.\./||" | sort -u)

if [[ -z $findings ]]; then
    echo "clang-tidy: no findings in include/."
    exit 0
fi

echo "$findings"
echo "clang-tidy: $(wc -l <<<"$findings" | tr -d ' ') finding(s) in include/."

if [[ -n ${GITHUB_ACTIONS:-} ]]; then
    sed -E 's/^([^:]+):([0-9]+):([0-9]+): (warning|error): (.*)$/::warning file=\1,line=\2,col=\3::\5/' <<<"$findings"
fi

[[ $report_only == --report-only ]]
