#!/usr/bin/env bash
# Differential fuzz across ALL backends (soft / sse2 / avx2 / avx512) in
# parallel.
#
# The runtime dispatcher normally picks one backend per machine, which would
# leave the other SIMD kernels unfuzzed; each instance therefore pins its
# backend at compile time via the RustCrypto-style RUSTFLAGS cfg
# `--cfg chacha20poly1305_backend=<name>` (zero probing cost in production
# builds, validated by build.rs against the target arch / target features).
# Every job gets its own CARGO_TARGET_DIR: different RUSTFLAGS cannot share
# one cargo build cache without thrashing.
#
# CPU usage is controlled by the caller: give each backend a core list
# (taskset) and, optionally, the fork-job count. Examples:
#   fuzz/run.sh                          # 1800s, 0-2:3-5:6-8:9-11
#   fuzz/run.sh 300 "0-2:3-5:6-8:9-11"   # fewer cores per backend
#   FUZZ_JOBS=2 fuzz/run.sh 300          # override fork jobs
#
# Usage: fuzz/run.sh [seconds-per-backend] [cpu-spec]
#   cpu-spec: core list for each backend, colon-separated (soft:sse2:avx2:avx512).
#             Each list is a taskset range (e.g. "0-3" or "0,2,4").
#   FUZZ_JOBS: fork jobs per backend; default = min(4, cores of that backend).
set -euo pipefail
cd "$(dirname "$0")"

TIME=${1:-1800}
SPEC=${2:-"0-2:3-5:6-8:9-11"}
MAXLEN=${MAXLEN:-8192}
TMP=$(mktemp -d /tmp/chacha-fuzz.XXXXXX)
trap 'rm -rf "$TMP"' EXIT

# Per-backend RUSTFLAGS: force our backend, and give SIMD tiers the matching
# target features build.rs demands (matches the production build contract).
flags_for() {
	case $1 in
	soft) printf '%s' '--cfg chacha20poly1305_backend="soft"' ;;
	sse2) printf '%s' '--cfg chacha20poly1305_backend="sse2"' ;;
	avx2) printf '%s' '--cfg chacha20poly1305_backend="avx2" -Ctarget-feature=+avx2' ;;
	avx512) printf '%s' '--cfg chacha20poly1305_backend="avx512" -Ctarget-feature=+avx2,+avx512f,+avx512vl' ;;
	*) exit 1 ;;
	esac
}

BACKENDS=(soft sse2 avx2 avx512)
IFS=: read -ra CORES <<<"$SPEC"
if [ "${#CORES[@]}" -ne 4 ]; then
	echo "cpu-spec must have 4 parts (soft:sse2:avx2:avx512), got: $SPEC" >&2
	exit 1
fi

count_cores() {
	local n=0 part lo hi
	IFS=, read -ra parts <<<"$1"
	for part in "${parts[@]}"; do
		if [[ "$part" == *-* ]]; then
			lo=${part%-*}
			hi=${part#*-}
			n=$((n + hi - lo + 1))
		else
			n=$((n + 1))
		fi
	done
	echo "$n"
}

PIDS=()
for i in "${!BACKENDS[@]}"; do
	be=${BACKENDS[$i]}
	cores=${CORES[$i]}
	cp -r "corpus/differential" "$TMP/corpus-$be"
	mkdir -p "$TMP/art-$be" "$TMP/target-$be"
	jobs=${FUZZ_JOBS:-}
	if [ -z "$jobs" ]; then
		c=$(count_cores "$cores")
		jobs=$((c < 4 ? c : 4))
	fi
	RUSTFLAGS="$(flags_for "$be")" CARGO_TARGET_DIR="$TMP/target-$be" \
		taskset -c "$cores" \
		cargo fuzz run -j "$jobs" differential "$TMP/corpus-$be" -- \
		-max_total_time="$TIME" -max_len="$MAXLEN" -artifact_prefix="$TMP/art-$be/" \
		>"$TMP/fuzz-$be.log" 2>&1 &
	PIDS+=($!)
	echo "[$be] pid $! cores $cores jobs $jobs"
done

FAIL=0
for i in "${!BACKENDS[@]}"; do
	be=${BACKENDS[$i]}
	if wait "${PIDS[$i]}"; then
		echo "[$be] finished: $(tail -1 "$TMP/fuzz-$be.log")"
	else
		echo "[$be] FAILED:"
		tail -30 "$TMP/fuzz-$be.log"
		FAIL=1
	fi
	if ls "$TMP/art-$be"/* >/dev/null 2>&1; then
		echo "[$be] artifacts:"
		cp -v "$TMP/art-$be"/* artifacts/differential/
	fi
done
# Preserve the per-backend libFuzzer logs so executed-run totals stay
# auditable after the TMP cleanup below (the final `#N` fork-mode progress
# line carries the cumulative execution count).
mkdir -p logs
for i in "${!BACKENDS[@]}"; do
	be=${BACKENDS[$i]}
	cp "$TMP/fuzz-$be.log" "logs/fuzz-$be-$(date +%Y%m%d-%H%M%S).log"
done
exit $FAIL
