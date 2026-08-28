#!/usr/bin/env python3
"""Collect criterion means into bench_results.json.

Each ISA tier is benched in its own CARGO_TARGET_DIR (different RUSTFLAGS
cannot share one cargo cache), produced by e.g.:

    RUSTFLAGS='--cfg chacha20poly1305_backend="avx512" --cfg chacha20_backend="avx512" \
        --cfg chacha20_avx512 -Ctarget-feature=+avx2,+avx512f,+avx512vl,+avx512ifma' \
    CARGO_TARGET_DIR=target/bench-avx512 cargo bench --bench aead -- \
        'seal/chacha20poly1305|open/chacha20poly1305'

(soft/sse2/avx2 analogous; see README "Performance".) OpenSSL medians come
from perf/openssl.csv (seal, EVP_chacha20_poly1305).
"""

import csv
import json
import os

SIZES = [16, 31, 64, 128, 256, 512, 1024, 4096, 16384, 65536, 262144, 1048576]

# tier -> (target dir, our bench id, RustCrypto bench id)
TIERS = {
    "soft": ("target/bench-soft", "soft", "rc-soft+poly-soft"),
    "sse2": ("target/bench-sse2", "sse2", "rc-sse2+poly-soft"),
    "avx2": ("target/bench-avx2", "avx2", "rc-avx2+poly-auto(avx2_soft)"),
    "avx512": ("target/bench-avx512", "avx512-ifma", "rc-avx512+poly-auto(avx2_soft)"),
}


def read_ns(root, group, bid, size):
    p = os.path.join(root, "criterion", group, f"{bid}_{size}", "new", "estimates.json")
    with open(p) as f:
        return json.load(f)["mean"]["point_estimate"]  # ns


data: dict = {"ours": {}, "rustcrypto": {}, "openssl": {}, "open_1mib": {}}
for tier, (root, ours_id, rc_id) in TIERS.items():
    data["ours"][tier] = [read_ns(root, "seal_chacha20poly1305", ours_id, s) for s in SIZES]
    data["rustcrypto"][tier] = [
        read_ns(root, "seal_chacha20poly1305", rc_id, s) for s in SIZES
    ]
    # Open is only benched at 1 MiB (criterion id label, not the raw size).
    data["open_1mib"][tier] = read_ns(root, "open_chacha20poly1305", ours_id, "1MiB")

with open("perf/openssl.csv") as f:
    r = list(csv.reader(f))[1:]
    data["openssl"]["auto"] = [float(row[1]) for row in r]

data["sizes"] = SIZES
with open("perf/bench_results.json", "w") as f:
    json.dump(data, f, indent=1)


def mib(ns, s):
    return s / ns * 1000 / 1.048576


print(f"{'size':>8} | " + " | ".join(f"{t:>7}" for t in TIERS) + " | openssl")
for i, s in enumerate(SIZES):
    row = [f"{mib(data['ours'][t][i], s):7.0f}" for t in TIERS]
    ossl = f"{mib(data['openssl']['auto'][i], s):7.0f}"
    print(f"{s:>8} | " + " | ".join(row) + f" | {ossl}")
print("open 1MiB (MiB/s):", {t: round(mib(v, 1 << 20)) for t, v in data["open_1mib"].items()})
