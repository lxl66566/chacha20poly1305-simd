import json, os, csv, math

SIZES = [16,31,64,128,256,512,1024,4096,16384,65536,262144,1048576]
base = "target/criterion/seal_chacha20poly1305"

def read_ns(dirname, size):
    p = os.path.join(base, f"{dirname}_{size}", "new", "estimates.json")
    with open(p) as f:
        return json.load(f)["mean"]["point_estimate"]  # ns

data = {"ours": {}, "rustcrypto": {}, "openssl": {}}
tiers = {
    "soft": ("soft", "rc-soft+poly-soft"),
    "sse2": ("sse2", "rc-sse2+poly-soft"),
    "avx2": ("avx2", "rc-avx2+poly-auto(avx2_soft)"),
    "avx512": ("avx512", "rc-avx512+poly-auto(avx2_soft)"),
}
for tier,(ours_id, rc_id) in tiers.items():
    data["ours"][tier] = [read_ns(ours_id, s) for s in SIZES]
    data["rustcrypto"][tier] = [read_ns(rc_id, s) for s in SIZES]
with open("perf/openssl.csv") as f:
    r = list(csv.reader(f))[1:]
    data["openssl"]["auto"] = [float(row[1]) for row in r]

data["sizes"] = SIZES
with open("perf/bench_results.json","w") as f:
    json.dump(data, f, indent=1)

# print throughput table (MiB/s)
def mib(ns, s): return s/ns*1000/1.048576
print(f"{'size':>8} | " + " | ".join(f"{t:>16}" for t in list(tiers)+["ossl"]))
for i,s in enumerate(SIZES):
    cells = [f"{mib(data['ours'][t][i],s):7.1f}/{mib(data['rustcrypto'][t][i],s):7.1f}" for t in tiers]
    print(f"{s:>8} | " + " | ".join(f"{c:>16}" for c in cells) + f" | {mib(data['openssl']['auto'][i],s):7.1f}")
