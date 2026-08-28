#!/usr/bin/env python3
"""Generate benchmark comparison SVG charts from bench_results.json.

No third-party deps; emits plain SVG so it renders on GitHub.
Usage: python3 perf/make_charts.py
"""

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "bench_results.json")) as f:
    DATA = json.load(f)

SIZES = DATA["sizes"]
MIB = 1.048576  # bytes-per-ns -> MiB/s factor: size/ns*1000/1.048576


def mib(ns, size):
    return size / ns * 1000 / MIB


def size_label(s):
    if s >= 1 << 20:
        return f"{s >> 20} MiB"
    if s >= 1 << 10:
        return f"{s >> 10} KiB"
    return f"{s} B"


FONT = "font-family='system-ui,-apple-system,Segoe UI,Roboto,sans-serif'"
MONO = "font-family='ui-monospace,SFMono-Regular,Menlo,monospace'"

TIERS = ["soft", "sse2", "avx2", "avx512"]
COLOR = {"soft": "#1f77b4", "sse2": "#2ca02c", "avx2": "#ff7f0e", "avx512": "#d62728"}
TIER_LABEL = {"soft": "scalar", "sse2": "SSE2", "avx2": "AVX2", "avx512": "AVX-512"}
OSSL_COLOR = "#9467bd"
AWSLC_COLOR = "#8c564b"


class Plot:
    """Single log-x / log-or-linear-y panel."""

    def __init__(self, w, h, xlog=True, ylog=True):
        self.w, self.h = w, h
        self.ml, self.mr, self.mt, self.mb = 78, 16, 40, 52
        self.xlog, self.ylog = xlog, ylog
        self.el = []

    def fit(self, xs, ys, ymin_floor=None):
        self.x0, self.x1 = min(xs), max(xs)
        self.y0, self.y1 = min(ys), max(ys)
        if self.ylog:
            self.ly0 = math.log10(self.y0)
            self.ly1 = math.log10(self.y1)
            span = self.ly1 - self.ly0
            self.ly0 -= span * 0.08
            self.ly1 += span * 0.10
            if ymin_floor is not None:
                self.ly0 = max(self.ly0, math.log10(ymin_floor))
        else:
            span = self.y1 - self.y0
            self.y0 -= span * 0.08
            self.y1 += span * 0.10
        self.lx0, self.lx1 = math.log10(self.x0), math.log10(self.x1)

    def X(self, x):
        f = (math.log10(x) - self.lx0) / (self.lx1 - self.lx0)
        return self.ml + f * (self.w - self.ml - self.mr)

    def Y(self, y):
        v = math.log10(y) if self.ylog else y
        lo, hi = (self.ly0, self.ly1) if self.ylog else (self.y0, self.y1)
        f = (v - lo) / (hi - lo)
        return self.mt + (1 - f) * (self.h - self.mt - self.mb)

    def line(self, xs, ys, color, dash=None, width=2.2, marker=True, opacity=1.0):
        pts = " ".join(f"{self.X(x):.1f},{self.Y(y):.1f}" for x, y in zip(xs, ys))
        d = f" stroke-dasharray='{dash}'" if dash else ""
        self.el.append(
            f"<polyline points='{pts}' fill='none' stroke='{color}' "
            f"stroke-width='{width}'{d} opacity='{opacity}'/>"
        )
        if marker:
            for x, y in zip(xs, ys):
                self.el.append(
                    f"<circle cx='{self.X(x):.1f}' cy='{self.Y(y):.1f}' r='2.6' "
                    f"fill='{color}' opacity='{opacity}'/>"
                )

    def text(
        self, x, y, s, size=12.0, anchor="middle", color="#333", mono=False, rotate=None
    ):
        rot = f" transform='rotate({rotate} {x} {y})'" if rotate else ""
        self.el.append(
            f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' text-anchor='{anchor}' "
            f"fill='{color}' {MONO if mono else FONT}{rot}>{s}</text>"
        )

    def frame(self, xticks, yticks, yfmt, xlabel="", ylabel=""):
        # gridlines + ticks
        for t in yticks:
            yy = self.Y(t)
            self.el.append(
                f"<line x1='{self.ml}' y1='{yy:.1f}' x2='{self.w - self.mr}' y2='{yy:.1f}' "
                f"stroke='#e3e3e3' stroke-width='1'/>"
            )
            self.text(self.ml - 8, yy + 4, yfmt(t), size=11, anchor="end", color="#555")
        for t in xticks:
            xx = self.X(t)
            self.el.append(
                f"<line x1='{xx:.1f}' y1='{self.mt}' x2='{xx:.1f}' "
                f"y2='{self.h - self.mb}' stroke='#efefef' stroke-width='1'/>"
            )
            self.text(xx, self.h - self.mb + 18, size_label(t), size=11, color="#555")
        # axes
        self.el.append(
            f"<rect x='{self.ml}' y='{self.mt}' width='{self.w - self.ml - self.mr}' "
            f"height='{self.h - self.mt - self.mb}' fill='none' stroke='#bbb'/>"
        )
        if xlabel:
            self.text(
                (self.ml + self.w - self.mr) / 2,
                self.h - 8,
                xlabel,
                size=12,
                color="#444",
            )
        if ylabel:
            self.text(
                18,
                (self.mt + self.h - self.mb) / 2,
                ylabel,
                size=12,
                color="#444",
                rotate=-90,
            )

    def render(self):
        body = "\n".join(self.el)
        return f"<rect width='{self.w}' height='{self.h}' fill='white'/>\n{body}"


def svg_doc(w, h, inner):
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' "
        f"viewBox='0 0 {w} {h}'>\n{inner}\n</svg>\n"
    )


def nice_yticks(y0, y1, log=True):
    if log:
        # Binary ticks so labels read as clean MiB/s / GiB/s values.
        ticks = []
        v = 32
        while v <= y1 * 1.1:
            if v >= y0 * 0.9:
                ticks.append(v)
            v *= 2
        return ticks
    span = y1 - y0
    step = 10 ** math.floor(math.log10(span / 4))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= 5:
            step *= m
            break
    t = math.ceil(y0 / step) * step
    ticks = []
    while t <= y1:
        ticks.append(t)
        t += step
    return ticks


def fmt_mib(v):
    if v >= 1024:
        return f"{v / 1024:g} GiB/s"
    return f"{v:g} MiB/s"


def bottom_legend(p, title, entries, ncols=4, row_h=17, font=11.0):
    """Title on top; legend in a bottom band below the x tick labels.

    Fixed 4-column grid so the two charts' legends line up column-by-column
    (same ml/mr/width); p.mb is enlarged to fit, keeping the legend out of
    the plot area. Must be called before fit()/frame().
    """
    p.mt = 40
    p.text(p.ml + 2, 16, title, size=13, anchor="start", color="#222")

    rows = -(-len(entries) // ncols)
    p.mb = 52 + rows * row_h
    col_w = (p.w - p.ml - p.mr) / ncols
    plot_bottom = p.h - p.mb
    for i, (label, color, dash) in enumerate(entries):
        r, c = divmod(i, ncols)
        x = p.ml + c * col_w
        y = plot_bottom + 36 + r * row_h
        dd = f" stroke-dasharray='{dash}'" if dash else ""
        p.el.append(
            f"<line x1='{x:.0f}' y1='{y - 4:.0f}' x2='{x + 22:.0f}' y2='{y - 4:.0f}' "
            f"stroke='{color}' stroke-width='2.4'{dd}/>"
        )
        p.text(x + 30, y, label, size=font, anchor="start", color="#333")


def chart_throughput():
    p = Plot(960, 560, xlog=True, ylog=True)
    series = {}
    for t in TIERS:
        series[f"ours-{t}"] = [mib(ns, s) for ns, s in zip(DATA["ours"][t], SIZES)]
        series[f"rc-{t}"] = [mib(ns, s) for ns, s in zip(DATA["rustcrypto"][t], SIZES)]
    series["awslc"] = [mib(ns, s) for ns, s in zip(DATA["awslc"]["auto"], SIZES)]
    series["openssl"] = [mib(ns, s) for ns, s in zip(DATA["openssl"]["auto"], SIZES)]

    ally = [y for ys in series.values() for y in ys]
    entries: list[tuple[str, str, str | None]] = [
        (f"this crate · {TIER_LABEL[t]}", COLOR[t], None) for t in TIERS
    ]
    for t in TIERS:
        entries.append((f"RustCrypto · {TIER_LABEL[t]}", COLOR[t], "6 4"))
    entries.append(("aws-lc-rs 1.18 · auto", AWSLC_COLOR, "4 2"))
    entries.append(("OpenSSL · auto (AVX-512/IFMA)", OSSL_COLOR, "10 4 2 4"))
    bottom_legend(
        p,
        "ChaCha20-Poly1305 seal throughput — this crate (solid) vs RustCrypto "
        "(dashed, same ISA tier) vs OpenSSL 4.1-dev / aws-lc-rs (auto-dispatch)",
        entries,
    )

    p.fit(SIZES, ally, ymin_floor=10)

    yticks = nice_yticks(min(ally), max(ally), log=True)
    p.frame(
        SIZES,
        yticks,
        fmt_mib,
        xlabel="message size",
        ylabel="throughput (seal, in-place)",
    )

    for t in TIERS:
        p.line(
            SIZES,
            series[f"rc-{t}"],
            COLOR[t],
            dash="6 4",
            width=1.8,
            opacity=0.55,
            marker=False,
        )
    p.line(SIZES, series["awslc"], AWSLC_COLOR, dash="4 2", width=2.2, marker=True)
    p.line(
        SIZES, series["openssl"], OSSL_COLOR, dash="10 4 2 4", width=2.6, marker=True
    )
    for t in TIERS:
        p.line(SIZES, series[f"ours-{t}"], COLOR[t], width=2.4)

    return svg_doc(p.w, p.h, p.render())


def chart_speedup():
    p = Plot(960, 500, xlog=True, ylog=False)
    series = {}
    for t in TIERS:
        series[t] = [
            DATA["rustcrypto"][t][i] / DATA["ours"][t][i] for i in range(len(SIZES))
        ]
    series["openssl"] = [
        DATA["openssl"]["auto"][i] / DATA["ours"]["avx512"][i]
        for i in range(len(SIZES))
    ]
    series["awslc"] = [
        DATA["awslc"]["auto"][i] / DATA["ours"]["avx512"][i] for i in range(len(SIZES))
    ]
    ally = [y for ys in series.values() for y in ys]
    entries: list[tuple[str, str, str | None]] = [
        (f"vs RustCrypto · {TIER_LABEL[t]}", COLOR[t], None) for t in TIERS
    ]
    entries.append(("vs OpenSSL 4.1-dev · auto", OSSL_COLOR, "10 4 2 4"))
    entries.append(("vs aws-lc-rs · auto", AWSLC_COLOR, "4 2"))
    bottom_legend(
        p,
        "Speedup: this crate vs RustCrypto (per ISA tier) and vs OpenSSL 4.1-dev "
        "/ aws-lc-rs (our AVX-512 vs their auto)",
        entries,
    )

    p.fit(SIZES, [0.0, max(ally) * 1.02])
    p.y0 = 0.0

    yticks = nice_yticks(0, max(ally), log=False)
    p.frame(
        SIZES,
        yticks,
        lambda v: f"{v:g}×",
        xlabel="message size",
        ylabel="speedup of this crate (higher = faster)",
    )

    # 1x reference
    yy = p.Y(1.0)
    p.el.append(
        f"<line x1='{p.ml}' y1='{yy:.1f}' x2='{p.w - p.mr}' y2='{yy:.1f}' stroke='#999' "
        f"stroke-width='1.2' stroke-dasharray='3 3'/>"
    )
    p.text(p.w - p.mr - 4, yy - 5, "parity", size=10.5, anchor="end", color="#777")

    for t in TIERS:
        p.line(SIZES, series[t], COLOR[t], width=2.4)
    p.line(
        SIZES, series["openssl"], OSSL_COLOR, dash="10 4 2 4", width=2.4, marker=True
    )
    p.line(SIZES, series["awslc"], AWSLC_COLOR, dash="4 2", width=2.4, marker=True)

    return svg_doc(p.w, p.h, p.render())


def main():
    out = {
        "chart-throughput.svg": chart_throughput(),
        "chart-speedup.svg": chart_speedup(),
    }
    for name, content in out.items():
        path = os.path.join(HERE, name)
        with open(path, "w") as f:
            f.write(content)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
