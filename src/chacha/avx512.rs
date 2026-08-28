//! AVX-512VL ChaCha20 kernel (YMM-wide, `vprold` rotations).
//!
//! Zen 4 executes 256-bit vector ops at full rate while 512-bit ops are
//! double-pumped, so the fastest shape on that uarch is 8 blocks spread over
//! 16 YMM registers with single-instruction rotations — AVX-512VL gives both
//! the 32 registers and `vprold` without paying 512-bit dispatch.
//!
//! Layout per quad `k` (2 blocks): `a|b|c|d` row registers where each 128-bit
//! lane holds one block's row; the classic rows↔cols shuffle trick turns
//! column rounds into diagonal rounds.

// ChaCha quarter-round rows are conventionally named a/b/c/d; state loads use
// unaligned `_mm_loadu_si128`, so the stricter pointer alignment is intentional.
#![allow(clippy::many_single_char_names, clippy::cast_ptr_alignment)]

use core::arch::x86_64::*;

use crate::chacha::{BLOCK, State};

/// Blocks per bulk batch (word-major zmm kernel).
pub(crate) const BATCH_BLOCKS: usize = 16;

/// Load the constant/key rows of `state` broadcast into both 128-bit lanes.
#[cfg_attr(debug_assertions, inline)]
#[cfg_attr(not(debug_assertions), inline(always))]
unsafe fn rows(state: &State) -> [__m256i; 3] {
    let p = state.words.as_ptr().cast::<__m128i>();
    [
        _mm256_broadcastsi128_si256(_mm_loadu_si128(p.add(0))),
        _mm256_broadcastsi128_si256(_mm_loadu_si128(p.add(1))),
        _mm256_broadcastsi128_si256(_mm_loadu_si128(p.add(2))),
    ]
}

/// Counter row for quad `k`: `[c0, w13, w14, w15 | c0+1, w13, w14, w15]`
/// where `c0 = base + 2k`. Built directly (the broadcast+add form double
/// counts `w12`, which only worked by accident at counter 0).
#[cfg_attr(debug_assertions, inline)]
#[cfg_attr(not(debug_assertions), inline(always))]
unsafe fn ctr_row(state: &State, base: u32, k: usize) -> __m256i {
    let w = state.words;
    let c0 = base.wrapping_add(2 * k as u32);
    _mm256_setr_epi32(
        c0 as i32,
        w[13] as i32,
        w[14] as i32,
        w[15] as i32,
        c0.wrapping_add(1) as i32,
        w[13] as i32,
        w[14] as i32,
        w[15] as i32,
    )
}

/// A full quarter round (rotations 16/12/8/7) on explicit row registers.
///
/// Macro form is load-bearing: the 8-block kernel keeps 16 YMM rows live, and
/// the previous array-based formulation (`[[__m256i; 4]; N]` + helper fns)
/// made LLVM spill the whole state to the stack on every double round.
///
/// Splitting the QR into (16,12) + lane shuffle + (8,7) is INVALID for
/// ChaCha (that shape belongs to BLAKE2's half-rounds): the whole QR must
/// run on the same word grouping before the rows↔cols shuffle.
macro_rules! qr {
    ($a:ident, $b:ident, $c:ident, $d:ident) => {{
        $a = _mm256_add_epi32($a, $b);
        $d = _mm256_rol_epi32::<16>(_mm256_xor_si256($d, $a));
        $c = _mm256_add_epi32($c, $d);
        $b = _mm256_rol_epi32::<12>(_mm256_xor_si256($b, $c));
        $a = _mm256_add_epi32($a, $b);
        $d = _mm256_rol_epi32::<8>(_mm256_xor_si256($d, $a));
        $c = _mm256_add_epi32($c, $d);
        $b = _mm256_rol_epi32::<7>(_mm256_xor_si256($b, $c));
    }};
}

/// Rotate rows to columns and back (the blake2-avx2 lane-shuffle trick):
/// shuffles are 128-bit-lane local, so they commute with the quad layout.
macro_rules! to_cols {
    ($a:ident, $b:ident, $c:ident, $d:ident) => {{
        let _ = $b;
        $c = _mm256_shuffle_epi32::<0b_00_11_10_01>($c);
        $d = _mm256_shuffle_epi32::<0b_01_00_11_10>($d);
        $a = _mm256_shuffle_epi32::<0b_10_01_00_11>($a);
    }};
}

macro_rules! to_rows {
    ($a:ident, $b:ident, $c:ident, $d:ident) => {{
        let _ = $b;
        $c = _mm256_shuffle_epi32::<0b_10_01_00_11>($c);
        $d = _mm256_shuffle_epi32::<0b_01_00_11_10>($d);
        $a = _mm256_shuffle_epi32::<0b_00_11_10_01>($a);
    }};
}

/// Full 20 rounds + feed-forward add on 4 quads (8 blocks), state held in
/// 16 explicit locals so it stays in registers.
#[cfg_attr(debug_assertions, inline)]
#[cfg_attr(not(debug_assertions), inline(always))]
unsafe fn rounds4(v: &[__m256i; 3], ctrs: &[__m256i; 4]) -> [[__m256i; 4]; 4] {
    let (mut a0, mut b0, mut c0, mut d0) = (v[0], v[1], v[2], ctrs[0]);
    let (mut a1, mut b1, mut c1, mut d1) = (v[0], v[1], v[2], ctrs[1]);
    let (mut a2, mut b2, mut c2, mut d2) = (v[0], v[1], v[2], ctrs[2]);
    let (mut a3, mut b3, mut c3, mut d3) = (v[0], v[1], v[2], ctrs[3]);
    for _ in 0..10 {
        qr!(a0, b0, c0, d0);
        qr!(a1, b1, c1, d1);
        qr!(a2, b2, c2, d2);
        qr!(a3, b3, c3, d3);
        to_cols!(a0, b0, c0, d0);
        to_cols!(a1, b1, c1, d1);
        to_cols!(a2, b2, c2, d2);
        to_cols!(a3, b3, c3, d3);
        qr!(a0, b0, c0, d0);
        qr!(a1, b1, c1, d1);
        qr!(a2, b2, c2, d2);
        qr!(a3, b3, c3, d3);
        to_rows!(a0, b0, c0, d0);
        to_rows!(a1, b1, c1, d1);
        to_rows!(a2, b2, c2, d2);
        to_rows!(a3, b3, c3, d3);
    }
    [
        [
            _mm256_add_epi32(a0, v[0]),
            _mm256_add_epi32(b0, v[1]),
            _mm256_add_epi32(c0, v[2]),
            _mm256_add_epi32(d0, ctrs[0]),
        ],
        [
            _mm256_add_epi32(a1, v[0]),
            _mm256_add_epi32(b1, v[1]),
            _mm256_add_epi32(c1, v[2]),
            _mm256_add_epi32(d1, ctrs[1]),
        ],
        [
            _mm256_add_epi32(a2, v[0]),
            _mm256_add_epi32(b2, v[1]),
            _mm256_add_epi32(c2, v[2]),
            _mm256_add_epi32(d2, ctrs[2]),
        ],
        [
            _mm256_add_epi32(a3, v[0]),
            _mm256_add_epi32(b3, v[1]),
            _mm256_add_epi32(c3, v[2]),
            _mm256_add_epi32(d3, ctrs[3]),
        ],
    ]
}

/// Single-quad (2-block) variant of [`rounds4`].
#[cfg_attr(debug_assertions, inline)]
#[cfg_attr(not(debug_assertions), inline(always))]
unsafe fn rounds1(v: &[__m256i; 3], ctr: __m256i) -> [__m256i; 4] {
    let (mut a, mut b, mut c, mut d) = (v[0], v[1], v[2], ctr);
    for _ in 0..10 {
        qr!(a, b, c, d);
        to_cols!(a, b, c, d);
        qr!(a, b, c, d);
        to_rows!(a, b, c, d);
    }
    [
        _mm256_add_epi32(a, v[0]),
        _mm256_add_epi32(b, v[1]),
        _mm256_add_epi32(c, v[2]),
        _mm256_add_epi32(d, ctr),
    ]
}

/// Two-quad (4-block) variant of [`rounds4`] — the OpenSSL 4x ladder tier
/// for 256..511-byte tails.
#[cfg_attr(debug_assertions, inline)]
#[cfg_attr(not(debug_assertions), inline(always))]
unsafe fn rounds2(v: &[__m256i; 3], ctrs: &[__m256i; 2]) -> [[__m256i; 4]; 2] {
    let (mut a0, mut b0, mut c0, mut d0) = (v[0], v[1], v[2], ctrs[0]);
    let (mut a1, mut b1, mut c1, mut d1) = (v[0], v[1], v[2], ctrs[1]);
    for _ in 0..10 {
        qr!(a0, b0, c0, d0);
        qr!(a1, b1, c1, d1);
        to_cols!(a0, b0, c0, d0);
        to_cols!(a1, b1, c1, d1);
        qr!(a0, b0, c0, d0);
        qr!(a1, b1, c1, d1);
        to_rows!(a0, b0, c0, d0);
        to_rows!(a1, b1, c1, d1);
    }
    [
        [
            _mm256_add_epi32(a0, v[0]),
            _mm256_add_epi32(b0, v[1]),
            _mm256_add_epi32(c0, v[2]),
            _mm256_add_epi32(d0, ctrs[0]),
        ],
        [
            _mm256_add_epi32(a1, v[0]),
            _mm256_add_epi32(b1, v[1]),
            _mm256_add_epi32(c1, v[2]),
            _mm256_add_epi32(d1, ctrs[1]),
        ],
    ]
}

/// Assemble one 64-byte block from finalized rows: `[a|b]` and `[c|d]`.
#[cfg_attr(debug_assertions, inline)]
#[cfg_attr(not(debug_assertions), inline(always))]
unsafe fn emit_lo(a: __m256i, b: __m256i, block: usize) -> __m256i {
    match block {
        0 => _mm256_permute2f128_si256::<0x20>(a, b),
        _ => _mm256_permute2f128_si256::<0x31>(a, b),
    }
}

#[cfg_attr(debug_assertions, inline)]
#[cfg_attr(not(debug_assertions), inline(always))]
unsafe fn emit_hi(c: __m256i, d: __m256i, block: usize) -> __m256i {
    match block {
        0 => _mm256_permute2f128_si256::<0x20>(c, d),
        _ => _mm256_permute2f128_si256::<0x31>(c, d),
    }
}

/// Generate `BATCH_BLOCKS` keystream blocks and XOR them into `buf`
/// (`buf.len() == BATCH_BLOCKS * BLOCK`), advancing the counter.
///
/// Counter wrap-around mirrors upstream wrapping semantics at u32.
#[cfg_attr(debug_assertions, inline)]
#[cfg_attr(not(debug_assertions), inline(always))]
pub(crate) unsafe fn xor_batch8(state: &mut State, buf: *mut u8) {
    let v = rows(state);
    let base = state.words[12];
    let ctrs = [
        ctr_row(state, base, 0),
        ctr_row(state, base, 1),
        ctr_row(state, base, 2),
        ctr_row(state, base, 3),
    ];
    state.advance(8);
    let vs = rounds4(&v, &ctrs);

    let mut p = buf;
    for quad in &vs {
        let [a, b, c, d] = *quad;
        for blk in 0..2 {
            let lo = emit_lo(a, b, blk);
            let hi = emit_hi(c, d, blk);
            let pt_lo = _mm256_loadu_si256(p.cast());
            let pt_hi = _mm256_loadu_si256(p.add(32).cast());
            _mm256_storeu_si256(p.cast(), _mm256_xor_si256(pt_lo, lo));
            _mm256_storeu_si256(p.add(32).cast(), _mm256_xor_si256(pt_hi, hi));
            p = p.add(BLOCK);
        }
    }
}

/// Two-block (128-byte) batch for mid-size tails.
#[cfg_attr(debug_assertions, inline)]
#[cfg_attr(not(debug_assertions), inline(always))]
pub(crate) unsafe fn xor_quad(state: &mut State, mut buf: *mut u8) {
    let v = rows(state);
    let c = ctr_row(state, state.words[12], 0);
    state.advance(2);
    let [a, b, cc, d] = rounds1(&v, c);
    for blk in 0..2 {
        let lo = emit_lo(a, b, blk);
        let hi = emit_hi(cc, d, blk);
        let pt_lo = _mm256_loadu_si256(buf.cast());
        let pt_hi = _mm256_loadu_si256(buf.add(32).cast());
        _mm256_storeu_si256(buf.cast(), _mm256_xor_si256(pt_lo, lo));
        _mm256_storeu_si256(buf.add(32).cast(), _mm256_xor_si256(pt_hi, hi));
        buf = buf.add(BLOCK);
    }
}

/// Four-block (256-byte) batch — 2 interleaved quads in one kernel call.
#[cfg_attr(debug_assertions, inline)]
#[cfg_attr(not(debug_assertions), inline(always))]
pub(crate) unsafe fn xor_quad2(state: &mut State, mut buf: *mut u8) {
    let v = rows(state);
    let base = state.words[12];
    let ctrs = [ctr_row(state, base, 0), ctr_row(state, base, 1)];
    state.advance(4);
    let vs = rounds2(&v, &ctrs);
    for quad in &vs {
        let [a, b, cc, d] = *quad;
        for blk in 0..2 {
            let lo = emit_lo(a, b, blk);
            let hi = emit_hi(cc, d, blk);
            let pt_lo = _mm256_loadu_si256(buf.cast());
            let pt_hi = _mm256_loadu_si256(buf.add(32).cast());
            _mm256_storeu_si256(buf.cast(), _mm256_xor_si256(pt_lo, lo));
            _mm256_storeu_si256(buf.add(32).cast(), _mm256_xor_si256(pt_hi, hi));
            buf = buf.add(BLOCK);
        }
    }
}

/// Fused prologue: blocks 0 and 1 in one quad call — block 0's first 32
/// bytes (the Poly1305 one-time key) to `key_out`, block 1's keystream
/// XORed into `b1` (a zeroed buffer yields the raw keystream).
#[cfg_attr(debug_assertions, inline)]
#[cfg_attr(not(debug_assertions), inline(always))]
pub(crate) unsafe fn gen_key_xor2(state: &mut State, key_out: &mut [u8; 32], b1: &mut [u8; BLOCK]) {
    let v = rows(state);
    let c = ctr_row(state, state.words[12], 0);
    state.advance(2);
    let [a, b, cc, d] = rounds1(&v, c);
    // block 0 words 0..7 (bytes 0..31) = [a|b] lane 0
    _mm256_storeu_si256(
        key_out.as_mut_ptr().cast(),
        _mm256_permute2f128_si256::<0x20>(a, b),
    );
    // block 1 = [a|b|c|d] lane 1, xored into b1
    let p = b1.as_mut_ptr().cast::<__m256i>();
    _mm256_storeu_si256(
        p,
        _mm256_xor_si256(
            _mm256_loadu_si256(p),
            _mm256_permute2f128_si256::<0x31>(a, b),
        ),
    );
    _mm256_storeu_si256(
        p.add(1),
        _mm256_xor_si256(
            _mm256_loadu_si256(p.add(1)),
            _mm256_permute2f128_si256::<0x31>(cc, d),
        ),
    );
}

/// Small-message fused op — same contract as the avx2 kernel's
/// `gen_ks_small` (block 0's first 32 bytes → one-time key, raw message
/// keystream blocks written to `ks`, one kernel call).
#[cfg_attr(debug_assertions, inline)]
#[cfg_attr(not(debug_assertions), inline(always))]
pub(crate) unsafe fn gen_ks_small(state: &mut State, key_out: &mut [u8; 32], ks: &mut [u8]) {
    debug_assert!(ks.len() <= 3 * BLOCK && ks.len().is_multiple_of(BLOCK));
    if ks.is_empty() {
        let v = rows(state);
        let c = ctr_row(state, state.words[12], 0);
        state.advance(1);
        let [a, b, _, _] = rounds1(&v, c);
        _mm256_storeu_si256(
            key_out.as_mut_ptr().cast(),
            _mm256_permute2f128_si256::<0x20>(a, b),
        );
        return;
    }
    let v = rows(state);
    if ks.len() == BLOCK {
        let c = ctr_row(state, state.words[12], 0);
        state.advance(2);
        let [a, b, cc, d] = rounds1(&v, c);
        _mm256_storeu_si256(
            key_out.as_mut_ptr().cast(),
            _mm256_permute2f128_si256::<0x20>(a, b),
        );
        let p = ks.as_mut_ptr().cast::<__m256i>();
        _mm256_storeu_si256(p, _mm256_permute2f128_si256::<0x31>(a, b));
        _mm256_storeu_si256(p.add(1), _mm256_permute2f128_si256::<0x31>(cc, d));
    } else {
        let base = state.words[12];
        let ctrs = [ctr_row(state, base, 0), ctr_row(state, base, 1)];
        state.advance(4);
        let vs = rounds2(&v, &ctrs);
        let [a0, b0, c0, d0] = vs[0];
        _mm256_storeu_si256(
            key_out.as_mut_ptr().cast(),
            _mm256_permute2f128_si256::<0x20>(a0, b0),
        );
        let p = ks.as_mut_ptr().cast::<__m256i>();
        // block 1 = quad 0 lane 1, blocks 2/3 = quad 1
        _mm256_storeu_si256(p, _mm256_permute2f128_si256::<0x31>(a0, b0));
        _mm256_storeu_si256(p.add(1), _mm256_permute2f128_si256::<0x31>(c0, d0));
        let [a1, b1, c1, d1] = vs[1];
        if ks.len() > BLOCK {
            _mm256_storeu_si256(p.add(2), _mm256_permute2f128_si256::<0x20>(a1, b1));
            _mm256_storeu_si256(p.add(3), _mm256_permute2f128_si256::<0x20>(c1, d1));
        }
        if ks.len() > 2 * BLOCK {
            _mm256_storeu_si256(p.add(4), _mm256_permute2f128_si256::<0x31>(a1, b1));
            _mm256_storeu_si256(p.add(5), _mm256_permute2f128_si256::<0x31>(c1, d1));
        }
    }
}

// ── 16-block word-major zmm kernel (OpenSSL `ChaCha20_16x` shape) ──
//
// Word `w` of the 16 block states lives broadcast across the 16 dword lanes
// of register `x_w` (lane = block index), so the diagonal round needs NO
// shuffles at all — diagonals are just different register indices. The
// YMM quad kernels above pay 6 shuffles per double round (~25% of their
// instruction mix on Zen 4); this layout pays a single 16×16 dword
// transpose at the end (~80 shuffle ops per KiB, amortized to noise).

/// Full quarter round on word-major zmm registers.
macro_rules! qrz {
    ($a:ident, $b:ident, $c:ident, $d:ident) => {{
        $a = _mm512_add_epi32($a, $b);
        $d = _mm512_rol_epi32::<16>(_mm512_xor_si512($d, $a));
        $c = _mm512_add_epi32($c, $d);
        $b = _mm512_rol_epi32::<12>(_mm512_xor_si512($b, $c));
        $a = _mm512_add_epi32($a, $b);
        $d = _mm512_rol_epi32::<8>(_mm512_xor_si512($d, $a));
        $c = _mm512_add_epi32($c, $d);
        $b = _mm512_rol_epi32::<7>(_mm512_xor_si512($b, $c));
    }};
}

/// In-lane 4×4 dword transpose of four word-major registers: afterwards
/// `$w_j`'s 128-bit lane ℓ holds words 4g..4g+3 of block 4ℓ+j (g = group).
macro_rules! tr4z {
    ($w0:ident, $w1:ident, $w2:ident, $w3:ident) => {{
        let t0 = _mm512_unpacklo_epi32($w0, $w1);
        let t1 = _mm512_unpackhi_epi32($w0, $w1);
        let t2 = _mm512_unpacklo_epi32($w2, $w3);
        let t3 = _mm512_unpackhi_epi32($w2, $w3);
        $w0 = _mm512_unpacklo_epi64(t0, t2);
        $w1 = _mm512_unpackhi_epi64(t0, t2);
        $w2 = _mm512_unpacklo_epi64(t1, t3);
        $w3 = _mm512_unpackhi_epi64(t1, t3);
    }};
}

/// Generate 16 keystream blocks and XOR them into `buf`
/// (`buf.len() == 16 * BLOCK`), advancing the counter.
// NOTE: #[inline(never)] + explicit target_feature — inlined into the fused
// engine, the 16 live zmm state vectors (plus the Poly1305 zmm state) exceed
// LLVM's register budget and it spills the whole state to the stack every
// round (~950 stack movs in seal_ifma). A standalone function gets the full
// register file; the per-1024-byte call is noise next to the batch cost.
/// One double round on sixteen word-major zmm locals.
macro_rules! dr16 {
    (
        $x0:ident,
        $x1:ident,
        $x2:ident,
        $x3:ident,
        $x4:ident,
        $x5:ident,
        $x6:ident,
        $x7:ident,
        $x8:ident,
        $x9:ident,
        $x10:ident,
        $x11:ident,
        $x12:ident,
        $x13:ident,
        $x14:ident,
        $x15:ident
    ) => {{
        qrz!($x0, $x4, $x8, $x12);
        qrz!($x1, $x5, $x9, $x13);
        qrz!($x2, $x6, $x10, $x14);
        qrz!($x3, $x7, $x11, $x15);
        qrz!($x0, $x5, $x10, $x15);
        qrz!($x1, $x6, $x11, $x12);
        qrz!($x2, $x7, $x8, $x13);
        qrz!($x3, $x4, $x9, $x14);
    }};
}

/// Feed-forward add ($bc = word broadcast closure, $ctr = original counter
/// row), 16×16 dword transpose, XOR into `buf`, store.
macro_rules! emit16 {
    (
        $buf:expr,
        $bc:expr,
        $ctr:expr,
        $x0:ident,
        $x1:ident,
        $x2:ident,
        $x3:ident,
        $x4:ident,
        $x5:ident,
        $x6:ident,
        $x7:ident,
        $x8:ident,
        $x9:ident,
        $x10:ident,
        $x11:ident,
        $x12:ident,
        $x13:ident,
        $x14:ident,
        $x15:ident
    ) => {{
        $x0 = _mm512_add_epi32($x0, ($bc)(0));
        $x1 = _mm512_add_epi32($x1, ($bc)(1));
        $x2 = _mm512_add_epi32($x2, ($bc)(2));
        $x3 = _mm512_add_epi32($x3, ($bc)(3));
        $x4 = _mm512_add_epi32($x4, ($bc)(4));
        $x5 = _mm512_add_epi32($x5, ($bc)(5));
        $x6 = _mm512_add_epi32($x6, ($bc)(6));
        $x7 = _mm512_add_epi32($x7, ($bc)(7));
        $x8 = _mm512_add_epi32($x8, ($bc)(8));
        $x9 = _mm512_add_epi32($x9, ($bc)(9));
        $x10 = _mm512_add_epi32($x10, ($bc)(10));
        $x11 = _mm512_add_epi32($x11, ($bc)(11));
        $x12 = _mm512_add_epi32($x12, $ctr);
        $x13 = _mm512_add_epi32($x13, ($bc)(13));
        $x14 = _mm512_add_epi32($x14, ($bc)(14));
        $x15 = _mm512_add_epi32($x15, ($bc)(15));

        // 16×16 dword transpose: in-lane 4×4 per word group, then a
        // 2-level `vpermt2d` gather assembling block b = 4ℓ+k from lane ℓ
        // of piece k of each group.
        tr4z!($x0, $x1, $x2, $x3);
        tr4z!($x4, $x5, $x6, $x7);
        tr4z!($x8, $x9, $x10, $x11);
        tr4z!($x12, $x13, $x14, $x15);

        let buf = $buf;
        let idx_b = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
        // Explicitly named registers (NOT an indexed array — the array form
        // made LLVM spill all 16 transposed states to the stack per block).
        macro_rules! emit {
            ($b: expr,$l: expr,$g0: ident,$g1: ident,$g2: ident,$g3: ident) => {{
                const L: i32 = 4 * $l;
                let idx_a = _mm512_setr_epi32(
                    L,
                    L + 1,
                    L + 2,
                    L + 3,
                    16 + L,
                    16 + L + 1,
                    16 + L + 2,
                    16 + L + 3,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                );
                let lo = _mm512_permutex2var_epi32($g0, idx_a, $g1);
                let hi = _mm512_permutex2var_epi32($g2, idx_a, $g3);
                let block = _mm512_permutex2var_epi32(lo, idx_b, hi);
                let p = buf.add($b * BLOCK);
                _mm512_storeu_si512(
                    p.cast(),
                    _mm512_xor_si512(_mm512_loadu_si512(p.cast()), block),
                );
            }};
        }
        emit!(0, 0, $x0, $x4, $x8, $x12);
        emit!(1, 0, $x1, $x5, $x9, $x13);
        emit!(2, 0, $x2, $x6, $x10, $x14);
        emit!(3, 0, $x3, $x7, $x11, $x15);
        emit!(4, 1, $x0, $x4, $x8, $x12);
        emit!(5, 1, $x1, $x5, $x9, $x13);
        emit!(6, 1, $x2, $x6, $x10, $x14);
        emit!(7, 1, $x3, $x7, $x11, $x15);
        emit!(8, 2, $x0, $x4, $x8, $x12);
        emit!(9, 2, $x1, $x5, $x9, $x13);
        emit!(10, 2, $x2, $x6, $x10, $x14);
        emit!(11, 2, $x3, $x7, $x11, $x15);
        emit!(12, 3, $x0, $x4, $x8, $x12);
        emit!(13, 3, $x1, $x5, $x9, $x13);
        emit!(14, 3, $x2, $x6, $x10, $x14);
        emit!(15, 3, $x3, $x7, $x11, $x15);
    }};
}

/// Broadcast the state into word-major locals x0..x15 + the counter row.
/// x0..x15 ← the (already broadcast) originals o0..o15 with the counter
/// row from `ctr`. Register-to-register copies only — the rename is free.
macro_rules! reload16 {
    (
        $o0:ident,
        $o1:ident,
        $o2:ident,
        $o3:ident,
        $o4:ident,
        $o5:ident,
        $o6:ident,
        $o7:ident,
        $o8:ident,
        $o9:ident,
        $o10:ident,
        $o11:ident,
        $o12:ident,
        $o13:ident,
        $o14:ident,
        $o15:ident,
        $x0:ident,
        $x1:ident,
        $x2:ident,
        $x3:ident,
        $x4:ident,
        $x5:ident,
        $x6:ident,
        $x7:ident,
        $x8:ident,
        $x9:ident,
        $x10:ident,
        $x11:ident,
        $x12:ident,
        $x13:ident,
        $x14:ident,
        $x15:ident
    ) => {
        let (mut $x0, mut $x1, mut $x2, mut $x3) = ($o0, $o1, $o2, $o3);
        let (mut $x4, mut $x5, mut $x6, mut $x7) = ($o4, $o5, $o6, $o7);
        let (mut $x8, mut $x9, mut $x10, mut $x11) = ($o8, $o9, $o10, $o11);
        let (mut $x12, mut $x13, mut $x14, mut $x15) = ($o12, $o13, $o14, $o15);
    };
}

/// emit16 variant taking the broadcast originals as registers (o12 = the
/// CURRENT iteration's counter row).
macro_rules! emit16o {
    (
        $buf:expr,
        $o0:ident,
        $o1:ident,
        $o2:ident,
        $o3:ident,
        $o4:ident,
        $o5:ident,
        $o6:ident,
        $o7:ident,
        $o8:ident,
        $o9:ident,
        $o10:ident,
        $o11:ident,
        $o12:ident,
        $o13:ident,
        $o14:ident,
        $o15:ident,
        $x0:ident,
        $x1:ident,
        $x2:ident,
        $x3:ident,
        $x4:ident,
        $x5:ident,
        $x6:ident,
        $x7:ident,
        $x8:ident,
        $x9:ident,
        $x10:ident,
        $x11:ident,
        $x12:ident,
        $x13:ident,
        $x14:ident,
        $x15:ident
    ) => {{
        $x0 = _mm512_add_epi32($x0, $o0);
        $x1 = _mm512_add_epi32($x1, $o1);
        $x2 = _mm512_add_epi32($x2, $o2);
        $x3 = _mm512_add_epi32($x3, $o3);
        $x4 = _mm512_add_epi32($x4, $o4);
        $x5 = _mm512_add_epi32($x5, $o5);
        $x6 = _mm512_add_epi32($x6, $o6);
        $x7 = _mm512_add_epi32($x7, $o7);
        $x8 = _mm512_add_epi32($x8, $o8);
        $x9 = _mm512_add_epi32($x9, $o9);
        $x10 = _mm512_add_epi32($x10, $o10);
        $x11 = _mm512_add_epi32($x11, $o11);
        $x12 = _mm512_add_epi32($x12, $o12);
        $x13 = _mm512_add_epi32($x13, $o13);
        $x14 = _mm512_add_epi32($x14, $o14);
        $x15 = _mm512_add_epi32($x15, $o15);
        tr4z!($x0, $x1, $x2, $x3);
        tr4z!($x4, $x5, $x6, $x7);
        tr4z!($x8, $x9, $x10, $x11);
        tr4z!($x12, $x13, $x14, $x15);
        let buf = $buf;
        let idx_b = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
        macro_rules! emit {
            ($b: expr,$l: expr,$g0: ident,$g1: ident,$g2: ident,$g3: ident) => {{
                const L: i32 = 4 * $l;
                let idx_a = _mm512_setr_epi32(
                    L,
                    L + 1,
                    L + 2,
                    L + 3,
                    16 + L,
                    16 + L + 1,
                    16 + L + 2,
                    16 + L + 3,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                );
                let lo = _mm512_permutex2var_epi32($g0, idx_a, $g1);
                let hi = _mm512_permutex2var_epi32($g2, idx_a, $g3);
                let block = _mm512_permutex2var_epi32(lo, idx_b, hi);
                let p = buf.add($b * BLOCK);
                _mm512_storeu_si512(
                    p.cast(),
                    _mm512_xor_si512(_mm512_loadu_si512(p.cast()), block),
                );
            }};
        }
        emit!(0, 0, $x0, $x4, $x8, $x12);
        emit!(1, 0, $x1, $x5, $x9, $x13);
        emit!(2, 0, $x2, $x6, $x10, $x14);
        emit!(3, 0, $x3, $x7, $x11, $x15);
        emit!(4, 1, $x0, $x4, $x8, $x12);
        emit!(5, 1, $x1, $x5, $x9, $x13);
        emit!(6, 1, $x2, $x6, $x10, $x14);
        emit!(7, 1, $x3, $x7, $x11, $x15);
        emit!(8, 2, $x0, $x4, $x8, $x12);
        emit!(9, 2, $x1, $x5, $x9, $x13);
        emit!(10, 2, $x2, $x6, $x10, $x14);
        emit!(11, 2, $x3, $x7, $x11, $x15);
        emit!(12, 3, $x0, $x4, $x8, $x12);
        emit!(13, 3, $x1, $x5, $x9, $x13);
        emit!(14, 3, $x2, $x6, $x10, $x14);
        emit!(15, 3, $x3, $x7, $x11, $x15);
    }};
}

// NOTE: no wrapping block — the `let` bindings must escape into the
// caller's scope (a `{{ }}` body would scope them to the block).
macro_rules! setup16 {
    (
        $state:expr,
        $w:ident,
        $bc:ident,
        $ctr:ident,
        $x0:ident,
        $x1:ident,
        $x2:ident,
        $x3:ident,
        $x4:ident,
        $x5:ident,
        $x6:ident,
        $x7:ident,
        $x8:ident,
        $x9:ident,
        $x10:ident,
        $x11:ident,
        $x12:ident,
        $x13:ident,
        $x14:ident,
        $x15:ident
    ) => {
        let $w = $state.words;
        let $bc = |i: usize| _mm512_set1_epi32($w[i] as i32);
        // x12 lane L = base + L; the other words are uniform across lanes.
        let $ctr = _mm512_add_epi32(
            $bc(12),
            _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        );
        $state.advance(16);
        let (mut $x0, mut $x1, mut $x2, mut $x3) = ($bc(0), $bc(1), $bc(2), $bc(3));
        let (mut $x4, mut $x5, mut $x6, mut $x7) = ($bc(4), $bc(5), $bc(6), $bc(7));
        let (mut $x8, mut $x9, mut $x10, mut $x11) = ($bc(8), $bc(9), $bc(10), $bc(11));
        let (mut $x12, mut $x13, mut $x14, mut $x15) = ($ctr, $bc(13), $bc(14), $bc(15));
    };
}

// NOTE: #[inline(never)] + explicit target_feature — inlined into the fused
// engine, the 16 live zmm state vectors (plus the Poly1305 zmm state) exceed
// LLVM's register budget and it spills the whole state to the stack every
// round (~950 stack movs in seal_ifma). A standalone function gets the full
// register file; the per-1024-byte call is noise next to the batch cost.
#[target_feature(enable = "avx512f")]
#[inline(never)]
pub(crate) unsafe fn xor_batch16(state: &mut State, buf: *mut u8) {
    setup16!(
        state, w, bcast, ctr, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
    );
    // Fully unrolled: the rolled loop's back-edge broke LLVM's scheduling
    // of the 16 independent register chains.
    dr16!(
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
    );
    dr16!(
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
    );
    dr16!(
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
    );
    dr16!(
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
    );
    dr16!(
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
    );
    dr16!(
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
    );
    dr16!(
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
    );
    dr16!(
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
    );
    dr16!(
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
    );
    dr16!(
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
    );
    emit16!(
        buf, bcast, ctr, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
    );
}

/// Fused seal bulk: loops [`xor_batch16_fused_seal`]'s interleave over the
/// whole bulk region, keeping the Poly1305 H/powers and (via the macro
/// expansion) the cipher schedule entirely in registers across batches —
/// no per-batch call, state reload, or loop-branch in between. Covers
/// `[off, len)` in 1024-byte steps while `poly_off + 1024 <= off` (the MAC
/// only reads ciphertext written by earlier iterations).
///
/// Caller contract: `off`/`poly_off` 64-byte aligned, poly window cache
/// empty (`pending_blocks() == 0`).
#[target_feature(enable = "avx512f,avx512ifma")]
#[inline(never)]
pub(crate) unsafe fn xor_batch16_seal_bulk(
    state: &mut State,
    msg: *mut u8,
    mut off: usize,
    mut poly_off: usize,
    len: usize,
    poly: &mut crate::poly1305::ifma::IfmaPoly,
) -> (usize, usize) {
    use crate::poly1305::ifma::{load8, mul_reduce};

    poly.ensure_stream();
    // SAFETY: just ensured.
    let st = unsafe { poly.stream.as_mut().unwrap_unchecked() };
    let (mut h0, mut h1, mut h2) = (st.h0, st.h1, st.h2);
    let (r0, r1, r2, s1, s2) = (
        st.powers.b_r0,
        st.powers.b_r1,
        st.powers.b_r2,
        st.powers.b_s1,
        st.powers.b_s2,
    );

    // Hoisted broadcast state: only the counter row changes per iteration.
    // (LLVM rematerializes these `o` registers into folded broadcast
    // load-ops instead of occupying 16 registers.)
    let w = state.words;
    let bcast = |i: usize| _mm512_set1_epi32(w[i] as i32);
    let (o0, o1, o2, o3) = (bcast(0), bcast(1), bcast(2), bcast(3));
    let (o4, o5, o6, o7) = (bcast(4), bcast(5), bcast(6), bcast(7));
    let (o8, o9, o10, o11) = (bcast(8), bcast(9), bcast(10), bcast(11));
    let (o13, o14, o15) = (bcast(13), bcast(14), bcast(15));
    let mut ctr = _mm512_add_epi32(
        bcast(12),
        _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    );
    let sixteen = _mm512_set1_epi32(16);
    let mut iters = 0u32;

    while len - off >= 1024 && poly_off + 1024 <= off {
        let mut pptr = msg.add(poly_off);
        reload16!(
            o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, ctr, o13, o14, o15, x0, x1, x2, x3,
            x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
        );

        macro_rules! poly_round {
            () => {{
                let (t0, t1, t2) = load8(pptr, pptr.add(64));
                let (g0, g1, g2) = mul_reduce(h0, h1, h2, r0, r1, r2, s1, s2);
                h0 = _mm512_add_epi64(g0, t0);
                h1 = _mm512_add_epi64(g1, t1);
                h2 = _mm512_add_epi64(g2, t2);
                pptr = pptr.add(128);
            }};
        }

        dr16!(
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
        );
        poly_round!();
        dr16!(
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
        );
        poly_round!();
        dr16!(
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
        );
        poly_round!();
        dr16!(
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
        );
        poly_round!();
        dr16!(
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
        );
        poly_round!();
        dr16!(
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
        );
        poly_round!();
        dr16!(
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
        );
        poly_round!();
        dr16!(
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
        );
        poly_round!();
        dr16!(
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
        );
        dr16!(
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
        );
        let _ = pptr;

        emit16o!(
            msg.add(off),
            o0,
            o1,
            o2,
            o3,
            o4,
            o5,
            o6,
            o7,
            o8,
            o9,
            o10,
            o11,
            ctr,
            o13,
            o14,
            o15,
            x0,
            x1,
            x2,
            x3,
            x4,
            x5,
            x6,
            x7,
            x8,
            x9,
            x10,
            x11,
            x12,
            x13,
            x14,
            x15
        );
        off += 1024;
        poly_off += 1024;
        iters += 1;
        ctr = _mm512_add_epi32(ctr, sixteen);
    }

    state.advance(16u32.wrapping_mul(iters));
    st.h0 = h0;
    st.h1 = h1;
    st.h2 = h2;
    (off, poly_off)
}

/// Fused open bulk: same interleave as [`xor_batch16_seal_bulk`], but the
/// MAC reads `[poly_off, poly_off + 1024)` — which overlaps the region THIS
/// call is about to overwrite. Sound because every poly load happens before
/// the terminal emit stores (in program order), and it never reads bytes
/// already overwritten by earlier calls (the cursor relation `poly_off >
/// off` is maintained by the engine's open prologue).
#[target_feature(enable = "avx512f,avx512ifma")]
#[inline(never)]
pub(crate) unsafe fn xor_batch16_open_bulk(
    state: &mut State,
    msg: *mut u8,
    mut off: usize,
    mut poly_off: usize,
    len: usize,
    poly: &mut crate::poly1305::ifma::IfmaPoly,
) -> (usize, usize) {
    use crate::poly1305::ifma::{load8, mul_reduce};

    poly.ensure_stream();
    // SAFETY: just ensured.
    let st = unsafe { poly.stream.as_mut().unwrap_unchecked() };
    let (mut h0, mut h1, mut h2) = (st.h0, st.h1, st.h2);
    let (r0, r1, r2, s1, s2) = (
        st.powers.b_r0,
        st.powers.b_r1,
        st.powers.b_r2,
        st.powers.b_s1,
        st.powers.b_s2,
    );

    let w = state.words;
    let bcast = |i: usize| _mm512_set1_epi32(w[i] as i32);
    let (o0, o1, o2, o3) = (bcast(0), bcast(1), bcast(2), bcast(3));
    let (o4, o5, o6, o7) = (bcast(4), bcast(5), bcast(6), bcast(7));
    let (o8, o9, o10, o11) = (bcast(8), bcast(9), bcast(10), bcast(11));
    let (o13, o14, o15) = (bcast(13), bcast(14), bcast(15));
    let mut ctr = _mm512_add_epi32(
        bcast(12),
        _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    );
    let sixteen = _mm512_set1_epi32(16);
    let mut iters = 0u32;

    while len - off >= 1024 && len - poly_off >= 1024 {
        let mut pptr = msg.add(poly_off);
        reload16!(
            o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, ctr, o13, o14, o15, x0, x1, x2, x3,
            x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
        );

        macro_rules! poly_round {
            () => {{
                let (t0, t1, t2) = load8(pptr, pptr.add(64));
                let (g0, g1, g2) = mul_reduce(h0, h1, h2, r0, r1, r2, s1, s2);
                h0 = _mm512_add_epi64(g0, t0);
                h1 = _mm512_add_epi64(g1, t1);
                h2 = _mm512_add_epi64(g2, t2);
                pptr = pptr.add(128);
            }};
        }

        dr16!(
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
        );
        poly_round!();
        dr16!(
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
        );
        poly_round!();
        dr16!(
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
        );
        poly_round!();
        dr16!(
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
        );
        poly_round!();
        dr16!(
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
        );
        poly_round!();
        dr16!(
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
        );
        poly_round!();
        dr16!(
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
        );
        poly_round!();
        dr16!(
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
        );
        poly_round!();
        dr16!(
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
        );
        dr16!(
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
        );
        let _ = pptr;

        emit16o!(
            msg.add(off),
            o0,
            o1,
            o2,
            o3,
            o4,
            o5,
            o6,
            o7,
            o8,
            o9,
            o10,
            o11,
            ctr,
            o13,
            o14,
            o15,
            x0,
            x1,
            x2,
            x3,
            x4,
            x5,
            x6,
            x7,
            x8,
            x9,
            x10,
            x11,
            x12,
            x13,
            x14,
            x15
        );
        off += 1024;
        poly_off += 1024;
        iters += 1;
        ctr = _mm512_add_epi32(ctr, sixteen);
    }

    state.advance(16u32.wrapping_mul(iters));
    st.h0 = h0;
    st.h1 = h1;
    st.h2 = h2;
    (off, poly_off)
}

/// Single-block (64-byte) kernel in XMM registers.
#[cfg_attr(debug_assertions, inline)]
#[cfg_attr(not(debug_assertions), inline(always))]
pub(crate) unsafe fn xor_single(state: &mut State, buf: *mut u8) {
    let p = state.words.as_ptr().cast::<__m128i>();
    let v0 = _mm_loadu_si128(p.add(0));
    let v1 = _mm_loadu_si128(p.add(1));
    let v2 = _mm_loadu_si128(p.add(2));
    let v3 = _mm_loadu_si128(p.add(3));
    let mut a = v0;
    let mut b = v1;
    let mut c = v2;
    let mut d = v3;
    for _ in 0..10 {
        // column + diagonal rounds, diagonal via the same lane shuffles
        quarter(&mut a, &mut b, &mut c, &mut d);
        c = _mm_shuffle_epi32::<0b_00_11_10_01>(c);
        d = _mm_shuffle_epi32::<0b_01_00_11_10>(d);
        a = _mm_shuffle_epi32::<0b_10_01_00_11>(a);
        quarter(&mut a, &mut b, &mut c, &mut d);
        c = _mm_shuffle_epi32::<0b_10_01_00_11>(c);
        d = _mm_shuffle_epi32::<0b_01_00_11_10>(d);
        a = _mm_shuffle_epi32::<0b_00_11_10_01>(a);
    }
    a = _mm_add_epi32(a, v0);
    b = _mm_add_epi32(b, v1);
    c = _mm_add_epi32(c, v2);
    d = _mm_add_epi32(d, v3);
    state.advance(1);

    // rows hold contiguous words already: block = [a|b|c|d] sequentially
    let pt0 = _mm_loadu_si128(buf.cast());
    let pt1 = _mm_loadu_si128(buf.add(16).cast());
    let pt2 = _mm_loadu_si128(buf.add(32).cast());
    let pt3 = _mm_loadu_si128(buf.add(48).cast());
    _mm_storeu_si128(buf.cast(), _mm_xor_si128(pt0, a));
    _mm_storeu_si128(buf.add(16).cast(), _mm_xor_si128(pt1, b));
    _mm_storeu_si128(buf.add(32).cast(), _mm_xor_si128(pt2, c));
    _mm_storeu_si128(buf.add(48).cast(), _mm_xor_si128(pt3, d));
}

#[cfg_attr(debug_assertions, inline)]
#[cfg_attr(not(debug_assertions), inline(always))]
unsafe fn quarter(a: &mut __m128i, b: &mut __m128i, c: &mut __m128i, d: &mut __m128i) {
    *a = _mm_add_epi32(*a, *b);
    *d = _mm_xor_si128(*d, *a);
    *d = _mm_rol_epi32::<16>(*d);
    *c = _mm_add_epi32(*c, *d);
    *b = _mm_xor_si128(*b, *c);
    *b = _mm_rol_epi32::<12>(*b);
    *a = _mm_add_epi32(*a, *b);
    *d = _mm_xor_si128(*d, *a);
    *d = _mm_rol_epi32::<8>(*d);
    *c = _mm_add_epi32(*c, *d);
    *b = _mm_xor_si128(*b, *c);
    *b = _mm_rol_epi32::<7>(*b);
}

/// Generate exactly one keystream block (no XOR, no advance). Test
/// reference target only — the engine uses the fused [`gen_key_xor2`].
#[cfg(test)]
#[cfg_attr(debug_assertions, inline)]
#[cfg_attr(not(debug_assertions), inline(always))]
pub(crate) unsafe fn gen_block(state: &State, out: &mut [u8; BLOCK]) {
    let p = state.words.as_ptr().cast::<__m128i>();
    let v0 = _mm_loadu_si128(p.add(0));
    let v1 = _mm_loadu_si128(p.add(1));
    let v2 = _mm_loadu_si128(p.add(2));
    let v3 = _mm_loadu_si128(p.add(3));
    let mut a = v0;
    let mut b = v1;
    let mut c = v2;
    let mut d = v3;
    for _ in 0..10 {
        quarter(&mut a, &mut b, &mut c, &mut d);
        c = _mm_shuffle_epi32::<0b_00_11_10_01>(c);
        d = _mm_shuffle_epi32::<0b_01_00_11_10>(d);
        a = _mm_shuffle_epi32::<0b_10_01_00_11>(a);
        quarter(&mut a, &mut b, &mut c, &mut d);
        c = _mm_shuffle_epi32::<0b_10_01_00_11>(c);
        d = _mm_shuffle_epi32::<0b_01_00_11_10>(d);
        a = _mm_shuffle_epi32::<0b_00_11_10_01>(a);
    }
    a = _mm_add_epi32(a, v0);
    b = _mm_add_epi32(b, v1);
    c = _mm_add_epi32(c, v2);
    d = _mm_add_epi32(d, v3);
    _mm_storeu_si128(out.as_mut_ptr().cast(), a);
    _mm_storeu_si128(out.as_mut_ptr().add(16).cast(), b);
    _mm_storeu_si128(out.as_mut_ptr().add(32).cast(), c);
    _mm_storeu_si128(out.as_mut_ptr().add(48).cast(), d);
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::chacha::State;

    /// Tests below call AVX-512 kernels directly (bypassing runtime dispatch);
    /// skip on CPUs without the features instead of executing illegal
    /// instructions.
    fn skip_unsupported() -> bool {
        !(std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512vl"))
    }

    #[test]
    fn gen_block_matches_soft() {
        if skip_unsupported() {
            return;
        }
        let key: [u8; 32] = core::array::from_fn(|i| (i * 7 + 3) as u8);
        let nonce: [u8; 12] = core::array::from_fn(|i| (i * 11 + 1) as u8);
        for ctr in [0u32, 1, 2, 0xffff_ffffu32.wrapping_add(1)] {
            let mut st = State::new_ietf(&key, &nonce);
            st.words[12] = ctr;
            let mut soft = [0u8; 64];
            let mut fast = [0u8; 64];
            unsafe {
                crate::chacha::soft::gen_block(&st, &mut soft);
                gen_block(&st, &mut fast);
            }
            assert_eq!(soft, fast, "ctr {ctr}");
        }
    }

    #[test]
    fn xor_batch16_matches_soft() {
        if skip_unsupported() {
            return;
        }
        let key: [u8; 32] = core::array::from_fn(|i| (i * 13 + 5) as u8);
        let nonce: [u8; 12] = core::array::from_fn(|i| (i * 17 + 9) as u8);
        for skip in [0u32, 1, 7] {
            let mut st = State::new_ietf(&key, &nonce);
            st.advance(skip);
            let mut ref_buf: Vec<u8> = (0..1024u32).map(|i| (i * 31 + 7) as u8).collect();
            let mut fast_buf = ref_buf.clone();
            let mut ref_st = st.clone_struct();
            unsafe {
                crate::chacha::soft::xor(&mut ref_st, &mut ref_buf);
                xor_batch16(&mut st, fast_buf.as_mut_ptr());
            }
            for b in 0..16 {
                assert_eq!(
                    &ref_buf[b * 64..(b + 1) * 64],
                    &fast_buf[b * 64..(b + 1) * 64],
                    "skip {skip} block {b}"
                );
            }
            assert_eq!(ref_st.words, st.words);
        }
    }

    #[test]
    fn xor_batch8_matches_soft() {
        if skip_unsupported() {
            return;
        }
        let key: [u8; 32] = core::array::from_fn(|i| (i * 13 + 5) as u8);
        let nonce: [u8; 12] = core::array::from_fn(|i| (i * 17 + 9) as u8);
        let mut st = State::new_ietf(&key, &nonce);
        let mut ref_buf: Vec<u8> = (0..512u32).map(|i| (i * 31 + 7) as u8).collect();
        let mut fast_buf = ref_buf.clone();
        let mut ref_st = st.clone_struct();
        unsafe {
            crate::chacha::soft::xor(&mut ref_st, &mut ref_buf);
            xor_batch8(&mut st, fast_buf.as_mut_ptr());
        }
        assert_eq!(ref_buf, fast_buf);
        assert_eq!(ref_st.words, st.words);
    }

    #[test]
    fn xor_quad_and_single_match_soft() {
        if skip_unsupported() {
            return;
        }
        let key: [u8; 32] = core::array::from_fn(|i| (i * 13 + 5) as u8);
        let nonce: [u8; 12] = core::array::from_fn(|i| (i * 17 + 9) as u8);
        let mut st = State::new_ietf(&key, &nonce);
        let mut ref_buf: Vec<u8> = (0..192u32).map(|i| (i * 31 + 7) as u8).collect();
        let mut fast_buf = ref_buf.clone();
        let mut ref_st = st.clone_struct();
        unsafe {
            crate::chacha::soft::xor(&mut ref_st, &mut ref_buf[..128]);
            xor_quad(&mut st, fast_buf.as_mut_ptr());
        }
        assert_eq!(&ref_buf[..128], &fast_buf[..128]);
        unsafe {
            crate::chacha::soft::xor(&mut ref_st, &mut ref_buf[128..]);
            xor_single(&mut st, fast_buf.as_mut_ptr().add(128));
        }
        assert_eq!(ref_buf, fast_buf);
        assert_eq!(ref_st.words, st.words);
    }

    #[test]
    fn xor_quad2_matches_soft() {
        if skip_unsupported() {
            return;
        }
        let key: [u8; 32] = core::array::from_fn(|i| (i * 13 + 5) as u8);
        let nonce: [u8; 12] = core::array::from_fn(|i| (i * 17 + 9) as u8);
        for (skip, len) in [(0usize, 256usize), (3, 256), (5, 448)] {
            let mut st = State::new_ietf(&key, &nonce);
            st.advance(skip as u32);
            let mut ref_st = st.clone_struct();
            let mut fast: Vec<u8> = (0..len).map(|i| (i * 37 + 3) as u8).collect();
            let mut expect = fast.clone();
            unsafe {
                let mut off = 0usize;
                while fast.len() - off >= 256 {
                    xor_quad2(&mut st, fast[off..].as_mut_ptr());
                    off += 256;
                }
                while fast.len() - off >= 128 {
                    xor_quad(&mut st, fast[off..].as_mut_ptr());
                    off += 128;
                }
                while fast.len() - off >= BLOCK {
                    xor_single(&mut st, fast[off..].as_mut_ptr());
                    off += BLOCK;
                }
                crate::chacha::soft::xor(&mut ref_st, &mut expect);
            }
            assert_eq!(expect, fast, "skip {skip} len {len}");
            assert_eq!(ref_st.words, st.words);
        }
    }
}

#[cfg(test)]
mod tail_tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::chacha::State;

    /// Reproduce the engine's 66-byte tail: 48-byte alignment chunk then
    /// chacha_xor on the remaining 66 bytes.
    #[test]
    fn tail_66() {
        if !(std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512vl"))
        {
            return;
        }
        let key: [u8; 32] = core::array::from_fn(|i| (i * 13 + 5) as u8);
        let nonce: [u8; 12] = core::array::from_fn(|i| (i * 17 + 9) as u8);
        let mut st = State::new_ietf(&key, &nonce);
        st.advance(1);
        let mut ref_st = st.clone_struct();
        let data: Vec<u8> = (0..114u32).map(|i| (i * 31 + 7) as u8).collect();
        let mut ref_buf = data.clone();
        let mut fast_buf = data;
        unsafe {
            crate::chacha::soft::xor(&mut ref_st, &mut ref_buf[..48]);
            crate::chacha::soft::xor(&mut ref_st, &mut ref_buf[48..]);
            // fast: engine prologue XORs the 48-byte alignment chunk through
            // chacha_xor (which falls back to soft for sub-block sizes)
            let buf = &mut fast_buf;
            crate::chacha::soft::xor(&mut st, &mut buf[..48]);
            let mut off = 48;
            while buf.len() - off >= 512 {
                xor_batch8(&mut st, buf[off..].as_mut_ptr());
                off += 512;
            }
            while buf.len() - off >= 128 {
                xor_quad(&mut st, buf[off..].as_mut_ptr());
                off += 128;
            }
            while buf.len() - off >= 64 {
                xor_single(&mut st, buf[off..].as_mut_ptr());
                off += 64;
            }
            if off < buf.len() {
                crate::chacha::soft::xor(&mut st, &mut buf[off..]);
            }
        }
        assert_eq!(ref_buf, fast_buf);
    }
}

#[cfg(test)]
mod quad_ctr_tests {
    use super::*;
    use crate::chacha::State;

    #[test]
    fn quad_at_various_counters() {
        if !(std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512vl"))
        {
            return;
        }
        let key: [u8; 32] = core::array::from_fn(|i| (i * 7 + 11) as u8);
        let nonce: [u8; 12] = core::array::from_fn(|i| (i * 5 + 2) as u8);
        for ctr in [0u32, 1, 2, 5, 0x7fff_ffff] {
            let mut st = State::new_ietf(&key, &nonce);
            st.words[12] = ctr;
            let mut ref_st = st.clone_struct();
            let mut fast = [0u8; 128];
            let mut expect = [0u8; 128];
            unsafe {
                xor_quad(&mut st, fast.as_mut_ptr());
                crate::chacha::soft::xor(&mut ref_st, &mut expect);
            }
            assert_eq!(expect, fast, "ctr {ctr}");
        }
    }

    #[test]
    fn three_quads_then_single() {
        if !(std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512vl"))
        {
            return;
        }
        let key: [u8; 32] = core::array::from_fn(|i| (i * 7 + 11) as u8);
        let nonce: [u8; 12] = core::array::from_fn(|i| (i * 5 + 2) as u8);
        let mut st = State::new_ietf(&key, &nonce);
        st.advance(2);
        let mut ref_st = st.clone_struct();
        let mut fast = [0u8; 448];
        let mut expect = [0u8; 448];
        unsafe {
            crate::chacha::soft::xor(&mut ref_st, &mut expect);
            xor_quad(&mut st, fast.as_mut_ptr());
            xor_quad(&mut st, fast.as_mut_ptr().add(128));
            xor_quad(&mut st, fast.as_mut_ptr().add(256));
            xor_single(&mut st, fast.as_mut_ptr().add(384));
        }
        assert_eq!(expect, fast);
    }
}
