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
#[cfg_attr(not(debug_assertions), inline(always))]
unsafe fn emit_lo(a: __m256i, b: __m256i, block: usize) -> __m256i {
    match block {
        0 => _mm256_permute2f128_si256::<0x20>(a, b),
        _ => _mm256_permute2f128_si256::<0x31>(a, b),
    }
}

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

// ── Fused medium-size kernels (512-byte quad batches + IFMA MAC) ──
//
// The 16-block fused loops above only engage once ≥ 1024 bytes remain past
// the engine prologue (message ≳ 2 KiB); below that the generic tail ran
// cipher and MAC serially — the dominant sub-2 KiB cost (see
// perf/1kib-gap-analysis.md). These kernels port BoringSSL's
// `chacha20_poly1305_seal_avx2` sub-batch shape: a cipher-ahead batch seeds
// a one-batch MAC lag, then 8-block quad rounds run with IFMA `mul_reduce`
// rounds woven between the double rounds, reading only ciphertext written
// by earlier batches (no store→load-forwarding hazards). A final partial
// batch stores through byte masks, so any tail length is covered without a
// scalar fallback.

/// One double round over four quads (8 blocks) — the [`rounds4`] loop body.
macro_rules! dr8q {
    (
        $a0:ident,
        $b0:ident,
        $c0:ident,
        $d0:ident,
        $a1:ident,
        $b1:ident,
        $c1:ident,
        $d1:ident,
        $a2:ident,
        $b2:ident,
        $c2:ident,
        $d2:ident,
        $a3:ident,
        $b3:ident,
        $c3:ident,
        $d3:ident
    ) => {{
        qr!($a0, $b0, $c0, $d0);
        qr!($a1, $b1, $c1, $d1);
        qr!($a2, $b2, $c2, $d2);
        qr!($a3, $b3, $c3, $d3);
        to_cols!($a0, $b0, $c0, $d0);
        to_cols!($a1, $b1, $c1, $d1);
        to_cols!($a2, $b2, $c2, $d2);
        to_cols!($a3, $b3, $c3, $d3);
        qr!($a0, $b0, $c0, $d0);
        qr!($a1, $b1, $c1, $d1);
        qr!($a2, $b2, $c2, $d2);
        qr!($a3, $b3, $c3, $d3);
        to_rows!($a0, $b0, $c0, $d0);
        to_rows!($a1, $b1, $c1, $d1);
        to_rows!($a2, $b2, $c2, $d2);
        to_rows!($a3, $b3, $c3, $d3);
    }};
}

/// 8-block quad rounds with up to eight IFMA poly rounds woven after
/// double-rounds 1..=8, then the feed-forward against the original row and
/// counter registers. `$n` (runtime, 0..=8) of the `$pr` slots fire; `$pr`
/// is the caller's poly-round macro (capturing `pptr`, `h0..h2` and the
/// hoisted power registers).
macro_rules! rounds8_fused {
    (
        $n:expr,
        $v0:expr,
        $v1:expr,
        $v2:expr,
        $pr:ident,
        $od0:expr,
        $od1:expr,
        $od2:expr,
        $od3:expr,
        $a0:ident,
        $b0:ident,
        $c0:ident,
        $d0:ident,
        $a1:ident,
        $b1:ident,
        $c1:ident,
        $d1:ident,
        $a2:ident,
        $b2:ident,
        $c2:ident,
        $d2:ident,
        $a3:ident,
        $b3:ident,
        $c3:ident,
        $d3:ident
    ) => {{
        let n = $n;
        dr8q!(
            $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, $a3, $b3, $c3, $d3
        );
        if n > 0 {
            $pr!();
        }
        dr8q!(
            $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, $a3, $b3, $c3, $d3
        );
        if n > 1 {
            $pr!();
        }
        dr8q!(
            $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, $a3, $b3, $c3, $d3
        );
        if n > 2 {
            $pr!();
        }
        dr8q!(
            $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, $a3, $b3, $c3, $d3
        );
        if n > 3 {
            $pr!();
        }
        dr8q!(
            $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, $a3, $b3, $c3, $d3
        );
        if n > 4 {
            $pr!();
        }
        dr8q!(
            $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, $a3, $b3, $c3, $d3
        );
        if n > 5 {
            $pr!();
        }
        dr8q!(
            $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, $a3, $b3, $c3, $d3
        );
        if n > 6 {
            $pr!();
        }
        dr8q!(
            $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, $a3, $b3, $c3, $d3
        );
        if n > 7 {
            $pr!();
        }
        dr8q!(
            $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, $a3, $b3, $c3, $d3
        );
        dr8q!(
            $a0, $b0, $c0, $d0, $a1, $b1, $c1, $d1, $a2, $b2, $c2, $d2, $a3, $b3, $c3, $d3
        );
        $a0 = _mm256_add_epi32($a0, $v0);
        $b0 = _mm256_add_epi32($b0, $v1);
        $c0 = _mm256_add_epi32($c0, $v2);
        $d0 = _mm256_add_epi32($d0, $od0);
        $a1 = _mm256_add_epi32($a1, $v0);
        $b1 = _mm256_add_epi32($b1, $v1);
        $c1 = _mm256_add_epi32($c1, $v2);
        $d1 = _mm256_add_epi32($d1, $od1);
        $a2 = _mm256_add_epi32($a2, $v0);
        $b2 = _mm256_add_epi32($b2, $v1);
        $c2 = _mm256_add_epi32($c2, $v2);
        $d2 = _mm256_add_epi32($d2, $od2);
        $a3 = _mm256_add_epi32($a3, $v0);
        $b3 = _mm256_add_epi32($b3, $v1);
        $c3 = _mm256_add_epi32($c3, $v2);
        $d3 = _mm256_add_epi32($d3, $od3);
    }};
}

/// XOR-store one 64-byte block (row registers of one quad, `lane` 0|1) with
/// `k` bytes written (`k == BLOCK`: plain stores; `k < BLOCK`: AVX-512VL
/// masked stores, so partial tails never touch memory past the message).
#[cfg_attr(not(debug_assertions), inline(always))]
unsafe fn store_blk(
    buf: *mut u8,
    idx: usize,
    a: __m256i,
    b: __m256i,
    c: __m256i,
    d: __m256i,
    lane: usize,
    k: usize,
) {
    unsafe {
        let lo = emit_lo(a, b, lane);
        let hi = emit_hi(c, d, lane);
        let p = buf.add(idx * BLOCK);
        if k == BLOCK {
            _mm256_storeu_si256(p.cast(), _mm256_xor_si256(_mm256_loadu_si256(p.cast()), lo));
            _mm256_storeu_si256(
                p.add(32).cast(),
                _mm256_xor_si256(_mm256_loadu_si256(p.add(32).cast()), hi),
            );
        } else {
            let (k0, k1) = (k.min(32), k - k.min(32));
            if k0 > 0 {
                let m = ((1u64 << k0) - 1) as __mmask32;
                _mm256_mask_storeu_epi8(
                    p.cast(),
                    m,
                    _mm256_xor_si256(_mm256_loadu_si256(p.cast()), lo),
                );
            }
            if k1 > 0 {
                let m = ((1u64 << k1) - 1) as __mmask32;
                _mm256_mask_storeu_epi8(
                    p.add(32).cast(),
                    m,
                    _mm256_xor_si256(_mm256_loadu_si256(p.add(32).cast()), hi),
                );
            }
        }
    }
}

/// XOR-store both blocks of one finalized quad (128 bytes at `buf`).
#[cfg_attr(not(debug_assertions), inline(always))]
unsafe fn store_quad(buf: *mut u8, a: __m256i, b: __m256i, c: __m256i, d: __m256i) {
    unsafe {
        store_blk(buf, 0, a, b, c, d, 0, BLOCK);
        store_blk(buf, 1, a, b, c, d, 1, BLOCK);
    }
}

/// XOR-store `nfull` whole blocks plus a `k`-byte masked partial block
/// (`k < 64`) from four finalized quads, covering any 1..=511-byte tail.
/// The runtime `nfull` indexing forces a one-time register spill — fine at
/// once per message.
#[cfg_attr(not(debug_assertions), inline(always))]
#[allow(clippy::too_many_arguments)]
unsafe fn emit_partial(
    buf: *mut u8,
    nfull: usize,
    k: usize,
    a0: __m256i,
    b0: __m256i,
    c0: __m256i,
    d0: __m256i,
    a1: __m256i,
    b1: __m256i,
    c1: __m256i,
    d1: __m256i,
    a2: __m256i,
    b2: __m256i,
    c2: __m256i,
    d2: __m256i,
    a3: __m256i,
    b3: __m256i,
    c3: __m256i,
    d3: __m256i,
) {
    let quads: [(__m256i, __m256i, __m256i, __m256i); 4] = [
        (a0, b0, c0, d0),
        (a1, b1, c1, d1),
        (a2, b2, c2, d2),
        (a3, b3, c3, d3),
    ];
    for b in 0..nfull {
        let (a, bb, c, d) = quads[b / 2];
        unsafe { store_blk(buf, b, a, bb, c, d, b % 2, BLOCK) };
    }
    if k > 0 {
        let (a, bb, c, d) = quads[nfull / 2];
        unsafe { store_blk(buf, nfull, a, bb, c, d, nfull % 2, k) };
    }
}

/// Fused medium-size seal (requires AVX-512F+VL+IFMA at runtime).
///
/// Phase A encrypts one 512-byte batch ahead (seeding the one-batch MAC lag
/// BoringSSL's assembly uses), the cached MAC blocks are folded against
/// already-written ciphertext, and a final partial batch runs the quad
/// rounds with the remaining IFMA MAC rounds woven between the double
/// rounds. Its last block stores through a byte mask, so any tail length is
/// covered without a scalar fallback; the counter advances by the blocks
/// actually stored.
///
/// Caller contract (engine-maintained): `off < len`, `poly_off <= off`, the
/// whole-block cache is a multiple of 4, and the MAC stream sits at a
/// 64-byte boundary at `poly_off`.
#[target_feature(enable = "avx512f,avx512vl,avx512ifma")]
#[inline(never)]
pub(crate) unsafe fn seal_medium(
    state: &mut State,
    msg: *mut u8,
    mut off: usize,
    mut poly_off: usize,
    len: usize,
    poly: &mut crate::poly1305::ifma::IfmaPoly,
) -> (usize, usize) {
    use crate::poly1305::{
        Backend,
        ifma::{load8, mul_reduce},
    };

    // Phase A: cipher-ahead.
    if len - off >= 512 {
        unsafe { xor_batch8(state, msg.add(off)) };
        off += 512;
    }

    // Fold the cached half/whole group so the fused rounds below load
    // straight off the message buffer. The pairing window must already be
    // ciphertext — written by the prologue or phase A — hence the `<= off`
    // guard (without phase A there is nothing fresh to pair with; the
    // generic tail folds the cache instead).
    match poly.pending_blocks() {
        4 if poly_off + 64 <= off => {
            poly.absorb_blocks(unsafe { core::slice::from_raw_parts(msg.add(poly_off), 64) });
            poly_off += 64;
        },
        8 => {
            poly.absorb_blocks(unsafe { core::slice::from_raw_parts(msg.add(poly_off), 0) });
        },
        _ => {},
    }
    // Create the streaming state (r^8 powers) up front: the key-power
    // computation is independent of the cipher, so out-of-order execution
    // hides it under the kernel rounds below instead of sitting serially
    // between the cipher and the first MAC fold.
    poly.ensure_stream();

    let r = len - off;
    if r == 0 {
        return (off, poly_off);
    }
    let nfull = r / BLOCK;
    let k = r % BLOCK;
    let rounds = ((off - poly_off) / 128).min((len - poly_off) / 128).min(8);
    // Below ~3 stored blocks the cascade tail beats an 8-block computation
    // with most of its results discarded.
    if r < 192 && rounds == 0 {
        return (off, poly_off);
    }
    // SAFETY: just ensured.
    let (mut h0, mut h1, mut h2, r0, r1, r2, s1, s2) = {
        let st = unsafe { poly.stream.as_ref().unwrap_unchecked() };
        (
            st.h0,
            st.h1,
            st.h2,
            st.powers.b_r0,
            st.powers.b_r1,
            st.powers.b_r2,
            st.powers.b_s1,
            st.powers.b_s2,
        )
    };
    let v = rows(state);
    let mut pptr = msg.add(poly_off);
    macro_rules! pr {
        () => {{
            let (t0, t1, t2) = load8(pptr, pptr.add(64));
            let (g0, g1, g2) = mul_reduce(h0, h1, h2, r0, r1, r2, s1, s2);
            h0 = _mm512_add_epi64(g0, t0);
            h1 = _mm512_add_epi64(g1, t1);
            h2 = _mm512_add_epi64(g2, t2);
            pptr = pptr.add(128);
        }};
    }
    let base = state.words[12];
    let od0 = ctr_row(state, base, 0);
    let od1 = ctr_row(state, base, 1);
    let od2 = ctr_row(state, base, 2);
    let od3 = ctr_row(state, base, 3);
    state.advance((nfull + usize::from(k > 0)) as u32);
    let (mut a0, mut b0, mut c0, mut d0) = (v[0], v[1], v[2], od0);
    let (mut a1, mut b1, mut c1, mut d1) = (v[0], v[1], v[2], od1);
    let (mut a2, mut b2, mut c2, mut d2) = (v[0], v[1], v[2], od2);
    let (mut a3, mut b3, mut c3, mut d3) = (v[0], v[1], v[2], od3);
    rounds8_fused!(
        rounds, v[0], v[1], v[2], pr, od0, od1, od2, od3, a0, b0, c0, d0, a1, b1, c1, d1, a2, b2,
        c2, d2, a3, b3, c3, d3
    );
    // pptr's final increment (inside `pr`) has no later consumer here.
    let _ = pptr;
    emit_partial(
        msg.add(off),
        nfull,
        k,
        a0,
        b0,
        c0,
        d0,
        a1,
        b1,
        c1,
        d1,
        a2,
        b2,
        c2,
        d2,
        a3,
        b3,
        c3,
        d3,
    );
    off = len;
    poly_off += 128 * rounds;
    // SAFETY: stream ensured above.
    let st = unsafe { poly.stream.as_mut().unwrap_unchecked() };
    st.h0 = h0;
    st.h1 = h1;
    st.h2 = h2;
    (off, poly_off)
}

/// Fused medium-size open (requires AVX-512F+VL+IFMA at runtime). The MAC
/// leads (loads pristine ciphertext before the xor cursor reaches it), so
/// the steady iterations have no ordering hazards at all: poly rounds woven
/// into the cipher rounds load windows at/past `off` that this iteration's
/// stores only touch afterwards. When the MAC runs dry before the cipher,
/// its 64-aligned bulk is absorbed wholesale and plain 8-block batches
/// finish the encryption; a masked partial batch covers the tail.
///
/// Caller contract: `poly_off >= off`, whole-block cache `% 4 == 0`, MAC
/// stream at a 64-byte boundary at `poly_off`.
#[target_feature(enable = "avx512f,avx512vl,avx512ifma")]
#[inline(never)]
pub(crate) unsafe fn open_medium(
    state: &mut State,
    msg: *mut u8,
    mut off: usize,
    mut poly_off: usize,
    len: usize,
    poly: &mut crate::poly1305::ifma::IfmaPoly,
) -> (usize, usize) {
    use crate::poly1305::{
        Backend,
        ifma::{load8, mul_reduce},
    };

    // Fold the cached half/whole group; the pairing window is pristine
    // ciphertext (the MAC leads the xor cursor).
    match poly.pending_blocks() {
        4 if poly_off + 64 <= len => {
            poly.absorb_blocks(unsafe { core::slice::from_raw_parts(msg.add(poly_off), 64) });
            poly_off += 64;
        },
        8 => {
            poly.absorb_blocks(unsafe { core::slice::from_raw_parts(msg.add(poly_off), 0) });
        },
        _ => {},
    }
    // See seal_medium: create the streaming state up front so the r^8
    // powers overlap the cipher rounds.
    poly.ensure_stream();

    // SAFETY: just ensured.
    let (mut h0, mut h1, mut h2, r0, r1, r2, s1, s2) = {
        let st = unsafe { poly.stream.as_ref().unwrap_unchecked() };
        (
            st.h0,
            st.h1,
            st.h2,
            st.powers.b_r0,
            st.powers.b_r1,
            st.powers.b_r2,
            st.powers.b_s1,
            st.powers.b_s2,
        )
    };
    let v = rows(state);
    let mut pptr = msg.add(poly_off);
    macro_rules! pr {
        () => {{
            let (t0, t1, t2) = load8(pptr, pptr.add(64));
            let (g0, g1, g2) = mul_reduce(h0, h1, h2, r0, r1, r2, s1, s2);
            h0 = _mm512_add_epi64(g0, t0);
            h1 = _mm512_add_epi64(g1, t1);
            h2 = _mm512_add_epi64(g2, t2);
            pptr = pptr.add(128);
        }};
    }

    // Steady fused iterations: 4 MAC windows (512 bytes) per 512 cipher
    // bytes, 1:1 like the 16-block loop. Each iteration is self-covering:
    // its loads ([poly_off, poly_off+512)) precede its stores
    // ([off, off+512)) in program order, and everything the stores touch
    // below `poly_off` was absorbed by earlier iterations — so the MAC
    // never reads decrypted bytes. `pptr` stays in lockstep with `poly_off`.
    while len - off >= 512 && len - poly_off >= 512 {
        let base = state.words[12];
        let od0 = ctr_row(state, base, 0);
        let od1 = ctr_row(state, base, 1);
        let od2 = ctr_row(state, base, 2);
        let od3 = ctr_row(state, base, 3);
        state.advance(8);
        let (mut a0, mut b0, mut c0, mut d0) = (v[0], v[1], v[2], od0);
        let (mut a1, mut b1, mut c1, mut d1) = (v[0], v[1], v[2], od1);
        let (mut a2, mut b2, mut c2, mut d2) = (v[0], v[1], v[2], od2);
        let (mut a3, mut b3, mut c3, mut d3) = (v[0], v[1], v[2], od3);
        rounds8_fused!(
            4, v[0], v[1], v[2], pr, od0, od1, od2, od3, a0, b0, c0, d0, a1, b1, c1, d1, a2, b2,
            c2, d2, a3, b3, c3, d3
        );
        unsafe {
            store_quad(msg.add(off), a0, b0, c0, d0);
            store_quad(msg.add(off + 128), a1, b1, c1, d1);
            store_quad(msg.add(off + 256), a2, b2, c2, d2);
            store_quad(msg.add(off + 384), a3, b3, c3, d3);
        }
        off += 512;
        poly_off += 512;
    }

    // Partial tail batch. Open-side ordering invariant: the xor cursor may
    // only cross a byte AFTER the MAC absorbed it, so this batch's stores
    // stop at the MAC coverage boundary `poly_off + 128·rounds` (everything
    // in [off, poly_off) was absorbed by earlier rounds). A mid-batch stop
    // must land on a block boundary — the engine tail continues the cipher
    // from `off` with the block counter — so a masked partial store is only
    // legal when it ends the message; anything else is left to the tail,
    // which absorbs before it xors.
    let r = len - off;
    if r > 0 && r <= 512 {
        let rounds = ((len - poly_off) / 128).min(4);
        let stored_end = (poly_off + 128 * rounds).min(len);
        if stored_end - off >= 384 || rounds > 0 {
            // Re-sync after the steady loop advanced pptr per iteration.
            pptr = msg.add(poly_off);
            let stored = stored_end - off;
            let k = if stored_end == len {
                stored % BLOCK
            } else {
                0
            };
            let nfull = stored / BLOCK;
            let base = state.words[12];
            let od0 = ctr_row(state, base, 0);
            let od1 = ctr_row(state, base, 1);
            let od2 = ctr_row(state, base, 2);
            let od3 = ctr_row(state, base, 3);
            state.advance((nfull + usize::from(k > 0)) as u32);
            let (mut a0, mut b0, mut c0, mut d0) = (v[0], v[1], v[2], od0);
            let (mut a1, mut b1, mut c1, mut d1) = (v[0], v[1], v[2], od1);
            let (mut a2, mut b2, mut c2, mut d2) = (v[0], v[1], v[2], od2);
            let (mut a3, mut b3, mut c3, mut d3) = (v[0], v[1], v[2], od3);
            rounds8_fused!(
                rounds, v[0], v[1], v[2], pr, od0, od1, od2, od3, a0, b0, c0, d0, a1, b1, c1, d1,
                a2, b2, c2, d2, a3, b3, c3, d3
            );
            emit_partial(
                msg.add(off),
                nfull,
                k,
                a0,
                b0,
                c0,
                d0,
                a1,
                b1,
                c1,
                d1,
                a2,
                b2,
                c2,
                d2,
                a3,
                b3,
                c3,
                d3,
            );
            off += nfull * BLOCK + k;
            poly_off += 128 * rounds;
        }
    }
    // pptr's final increment (inside `pr`) has no later consumer.
    let _ = pptr;

    // SAFETY: stream ensured above.
    let st = unsafe { poly.stream.as_mut().unwrap_unchecked() };
    st.h0 = h0;
    st.h1 = h1;
    st.h2 = h2;
    (off, poly_off)
}

// ── Whole-message direct seal (16-block zmm batches from counter 0) ──
//
// The engine's generic prologue pays a serialized 2-block key kernel
// (gen_key_xor2) before any bulk kernel; on sub-2 KiB messages the medium
// path above adds another quad batch, and with no ciphertext lag the MAC
// runs fully serial after the cipher. Here ONE word-major zmm kernel
// derives the Poly1305 key from block 0 AND encrypts blocks 1..=15 (masked
// partial store for the final block); messages past 960 bytes get a second
// zmm batch with up to 7 IFMA MAC rounds woven between the double rounds,
// reading only kernel-1 ciphertext (< 960) — no store→load hazards.

/// emit16 variant for kernel 1: block 0 goes (raw keystream) to `key64`,
/// blocks `1..=n1` XOR-store at `(b-1)*BLOCK`, plus a `k1`-byte masked
/// partial at block `n1+1` — runtime `n1`/`k1`, so any length ≤ 15 blocks
/// is covered by the single kernel.
macro_rules! emit16k {
    (
        $buf:expr,
        $key64:expr,
        $bc:expr,
        $ctr:expr,
        $n1:expr,
        $k1:expr,
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

        tr4z!($x0, $x1, $x2, $x3);
        tr4z!($x4, $x5, $x6, $x7);
        tr4z!($x8, $x9, $x10, $x11);
        tr4z!($x12, $x13, $x14, $x15);

        let buf = $buf;
        let key64 = $key64;
        let n1 = $n1;
        let k1 = $k1;
        let idx_b = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
        macro_rules! emitk {
            ($b: literal,$l: literal,$g0: ident,$g1: ident,$g2: ident,$g3: ident) => {{
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
                if $b == 0 {
                    _mm512_storeu_si512(key64.as_mut_ptr().cast(), block);
                } else {
                    let bi = $b - 1;
                    let p = buf.add(bi * BLOCK);
                    if bi < n1 {
                        _mm512_storeu_si512(
                            p.cast(),
                            _mm512_xor_si512(_mm512_loadu_si512(p.cast()), block),
                        );
                    } else if bi == n1 && k1 > 0 {
                        let m = ((1u64 << k1) - 1) as __mmask64;
                        _mm512_mask_storeu_epi8(
                            p.cast(),
                            m,
                            _mm512_xor_si512(_mm512_loadu_si512(p.cast()), block),
                        );
                    }
                }
            }};
        }
        emitk!(0, 0, $x0, $x4, $x8, $x12);
        emitk!(1, 0, $x1, $x5, $x9, $x13);
        emitk!(2, 0, $x2, $x6, $x10, $x14);
        emitk!(3, 0, $x3, $x7, $x11, $x15);
        emitk!(4, 1, $x0, $x4, $x8, $x12);
        emitk!(5, 1, $x1, $x5, $x9, $x13);
        emitk!(6, 1, $x2, $x6, $x10, $x14);
        emitk!(7, 1, $x3, $x7, $x11, $x15);
        emitk!(8, 2, $x0, $x4, $x8, $x12);
        emitk!(9, 2, $x1, $x5, $x9, $x13);
        emitk!(10, 2, $x2, $x6, $x10, $x14);
        emitk!(11, 2, $x3, $x7, $x11, $x15);
        emitk!(12, 3, $x0, $x4, $x8, $x12);
        emitk!(13, 3, $x1, $x5, $x9, $x13);
        emitk!(14, 3, $x2, $x6, $x10, $x14);
        emitk!(15, 3, $x3, $x7, $x11, $x15);
    }};
}

/// Kernel 1 of [`seal_direct`]: 16 blocks from the initial counter; block
/// 0 → `key64` (raw keystream), blocks `1..=n1` (+`k1` partial) → message.
/// Advances the counter by the blocks actually stored.
#[target_feature(enable = "avx512f")]
#[inline(never)]
unsafe fn seal_direct_k1(
    state: &mut State,
    msg: *mut u8,
    key64: &mut [u8; 64],
    n1: usize,
    k1: usize,
) {
    let w = state.words;
    let bcast = |i: usize| _mm512_set1_epi32(w[i] as i32);
    let ctr = _mm512_add_epi32(
        bcast(12),
        _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    );
    let (mut x0, mut x1, mut x2, mut x3) = (bcast(0), bcast(1), bcast(2), bcast(3));
    let (mut x4, mut x5, mut x6, mut x7) = (bcast(4), bcast(5), bcast(6), bcast(7));
    let (mut x8, mut x9, mut x10, mut x11) = (bcast(8), bcast(9), bcast(10), bcast(11));
    let (mut x12, mut x13, mut x14, mut x15) = (ctr, bcast(13), bcast(14), bcast(15));
    state.words[12] = 1 + n1 as u32 + u32::from(k1 > 0);
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
    emit16k!(
        msg, key64, bcast, ctr, n1, k1, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13,
        x14, x15
    );
}

/// Whole-message fused seal for the IFMA tier (requires AVX-512F+VL+IFMA).
///
/// Two kernel shapes cover the band (measured on Zen 4, where 512-bit ops
/// are double-pumped and quads beat zmm per block at small counts):
///
/// - ≤ 448 B: one quad `rounds4` batch from counter 0 — block 0 yields the key, blocks 1..7 (+
///   masked partial) the message.
/// - ≤ 960 B: one word-major zmm batch (16 blocks) via [`seal_direct_k1`] — the wider batch
///   amortizes its shuffle-free rounds from ~9 blocks on.
///
/// Either way the whole cipher is ONE pass (the generic prologue's separate
/// 2-block key kernel disappears) and the MAC — with no ciphertext lag to
/// interleave against on this band — folds through the wide absorb path
/// right after the kernel.
///
/// Engine contract: `SMALL_MAX < len <= 15·BLOCK` (192 < len ≤ 960).
#[target_feature(enable = "avx512f,avx512vl,avx512ifma")]
#[inline(never)]
pub(crate) unsafe fn seal_direct(
    state: &mut State,
    aad: &[u8],
    msg: *mut u8,
    len: usize,
    tag_out: &mut [u8; 16],
) {
    debug_assert!(len > crate::aead::SMALL_MAX);
    debug_assert!(len <= 15 * BLOCK);
    let nfull = len / BLOCK; // 4..=15 whole message blocks
    let k = len % BLOCK;

    let mut key32 = [0u8; 32];
    if len <= 7 * BLOCK {
        // Quad batch: blocks 0..8 (key + up to 7 message blocks).
        let v = rows(state);
        let base = state.words[12];
        let ctrs = [
            ctr_row(state, base, 0),
            ctr_row(state, base, 1),
            ctr_row(state, base, 2),
            ctr_row(state, base, 3),
        ];
        state.words[12] = 1 + nfull as u32 + u32::from(k > 0);
        let vs = rounds4(&v, &ctrs);
        let quads: [(__m256i, __m256i, __m256i, __m256i); 4] = [
            (vs[0][0], vs[0][1], vs[0][2], vs[0][3]),
            (vs[1][0], vs[1][1], vs[1][2], vs[1][3]),
            (vs[2][0], vs[2][1], vs[2][2], vs[2][3]),
            (vs[3][0], vs[3][1], vs[3][2], vs[3][3]),
        ];
        // Block 0's first 32 bytes = the one-time key (quad 0, lane 0).
        unsafe {
            _mm256_storeu_si256(
                key32.as_mut_ptr().cast(),
                _mm256_permute2f128_si256::<0x20>(quads[0].0, quads[0].1),
            );
        }
        // Message block i (buffer offset i·64) = state block i+1
        // = quad (i+1)/2, lane (i+1)%2.
        for i in 0..nfull {
            let b = i + 1;
            let (a, bb, c, d) = quads[b / 2];
            unsafe { store_blk(msg, i, a, bb, c, d, b % 2, BLOCK) };
        }
        if k > 0 {
            let b = nfull + 1;
            let (a, bb, c, d) = quads[b / 2];
            unsafe { store_blk(msg, nfull, a, bb, c, d, b % 2, k) };
        }
    } else {
        let mut key64 = [0u8; 64];
        unsafe { seal_direct_k1(state, msg, &mut key64, nfull, k) };
        key32.copy_from_slice(&key64[..32]);
    }

    let mut poly = crate::aead::poly_with_aad::<crate::poly1305::ifma::IfmaPoly>(&mut key32, aad);

    // MAC alignment window (the whole buffer is ciphertext by now).
    let m = poly.pending_blocks();
    let s = (4 - m % 4) % 4 * 16;
    if s > 0 {
        unsafe { poly.update(core::slice::from_raw_parts(msg, s)) };
    }
    unsafe { poly.update_tail(core::slice::from_raw_parts(msg.add(s), len - s)) };
    unsafe { crate::aead::finish_tag(&mut poly, aad.len(), len, tag_out) };
}

/// Single-block (64-byte) kernel in XMM registers.
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
