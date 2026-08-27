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

/// Blocks per bulk batch (4 quads × 2).
pub(crate) const BATCH_BLOCKS: usize = 8;

/// Load the constant/key rows of `state` broadcast into both 128-bit lanes.
#[inline(always)]
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
#[inline(always)]
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
#[inline(always)]
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
#[inline(always)]
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
#[inline(always)]
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
#[inline(always)]
unsafe fn emit_lo(a: __m256i, b: __m256i, block: usize) -> __m256i {
    match block {
        0 => _mm256_permute2f128_si256::<0x20>(a, b),
        _ => _mm256_permute2f128_si256::<0x31>(a, b),
    }
}

#[inline(always)]
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
#[inline(always)]
pub(crate) unsafe fn xor_batch8(state: &mut State, buf: *mut u8) {
    let v = rows(state);
    let base = state.words[12];
    let ctrs = [
        ctr_row(state, base, 0),
        ctr_row(state, base, 1),
        ctr_row(state, base, 2),
        ctr_row(state, base, 3),
    ];
    state.advance(BATCH_BLOCKS as u32);
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
#[inline(always)]
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
#[inline(always)]
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
#[inline(always)]
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

/// Single-block (64-byte) kernel in XMM registers.
#[inline(always)]
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

#[inline(always)]
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
#[inline(always)]
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
