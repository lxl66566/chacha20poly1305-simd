//! SSE2 ChaCha20 kernel: 4 blocks per batch in XMM registers.
//!
//! SSE2 is x86-64 baseline, so this is the always-available SIMD tier between
//! the scalar `soft` code and AVX2. Rotations are shift-xor pairs only —
//! `pshufb` (the AVX2 kernel's byte-shuffle trick) is SSSE3, not SSE2.

// ChaCha quarter-round rows are conventionally named a/b/c/d; state loads use
// unaligned `_mm_loadu_si128`, so the stricter pointer alignment is intentional.
#![allow(clippy::many_single_char_names, clippy::cast_ptr_alignment)]

use core::arch::x86_64::*;

use crate::chacha::{BLOCK, State};

/// Blocks per bulk batch.
pub(crate) const BATCH_BLOCKS: usize = 4;

/// Load the state rows [a, b, c]; word 12 (block counter) sits in row d and
/// is materialized separately by [`ctr_row`].
#[inline(always)]
unsafe fn rows(state: &State) -> [__m128i; 3] {
    let p = state.words.as_ptr().cast::<__m128i>();
    [
        _mm_loadu_si128(p.add(0)),
        _mm_loadu_si128(p.add(1)),
        _mm_loadu_si128(p.add(2)),
    ]
}

/// Row d (counter || nonce) with the block counter set to `base + k`.
#[inline(always)]
unsafe fn ctr_row(state: &State, base: u32, k: u32) -> __m128i {
    let w = state.words;
    _mm_setr_epi32(
        base.wrapping_add(k) as i32,
        w[13] as i32,
        w[14] as i32,
        w[15] as i32,
    )
}

/// Full quarter round (rotations 16/12/8/7) on row registers. SSE2 has no
/// rotate, so every rotation is a shift-left / shift-right / xor triple.
///
/// Macro form for the same register-pressure reason as the AVX2 kernel: the
/// 4-block batch keeps 16 XMM rows live, exactly the whole register file.
macro_rules! qr {
    ($a:ident, $b:ident, $c:ident, $d:ident) => {{
        $a = _mm_add_epi32($a, $b);
        let x = _mm_xor_si128($d, $a);
        $d = _mm_xor_si128(_mm_slli_epi32(x, 16), _mm_srli_epi32(x, 16));
        $c = _mm_add_epi32($c, $d);
        let x = _mm_xor_si128($b, $c);
        $b = _mm_xor_si128(_mm_slli_epi32(x, 12), _mm_srli_epi32(x, 20));
        $a = _mm_add_epi32($a, $b);
        let x = _mm_xor_si128($d, $a);
        $d = _mm_xor_si128(_mm_slli_epi32(x, 8), _mm_srli_epi32(x, 24));
        $c = _mm_add_epi32($c, $d);
        let x = _mm_xor_si128($b, $c);
        $b = _mm_xor_si128(_mm_slli_epi32(x, 7), _mm_srli_epi32(x, 25));
    }};
}

/// Rotate rows to columns and back (diagonal rounds); same lane-rotation
/// trick as the AVX2 kernel / floodyberry's SSE2 ChaCha.
macro_rules! to_cols {
    ($a:ident, $b:ident, $c:ident, $d:ident) => {{
        let _ = $b;
        $c = _mm_shuffle_epi32::<0b_00_11_10_01>($c);
        $d = _mm_shuffle_epi32::<0b_01_00_11_10>($d);
        $a = _mm_shuffle_epi32::<0b_10_01_00_11>($a);
    }};
}

macro_rules! to_rows {
    ($a:ident, $b:ident, $c:ident, $d:ident) => {{
        let _ = $b;
        $c = _mm_shuffle_epi32::<0b_10_01_00_11>($c);
        $d = _mm_shuffle_epi32::<0b_01_00_11_10>($d);
        $a = _mm_shuffle_epi32::<0b_00_11_10_01>($a);
    }};
}

/// Full 20 rounds + feed-forward add on 4 blocks (256 bytes of keystream).
#[inline(always)]
unsafe fn rounds4(v: &[__m128i; 3], ctrs: &[__m128i; 4]) -> [[__m128i; 4]; BATCH_BLOCKS] {
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
            _mm_add_epi32(a0, v[0]),
            _mm_add_epi32(b0, v[1]),
            _mm_add_epi32(c0, v[2]),
            _mm_add_epi32(d0, ctrs[0]),
        ],
        [
            _mm_add_epi32(a1, v[0]),
            _mm_add_epi32(b1, v[1]),
            _mm_add_epi32(c1, v[2]),
            _mm_add_epi32(d1, ctrs[1]),
        ],
        [
            _mm_add_epi32(a2, v[0]),
            _mm_add_epi32(b2, v[1]),
            _mm_add_epi32(c2, v[2]),
            _mm_add_epi32(d2, ctrs[2]),
        ],
        [
            _mm_add_epi32(a3, v[0]),
            _mm_add_epi32(b3, v[1]),
            _mm_add_epi32(c3, v[2]),
            _mm_add_epi32(d3, ctrs[3]),
        ],
    ]
}

/// Full 20 rounds + feed-forward on 2 interleaved blocks (the OpenSSL
/// `ChaCha20_128` shape: dedicated 2-block kernel, no loop machinery).
#[inline(always)]
unsafe fn rounds2(v: &[__m128i; 3], ctrs: &[__m128i; 2]) -> [[__m128i; 4]; 2] {
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
            _mm_add_epi32(a0, v[0]),
            _mm_add_epi32(b0, v[1]),
            _mm_add_epi32(c0, v[2]),
            _mm_add_epi32(d0, ctrs[0]),
        ],
        [
            _mm_add_epi32(a1, v[0]),
            _mm_add_epi32(b1, v[1]),
            _mm_add_epi32(c1, v[2]),
            _mm_add_epi32(d1, ctrs[1]),
        ],
    ]
}

/// XOR one finished block's rows into `buf` (64 bytes).
#[inline(always)]
unsafe fn emit_xor_block(quad: &[__m128i; 4], buf: *mut u8) {
    let p = buf.cast::<__m128i>();
    for i in 0..4 {
        let pt = _mm_loadu_si128(p.add(i));
        _mm_storeu_si128(p.add(i), _mm_xor_si128(pt, quad[i]));
    }
}

/// Generate 4 keystream blocks and XOR them into `buf` (256 bytes).
#[inline(always)]
pub(crate) unsafe fn xor_batch4(state: &mut State, buf: *mut u8) {
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
        emit_xor_block(quad, p);
        p = p.add(BLOCK);
    }
}

/// Fused prologue: blocks 0 and 1 in one interleaved kernel call — block
/// 0's first 32 bytes (the Poly1305 one-time key) to `key_out`, block 1's
/// keystream XORed into `b1` (a zeroed buffer yields the raw keystream).
#[inline(always)]
pub(crate) unsafe fn gen_key_xor2(state: &mut State, key_out: &mut [u8; 32], b1: &mut [u8; BLOCK]) {
    let v = rows(state);
    let base = state.words[12];
    let ctrs = [ctr_row(state, base, 0), ctr_row(state, base, 1)];
    state.advance(2);
    let [blk0, blk1] = rounds2(&v, &ctrs);
    // key = block 0 bytes 0..31 = rows a|b
    let k = key_out.as_mut_ptr().cast::<__m128i>();
    _mm_storeu_si128(k, blk0[0]);
    _mm_storeu_si128(k.add(1), blk0[1]);
    // block 1 xored into b1
    let p = b1.as_mut_ptr().cast::<__m128i>();
    for i in 0..4 {
        _mm_storeu_si128(p.add(i), _mm_xor_si128(_mm_loadu_si128(p.add(i)), blk1[i]));
    }
}

/// Generate exactly one keystream block (no XOR, no advance). Test
/// reference target only — the engine uses the fused [`gen_key_xor2`].
#[cfg(test)]
#[inline(always)]
pub(crate) unsafe fn gen_block(state: &State, out: &mut [u8; BLOCK]) {
    let [a, b, c, d] = rounds1(state);
    let p = out.as_mut_ptr().cast::<__m128i>();
    _mm_storeu_si128(p.add(0), a);
    _mm_storeu_si128(p.add(1), b);
    _mm_storeu_si128(p.add(2), c);
    _mm_storeu_si128(p.add(3), d);
}

/// Single-block (64-byte) kernel: one block in 4 XMM registers.
#[inline(always)]
pub(crate) unsafe fn xor_single(state: &mut State, buf: *mut u8) {
    let [a, b, c, d] = rounds1(state);
    state.advance(1);
    let quad = [a, b, c, d];
    emit_xor_block(&quad, buf);
}

/// 20 rounds + feed-forward on a single block, returned as rows.
#[inline(always)]
unsafe fn rounds1(state: &State) -> [__m128i; 4] {
    let p = state.words.as_ptr().cast::<__m128i>();
    let v0 = _mm_loadu_si128(p.add(0));
    let v1 = _mm_loadu_si128(p.add(1));
    let v2 = _mm_loadu_si128(p.add(2));
    let v3 = _mm_loadu_si128(p.add(3));
    let (mut a, mut b, mut c, mut d) = (v0, v1, v2, v3);
    for _ in 0..10 {
        qr!(a, b, c, d);
        to_cols!(a, b, c, d);
        qr!(a, b, c, d);
        to_rows!(a, b, c, d);
    }
    [
        _mm_add_epi32(a, v0),
        _mm_add_epi32(b, v1),
        _mm_add_epi32(c, v2),
        _mm_add_epi32(d, v3),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chacha::State;

    // SSE2 is x86-64 baseline: no runtime-feature skip is needed here, these
    // run everywhere the module compiles.

    #[test]
    fn gen_block_matches_soft() {
        let key: [u8; 32] = core::array::from_fn(|i| (i * 19 + 7) as u8);
        let nonce: [u8; 12] = core::array::from_fn(|i| (i * 7 + 11) as u8);
        let mut st = State::new_ietf(&key, &nonce);
        st.words[12] = 3;
        let mut fast = [0u8; 64];
        unsafe { gen_block(&st, &mut fast) };
        let mut expect = [0u8; 64];
        unsafe { crate::chacha::soft::gen_block(&st, &mut expect) };
        assert_eq!(expect, fast);
    }

    /// Batch and single kernels must be cursor-exact vs the scalar reference
    /// across the 256-byte batch grid, the 64-byte grid and a partial tail.
    #[test]
    fn xor_paths_match_soft() {
        let key: [u8; 32] = core::array::from_fn(|i| (i * 23 + 5) as u8);
        let nonce: [u8; 12] = core::array::from_fn(|i| (i * 11 + 3) as u8);
        for (skip, len) in [(0usize, 1024usize), (1, 320), (5, 449), (2, 114)] {
            let mut st = State::new_ietf(&key, &nonce);
            st.advance(skip as u32);
            let mut ref_st = st.clone_struct();
            let mut fast: alloc::vec::Vec<u8> = (0..len).map(|i| (i * 37 + 3) as u8).collect();
            let mut expect = fast.clone();
            unsafe {
                // emulate the backend's chacha_xor shape
                let mut off = 0usize;
                while fast.len() - off >= 256 {
                    xor_batch4(&mut st, fast[off..].as_mut_ptr());
                    off += 256;
                }
                while fast.len() - off >= BLOCK {
                    xor_single(&mut st, fast[off..].as_mut_ptr());
                    off += BLOCK;
                }
                if off < fast.len() {
                    crate::chacha::soft::xor(&mut st, &mut fast[off..]);
                }
                crate::chacha::soft::xor(&mut ref_st, &mut expect);
            }
            assert_eq!(expect, fast, "skip {skip} len {len}");
        }
    }
}
