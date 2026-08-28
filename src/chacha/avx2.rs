//! AVX2 ChaCha20 kernel: 8 blocks per batch in YMM registers.
//!
//! Same quad layout as the AVX-512VL kernel; rotations without `vprold`
//! use the byte-shuffle trick (16/8) and shift-xor pairs (12/7).

// ChaCha quarter-round rows are conventionally named a/b/c/d; state loads use
// unaligned `_mm_loadu_si128`, so the stricter pointer alignment is intentional.
#![allow(clippy::many_single_char_names, clippy::cast_ptr_alignment)]

use core::arch::x86_64::*;

use crate::chacha::{BLOCK, State};

/// Blocks per bulk batch (4 quads × 2).
pub(crate) const BATCH_BLOCKS: usize = 8;

#[cfg_attr(not(debug_assertions), inline(always))]
unsafe fn rol16_mask() -> __m256i {
    _mm256_set_epi64x(
        0x0d0c_0f0e_0908_0b0a,
        0x0504_0706_0100_0302,
        0x0d0c_0f0e_0908_0b0a,
        0x0504_0706_0100_0302,
    )
}

#[cfg_attr(not(debug_assertions), inline(always))]
unsafe fn rol8_mask() -> __m256i {
    _mm256_set_epi64x(
        0x0e0d_0c0f_0a09_080b,
        0x0605_0407_0201_0003,
        0x0e0d_0c0f_0a09_080b,
        0x0605_0407_0201_0003,
    )
}

#[cfg_attr(not(debug_assertions), inline(always))]
unsafe fn rows(state: &State) -> [__m256i; 3] {
    let p = state.words.as_ptr().cast::<__m128i>();
    [
        _mm256_broadcastsi128_si256(_mm_loadu_si128(p.add(0))),
        _mm256_broadcastsi128_si256(_mm_loadu_si128(p.add(1))),
        _mm256_broadcastsi128_si256(_mm_loadu_si128(p.add(2))),
    ]
}

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

/// Full quarter round (rotations 16/12/8/7) on explicit row registers;
/// AVX2 lacks `vprold`, so 16/8 use byte-shuffle masks and 12/7 shift pairs.
///
/// Macro form is load-bearing: the 8-block kernel keeps 16 YMM rows live, and
/// the previous array-based formulation (`[[__m256i; 4]; N]` + helper fns)
/// made LLVM spill the whole state to the stack on every double round.
macro_rules! qr {
    ($a:ident, $b:ident, $c:ident, $d:ident, $m16:ident, $m8:ident) => {{
        $a = _mm256_add_epi32($a, $b);
        $d = _mm256_shuffle_epi8(_mm256_xor_si256($d, $a), $m16);
        $c = _mm256_add_epi32($c, $d);
        let x = _mm256_xor_si256($b, $c);
        $b = _mm256_xor_si256(_mm256_slli_epi32(x, 12), _mm256_srli_epi32(x, 20));
        $a = _mm256_add_epi32($a, $b);
        $d = _mm256_shuffle_epi8(_mm256_xor_si256($d, $a), $m8);
        $c = _mm256_add_epi32($c, $d);
        let x = _mm256_xor_si256($b, $c);
        $b = _mm256_xor_si256(_mm256_slli_epi32(x, 7), _mm256_srli_epi32(x, 25));
    }};
}

/// Rotate rows to columns and back (lane-local shuffles commute with the
/// quad layout).
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
    let m16 = rol16_mask();
    let m8 = rol8_mask();
    let (mut a0, mut b0, mut c0, mut d0) = (v[0], v[1], v[2], ctrs[0]);
    let (mut a1, mut b1, mut c1, mut d1) = (v[0], v[1], v[2], ctrs[1]);
    let (mut a2, mut b2, mut c2, mut d2) = (v[0], v[1], v[2], ctrs[2]);
    let (mut a3, mut b3, mut c3, mut d3) = (v[0], v[1], v[2], ctrs[3]);
    for _ in 0..10 {
        qr!(a0, b0, c0, d0, m16, m8);
        qr!(a1, b1, c1, d1, m16, m8);
        qr!(a2, b2, c2, d2, m16, m8);
        qr!(a3, b3, c3, d3, m16, m8);
        to_cols!(a0, b0, c0, d0);
        to_cols!(a1, b1, c1, d1);
        to_cols!(a2, b2, c2, d2);
        to_cols!(a3, b3, c3, d3);
        qr!(a0, b0, c0, d0, m16, m8);
        qr!(a1, b1, c1, d1, m16, m8);
        qr!(a2, b2, c2, d2, m16, m8);
        qr!(a3, b3, c3, d3, m16, m8);
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
    let m16 = rol16_mask();
    let m8 = rol8_mask();
    let (mut a, mut b, mut c, mut d) = (v[0], v[1], v[2], ctr);
    for _ in 0..10 {
        qr!(a, b, c, d, m16, m8);
        to_cols!(a, b, c, d);
        qr!(a, b, c, d, m16, m8);
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
    let m16 = rol16_mask();
    let m8 = rol8_mask();
    let (mut a0, mut b0, mut c0, mut d0) = (v[0], v[1], v[2], ctrs[0]);
    let (mut a1, mut b1, mut c1, mut d1) = (v[0], v[1], v[2], ctrs[1]);
    for _ in 0..10 {
        qr!(a0, b0, c0, d0, m16, m8);
        qr!(a1, b1, c1, d1, m16, m8);
        to_cols!(a0, b0, c0, d0);
        to_cols!(a1, b1, c1, d1);
        qr!(a0, b0, c0, d0, m16, m8);
        qr!(a1, b1, c1, d1, m16, m8);
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

#[cfg_attr(not(debug_assertions), inline(always))]
unsafe fn emit_xor_quad(quad: &[__m256i; 4], mut buf: *mut u8) {
    let [a, b, c, d] = *quad;
    for blk in 0..2 {
        let lo = match blk {
            0 => _mm256_permute2f128_si256::<0x20>(a, b),
            _ => _mm256_permute2f128_si256::<0x31>(a, b),
        };
        let hi = match blk {
            0 => _mm256_permute2f128_si256::<0x20>(c, d),
            _ => _mm256_permute2f128_si256::<0x31>(c, d),
        };
        let pt_lo = _mm256_loadu_si256(buf.cast());
        let pt_hi = _mm256_loadu_si256(buf.add(32).cast());
        _mm256_storeu_si256(buf.cast(), _mm256_xor_si256(pt_lo, lo));
        _mm256_storeu_si256(buf.add(32).cast(), _mm256_xor_si256(pt_hi, hi));
        buf = buf.add(BLOCK);
    }
}

/// Generate 8 keystream blocks and XOR them into `buf` (512 bytes).
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
    state.advance(BATCH_BLOCKS as u32);
    let vs = rounds4(&v, &ctrs);
    let mut p = buf;
    for quad in &vs {
        emit_xor_quad(quad, p);
        p = p.add(2 * BLOCK);
    }
}

/// Four-block (256-byte) batch — 2 interleaved quads in one kernel call.
#[cfg_attr(not(debug_assertions), inline(always))]
pub(crate) unsafe fn xor_quad2(state: &mut State, buf: *mut u8) {
    let v = rows(state);
    let base = state.words[12];
    let ctrs = [ctr_row(state, base, 0), ctr_row(state, base, 1)];
    state.advance(4);
    let vs = rounds2(&v, &ctrs);
    let mut p = buf;
    for quad in &vs {
        emit_xor_quad(quad, p);
        p = p.add(2 * BLOCK);
    }
}

/// Two-block (128-byte) batch.
#[cfg_attr(not(debug_assertions), inline(always))]
pub(crate) unsafe fn xor_quad(state: &mut State, buf: *mut u8) {
    let v = rows(state);
    let c = ctr_row(state, state.words[12], 0);
    state.advance(2);
    let vs = rounds1(&v, c);
    emit_xor_quad(&vs, buf);
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

/// Small-message fused op (the OpenSSL `chacha20_poly1305_tls_cipher`
/// shape): ONE kernel call computes block 0 (first 32 bytes = the Poly1305
/// one-time key → `key_out`) and the RAW keystream blocks covering a
/// ≤ 3·BLOCK-byte message, stored to `ks` (`ks.len()` must be
/// `ceil(msg_len / BLOCK) * BLOCK`, 0 for an empty message). Advances the
/// counter by `1 + ks.len() / BLOCK`; the engine XORs `ks` into the
/// message at whichever pipeline stage the MAC ordering requires.
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

/// Generate exactly one keystream block (no XOR, no advance). Test
/// reference target only — the engine uses the fused [`gen_key_xor2`].
#[cfg(test)]
#[cfg_attr(not(debug_assertions), inline(always))]
pub(crate) unsafe fn gen_block(state: &State, out: &mut [u8; BLOCK]) {
    // one quad (2 blocks), second block discarded
    let v = rows(state);
    let c = ctr_row(state, state.words[12], 0);
    let [a, b, cc, d] = rounds1(&v, c); // includes the feed-forward add
    let lo = _mm256_permute2f128_si256::<0x20>(a, b);
    let hi = _mm256_permute2f128_si256::<0x20>(cc, d);
    _mm256_storeu_si256(out.as_mut_ptr().cast(), lo);
    _mm256_storeu_si256(out.as_mut_ptr().add(32).cast(), hi);
}

/// Single-block (64-byte) kernel: one block in XMM registers, AVX2 rotations.
#[cfg_attr(not(debug_assertions), inline(always))]
pub(crate) unsafe fn xor_single(state: &mut State, buf: *mut u8) {
    let p = state.words.as_ptr().cast::<__m128i>();
    let v0 = _mm_loadu_si128(p.add(0));
    let v1 = _mm_loadu_si128(p.add(1));
    let v2 = _mm_loadu_si128(p.add(2));
    let v3 = _mm_loadu_si128(p.add(3));
    let m16 = _mm_set_epi64x(0x0d0c_0f0e_0908_0b0a, 0x0504_0706_0100_0302);
    let m8 = _mm_set_epi64x(0x0e0d_0c0f_0a09_080b, 0x0605_0407_0201_0003);
    let mut a = v0;
    let mut b = v1;
    let mut c = v2;
    let mut d = v3;
    for _ in 0..10 {
        quarter_xmm(&mut a, &mut b, &mut c, &mut d, m16, m8);
        c = _mm_shuffle_epi32::<0b_00_11_10_01>(c);
        d = _mm_shuffle_epi32::<0b_01_00_11_10>(d);
        a = _mm_shuffle_epi32::<0b_10_01_00_11>(a);
        quarter_xmm(&mut a, &mut b, &mut c, &mut d, m16, m8);
        c = _mm_shuffle_epi32::<0b_10_01_00_11>(c);
        d = _mm_shuffle_epi32::<0b_01_00_11_10>(d);
        a = _mm_shuffle_epi32::<0b_00_11_10_01>(a);
    }
    a = _mm_add_epi32(a, v0);
    b = _mm_add_epi32(b, v1);
    c = _mm_add_epi32(c, v2);
    d = _mm_add_epi32(d, v3);
    state.advance(1);
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
unsafe fn quarter_xmm(
    a: &mut __m128i,
    b: &mut __m128i,
    c: &mut __m128i,
    d: &mut __m128i,
    m16: __m128i,
    m8: __m128i,
) {
    *a = _mm_add_epi32(*a, *b);
    *d = _mm_shuffle_epi8(_mm_xor_si128(*d, *a), m16);
    *c = _mm_add_epi32(*c, *d);
    let x = _mm_xor_si128(*b, *c);
    *b = _mm_xor_si128(_mm_slli_epi32(x, 12), _mm_srli_epi32(x, 20));
    *a = _mm_add_epi32(*a, *b);
    *d = _mm_shuffle_epi8(_mm_xor_si128(*d, *a), m8);
    *c = _mm_add_epi32(*c, *d);
    let x = _mm_xor_si128(*b, *c);
    *b = _mm_xor_si128(_mm_slli_epi32(x, 7), _mm_srli_epi32(x, 25));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chacha::State;

    // Calls AVX2 kernels directly (bypassing runtime dispatch); skip on CPUs
    // without AVX2 instead of executing illegal instructions.
    #[test]
    fn gen_block_and_paths_match_soft() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
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

    #[test]
    fn xor_quad2_matches_soft() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let key: [u8; 32] = core::array::from_fn(|i| (i * 23 + 5) as u8);
        let nonce: [u8; 12] = core::array::from_fn(|i| (i * 11 + 3) as u8);
        for (skip, len) in [(0usize, 256usize), (3, 256), (5, 448), (2, 448)] {
            let mut st = State::new_ietf(&key, &nonce);
            st.advance(skip as u32);
            let mut ref_st = st.clone_struct();
            let mut fast: alloc::vec::Vec<u8> = (0..len).map(|i| (i * 37 + 3) as u8).collect();
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
        }
    }
}
