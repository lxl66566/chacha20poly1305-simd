//! NEON ChaCha20 kernel: 4 blocks per batch in 128-bit registers.
//!
//! Register = word position, lane = block (counters differ per lane), so the
//! column and diagonal quarter rounds operate on plain register quadruples
//! without any lane shuffling. Rotations use the `shl + vsra` pair (disjoint
//! bits make the accumulate act as an OR).

// ChaCha quarter-round rows are conventionally named a/b/c/d; state loads use
// unaligned `vld1q_u32`, so the stricter pointer alignment is intentional.
#![allow(clippy::many_single_char_names, clippy::cast_ptr_alignment)]

use core::arch::aarch64::*;

use super::{BLOCK, State};

/// Blocks per bulk batch (one uint32x4 across 4 counter lanes).
pub(crate) const BATCH_BLOCKS: usize = 4;

#[inline(always)]
fn rol7(x: uint32x4_t) -> uint32x4_t {
    unsafe { vsraq_n_u32::<25>(vshlq_n_u32::<7>(x), x) }
}

#[inline(always)]
fn rol8(x: uint32x4_t) -> uint32x4_t {
    unsafe { vsraq_n_u32::<24>(vshlq_n_u32::<8>(x), x) }
}

#[inline(always)]
fn rol12(x: uint32x4_t) -> uint32x4_t {
    unsafe { vsraq_n_u32::<20>(vshlq_n_u32::<12>(x), x) }
}

#[inline(always)]
fn rol16(x: uint32x4_t) -> uint32x4_t {
    unsafe { vsraq_n_u32::<16>(vshlq_n_u32::<16>(x), x) }
}

/// Full quarter round (rotations 16/12/8/7) on word-position registers.
///
/// Macro form (like the AVX2 kernel): the 4-block state is 16 explicit
/// locals; helpers taking `&mut` would make LLVM spill.
macro_rules! qr {
    ($a:ident, $b:ident, $c:ident, $d:ident) => {{
        $a = vaddq_u32($a, $b);
        $d = rol16(veorq_u32($d, $a));
        $c = vaddq_u32($c, $d);
        $b = rol12(veorq_u32($b, $c));
        $a = vaddq_u32($a, $b);
        $d = rol8(veorq_u32($d, $a));
        $c = vaddq_u32($c, $d);
        $b = rol7(veorq_u32($b, $c));
    }};
}

/// Double round on the 16-word state (column + diagonal rounds).
macro_rules! double_round {
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
        qr!($x0, $x4, $x8, $x12);
        qr!($x1, $x5, $x9, $x13);
        qr!($x2, $x6, $x10, $x14);
        qr!($x3, $x7, $x11, $x15);
        qr!($x0, $x5, $x10, $x15);
        qr!($x1, $x6, $x11, $x12);
        qr!($x2, $x7, $x8, $x13);
        qr!($x3, $x4, $x9, $x14);
    }};
}

/// Load state word `i` broadcast across the 4 block lanes.
#[inline(always)]
fn dup_word(state: &State, i: usize) -> uint32x4_t {
    unsafe { vdupq_n_u32(state.words[i]) }
}

/// Full 20 rounds + feed-forward on 4 blocks; returns the finished state as
/// 16 word-position vectors (lane = block).
#[inline(always)]
unsafe fn rounds4(state: &State, base: u32) -> [uint32x4_t; 16] {
    let mut x0 = dup_word(state, 0);
    let mut x1 = dup_word(state, 1);
    let mut x2 = dup_word(state, 2);
    let mut x3 = dup_word(state, 3);
    let mut x4 = dup_word(state, 4);
    let mut x5 = dup_word(state, 5);
    let mut x6 = dup_word(state, 6);
    let mut x7 = dup_word(state, 7);
    let mut x8 = dup_word(state, 8);
    let mut x9 = dup_word(state, 9);
    let mut x10 = dup_word(state, 10);
    let mut x11 = dup_word(state, 11);
    let ctr = vld1q_u32(
        [
            base,
            base.wrapping_add(1),
            base.wrapping_add(2),
            base.wrapping_add(3),
        ]
        .as_ptr(),
    );
    let mut x12 = ctr;
    let mut x13 = dup_word(state, 13);
    let mut x14 = dup_word(state, 14);
    let mut x15 = dup_word(state, 15);
    for _ in 0..10 {
        double_round!(
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
        );
    }
    // Feed-forward: re-derive the input words from the (untouched) state.
    x0 = vaddq_u32(x0, dup_word(state, 0));
    x1 = vaddq_u32(x1, dup_word(state, 1));
    x2 = vaddq_u32(x2, dup_word(state, 2));
    x3 = vaddq_u32(x3, dup_word(state, 3));
    x4 = vaddq_u32(x4, dup_word(state, 4));
    x5 = vaddq_u32(x5, dup_word(state, 5));
    x6 = vaddq_u32(x6, dup_word(state, 6));
    x7 = vaddq_u32(x7, dup_word(state, 7));
    x8 = vaddq_u32(x8, dup_word(state, 8));
    x9 = vaddq_u32(x9, dup_word(state, 9));
    x10 = vaddq_u32(x10, dup_word(state, 10));
    x11 = vaddq_u32(x11, dup_word(state, 11));
    x12 = vaddq_u32(x12, ctr);
    x13 = vaddq_u32(x13, dup_word(state, 13));
    x14 = vaddq_u32(x14, dup_word(state, 14));
    x15 = vaddq_u32(x15, dup_word(state, 15));
    [
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15,
    ]
}

/// Transpose one word-group (4 registers × 4 block lanes) into per-block
/// vectors, XOR with the plaintext and store 4×16 bytes. Group `g` of block
/// `k` lands at `buf + k*64 + g*16`.
#[inline(always)]
unsafe fn emit_xor_group(
    w0: uint32x4_t,
    w1: uint32x4_t,
    w2: uint32x4_t,
    w3: uint32x4_t,
    ptr: *mut u8,
) {
    let t0 = vreinterpretq_u64_u32(vzip1q_u32(w0, w1));
    let t1 = vreinterpretq_u64_u32(vzip2q_u32(w0, w1));
    let u0 = vreinterpretq_u64_u32(vzip1q_u32(w2, w3));
    let u1 = vreinterpretq_u64_u32(vzip2q_u32(w2, w3));
    // Lane k of each result = block k's words for this group.
    let blocks = [
        vreinterpretq_u32_u64(vzip1q_u64(t0, u0)),
        vreinterpretq_u32_u64(vzip2q_u64(t0, u0)),
        vreinterpretq_u32_u64(vzip1q_u64(t1, u1)),
        vreinterpretq_u32_u64(vzip2q_u64(t1, u1)),
    ];
    for (blk, v) in blocks.iter().enumerate() {
        let p = ptr.add(blk * BLOCK).cast::<u32>();
        vst1q_u32(p, veorq_u32(vld1q_u32(p), *v));
    }
}

/// Generate 4 keystream blocks and XOR them into `buf` (256 bytes).
#[inline(always)]
pub(crate) unsafe fn xor_batch4(state: &mut State, buf: *mut u8) {
    let base = state.words[12];
    let x = rounds4(state, base);
    state.advance(BATCH_BLOCKS as u32);
    for g in 0..4 {
        emit_xor_group(
            x[g * 4],
            x[g * 4 + 1],
            x[g * 4 + 2],
            x[g * 4 + 3],
            buf.add(g * 16),
        );
    }
}

/// Single-block (64-byte) kernel: lane = word index, so a quarter round on
/// (a,b,c,d) runs all four scalar quarter rounds of one round at once; the
/// diagonal round rotates the b/c/d lanes with `vext`.
#[inline(always)]
unsafe fn rounds1(state: &State) -> [uint32x4_t; 4] {
    let p = state.words.as_ptr().cast::<u32>();
    let (a0, b0, c0, d0) = (
        vld1q_u32(p.add(0)),
        vld1q_u32(p.add(4)),
        vld1q_u32(p.add(8)),
        vld1q_u32(p.add(12)),
    );
    let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
    for _ in 0..10 {
        qr!(a, b, c, d);
        // Rotate lanes: diagonal rounds (0,5,10,15),(1,6,11,12),...
        b = vextq_u32::<1>(b, b);
        c = vextq_u32::<2>(c, c);
        d = vextq_u32::<3>(d, d);
        qr!(a, b, c, d);
        b = vextq_u32::<3>(b, b);
        c = vextq_u32::<2>(c, c);
        d = vextq_u32::<1>(d, d);
    }
    [
        vaddq_u32(a, a0),
        vaddq_u32(b, b0),
        vaddq_u32(c, c0),
        vaddq_u32(d, d0),
    ]
}

/// XOR exactly one 64-byte block with the keystream, advancing by 1.
#[inline(always)]
pub(crate) unsafe fn xor_single(state: &mut State, buf: *mut u8) {
    let v = rounds1(state);
    state.advance(1);
    let p = buf.cast::<u32>();
    for (i, w) in v.iter().enumerate() {
        vst1q_u32(p.add(i * 4), veorq_u32(vld1q_u32(p.add(i * 4)), *w));
    }
}

/// Generate exactly one keystream block (no XOR, no advance).
#[inline(always)]
pub(crate) unsafe fn gen_block(state: &State, out: &mut [u8; BLOCK]) {
    let v = rounds1(state);
    let p = out.as_mut_ptr().cast::<u32>();
    for (i, w) in v.iter().enumerate() {
        vst1q_u32(p.add(i * 4), *w);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chacha::State;

    // NEON is baseline on aarch64; no runtime skip needed.
    #[test]
    fn gen_block_and_paths_match_soft() {
        let key: [u8; 32] = core::array::from_fn(|i| (i * 19 + 7) as u8);
        let nonce: [u8; 12] = core::array::from_fn(|i| (i * 7 + 11) as u8);
        let mut st = State::new_ietf(&key, &nonce);
        st.words[12] = 3;
        let mut fast = [0u8; 64];
        unsafe { gen_block(&st, &mut fast) };
        let mut expect = [0u8; 64];
        unsafe { crate::chacha::soft::gen_block(&st, &mut expect) };
        assert_eq!(expect, fast);

        // 4-block batch against the soft reference.
        let mut pt = [0u8; 256];
        for (i, b) in pt.iter_mut().enumerate() {
            *b = (i * 31 + 5) as u8;
        }
        let mut expect = pt;
        unsafe { crate::chacha::soft::xor(&mut st.clone_struct(), &mut expect) };
        unsafe { xor_batch4(&mut st, pt.as_mut_ptr()) };
        assert_eq!(expect, pt);
        assert_eq!(st.words[12], 3 + 4);
    }
}
