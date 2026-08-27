//! SSE2 backend: 4-block ChaCha20 + scalar Poly1305, fused.
//!
//! SSE2 is x86-64 baseline — this is the no-probe fallback when AVX2 is
//! unavailable, mirroring RustCrypto's sse2 configuration (SIMD ChaCha +
//! donna Poly1305) but with the fused AEAD pipeline on top.

use crate::{
    Tag,
    aead::Ops,
    chacha::{BLOCK, State},
};

pub(crate) struct Sse2Ops;

impl Ops for Sse2Ops {
    type Poly = crate::poly1305::soft::SoftPoly;

    const CHACHA_BATCH: usize = crate::chacha::sse2::BATCH_BLOCKS;

    #[inline(always)]
    unsafe fn gen_block(state: &State, out: &mut [u8; BLOCK]) {
        crate::chacha::sse2::gen_block(state, out);
    }

    #[inline(always)]
    unsafe fn chacha_xor(state: &mut State, buf: &mut [u8]) {
        let mut off = 0usize;
        while buf.len() - off >= 256 {
            crate::chacha::sse2::xor_batch4(state, buf[off..].as_mut_ptr());
            off += 256;
        }
        while buf.len() - off >= BLOCK {
            crate::chacha::sse2::xor_single(state, buf[off..].as_mut_ptr());
            off += BLOCK;
        }
        // scalar fallback for sub-block tails (never reached: callers feed
        // whole blocks through this path only via the alignment prologue)
        if off < buf.len() {
            crate::chacha::soft::xor(state, &mut buf[off..]);
        }
    }

    #[inline(always)]
    unsafe fn chacha_xor_batch(state: &mut State, buf: &mut [u8]) {
        debug_assert_eq!(buf.len(), 256);
        crate::chacha::sse2::xor_batch4(state, buf.as_mut_ptr());
    }

    #[inline(always)]
    unsafe fn xor_block1(state: &mut State, buf: &mut [u8]) {
        debug_assert_eq!(buf.len(), BLOCK);
        crate::chacha::sse2::xor_single(state, buf.as_mut_ptr());
    }
}

/// Fused seal entry (SSE2 is x86-64 baseline, always available).
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn seal(state: &mut State, aad: &[u8], msg: &mut [u8], tag: &mut Tag) {
    crate::aead::seal::<Sse2Ops>(state, aad, msg, tag);
}

/// Fused open entry (SSE2 is x86-64 baseline, always available).
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn open(state: &mut State, aad: &[u8], buf: &mut [u8], tag: &Tag) -> bool {
    crate::aead::open::<Sse2Ops>(state, aad, buf, tag)
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::chacha::State;

    fn run_seal<O: Ops>(key: &[u8; 32], nonce: &[u8; 12], aad: &[u8], msg: &mut [u8]) -> [u8; 16] {
        let mut st = State::new_ietf(key, nonce);
        let mut tag = [0u8; 16];
        crate::aead::seal::<O>(&mut st, aad, msg, &mut tag);
        tag
    }

    fn run_open<O: Ops>(
        key: &[u8; 32],
        nonce: &[u8; 12],
        aad: &[u8],
        buf: &mut [u8],
        tag: &[u8; 16],
    ) -> bool {
        let mut st = State::new_ietf(key, nonce);
        crate::aead::open::<O>(&mut st, aad, buf, tag)
    }

    // SSE2 is x86-64 baseline: unlike the AVX2/AVX-512 suites these run on
    // every x86-64 host, so the backend gets unconditional CI coverage.

    // Boundary set covers the alignment prologue (s ≠ 0 shifts), the 4-block
    // batch grid and partial-block tails (same matrix as the AVX-512 suite).
    #[test]
    fn seal_and_open_match_soft() {
        let key: [u8; 32] = core::array::from_fn(|i| (i * 7 + 11) as u8);
        let nonce: [u8; 12] = core::array::from_fn(|i| (i * 5 + 2) as u8);
        for (len, aad_len) in [
            (0usize, 0usize),
            (1, 0),
            (1, 1),
            (16, 12),
            (48, 12),
            (49, 0),
            (114, 12),
            (256, 0),
            (320, 12),
            (321, 0),
            (1023, 0),
            (1023, 16),
            (1024, 5),
            (1024, 16),
            (1024, 100),
            (1025, 0),
            (1025, 12),
            (1088, 16),
            (2048, 0),
            (4097, 5),
        ] {
            let msg: Vec<u8> = (0..len).map(|i| (i * 37 + 3) as u8).collect();
            let aad: Vec<u8> = (0..aad_len).map(|i| (i * 41 + 1) as u8).collect();
            let mut soft_buf = msg.clone();
            let soft_tag =
                run_seal::<crate::backend::soft::SoftOps>(&key, &nonce, &aad, &mut soft_buf);
            let mut fast_buf = msg.clone();
            let fast_tag = run_seal::<Sse2Ops>(&key, &nonce, &aad, &mut fast_buf);
            assert_eq!(soft_buf, fast_buf, "ct len {len} aad {aad_len}");
            assert_eq!(soft_tag, fast_tag, "tag len {len} aad {aad_len}");
            assert!(
                run_open::<Sse2Ops>(&key, &nonce, &aad, &mut fast_buf, &fast_tag),
                "open len {len}"
            );
            assert_eq!(fast_buf, msg, "pt len {len} aad {aad_len}");
        }
    }

    /// `Ops::chacha_xor` must be cursor-exact vs the scalar reference, both at
    /// block boundaries and inside a block (the 66-byte tail of the len=114
    /// engine sequence).
    #[test]
    fn chacha_xor_matches_soft() {
        let key: [u8; 32] = core::array::from_fn(|i| (i * 7 + 11) as u8);
        let nonce: [u8; 12] = core::array::from_fn(|i| (i * 5 + 2) as u8);
        for (skip, len) in [(0usize, 448usize), (2, 448), (0, 114)] {
            let mut st = State::new_ietf(&key, &nonce);
            st.advance(skip as u32);
            let mut ref_st = st.clone_struct();
            let mut fast: Vec<u8> = (0..len).map(|i| (i * 37 + 3) as u8).collect();
            let mut expect = fast.clone();
            unsafe {
                <Sse2Ops as Ops>::chacha_xor(&mut st, &mut fast);
                crate::chacha::soft::xor(&mut ref_st, &mut expect);
            }
            assert_eq!(expect, fast, "skip {skip} len {len}");
        }
    }
}
