//! AVX2 backend: 8-block ChaCha20 + Goll–Gueron batched Poly1305, fused.

use crate::{
    Tag,
    aead::Ops,
    chacha::{BLOCK, State},
};

pub(crate) struct Avx2Ops;

impl Ops for Avx2Ops {
    type Poly = crate::poly1305::avx2::Avx2Poly;

    const CHACHA_BATCH: usize = crate::chacha::avx2::BATCH_BLOCKS;

    #[cfg_attr(debug_assertions, inline)]
    #[cfg_attr(not(debug_assertions), inline(always))]
    unsafe fn gen_key_xor2(state: &mut State, key_out: &mut [u8; 32], b1: &mut [u8]) {
        debug_assert_eq!(b1.len(), BLOCK);
        crate::chacha::avx2::gen_key_xor2(state, key_out, b1.try_into().unwrap());
    }

    #[cfg_attr(debug_assertions, inline)]
    #[cfg_attr(not(debug_assertions), inline(always))]
    unsafe fn gen_ks_small(state: &mut State, key_out: &mut [u8; 32], ks: &mut [u8]) {
        crate::chacha::avx2::gen_ks_small(state, key_out, ks);
    }

    #[cfg_attr(debug_assertions, inline)]
    #[cfg_attr(not(debug_assertions), inline(always))]
    unsafe fn chacha_xor(state: &mut State, buf: &mut [u8]) {
        let mut off = 0usize;
        while buf.len() - off >= 512 {
            crate::chacha::avx2::xor_batch8(state, buf[off..].as_mut_ptr());
            off += 512;
        }
        while buf.len() - off >= 256 {
            crate::chacha::avx2::xor_quad2(state, buf[off..].as_mut_ptr());
            off += 256;
        }
        while buf.len() - off >= 128 {
            crate::chacha::avx2::xor_quad(state, buf[off..].as_mut_ptr());
            off += 128;
        }
        while buf.len() - off >= BLOCK {
            crate::chacha::avx2::xor_single(state, buf[off..].as_mut_ptr());
            off += BLOCK;
        }
        // scalar fallback for sub-block tails (never reached: callers feed
        // whole blocks through this path only via the alignment prologue)
        if off < buf.len() {
            crate::chacha::soft::xor(state, &mut buf[off..]);
        }
    }

    #[cfg_attr(debug_assertions, inline)]
    #[cfg_attr(not(debug_assertions), inline(always))]
    unsafe fn chacha_xor_batch(state: &mut State, buf: &mut [u8]) {
        debug_assert_eq!(buf.len(), 512);
        crate::chacha::avx2::xor_batch8(state, buf.as_mut_ptr());
    }
}

/// Fused seal entry (requires AVX2 at runtime).
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn seal(state: &mut State, aad: &[u8], msg: &mut [u8], tag: &mut Tag) {
    crate::aead::seal::<Avx2Ops>(state, aad, msg, tag);
}

/// Fused open entry (requires AVX2 at runtime).
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn open(state: &mut State, aad: &[u8], buf: &mut [u8], tag: &Tag) -> bool {
    crate::aead::open::<Avx2Ops>(state, aad, buf, tag)
}

#[cfg(test)]
mod tests {
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

    #[test]
    fn seal_and_open_match_soft() {
        // Calls AVX2 kernels directly; skip on CPUs without AVX2.
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let key: [u8; 32] = core::array::from_fn(|i| (i * 23 + 5) as u8);
        let nonce: [u8; 12] = core::array::from_fn(|i| (i * 13 + 3) as u8);
        for (len, aad_len) in [
            (0usize, 0usize),
            (1, 1),
            (16, 12),
            (114, 12),
            (512, 0),
            (576, 12),
            (577, 0),
            (1023, 16),
            (1024, 100),
            (1025, 12),
            (2048, 0),
            (4097, 5),
        ] {
            let msg: alloc::vec::Vec<u8> = (0..len).map(|i| (i * 37 + 3) as u8).collect();
            let aad: alloc::vec::Vec<u8> = (0..aad_len).map(|i| (i * 41 + 1) as u8).collect();
            let mut soft_buf = msg.clone();
            let soft_tag =
                run_seal::<crate::backend::soft::SoftOps>(&key, &nonce, &aad, &mut soft_buf);
            let mut fast_buf = soft_buf.clone();
            assert!(
                run_open::<Avx2Ops>(&key, &nonce, &aad, &mut fast_buf, &soft_tag),
                "open len {len}"
            );
            assert_eq!(fast_buf, msg, "pt len {len}");
            let mut fast_buf = msg.clone();
            let fast_tag = run_seal::<Avx2Ops>(&key, &nonce, &aad, &mut fast_buf);
            assert_eq!(soft_buf, fast_buf, "ct len {len} aad {aad_len}");
            assert_eq!(soft_tag, fast_tag, "tag len {len} aad {aad_len}");
        }
    }
}
