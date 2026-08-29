//! AVX-512 backend: 8-block YMM/vprold ChaCha20 + Goll–Gueron Poly1305, fused.

use crate::{
    Tag,
    aead::Ops,
    chacha::{BLOCK, State},
};

pub(crate) struct Avx512Ops;

impl Ops for Avx512Ops {
    type Poly = crate::poly1305::avx2::Avx2Poly;

    const CHACHA_BATCH: usize = crate::chacha::avx512::BATCH_BLOCKS;

    #[cfg_attr(not(debug_assertions), inline(always))]
    unsafe fn gen_key_xor2(state: &mut State, key_out: &mut [u8; 32], b1: &mut [u8]) {
        debug_assert_eq!(b1.len(), BLOCK);
        crate::chacha::avx512::gen_key_xor2(state, key_out, b1.try_into().unwrap());
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    unsafe fn gen_ks_small(state: &mut State, key_out: &mut [u8; 32], ks: &mut [u8]) {
        crate::chacha::avx512::gen_ks_small(state, key_out, ks);
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    unsafe fn chacha_xor(state: &mut State, buf: &mut [u8]) {
        let mut off = 0usize;
        while buf.len() - off >= 1024 {
            crate::chacha::avx512::xor_batch16(state, buf[off..].as_mut_ptr());
            off += 1024;
        }
        while buf.len() - off >= 512 {
            crate::chacha::avx512::xor_batch8(state, buf[off..].as_mut_ptr());
            off += 512;
        }
        while buf.len() - off >= 256 {
            crate::chacha::avx512::xor_quad2(state, buf[off..].as_mut_ptr());
            off += 256;
        }
        while buf.len() - off >= 128 {
            crate::chacha::avx512::xor_quad(state, buf[off..].as_mut_ptr());
            off += 128;
        }
        while buf.len() - off >= BLOCK {
            crate::chacha::avx512::xor_single(state, buf[off..].as_mut_ptr());
            off += BLOCK;
        }
        if off < buf.len() {
            // sub-block remainder: scalar over the last partial block
            crate::chacha::soft::xor(state, &mut buf[off..]);
        }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    unsafe fn chacha_xor_batch(state: &mut State, buf: &mut [u8]) {
        debug_assert_eq!(buf.len(), 1024);
        crate::chacha::avx512::xor_batch16(state, buf.as_mut_ptr());
    }
}

/// Fused seal entry (requires AVX-512F+VL at runtime).
#[target_feature(enable = "avx512f,avx512vl,avx2")]
pub(crate) unsafe fn seal(state: &mut State, aad: &[u8], msg: &mut [u8], tag: &mut Tag) {
    crate::aead::seal::<Avx512Ops>(state, aad, msg, tag);
}

/// Fused open entry (requires AVX-512F+VL at runtime).
#[target_feature(enable = "avx512f,avx512vl,avx2")]
pub(crate) unsafe fn open(state: &mut State, aad: &[u8], buf: &mut [u8], tag: &Tag) -> bool {
    crate::aead::open::<Avx512Ops>(state, aad, buf, tag)
}

/// Same ChaCha20 kernels, but the Poly1305 side is the 8-lane `vpmadd52`
/// (IFMA) engine. Selected at runtime on CPUs with AVX-512IFMA.
pub(crate) struct Avx512IfmaOps;

impl Ops for Avx512IfmaOps {
    type Poly = crate::poly1305::ifma::IfmaPoly;

    const CHACHA_BATCH: usize = crate::chacha::avx512::BATCH_BLOCKS;
    const DIRECT_SEAL_MAX: usize = 15 * BLOCK;
    const FUSED_MEDIUM: bool = true;
    const FUSED_OPEN: bool = true;

    #[cfg_attr(not(debug_assertions), inline(always))]
    unsafe fn seal_direct(state: &mut State, aad: &[u8], msg: &mut [u8], tag_out: &mut [u8; 16]) {
        // SAFETY: callers reached this through the ifma entry points.
        unsafe {
            crate::chacha::avx512::seal_direct(state, aad, msg.as_mut_ptr(), msg.len(), tag_out);
        }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    unsafe fn gen_key_xor2(state: &mut State, key_out: &mut [u8; 32], b1: &mut [u8]) {
        unsafe { <Avx512Ops as Ops>::gen_key_xor2(state, key_out, b1) }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    unsafe fn gen_ks_small(state: &mut State, key_out: &mut [u8; 32], ks: &mut [u8]) {
        unsafe { <Avx512Ops as Ops>::gen_ks_small(state, key_out, ks) }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    unsafe fn chacha_xor(state: &mut State, buf: &mut [u8]) {
        unsafe { <Avx512Ops as Ops>::chacha_xor(state, buf) }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    unsafe fn chacha_xor_batch(state: &mut State, buf: &mut [u8]) {
        unsafe { <Avx512Ops as Ops>::chacha_xor_batch(state, buf) }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    unsafe fn seal_bulk(
        state: &mut State,
        msg: &mut [u8],
        off: usize,
        poly_off: usize,
        poly: &mut crate::poly1305::Poly<Self::Poly>,
    ) -> (usize, usize) {
        // SAFETY: callers reached this through the ifma entry points.
        unsafe {
            crate::chacha::avx512::xor_batch16_seal_bulk(
                state,
                msg.as_mut_ptr(),
                off,
                poly_off,
                msg.len(),
                poly.inner_mut(),
            )
        }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    unsafe fn open_bulk(
        state: &mut State,
        msg: &mut [u8],
        off: usize,
        poly_off: usize,
        poly: &mut crate::poly1305::Poly<Self::Poly>,
    ) -> (usize, usize) {
        // SAFETY: callers reached this through the ifma entry points.
        unsafe {
            crate::chacha::avx512::xor_batch16_open_bulk(
                state,
                msg.as_mut_ptr(),
                off,
                poly_off,
                msg.len(),
                poly.inner_mut(),
            )
        }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    unsafe fn seal_medium(
        state: &mut State,
        msg: &mut [u8],
        off: usize,
        poly_off: usize,
        poly: &mut crate::poly1305::Poly<Self::Poly>,
    ) -> (usize, usize) {
        // SAFETY: callers reached this through the ifma entry points.
        unsafe {
            crate::chacha::avx512::seal_medium(
                state,
                msg.as_mut_ptr(),
                off,
                poly_off,
                msg.len(),
                poly.inner_mut(),
            )
        }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    unsafe fn open_medium(
        state: &mut State,
        msg: &mut [u8],
        off: usize,
        poly_off: usize,
        poly: &mut crate::poly1305::Poly<Self::Poly>,
    ) -> (usize, usize) {
        // SAFETY: callers reached this through the ifma entry points.
        unsafe {
            crate::chacha::avx512::open_medium(
                state,
                msg.as_mut_ptr(),
                off,
                poly_off,
                msg.len(),
                poly.inner_mut(),
            )
        }
    }
}

/// Fused seal entry (requires AVX-512F+VL+IFMA at runtime).
#[target_feature(enable = "avx512f,avx512vl,avx512ifma,avx2")]
pub(crate) unsafe fn seal_ifma(state: &mut State, aad: &[u8], msg: &mut [u8], tag: &mut Tag) {
    crate::aead::seal::<Avx512IfmaOps>(state, aad, msg, tag);
}

/// Fused open entry (requires AVX-512F+VL+IFMA at runtime).
#[target_feature(enable = "avx512f,avx512vl,avx512ifma,avx2")]
pub(crate) unsafe fn open_ifma(state: &mut State, aad: &[u8], buf: &mut [u8], tag: &Tag) -> bool {
    crate::aead::open::<Avx512IfmaOps>(state, aad, buf, tag)
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

    /// Tests below call AVX-512 kernels directly (bypassing runtime
    /// dispatch); skip on CPUs without the features.
    fn skip_unsupported() -> bool {
        !(std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512vl"))
    }

    // Boundary set covers the alignment prologue (s ≠ 0 shifts), the 8-block
    // batch grid and partial-block tails; len=114/aad=12 was a real BUGFIX
    // case (prologue straddling keystream blocks 1 and 2).
    #[test]
    fn seal_and_open_match_soft() {
        if skip_unsupported() {
            return;
        }
        let key: [u8; 32] = core::array::from_fn(|i| (i * 7 + 11) as u8);
        let nonce: [u8; 12] = core::array::from_fn(|i| (i * 5 + 2) as u8);
        for (len, aad_len) in [
            (0usize, 0usize),
            (1, 0),
            (1, 1),
            (16, 12),
            (48, 12),
            (49, 0),
            (111, 12),
            (112, 12),
            (114, 12),
            (512, 0),
            (576, 12),
            (577, 0),
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
            // Medium-path boundaries: phase A threshold, fused-C rounds
            // 0..=4, block-alignment edges, the steady-loop entry/exit, and
            // the 16-block fused loop hand-off around 2 KiB.
            (193, 0),
            (193, 16),
            (255, 16),
            (256, 0),
            (319, 16),
            (320, 0),
            (383, 16),
            (384, 1),
            (447, 32),
            (448, 16),
            (449, 0),
            (511, 16),
            (512, 1),
            (575, 0),
            (576, 0),
            (576, 1),
            (576, 16),
            (577, 16),
            (639, 16),
            (640, 0),
            (640, 16),
            (641, 16),
            (703, 0),
            (704, 32),
            (767, 16),
            (768, 0),
            (831, 16),
            (832, 5),
            (959, 16),
            (960, 0),
            (1024, 0),
            (1024, 32),
            (1087, 16),
            (1088, 0),
            (1089, 16),
            (1152, 16),
            (1216, 0),
            (1279, 16),
            (1536, 16),
            (1600, 0),
            (1984, 16),
            (2047, 16),
            (2048, 16),
            (2049, 0),
            (2111, 16),
            (2112, 0),
            (2112, 16),
            (2113, 1),
            (2176, 16),
            (3072, 64),
            (4096, 16),
            (8192, 48),
            // AAD sizes driving the flush/pairing guard (s = 0/16/32/48)
            // plus cached-8 flush cases.
            (640, 3),
            (640, 15),
            (640, 17),
            (640, 33),
            (640, 47),
            (640, 48),
            (640, 49),
            (640, 63),
            (640, 64),
            (640, 65),
            (640, 79),
            (640, 80),
            (640, 81),
            (1024, 3),
            (1024, 47),
            (1024, 65),
            (1024, 79),
            (1024, 80),
            (1024, 128),
            (1024, 129),
        ] {
            let msg: Vec<u8> = (0..len).map(|i| (i * 37 + 3) as u8).collect();
            let aad: Vec<u8> = (0..aad_len).map(|i| (i * 41 + 1) as u8).collect();
            let mut soft_buf = msg.clone();
            let soft_tag =
                run_seal::<crate::backend::soft::SoftOps>(&key, &nonce, &aad, &mut soft_buf);
            let mut fast_buf = msg.clone();
            let fast_tag = run_seal::<Avx512Ops>(&key, &nonce, &aad, &mut fast_buf);
            assert_eq!(soft_buf, fast_buf, "ct len {len} aad {aad_len}");
            assert_eq!(soft_tag, fast_tag, "tag len {len} aad {aad_len}");
            assert!(
                run_open::<Avx512Ops>(&key, &nonce, &aad, &mut fast_buf, &fast_tag),
                "open len {len}"
            );
            assert_eq!(fast_buf, msg, "pt len {len} aad {aad_len}");
        }
    }

    #[test]
    fn seal_and_open_match_soft_ifma() {
        if skip_unsupported() || !std::arch::is_x86_feature_detected!("avx512ifma") {
            return;
        }
        let key: [u8; 32] = core::array::from_fn(|i| (i * 7 + 11) as u8);
        let nonce: [u8; 12] = core::array::from_fn(|i| (i * 5 + 2) as u8);
        for (len, aad_len) in [
            (0usize, 0usize),
            (1, 0),
            (16, 12),
            (111, 12),
            (114, 12),
            (512, 0),
            (576, 12),
            (577, 0),
            (1023, 0),
            (1024, 100),
            (1025, 12),
            (2048, 0),
            (4097, 5),
            // Medium fused-path boundaries (phase A threshold, fused-C
            // rounds 0..=4, block-alignment edges, steady-loop entry/exit,
            // 16-block loop hand-off).
            (193, 0),
            (193, 16),
            (256, 16),
            (320, 0),
            (384, 16),
            (448, 1),
            (511, 16),
            (575, 0),
            (576, 0),
            (576, 1),
            (576, 16),
            (577, 16),
            (640, 0),
            (640, 16),
            (641, 16),
            (704, 32),
            (768, 0),
            (832, 5),
            (960, 16),
            (1024, 0),
            (1024, 32),
            (1088, 0),
            (1089, 16),
            (1216, 0),
            (1536, 16),
            (2047, 16),
            (2048, 16),
            (2049, 0),
            (2112, 0),
            (2112, 16),
            (2113, 1),
            (3072, 64),
            (8192, 48),
            // seal_direct band boundaries: quad/zmm kernel switch at 448,
            // single-kernel cap at 960 (961 falls back to the medium path),
            // and partial-block cases in each band.
            (447, 16),
            (448, 0),
            (448, 16),
            (449, 0),
            (449, 16),
            (450, 1),
            (511, 16),
            (959, 16),
            (960, 0),
            (960, 16),
            (961, 0),
            (961, 16),
            (1000, 33),
            (1023, 0),
            (1023, 5),
            (1087, 0),
            (1217, 16),
            (1537, 1),
            (1919, 16),
            (1920, 0),
            (1920, 16),
            (1921, 0),
            (1921, 16),
            (1984, 16),
            // AAD sizes driving the direct path's cached-fold guards.
            (640, 3),
            (640, 15),
            (640, 17),
            (640, 47),
            (640, 48),
            (640, 63),
            (640, 64),
            (640, 65),
            (640, 79),
            (640, 80),
            (640, 81),
            (1024, 3),
            (1024, 65),
            (1024, 80),
            (1024, 128),
            (1024, 129),
        ] {
            let msg: Vec<u8> = (0..len).map(|i| (i * 37 + 3) as u8).collect();
            let aad: Vec<u8> = (0..aad_len).map(|i| (i * 41 + 1) as u8).collect();
            let mut soft_buf = msg.clone();
            let soft_tag =
                run_seal::<crate::backend::soft::SoftOps>(&key, &nonce, &aad, &mut soft_buf);
            let mut fast_buf = msg.clone();
            let fast_tag = run_seal::<Avx512IfmaOps>(&key, &nonce, &aad, &mut fast_buf);
            assert_eq!(soft_buf, fast_buf, "ct len {len} aad {aad_len}");
            assert_eq!(soft_tag, fast_tag, "tag len {len} aad {aad_len}");
            assert!(
                run_open::<Avx512IfmaOps>(&key, &nonce, &aad, &mut fast_buf, &fast_tag),
                "open len {len}"
            );
            assert_eq!(fast_buf, msg, "pt len {len} aad {aad_len}");
        }
    }

    /// `Ops::chacha_xor` must be cursor-exact vs the scalar reference, both at
    /// block boundaries (448 bytes from counter 2) and inside a block (the
    /// 66-byte tail of the len=114 engine sequence).
    #[test]
    fn chacha_xor_matches_soft() {
        if skip_unsupported() {
            return;
        }
        let key: [u8; 32] = core::array::from_fn(|i| (i * 7 + 11) as u8);
        let nonce: [u8; 12] = core::array::from_fn(|i| (i * 5 + 2) as u8);
        for (skip, len) in [(0usize, 448usize), (2, 448), (0, 114)] {
            let mut st = State::new_ietf(&key, &nonce);
            st.advance(skip as u32);
            let mut ref_st = st.clone_struct();
            let mut fast: Vec<u8> = (0..len).map(|i| (i * 37 + 3) as u8).collect();
            let mut expect = fast.clone();
            unsafe {
                <Avx512Ops as Ops>::chacha_xor(&mut st, &mut fast);
                crate::chacha::soft::xor(&mut ref_st, &mut expect);
            }
            assert_eq!(expect, fast, "skip {skip} len {len}");
        }
    }
}
