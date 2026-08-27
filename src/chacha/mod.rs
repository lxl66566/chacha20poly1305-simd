//! ChaCha20 core (RFC 8439) + HChaCha20.
//!
//! Backends are chosen by [`crate::backend`]; the scalar code here doubles as
//! the correctness reference for the SIMD backends.

// ChaCha quarter-round indices are conventionally named a/b/c/d.
#![allow(clippy::many_single_char_names)]

// Scalar kernels stay compiled everywhere: every SIMD backend tail-handles
// sub-batch remainders through `soft::xor`, and the test suites use it as the
// reference.
pub(crate) mod soft;

// cfg aliases are computed by `build.rs` (forced: only the requested backend
// is compiled; auto: everything reachable on the target arch).
#[cfg(backend_avx2)]
pub(crate) mod avx2;
#[cfg(backend_avx512)]
pub(crate) mod avx512;
#[cfg(backend_neon)]
pub(crate) mod neon;
#[cfg(backend_sse2)]
pub(crate) mod sse2;

/// RFC 8439 block size in bytes.
pub(crate) const BLOCK: usize = 64;
/// RFC 8439 sigma constants.
pub(crate) const CONSTANTS: [u32; 4] = [0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574];

/// Working ChaCha20 state: `constants || key || counter || nonce` (16 words).
///
/// The counter is a full u32 kept in `words[12]`; callers advance it by whole
/// blocks. Messages never exceed u32::MAX blocks (checked in the engine).
pub(crate) struct State {
    pub(crate) words: [u32; 16],
}

/// Load little-endian u32 words from `src` into `dst`.
#[inline]
fn load_le_words(src: &[u8], dst: &mut [u32]) {
    for (d, chunk) in dst.iter_mut().zip(src.as_chunks::<4>().0) {
        *d = u32::from_le_bytes(*chunk);
    }
}

/// `dst ^= src` over equal-length slices, 8 bytes at a time so LLVM lowers
/// it to wide loads/stores instead of byte ops. Shared by the AEAD engine
/// and the small-message kernels' partial-block handling.
#[inline(always)]
pub(crate) fn xor_bytes(dst: &mut [u8], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    let (chunks, tail) = dst.as_chunks_mut::<8>();
    let (srcs, _) = src.as_chunks::<8>();
    for (d, s) in chunks.iter_mut().zip(srcs) {
        *d = (u64::from_le_bytes(*d) ^ u64::from_le_bytes(*s)).to_le_bytes();
    }
    let base = src.len() - tail.len();
    for (i, d) in tail.iter_mut().enumerate() {
        *d ^= src[base + i];
    }
}

impl State {
    /// IETF ChaCha20 state with counter = 0.
    pub(crate) fn new_ietf(key: &[u8; 32], nonce: &[u8; 12]) -> Self {
        let mut words = [0u32; 16];
        words[..4].copy_from_slice(&CONSTANTS);
        load_le_words(key, &mut words[4..12]);
        words[12] = 0;
        load_le_words(nonce, &mut words[13..]);
        Self { words }
    }

    /// Test-only deep copy helper.
    #[cfg(test)]
    pub(crate) fn clone_struct(&self) -> Self {
        Self { words: self.words }
    }

    /// Advance the block counter (wrapping, mirroring upstream semantics at
    /// the u32 boundary; the engine prevents reaching it).
    #[inline(always)]
    pub(crate) fn advance(&mut self, blocks: u32) {
        self.words[12] = self.words[12].wrapping_add(blocks);
    }

    /// Scrub the state — `words[4..12]` holds the cipher key.
    #[cfg(feature = "zeroize")]
    #[inline]
    pub(crate) fn zeroize(&mut self) {
        zeroize::Zeroize::zeroize(&mut self.words);
    }
}

/// HChaCha20: derive a 32-byte subkey from `key` and the first 16 bytes of
/// `xnonce` (draft-irtf-cfrg-xchacha §2.2).
///
/// Outputs the pre-addition state words `0..4 || 12..16`.
pub(crate) fn hchacha20(key: &[u8; 32], xnonce: &[u8; 24], out: &mut [u8; 32]) {
    let mut x = [0u32; 16];
    x[..4].copy_from_slice(&CONSTANTS);
    load_le_words(key, &mut x[4..12]);
    load_le_words(&xnonce[..16], &mut x[12..]);

    // HChaCha20 skips the final state addition: output is the raw post-round
    // words `0..4 || 12..16`.
    for _ in 0..10 {
        quarter_round(&mut x, 0, 4, 8, 12);
        quarter_round(&mut x, 1, 5, 9, 13);
        quarter_round(&mut x, 2, 6, 10, 14);
        quarter_round(&mut x, 3, 7, 11, 15);
        quarter_round(&mut x, 0, 5, 10, 15);
        quarter_round(&mut x, 1, 6, 11, 12);
        quarter_round(&mut x, 2, 7, 8, 13);
        quarter_round(&mut x, 3, 4, 9, 14);
    }

    for (i, &w) in x[..4].iter().chain(x[12..].iter()).enumerate() {
        out[i * 4..][..4].copy_from_slice(&w.to_le_bytes());
    }
}

/// Scalar ChaCha20 quarter round on word indices.
#[inline(always)]
fn quarter_round(x: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    x[a] = x[a].wrapping_add(x[b]);
    x[d] = (x[d] ^ x[a]).rotate_left(16);
    x[c] = x[c].wrapping_add(x[d]);
    x[b] = (x[b] ^ x[c]).rotate_left(12);
    x[a] = x[a].wrapping_add(x[b]);
    x[d] = (x[d] ^ x[a]).rotate_left(8);
    x[c] = x[c].wrapping_add(x[d]);
    x[b] = (x[b] ^ x[c]).rotate_left(7);
}
