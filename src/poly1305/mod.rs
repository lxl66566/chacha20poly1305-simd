//! Poly1305 facade + backend contract.
//!
//! Streaming shape tailored to the RFC 8439 AEAD construction: every absorbed
//! 16-byte block carries the 2^128 high bit (the AEAD zero-pads AAD and
//! ciphertext; there is no `2^(8·len)` final-partial block here), and the
//! engine inserts the explicit zero padding, so [`Poly::finalize_into`]
//! assumes the stream ended on a block boundary.

// Soft Poly1305 serves the soft / sse2 backends and doubles as the test
// reference (cfg aliases computed by `build.rs`).
#[cfg(any(backend_soft, backend_sse2, test))]
pub(crate) mod soft;

#[cfg(test)]
mod test_common;

// Shared by the avx2 and avx512 AEAD backends.
#[cfg(any(backend_avx2, backend_avx512))]
pub(crate) mod avx2;
// AVX-512 IFMA (vpmadd52) Poly1305 for the avx512-ifma dispatch tier.
#[cfg(backend_avx512)]
pub(crate) mod ifma;
#[cfg(backend_neon)]
pub(crate) mod neon;

/// Primitive operations each Poly1305 backend must provide.
pub(crate) trait Backend {
    /// Init from a one-time key: clamped `r`, addition key `s`.
    unsafe fn init(key: &[u8; 32]) -> Self
    where
        Self: Sized;
    /// Absorb one full block with the 2^128 high bit set.
    unsafe fn absorb_block(&mut self, block: &[u8; 16]);
    /// Hot path: absorb exactly 4 blocks at a 64-byte aligned stream
    /// position. Only called when no partial block is buffered and
    /// [`Backend::pending_blocks`] is a multiple of 4 (any deferred batches
    /// are drained first, in order).
    unsafe fn absorb4(&mut self, blocks: &[u8; 64]);
    /// Hot path: absorb `blocks.len()` bytes (`% 64 == 0`) at a 64-byte
    /// aligned stream position. Default: per-window [`Backend::absorb4`];
    /// backends with wide batching override it to keep state in registers
    /// across windows.
    #[inline(always)]
    unsafe fn absorb_blocks(&mut self, blocks: &[u8]) {
        debug_assert_eq!(blocks.len() % 64, 0);
        for w in blocks.chunks_exact(64) {
            unsafe { self.absorb4(w.try_into().unwrap()) };
        }
    }
    /// Whole blocks currently batched but not yet folded (0..=8).
    fn pending_blocks(&self) -> usize;
    /// Flush batching, reduce, add `s` and emit the tag.
    unsafe fn finalize_into(&mut self, out: &mut [u8; 16]);
    /// Best-effort scrub of secret state (clamped `r`, powers, accumulator,
    /// pad key). Volatile writes so the compiler cannot elide them.
    #[cfg(feature = "zeroize")]
    unsafe fn zeroize_secrets(&mut self);
}

/// Streaming wrapper: forwards whole blocks, buffers a partial tail until
/// more data (or the engine's explicit zero padding) completes it.
pub(crate) struct Poly<B: Backend> {
    inner: B,
    pending: [u8; 16],
    have: usize,
    #[cfg(test)]
    pub(crate) dbg_absorbed: usize,
    #[cfg(test)]
    pub(crate) dbg_digest: u64,
}

impl<B: Backend> Poly<B> {
    #[inline(always)]
    pub(crate) unsafe fn new(key: &[u8; 32]) -> Self {
        Self {
            inner: B::init(key),
            pending: [0; 16],
            have: 0,
            #[cfg(test)]
            dbg_absorbed: 0,
            #[cfg(test)]
            dbg_digest: 0,
        }
    }

    /// Absorb an arbitrary-length segment (no padding).
    #[inline(always)]
    pub(crate) unsafe fn update(&mut self, mut data: &[u8]) {
        if self.have > 0 {
            // BUGFIX: fill up to the block boundary, not `have` bytes — the
            // old `self.have.min(data.len())` silently dropped data when
            // `data` was longer than the pending remainder.
            let n = (16 - self.have).min(data.len());
            self.pending[self.have..][..n].copy_from_slice(&data[..n]);
            self.have += n;
            data = &data[n..];
            if self.have == 16 {
                let block = self.pending;
                self.inner.absorb_block(&block);
                self.have = 0;
                debug_assert!(data.is_empty() || self.have == 0);
            } else {
                debug_assert_eq!(data.len(), 0);
                return;
            }
        }
        debug_assert_eq!(self.have, 0);
        let (chunks, rem) = data.as_chunks::<16>();
        for block in chunks {
            #[cfg(test)]
            {
                self.dbg_absorbed += 16;
                for &b in block {
                    self.dbg_digest = self.dbg_digest.rotate_left(7) ^ u64::from(b);
                }
            }
            self.inner.absorb_block(block);
        }
        self.pending = [0; 16];
        self.pending[..rem.len()].copy_from_slice(rem);
        self.have = rem.len();
    }

    /// Hot-path absorb of exactly 4 blocks at a 64-byte aligned stream
    /// position (no partial tail buffered; `pending_blocks() % 4 == 0`).
    #[inline(always)]
    pub(crate) unsafe fn absorb4(&mut self, blocks: &[u8; 64]) {
        debug_assert_eq!(self.have, 0);
        debug_assert_eq!(self.inner.pending_blocks() % 4, 0);
        #[cfg(test)]
        {
            self.dbg_absorbed += 64;
            for &b in blocks {
                self.dbg_digest = self.dbg_digest.rotate_left(7) ^ u64::from(b);
            }
        }
        self.inner.absorb4(blocks);
    }

    /// Hot-path absorb of `blocks.len()` bytes (`% 64 == 0`) at a 64-byte
    /// aligned stream position.
    #[inline(always)]
    pub(crate) unsafe fn absorb_blocks(&mut self, blocks: &[u8]) {
        debug_assert_eq!(self.have, 0);
        debug_assert_eq!(self.inner.pending_blocks() % 4, 0);
        debug_assert_eq!(blocks.len() % 64, 0);
        #[cfg(test)]
        {
            self.dbg_absorbed += blocks.len();
            for &b in blocks {
                self.dbg_digest = self.dbg_digest.rotate_left(7) ^ u64::from(b);
            }
        }
        unsafe { self.inner.absorb_blocks(blocks) };
    }

    /// Whole blocks not yet folded: batched blocks plus an implicit one if a
    /// partial block is buffered. The engine aligns the bulk stream to
    /// 64-byte boundaries with this.
    #[inline(always)]
    pub(crate) fn pending_blocks(&self) -> usize {
        self.inner.pending_blocks() + usize::from(self.have > 0)
    }

    /// Flush everything and emit the tag. The stream must already be
    /// zero-padded to a block boundary by the engine.
    #[inline(always)]
    pub(crate) unsafe fn finalize_into(&mut self, out: &mut [u8; 16]) {
        debug_assert_eq!(self.have, 0, "engine must zero-pad before finalize");
        self.inner.finalize_into(out);
    }
}

#[cfg(feature = "zeroize")]
impl<B: Backend> Drop for Poly<B> {
    fn drop(&mut self) {
        zeroize::Zeroize::zeroize(&mut self.pending);
        // SAFETY: best-effort scrub; all backend fields are plain data.
        unsafe { self.inner.zeroize_secrets() };
    }
}

/// Zeroize the one-time Poly1305 key bytes when the feature is enabled.
#[cfg(feature = "zeroize")]
#[inline]
pub(crate) fn clear_key(bytes: &mut [u8]) {
    zeroize::Zeroize::zeroize(bytes);
}

#[cfg(not(feature = "zeroize"))]
#[inline]
pub(crate) fn clear_key(_bytes: &mut [u8]) {}
