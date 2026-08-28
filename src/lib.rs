//! Performance-first [`ChaCha20-Poly1305`][rfc] AEAD implementation with
//! [`XChaCha20-Poly1305`][xchacha] support, hand-tuned for x86_64 and aarch64.
//!
//! This crate is a from-scratch, perf-first rewrite of the
//! [RustCrypto `chacha20poly1305`][rustcrypto] crate:
//!
//! - Fused pipeline: ChaCha20 keystream generation, message XOR and Poly1305 absorption are
//!   interleaved in a single loop so their independent latency chains overlap (the upstream TODO).
//! - Hand-written AVX2 (8-block ChaCha20, Goll–Gueron batched Poly1305) and AVX-512VL (8-block
//!   ChaCha20 in YMM registers with single-instruction `vprold` rotations) backends on x86_64, an
//!   SSE2 backend (4-block ChaCha20, x86-64 baseline — the no-AVX2 fallback matching upstream's
//!   sse2 tier) with scalar Poly1305, and a NEON backend (4-block ChaCha20, lane-parallel 26-bit
//!   limb Poly1305) on aarch64 — all runtime dispatched and cached in a single atomic load.
//! - Zero-copy detached API operating directly on caller buffers; no trait objects, no per-block
//!   buffering on the bulk path.
//! - Backends are runtime-dispatched and cached in a single atomic load; they can also be pinned at
//!   compile time the RustCrypto way — `RUSTFLAGS='--cfg chacha20poly1305_backend="avx2"
//!   -Ctarget-feature=+avx2'` — which compiles exactly one backend, removes the rest from the
//!   binary and resolves dispatch to a compile-time constant. The AVX-512 tier sits behind the
//!   default-on [`avx512`](crate#features) feature; disable it with `default-features = false` to
//!   keep AVX-512 instructions out of the binary (dispatch then stops at AVX2).
//!
//! # Features
//!
//! | feature   | default | description                                                      |
//! | --------- | ------- | ---------------------------------------------------------------- |
//! | `std`     | ✓       | runtime CPU detection (otherwise compile-time `target_feature`s) |
//! | `alloc`   | ✓       | allocating API                                                   |
//! | `avx512`  | ✓       | x86-64 AVX-512 backend; disable to keep the ISA out of the binary |
//! | `zeroize` | –       | zeroize keys and intermediate secrets on drop                    |
//! | `hotpath` | –       | [hotpath](https://crates.io/crates/hotpath) probes               |
//!
//! # Usage
//!
//! ```
//! use chacha20poly1305_simd::{ChaCha20Poly1305, Key, Nonce, Tag};
//!
//! let key = [0x42u8; 32];
//! let nonce = [7u8; 12];
//! let cipher = ChaCha20Poly1305::new(&key);
//!
//! let mut buffer = *b"hello world";
//! let tag: Tag = cipher
//!     .encrypt_in_place_detached(&nonce, b"aad", &mut buffer)
//!     .unwrap();
//! assert_ne!(&buffer, b"hello world");
//!
//! cipher
//!     .decrypt_in_place_detached(&nonce, b"aad", &mut buffer, &tag)
//!     .unwrap();
//! assert_eq!(&buffer, b"hello world");
//! ```
//!
//! Allocated variants appends / expects the 16-byte tag at the end of the
//! buffer, mirroring the upstream crate's wire format:
//!
//! ```
//! # use chacha20poly1305_simd::{XChaCha20Poly1305, XNonce};
//! let cipher = XChaCha20Poly1305::new(&[1u8; 32]);
//! let nonce = [2u8; 24];
//! let ct = cipher
//!     .encrypt(&nonce, b"header", b"secret payload")
//!     .unwrap();
//! assert_eq!(ct.len(), b"secret payload".len() + 16);
//! assert_eq!(
//!     cipher.decrypt(&nonce, b"header", &ct).unwrap(),
//!     b"secret payload"
//! );
//! ```
//!
//! On decryption failure the in-place buffer contents are unspecified
//! (decryption is fused with authentication and proceeds before the tag is
//! verified, as in BoringSSL).
//!
//! [rfc]: https://tools.ietf.org/html/rfc8439
//! [xchacha]: https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-xchacha
//! [rustcrypto]: https://github.com/RustCrypto/AEADs/tree/master/chacha20poly1305
#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(
    clippy::inline_always,
    clippy::needless_range_loop,
    unsafe_op_in_unsafe_fn,
    // All unsafe fns are crate-internal; safety invariants are documented at
    // the call sites / module docs rather than repeated per function.
    clippy::missing_safety_doc,
    // Deliberate truncating casts in the SIMD counter/lane math and in tests.
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
)]
// BUGFIX: every `#[inline(always)]` below is downgraded to a plain `#[inline]`
// hint under `debug_assertions`. LLVM's always-inliner runs even at -O0, so
// debug builds merged the whole fused pipeline (engine + kernels + Poly1305
// scalar helpers) into one function whose un-coalesced frame reached ~1.5 MiB
// and overflowed the default 1 MiB main-thread stack (doctests exited with
// STATUS_STACK_OVERFLOW on Windows MSVC; test threads pass only because they
// get 2 MiB). Release codegen is unchanged (byte-identical disassembly).

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

mod aead;
mod backend;
mod chacha;
mod poly1305;

use core::fmt;

use crate::backend::with_backend;
pub use crate::{aead::Error, backend::active_backend};

/// 256-bit secret key, shared by every cipher in this crate.
pub type Key = [u8; 32];
/// 96-bit (RFC 8439) nonce.
pub type Nonce = [u8; 12];
/// 192-bit extended nonce (XChaCha20).
pub type XNonce = [u8; 24];
/// Poly1305 authentication tag.
pub type Tag = [u8; 16];

/// ChaCha20Poly1305 AEAD (RFC 8439).
// Copy is unavailable under `zeroize`: the Drop impl zeroizes the key.
#[derive(Clone)]
#[cfg_attr(not(feature = "zeroize"), derive(Copy))]
pub struct ChaCha20Poly1305 {
    key: Key,
}

impl ChaCha20Poly1305 {
    /// Create a new cipher instance from a 256-bit key.
    #[must_use]
    pub const fn new(key: &Key) -> Self {
        Self { key: *key }
    }

    /// Derive the ChaCha20 state used to seal / open `nonce` (counter = 0).
    #[inline]
    fn init(&self, nonce: &Nonce) -> chacha::State {
        chacha::State::new_ietf(&self.key, nonce)
    }

    /// Encrypt `plaintext` allocating a `ciphertext || tag` buffer.
    ///
    /// Returns [`Error`] if `plaintext.len()` exceeds the ChaCha20 counter
    /// space (256 GiB - 64 KiB).
    #[cfg(feature = "alloc")]
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    pub fn encrypt(
        &self,
        nonce: &Nonce,
        aad: &[u8],
        plaintext: &[u8],
    ) -> Result<alloc::vec::Vec<u8>, Error> {
        let mut buf = alloc::vec![0u8; plaintext.len() + 16];
        buf[..plaintext.len()].copy_from_slice(plaintext);
        let tag = self.encrypt_in_place_detached(nonce, aad, &mut buf[..plaintext.len()])?;
        buf[plaintext.len()..].copy_from_slice(&tag);
        Ok(buf)
    }

    /// Decrypt `ciphertext || tag`, returning the plaintext, or [`Error`] on
    /// authentication failure.
    #[cfg(feature = "alloc")]
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    pub fn decrypt(
        &self,
        nonce: &Nonce,
        aad: &[u8],
        ciphertext_and_tag: &[u8],
    ) -> Result<alloc::vec::Vec<u8>, Error> {
        let (ct, tag) =
            ciphertext_and_tag.split_at(ciphertext_and_tag.len().checked_sub(16).ok_or(Error)?);
        let mut buf = alloc::vec![0u8; ct.len()];
        buf.copy_from_slice(ct);
        self.decrypt_in_place_detached(
            nonce,
            aad,
            &mut buf,
            tag.try_into().expect("length checked"),
        )?;
        Ok(buf)
    }

    /// In-place encryption returning the detached tag.
    ///
    /// `msg` is replaced by the ciphertext of the same length. Returns
    /// [`Error`] if `msg.len()` exceeds the ChaCha20 counter space.
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    pub fn encrypt_in_place_detached(
        &self,
        nonce: &Nonce,
        aad: &[u8],
        msg: &mut [u8],
    ) -> Result<Tag, Error> {
        if msg.len() >= aead::MAX_LEN {
            return Err(Error);
        }
        let mut tag = Tag::default();
        let mut st = self.init(nonce);
        with_backend(|b| b.seal(&mut st, aad, msg, &mut tag));
        #[cfg(feature = "zeroize")]
        st.zeroize();
        Ok(tag)
    }

    /// In-place decryption of a detached tag.
    ///
    /// Returns [`Error`] if `tag` does not authenticate; `buf` contents are
    /// unspecified after failure.
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    pub fn decrypt_in_place_detached(
        &self,
        nonce: &Nonce,
        aad: &[u8],
        buf: &mut [u8],
        tag: &Tag,
    ) -> Result<(), Error> {
        let mut st = self.init(nonce);
        let r = if buf.len() >= aead::MAX_LEN {
            Err(Error)
        } else {
            with_backend(|b| {
                if b.open(&mut st, aad, buf, tag) {
                    Ok(())
                } else {
                    Err(Error)
                }
            })
        };
        #[cfg(feature = "zeroize")]
        st.zeroize();
        r
    }

    /// In-place encryption appending the tag to a `Vec`-like buffer.
    ///
    /// Returns [`Error`] if `buffer.len()` exceeds the ChaCha20 counter space.
    #[cfg(feature = "alloc")]
    pub fn encrypt_in_place(
        &self,
        nonce: &Nonce,
        aad: &[u8],
        buffer: &mut alloc::vec::Vec<u8>,
    ) -> Result<(), Error> {
        let tag = self.encrypt_in_place_detached(nonce, aad, buffer)?;
        buffer.extend_from_slice(&tag);
        Ok(())
    }

    /// In-place decryption stripping a trailing tag from a `Vec`-like buffer.
    #[cfg(feature = "alloc")]
    pub fn decrypt_in_place(
        &self,
        nonce: &Nonce,
        aad: &[u8],
        buffer: &mut alloc::vec::Vec<u8>,
    ) -> Result<(), Error> {
        if buffer.len() < 16 {
            return Err(Error);
        }
        let tag: Tag = buffer
            .split_off(buffer.len() - 16)
            .try_into()
            .expect("16 bytes");
        self.decrypt_in_place_detached(nonce, aad, buffer, &tag)
    }
}

/// XChaCha20Poly1305 AEAD (extended 192-bit nonce).
// Copy is unavailable under `zeroize`: the Drop impl zeroizes the key.
#[derive(Clone)]
#[cfg_attr(not(feature = "zeroize"), derive(Copy))]
pub struct XChaCha20Poly1305 {
    key: Key,
}

impl XChaCha20Poly1305 {
    /// Create a new cipher instance from a 256-bit key.
    #[must_use]
    pub const fn new(key: &Key) -> Self {
        Self { key: *key }
    }

    #[cfg(feature = "alloc")]
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    pub fn encrypt(
        &self,
        nonce: &XNonce,
        aad: &[u8],
        plaintext: &[u8],
    ) -> Result<alloc::vec::Vec<u8>, Error> {
        #[allow(unused_mut)] // mut needed only under `zeroize`
        let (mut key, iv) = self.subkey(nonce);
        let cipher = ChaCha20Poly1305::new(&key);
        #[cfg(feature = "zeroize")]
        zeroize::Zeroize::zeroize(&mut key);
        cipher.encrypt(&iv, aad, plaintext)
    }

    #[cfg(feature = "alloc")]
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    pub fn decrypt(
        &self,
        nonce: &XNonce,
        aad: &[u8],
        ciphertext_and_tag: &[u8],
    ) -> Result<alloc::vec::Vec<u8>, Error> {
        #[allow(unused_mut)] // mut needed only under `zeroize`
        let (mut key, iv) = self.subkey(nonce);
        let cipher = ChaCha20Poly1305::new(&key);
        #[cfg(feature = "zeroize")]
        zeroize::Zeroize::zeroize(&mut key);
        cipher.decrypt(&iv, aad, ciphertext_and_tag)
    }

    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    pub fn encrypt_in_place_detached(
        &self,
        nonce: &XNonce,
        aad: &[u8],
        msg: &mut [u8],
    ) -> Result<Tag, Error> {
        #[allow(unused_mut)] // mut needed only under `zeroize`
        let (mut key, iv) = self.subkey(nonce);
        let cipher = ChaCha20Poly1305::new(&key);
        #[cfg(feature = "zeroize")]
        zeroize::Zeroize::zeroize(&mut key);
        cipher.encrypt_in_place_detached(&iv, aad, msg)
    }

    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    pub fn decrypt_in_place_detached(
        &self,
        nonce: &XNonce,
        aad: &[u8],
        buf: &mut [u8],
        tag: &Tag,
    ) -> Result<(), Error> {
        #[allow(unused_mut)] // mut needed only under `zeroize`
        let (mut key, iv) = self.subkey(nonce);
        let cipher = ChaCha20Poly1305::new(&key);
        #[cfg(feature = "zeroize")]
        zeroize::Zeroize::zeroize(&mut key);
        cipher.decrypt_in_place_detached(&iv, aad, buf, tag)
    }

    /// XChaCha20 key derivation: HChaCha20 subkey + `0^4 || nonce[16..24]` IV.
    #[inline]
    fn subkey(&self, nonce: &XNonce) -> (Key, Nonce) {
        let mut key = [0u8; 32];
        chacha::hchacha20(&self.key, nonce, &mut key);
        let mut iv = [0u8; 12];
        iv[4..].copy_from_slice(&nonce[16..]);
        (key, iv)
    }
}

impl fmt::Debug for ChaCha20Poly1305 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChaCha20Poly1305").finish_non_exhaustive()
    }
}

impl fmt::Debug for XChaCha20Poly1305 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("XChaCha20Poly1305").finish_non_exhaustive()
    }
}

#[cfg(feature = "zeroize")]
mod zeroize_impls {
    use zeroize::Zeroize;

    use super::{ChaCha20Poly1305, XChaCha20Poly1305};

    // NOTE: both structs hold the raw key; the per-message derived secrets
    // (Poly1305 key/state) are zeroized inside the engine when the feature is on.
    impl Drop for ChaCha20Poly1305 {
        fn drop(&mut self) {
            self.key.zeroize();
        }
    }

    impl Drop for XChaCha20Poly1305 {
        fn drop(&mut self) {
            self.key.zeroize();
        }
    }

    impl zeroize::ZeroizeOnDrop for ChaCha20Poly1305 {}
    impl zeroize::ZeroizeOnDrop for XChaCha20Poly1305 {}
}
