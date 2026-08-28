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
//! | feature     | default | description                                                       |
//! | ----------- | ------- | ----------------------------------------------------------------- |
//! | `std`       | ✓       | runtime CPU detection (otherwise compile-time `target_feature`s)  |
//! | `alloc`     | ✓       | allocating API                                                    |
//! | `avx512`    | ✓       | x86-64 AVX-512 backend; disable to keep the ISA out of the binary |
//! | `getrandom` | –       | `Key`/`Nonce` random generation via [`Generate`]                  |
//! | `aead`      | –       | RustCrypto `aead` 0.6 trait impls ([`aead_compat`])               |
//! | `bytes`     | –       | [`Buffer`] impl for `bytes::BytesMut`                             |
//! | `zeroize`   | –       | zeroize keys and intermediate secrets on drop                     |
//! | `hotpath`   | –       | [hotpath](https://crates.io/crates/hotpath) probes                |
//!
//! # Usage
//!
//! ```
//! use chacha20poly1305_simd::{ChaCha20Poly1305, Key, Nonce, Tag};
//!
//! let key = [0x42u8; 32];
//! let nonce = [7u8; 12];
//! let cipher = ChaCha20Poly1305::new(key);
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
//! Allocated variants append / expect the 16-byte tag at the end of the
//! buffer, mirroring the upstream crate's wire format. AAD is optional — a
//! bare `&[u8]` message auto-coerces into a [`Payload`] with empty AAD:
#![cfg_attr(feature = "alloc", doc = "```")]
#![cfg_attr(not(feature = "alloc"), doc = "```ignore")]
//! # use chacha20poly1305_simd::{Payload, XChaCha20Poly1305, XNonce};
//! let cipher = XChaCha20Poly1305::new([1u8; 32]);
//! let nonce = [2u8; 24];
//! // no AAD:
//! let ct = cipher.encrypt(&nonce, b"secret payload".as_ref()).unwrap();
//! assert_eq!(ct.len(), b"secret payload".len() + 16);
//! assert_eq!(
//!     cipher.decrypt(&nonce, ct.as_slice()).unwrap(),
//!     b"secret payload"
//! );
//!
//! // with AAD:
//! let ct = cipher
//!     .encrypt(&nonce, Payload { msg: b"secret payload", aad: b"header" })
//!     .unwrap();
//! assert_eq!(
//!     cipher.decrypt(&nonce, Payload { msg: &ct, aad: b"header" }).unwrap(),
//!     b"secret payload"
//! );
//! ```
//! 
//! Keys and nonces can be generated from the OS CSPRNG with the (optional)
//! `getrandom` feature:
#![cfg_attr(feature = "getrandom", doc = "```")]
#![cfg_attr(not(feature = "getrandom"), doc = "```ignore")]
//! # use chacha20poly1305_simd::{ChaCha20Poly1305, Generate, Key, Nonce};
//! let cipher = ChaCha20Poly1305::new(Key::generate());
//! let nonce = Nonce::generate();
//! # let _ = cipher;
//! # let _ = nonce;
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
// Use `#[cfg_attr(not(debug_assertions), inline(always))]` instead.

#[cfg(any(feature = "alloc", test))]
extern crate alloc;

#[cfg(any(feature = "std", test))]
extern crate std;

mod aead;
mod backend;
mod chacha;
mod poly1305;

#[cfg(feature = "aead")]
#[cfg_attr(docsrs, doc(cfg(feature = "aead")))]
pub mod aead_compat;

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

/// Split a trailing 16-byte Poly1305 tag off `buffer` — the recurring
/// boilerplate when driving the `*_detached` APIs over a `ciphertext || tag`
/// wire format.
///
/// # Errors
///
/// Returns [`Error::InvalidLength`] if `buffer` is shorter than the tag.
///
/// # Example
///
/// ```
/// use chacha20poly1305_simd::{ChaCha20Poly1305, split_tag};
///
/// let cipher = ChaCha20Poly1305::new([0x42u8; 32]);
/// let nonce = [7u8; 12];
///
/// let mut msg = *b"hello world";
/// let tag = cipher
///     .encrypt_in_place_detached(&nonce, b"aad", &mut msg)
///     .unwrap();
/// let mut wire = [&msg[..], &tag[..]].concat();
///
/// let (ct, tag) = split_tag(&mut wire).unwrap();
/// cipher
///     .decrypt_in_place_detached(&nonce, b"aad", ct, tag)
///     .unwrap();
/// ```
pub fn split_tag(buffer: &mut [u8]) -> Result<(&mut [u8], &Tag), Error> {
    let len = buffer.len().checked_sub(16).ok_or(Error::InvalidLength)?;
    let (body, tag) = buffer.split_at_mut(len);
    let tag: &Tag = (&*tag).try_into().expect("16 bytes");
    Ok((body, tag))
}

/// Payload for the allocating [`encrypt`](ChaCha20Poly1305::encrypt) /
/// [`decrypt`](ChaCha20Poly1305::decrypt) methods.
///
/// A bare `&[u8]` coerces to a payload with empty AAD via [`From`], so
/// associated data can be omitted entirely — pass an explicit [`Payload`]
/// to authenticate associated data:
///
/// ```
/// use chacha20poly1305_simd::Payload;
///
/// let p: Payload = (&b"hello"[..]).into();
/// assert_eq!(p.msg, b"hello");
/// assert!(p.aad.is_empty());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Payload<'msg, 'aad> {
    /// Message to encrypt / decrypt.
    pub msg: &'msg [u8],
    /// Additional authenticated data (not encrypted).
    pub aad: &'aad [u8],
}

impl<'msg> From<&'msg [u8]> for Payload<'msg, '_> {
    fn from(msg: &'msg [u8]) -> Self {
        Self { msg, aad: b"" }
    }
}

/// Growable in-place byte buffer for
/// [`encrypt_in_place`](ChaCha20Poly1305::encrypt_in_place) /
/// [`decrypt_in_place`](ChaCha20Poly1305::decrypt_in_place).
///
/// Implemented for `Vec<u8>` and — behind the respective features — for
/// [`Zeroizing<Vec<u8>>`](zeroize::Zeroizing) and
/// [`BytesMut`](bytes::BytesMut); implement it for your own storage to run
/// the in-place API directly on your buffer without copying into a `Vec`.
///
/// Slice access is expressed through the trait's own methods (not
/// `AsRef`/`AsMut` supertraits), so foreign wrapper types can never satisfy
/// the bound via the orphan rule alone — the impls above ship first-party
/// instead.
///
/// # Invariants
///
/// Implementations must uphold:
///
/// - `as_slice().len() == len()` (likewise `as_mut_slice`),
/// - `extend_from_slice` appends `other` to the end of the buffer,
/// - `truncate(n)` shrinks the buffer to `n` bytes if it is currently longer, and is a no-op
///   otherwise.
pub trait Buffer {
    /// Current length in bytes.
    #[must_use]
    fn len(&self) -> usize;

    /// Whether the buffer is empty.
    #[must_use]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow the initialized contents.
    #[must_use]
    fn as_slice(&self) -> &[u8];

    /// Mutably borrow the initialized contents.
    fn as_mut_slice(&mut self) -> &mut [u8];

    /// Append bytes, returning [`Error`] if there is no capacity.
    fn extend_from_slice(&mut self, other: &[u8]) -> Result<(), Error>;

    /// Shrink the buffer to `len` bytes.
    fn truncate(&mut self, len: usize);
}

#[cfg(feature = "alloc")]
impl Buffer for alloc::vec::Vec<u8> {
    fn len(&self) -> usize {
        alloc::vec::Vec::len(self)
    }

    fn is_empty(&self) -> bool {
        alloc::vec::Vec::is_empty(self)
    }

    fn as_slice(&self) -> &[u8] {
        alloc::vec::Vec::as_slice(self)
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        alloc::vec::Vec::as_mut_slice(self)
    }

    fn extend_from_slice(&mut self, other: &[u8]) -> Result<(), Error> {
        alloc::vec::Vec::extend_from_slice(self, other);
        Ok(())
    }

    fn truncate(&mut self, len: usize) {
        alloc::vec::Vec::truncate(self, len);
    }
}

// NOTE: foreign wrapper types (Zeroizing, BytesMut) can only get a `Buffer`
// impl from us — the local trait satisfies the orphan rule; their own crates
// cannot implement our trait for these same types.
//
// The inner accessors below are single deref-coercion sites: writing
// `(**self).len()` trips `explicit_auto_deref`, while `self.len()` trips a
// false-positive `unconditional_recursion` (resolution picks the inherent
// `Vec` method through Deref at runtime — verified by the round-trip tests).
#[cfg(all(feature = "alloc", feature = "zeroize"))]
fn zeroizing_inner(buf: &zeroize::Zeroizing<alloc::vec::Vec<u8>>) -> &alloc::vec::Vec<u8> {
    buf
}

#[cfg(all(feature = "alloc", feature = "zeroize"))]
fn zeroizing_inner_mut(
    buf: &mut zeroize::Zeroizing<alloc::vec::Vec<u8>>,
) -> &mut alloc::vec::Vec<u8> {
    buf
}

#[cfg(all(feature = "alloc", feature = "zeroize"))]
#[cfg_attr(docsrs, doc(cfg(all(feature = "alloc", feature = "zeroize"))))]
impl Buffer for zeroize::Zeroizing<alloc::vec::Vec<u8>> {
    fn len(&self) -> usize {
        zeroizing_inner(self).len()
    }

    fn is_empty(&self) -> bool {
        zeroizing_inner(self).is_empty()
    }

    fn as_slice(&self) -> &[u8] {
        zeroizing_inner(self).as_slice()
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        zeroizing_inner_mut(self).as_mut_slice()
    }

    fn extend_from_slice(&mut self, other: &[u8]) -> Result<(), Error> {
        zeroizing_inner_mut(self).extend_from_slice(other);
        Ok(())
    }

    fn truncate(&mut self, len: usize) {
        zeroizing_inner_mut(self).truncate(len);
    }
}

// The `bytes` feature implies `alloc`, so no `alloc` term is needed here.
#[cfg(feature = "bytes")]
#[cfg_attr(docsrs, doc(cfg(feature = "bytes")))]
impl Buffer for bytes::BytesMut {
    fn len(&self) -> usize {
        bytes::BytesMut::len(self)
    }

    fn is_empty(&self) -> bool {
        bytes::BytesMut::is_empty(self)
    }

    fn as_slice(&self) -> &[u8] {
        self
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        self
    }

    fn extend_from_slice(&mut self, other: &[u8]) -> Result<(), Error> {
        bytes::BytesMut::extend_from_slice(self, other);
        Ok(())
    }

    fn truncate(&mut self, len: usize) {
        bytes::BytesMut::truncate(self, len);
    }
}

/// Random generation for keys and nonces (requires the `getrandom` feature).
///
/// ```
/// use chacha20poly1305_simd::{Generate, Key, Nonce};
///
/// let key = Key::generate();
/// let nonce = Nonce::generate();
/// # let _ = (key, nonce);
/// ```
#[cfg(feature = "getrandom")]
pub trait Generate: Sized {
    /// Generate a random value (e.g. a [`Key`] or [`Nonce`]) from the OS
    /// CSPRNG, surfacing RNG failure instead of panicking.
    ///
    /// # Errors
    ///
    /// Returns an error if the OS RNG is unavailable.
    fn try_generate() -> Result<Self, getrandom::Error>;

    /// Generate a random value (e.g. a [`Key`] or [`Nonce`]) from the OS
    /// CSPRNG.
    ///
    /// # Panics
    ///
    /// Panics if the OS RNG is unavailable — see [`Generate::try_generate`]
    /// for the fallible variant.
    #[must_use]
    fn generate() -> Self {
        Self::try_generate().expect("OS RNG failure")
    }
}

#[cfg(feature = "getrandom")]
impl<const N: usize> Generate for [u8; N] {
    fn try_generate() -> Result<Self, getrandom::Error> {
        let mut buf = [0u8; N];
        getrandom::fill(&mut buf)?;
        Ok(buf)
    }
}

/// Tag-appending `Buffer` methods shared by both cipher types (identical
/// bodies modulo the nonce type).
macro_rules! buffer_aead_methods {
    ($nonce:ty) => {
        /// In-place encryption appending the tag to a [`Buffer`] (e.g.
        /// `Vec`).
        ///
        /// Returns [`Error::MessageTooLong`] if `buffer.len()` exceeds the
        /// ChaCha20 counter space.
        #[cfg(feature = "alloc")]
        pub fn encrypt_in_place(
            &self,
            nonce: &$nonce,
            aad: &[u8],
            buffer: &mut dyn Buffer,
        ) -> Result<(), Error> {
            let tag = self.encrypt_in_place_detached(nonce, aad, buffer.as_mut_slice())?;
            buffer.extend_from_slice(&tag)
        }

        /// In-place decryption stripping a trailing tag from a [`Buffer`]
        /// (e.g. `Vec`). On error the buffer *length* is unchanged and its
        /// contents are unspecified: decryption is fused with authentication
        /// and only verified after it has run.
        ///
        /// Returns [`Error::InvalidLength`] if the buffer is shorter than
        /// the 16-byte tag.
        #[cfg(feature = "alloc")]
        pub fn decrypt_in_place(
            &self,
            nonce: &$nonce,
            aad: &[u8],
            buffer: &mut dyn Buffer,
        ) -> Result<(), Error> {
            let ct_len = buffer.len().checked_sub(16).ok_or(Error::InvalidLength)?;
            let tag: Tag = buffer.as_slice()[ct_len..].try_into().expect("16 bytes");
            self.decrypt_in_place_detached(nonce, aad, &mut buffer.as_mut_slice()[..ct_len], &tag)?;
            buffer.truncate(ct_len);
            Ok(())
        }
    };
}

/// Out-of-place detached methods shared by both cipher types: `src` is
/// never written to (read-only `mmap`, shared buffers); `dst` receives
/// exactly `src.len()` bytes.
macro_rules! detached_into_methods {
    ($nonce:ty) => {
        /// Encrypt `src` into `dst`, returning the detached tag. `src` is
        /// never modified — useful for ciphertext targets that cannot be
        /// mutated (read-only `mmap`, shared memory), at the cost of one
        /// memcpy into `dst`.
        ///
        /// `dst.len()` must be at least `src.len()` ([`Error::InvalidLength`]
        /// otherwise); exactly `src.len()` bytes are written, leaving any
        /// headroom untouched.
        #[cfg_attr(feature = "hotpath", hotpath::measure)]
        pub fn encrypt_into_detached(
            &self,
            nonce: &$nonce,
            aad: &[u8],
            src: &[u8],
            dst: &mut [u8],
        ) -> Result<Tag, Error> {
            let msg = dst.get_mut(..src.len()).ok_or(Error::InvalidLength)?;
            msg.copy_from_slice(src);
            self.encrypt_in_place_detached(nonce, aad, msg)
        }

        /// Decrypt `src` into `dst`, verifying `tag`. As with
        /// [`encrypt_into_detached`](Self::encrypt_into_detached), `src` is
        /// never modified and `dst.len()` must be at least `src.len()`.
        ///
        /// On [`Error::TagMismatch`] the first `src.len()` bytes of `dst`
        /// hold unspecified data: decryption is fused with authentication
        /// and only verified after it has run.
        #[cfg_attr(feature = "hotpath", hotpath::measure)]
        pub fn decrypt_into_detached(
            &self,
            nonce: &$nonce,
            aad: &[u8],
            src: &[u8],
            dst: &mut [u8],
            tag: &Tag,
        ) -> Result<(), Error> {
            let msg = dst.get_mut(..src.len()).ok_or(Error::InvalidLength)?;
            msg.copy_from_slice(src);
            self.decrypt_in_place_detached(nonce, aad, msg, tag)
        }
    };
}

/// ChaCha20Poly1305 AEAD (RFC 8439).
// Copy is unavailable under `zeroize`: the Drop impl zeroizes the key.
#[derive(Clone)]
#[cfg_attr(not(feature = "zeroize"), derive(Copy))]
pub struct ChaCha20Poly1305 {
    key: Key,
}

impl ChaCha20Poly1305 {
    detached_into_methods!(Nonce);

    buffer_aead_methods!(Nonce);

    /// Create a new cipher instance from a 256-bit key.
    #[must_use]
    pub const fn new(key: Key) -> Self {
        Self { key }
    }

    /// Derive the ChaCha20 state used to seal / open `nonce` (counter = 0).
    #[inline]
    fn init(&self, nonce: &Nonce) -> chacha::State {
        chacha::State::new_ietf(&self.key, nonce)
    }

    /// Encrypt `payload.msg`, allocating a `ciphertext || tag` buffer.
    ///
    /// A bare `&[u8]` (empty AAD) auto-coerces into a [`Payload`]. Returns
    /// [`Error`] if `payload.msg.len()` exceeds the ChaCha20 counter space
    /// (256 GiB - 64 KiB).
    #[cfg(feature = "alloc")]
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    pub fn encrypt<'msg, 'aad>(
        &self,
        nonce: &Nonce,
        payload: impl Into<Payload<'msg, 'aad>>,
    ) -> Result<alloc::vec::Vec<u8>, Error> {
        let payload = payload.into();
        let mut buf = alloc::vec![0u8; payload.msg.len() + 16];
        buf[..payload.msg.len()].copy_from_slice(payload.msg);
        let tag =
            self.encrypt_in_place_detached(nonce, payload.aad, &mut buf[..payload.msg.len()])?;
        buf[payload.msg.len()..].copy_from_slice(&tag);
        Ok(buf)
    }

    /// Decrypt `payload.msg` (`ciphertext || tag`), returning the plaintext,
    /// or [`Error`] on authentication failure.
    #[cfg(feature = "alloc")]
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    pub fn decrypt<'msg, 'aad>(
        &self,
        nonce: &Nonce,
        ciphertext_and_tag: impl Into<Payload<'msg, 'aad>>,
    ) -> Result<alloc::vec::Vec<u8>, Error> {
        let payload = ciphertext_and_tag.into();
        let (ct, tag) = payload.msg.split_at(
            payload
                .msg
                .len()
                .checked_sub(16)
                .ok_or(Error::InvalidLength)?,
        );
        let mut buf = alloc::vec![0u8; ct.len()];
        buf.copy_from_slice(ct);
        self.decrypt_in_place_detached(
            nonce,
            payload.aad,
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
            return Err(Error::MessageTooLong);
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
            Err(Error::MessageTooLong)
        } else {
            with_backend(|b| {
                if b.open(&mut st, aad, buf, tag) {
                    Ok(())
                } else {
                    Err(Error::TagMismatch)
                }
            })
        };
        #[cfg(feature = "zeroize")]
        st.zeroize();
        r
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
    detached_into_methods!(XNonce);

    buffer_aead_methods!(XNonce);

    /// Create a new cipher instance from a 256-bit key.
    #[must_use]
    pub const fn new(key: Key) -> Self {
        Self { key }
    }

    #[cfg(feature = "alloc")]
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    pub fn encrypt<'msg, 'aad>(
        &self,
        nonce: &XNonce,
        payload: impl Into<Payload<'msg, 'aad>>,
    ) -> Result<alloc::vec::Vec<u8>, Error> {
        // The subkey is moved into the cipher, which zeroizes it on drop
        // under the `zeroize` feature.
        let (key, iv) = self.subkey(nonce);
        let cipher = ChaCha20Poly1305::new(key);
        cipher.encrypt(&iv, payload)
    }

    #[cfg(feature = "alloc")]
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    pub fn decrypt<'msg, 'aad>(
        &self,
        nonce: &XNonce,
        ciphertext_and_tag: impl Into<Payload<'msg, 'aad>>,
    ) -> Result<alloc::vec::Vec<u8>, Error> {
        // The subkey is moved into the cipher, which zeroizes it on drop
        // under the `zeroize` feature.
        let (key, iv) = self.subkey(nonce);
        let cipher = ChaCha20Poly1305::new(key);
        cipher.decrypt(&iv, ciphertext_and_tag)
    }

    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    pub fn encrypt_in_place_detached(
        &self,
        nonce: &XNonce,
        aad: &[u8],
        msg: &mut [u8],
    ) -> Result<Tag, Error> {
        // The subkey is moved into the cipher, which zeroizes it on drop
        // under the `zeroize` feature.
        let (key, iv) = self.subkey(nonce);
        let cipher = ChaCha20Poly1305::new(key);
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
        // The subkey is moved into the cipher, which zeroizes it on drop
        // under the `zeroize` feature.
        let (key, iv) = self.subkey(nonce);
        let cipher = ChaCha20Poly1305::new(key);
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
