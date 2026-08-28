//! RustCrypto [`aead`] 0.6 trait adapters (feature `aead`).
//!
//! Newtype wrappers exposing the native ciphers as [`KeyInit`] /
//! [`AeadInPlace`] implementors, so they drop into anything generic over
//! the RustCrypto AEAD traits. Conversion cost is confined to the boundary
//! (array copies of at most 32 bytes); the bulk path runs the same SIMD
//! kernels as the native API.
//!
//! The allocating [`Aead`](aead::Aead) blanket impl becomes available when
//! the *consumer* enables `aead/alloc`; this crate keeps the dependency
//! `default-features = false` to stay `no_std`-friendly.
//!
//! ```
//! use aead::{AeadInPlace, KeyInit};
//! use chacha20poly1305_simd::aead_compat::AeadXChaCha20Poly1305;
//!
//! let key = aead::Key::<AeadXChaCha20Poly1305>::from([1u8; 32]);
//! let cipher = AeadXChaCha20Poly1305::new(&key);
//! let nonce = aead::Nonce::<AeadXChaCha20Poly1305>::default();
//!
//! let mut msg = *b"hello world";
//! let tag = cipher
//!     .encrypt_in_place_detached(&nonce, b"aad", &mut msg)
//!     .unwrap();
//! assert_ne!(&msg, b"hello world");
//!
//! cipher
//!     .decrypt_in_place_detached(&nonce, b"aad", &mut msg, &tag)
//!     .unwrap();
//! assert_eq!(&msg, b"hello world");
//! ```

use aead::{
    AeadCore, AeadInPlace, Error as AeadError, KeyInit, KeySizeUser,
    consts::{U12, U16, U24, U32},
    generic_array::GenericArray,
};

macro_rules! aead_compat {
    ($name:ident, $inner:ty, $nonce_size:ty, $native_nonce:ty $(, $doc:expr)?) => {
        $(#[doc = $doc])?
        #[derive(Clone, Debug)]
        pub struct $name {
            inner: $inner,
        }

        impl From<$inner> for $name {
            fn from(inner: $inner) -> Self {
                Self { inner }
            }
        }

        impl KeySizeUser for $name {
            type KeySize = U32;
        }

        impl KeyInit for $name {
            fn new(key: &aead::Key<Self>) -> Self {
                let key: crate::Key = key
                    .as_slice()
                    .try_into()
                    .expect("KeySize is U32");
                Self {
                    inner: <$inner>::new(key),
                }
            }
        }

        impl AeadCore for $name {
            type NonceSize = $nonce_size;
            type TagSize = U16;
            type CiphertextOverhead = U16;
        }

        impl AeadInPlace for $name {
            fn encrypt_in_place_detached(
                &self,
                nonce: &aead::Nonce<Self>,
                aad: &[u8],
                buffer: &mut [u8],
            ) -> Result<aead::Tag<Self>, AeadError> {
                let nonce: $native_nonce = nonce
                    .as_slice()
                    .try_into()
                    .expect("NonceSize matches the cipher");
                self.inner
                    .encrypt_in_place_detached(&nonce, aad, buffer)
                    .map(GenericArray::from)
                    .map_err(|_| AeadError)
            }

            fn decrypt_in_place_detached(
                &self,
                nonce: &aead::Nonce<Self>,
                aad: &[u8],
                buffer: &mut [u8],
                tag: &aead::Tag<Self>,
            ) -> Result<(), AeadError> {
                let nonce: $native_nonce = nonce
                    .as_slice()
                    .try_into()
                    .expect("NonceSize matches the cipher");
                let tag: &crate::Tag = tag.as_slice().try_into().expect("TagSize is U16");
                self.inner
                    .decrypt_in_place_detached(&nonce, aad, buffer, tag)
                    .map_err(|_| AeadError)
            }
        }
    };
}

aead_compat!(
    AeadChaCha20Poly1305,
    crate::ChaCha20Poly1305,
    U12,
    crate::Nonce,
    "RustCrypto-trait wrapper around [`ChaCha20Poly1305`](crate::ChaCha20Poly1305)."
);
aead_compat!(
    AeadXChaCha20Poly1305,
    crate::XChaCha20Poly1305,
    U24,
    crate::XNonce,
    "RustCrypto-trait wrapper around [`XChaCha20Poly1305`](crate::XChaCha20Poly1305)."
);
