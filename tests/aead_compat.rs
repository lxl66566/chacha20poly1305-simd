//! `aead_compat` wrappers interop byte-for-byte with the native API.

#![cfg(feature = "aead")]

use aead::{AeadInOut, KeyInit, inout::InOutBuf};
use chacha20poly1305_simd::{Tag, XChaCha20Poly1305, aead_compat::AeadXChaCha20Poly1305 as XC};

const KEY: [u8; 32] = [0x42; 32];
const MSG: &[u8] = b"interop payload";

#[test]
fn xchacha_compat_interops_with_native() {
    let compat = XC::new(&aead::Key::<XC>::from(KEY));
    let native = XChaCha20Poly1305::new(KEY);
    let nonce = [7u8; 24];
    let rc_nonce = aead::Nonce::<XC>::from(nonce);

    // compat-encrypt, native-decrypt
    let mut msg = MSG.to_vec();
    let tag = compat
        .encrypt_inout_detached(&rc_nonce, b"aad", msg.as_mut_slice().into())
        .unwrap();
    let native_tag: Tag = tag.as_slice().try_into().unwrap();
    native
        .decrypt_in_place_detached(&nonce, b"aad", &mut msg, &native_tag)
        .unwrap();
    assert_eq!(&msg, MSG);

    // native-encrypt, compat-decrypt
    let tag = native
        .encrypt_in_place_detached(&nonce, b"aad", &mut msg)
        .unwrap();
    compat
        .decrypt_inout_detached(
            &rc_nonce,
            b"aad",
            msg.as_mut_slice().into(),
            &aead::Tag::<XC>::from(tag),
        )
        .unwrap();
    assert_eq!(&msg, MSG);

    // tampered tag must fail
    let mut bad_tag = tag;
    bad_tag[0] ^= 1;
    assert!(
        compat
            .decrypt_inout_detached(
                &rc_nonce,
                b"aad",
                msg.as_mut_slice().into(),
                &aead::Tag::<XC>::from(bad_tag),
            )
            .is_err()
    );
}

#[test]
fn chacha_compat_roundtrip() {
    use chacha20poly1305_simd::aead_compat::AeadChaCha20Poly1305 as C;

    let compat = C::new(&aead::Key::<C>::from(KEY));
    let nonce = aead::Nonce::<C>::from([7u8; 12]);

    let mut msg = MSG.to_vec();
    let tag = compat
        .encrypt_inout_detached(&nonce, b"aad", msg.as_mut_slice().into())
        .unwrap();
    compat
        .decrypt_inout_detached(&nonce, b"aad", msg.as_mut_slice().into(), &tag)
        .unwrap();
    assert_eq!(&msg, MSG);
}

/// Disjoint input/output buffers must behave like an out-of-place call:
/// `src` stays pristine, `dst` receives the ciphertext.
#[test]
fn disjoint_in_out_buffers() {
    let compat = XC::new(&aead::Key::<XC>::from(KEY));
    let nonce = aead::Nonce::<XC>::from([7u8; 24]);

    let src = MSG.to_vec();
    let mut dst = vec![0u8; MSG.len()];
    let tag = compat
        .encrypt_inout_detached(&nonce, b"aad", InOutBuf::new(&src, &mut dst).unwrap())
        .unwrap();
    assert_eq!(&src, MSG, "disjoint input must be untouched");
    assert_ne!(&dst, MSG);

    let mut plain = vec![0u8; MSG.len()];
    compat
        .decrypt_inout_detached(
            &nonce,
            b"aad",
            InOutBuf::new(&dst, &mut plain).unwrap(),
            &tag,
        )
        .unwrap();
    assert_eq!(&plain, MSG);
}
