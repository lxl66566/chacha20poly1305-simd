//! `Buffer` implementations beyond `Vec<u8>` and the tag-appending in-place
//! API on top of them.

#![cfg(feature = "alloc")]

use chacha20poly1305_simd::{ChaCha20Poly1305, Error, XChaCha20Poly1305};

const KEY: [u8; 32] = [0x42; 32];
const NONCE: [u8; 12] = [7; 12];
const AAD: &[u8] = b"header";

/// `Zeroizing<Vec<u8>>` must round-trip through the in-place API.
#[test]
#[cfg(feature = "zeroize")]
fn zeroizing_buffer_roundtrip() {
    use zeroize::Zeroizing;

    let cipher = ChaCha20Poly1305::new(KEY);
    let mut buf: Zeroizing<Vec<u8>> = Zeroizing::new(b"hello world".to_vec());
    cipher.encrypt_in_place(&NONCE, AAD, &mut buf).unwrap();
    assert_eq!(buf.len(), b"hello world".len() + 16);
    cipher.decrypt_in_place(&NONCE, AAD, &mut buf).unwrap();
    assert_eq!(&*buf, b"hello world");
}

/// `bytes::BytesMut` must round-trip through the in-place API.
#[test]
#[cfg(feature = "bytes")]
fn bytes_mut_buffer_roundtrip() {
    use bytes::BytesMut;

    let cipher = ChaCha20Poly1305::new(KEY);
    let mut buf = BytesMut::from(&b"hello world"[..]);
    cipher.encrypt_in_place(&NONCE, AAD, &mut buf).unwrap();
    assert_eq!(buf.len(), b"hello world".len() + 16);
    cipher.decrypt_in_place(&NONCE, AAD, &mut buf).unwrap();
    assert_eq!(&buf[..], b"hello world");
}

/// XChaCha20Poly1305 exposes the same tag-appending Buffer API as
/// ChaCha20Poly1305; its wire layout must equal `ciphertext || tag` from the
/// detached API.
#[test]
fn xchacha_in_place_matches_detached() {
    let cipher = XChaCha20Poly1305::new(KEY);
    let nonce = [2u8; 24];

    let mut via_buffer = b"payload".to_vec();
    cipher
        .encrypt_in_place(&nonce, AAD, &mut via_buffer)
        .unwrap();

    let mut detached = b"payload".to_vec();
    let tag = cipher
        .encrypt_in_place_detached(&nonce, AAD, &mut detached)
        .unwrap();
    assert_eq!(via_buffer, [&detached[..], &tag[..]].concat());

    cipher
        .decrypt_in_place(&nonce, AAD, &mut via_buffer)
        .unwrap();
    assert_eq!(via_buffer, b"payload");

    let mut short = vec![0u8; 15];
    assert_eq!(
        cipher.decrypt_in_place(&nonce, AAD, &mut short),
        Err(Error::InvalidLength)
    );
}
