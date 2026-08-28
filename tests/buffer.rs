//! `Buffer` implementations beyond `Vec<u8>` and the tag-appending in-place
//! API on top of them.

#![cfg(feature = "alloc")]

use chacha20poly1305_simd::ChaCha20Poly1305;

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
