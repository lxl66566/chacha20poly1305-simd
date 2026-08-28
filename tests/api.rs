//! Out-of-place detached API (`*_into_detached`) and the `split_tag`
//! wire-format helper.

use chacha20poly1305_simd::{ChaCha20Poly1305, Error, XChaCha20Poly1305, split_tag};

const KEY: [u8; 32] = [0x42; 32];
const AAD: &[u8] = b"header";
const MSG: &[u8] = b"hello world";

#[test]
fn chacha_out_of_place_roundtrip() {
    let cipher = ChaCha20Poly1305::new(KEY);
    let nonce = [7u8; 12];
    let src = MSG.to_vec();

    // dst with headroom: exactly src.len() bytes written, tail untouched.
    let mut dst = vec![0xeeu8; MSG.len() + 8];
    let tag = cipher
        .encrypt_into_detached(&nonce, AAD, &src, &mut dst)
        .unwrap();
    assert_eq!(&dst[MSG.len()..], &[0xee; 8]);
    assert_eq!(&src, MSG, "src must be untouched");

    // Byte-identical to the in-place detached path.
    let mut in_place = MSG.to_vec();
    let in_place_tag = cipher
        .encrypt_in_place_detached(&nonce, AAD, &mut in_place)
        .unwrap();
    assert_eq!(&dst[..MSG.len()], &in_place);
    assert_eq!(tag, in_place_tag);

    let mut plain = vec![0u8; MSG.len() + 8];
    cipher
        .decrypt_into_detached(&nonce, AAD, &dst[..MSG.len()], &mut plain, &tag)
        .unwrap();
    assert_eq!(&plain[..MSG.len()], MSG);

    // dst smaller than src must fail, not panic.
    let mut small = vec![0u8; MSG.len() - 1];
    assert_eq!(
        cipher.encrypt_into_detached(&nonce, AAD, &src, &mut small),
        Err(Error::InvalidLength)
    );
}

#[test]
fn xchacha_out_of_place_roundtrip() {
    let cipher = XChaCha20Poly1305::new(KEY);
    let nonce = [2u8; 24];
    let mut dst = vec![0u8; MSG.len()];
    let tag = cipher
        .encrypt_into_detached(&nonce, AAD, MSG, &mut dst)
        .unwrap();
    let mut plain = vec![0u8; MSG.len()];
    cipher
        .decrypt_into_detached(&nonce, AAD, &dst, &mut plain, &tag)
        .unwrap();
    assert_eq!(&plain, MSG);
}

#[test]
fn split_tag_helper() {
    let cipher = XChaCha20Poly1305::new(KEY);
    let nonce = [2u8; 24];
    let mut wire = MSG.to_vec();
    let tag = cipher
        .encrypt_in_place_detached(&nonce, AAD, &mut wire)
        .unwrap();
    wire.extend_from_slice(&tag);

    let (ct, t) = split_tag(&mut wire).unwrap();
    assert_eq!(ct.len(), MSG.len());
    cipher
        .decrypt_in_place_detached(&nonce, AAD, ct, t)
        .unwrap();
    assert_eq!(ct, MSG);

    let mut short = vec![0u8; 15];
    assert_eq!(split_tag(&mut short), Err(Error::InvalidLength));
}
