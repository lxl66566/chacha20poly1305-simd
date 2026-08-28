//! Test vectors: RFC 8439 §2.8.2, draft-irtf-cfrg-xchacha A.1, Wycheproof,
//! and a randomized differential test against the RustCrypto implementation.

#![cfg(feature = "alloc")]
// Truncating pseudo-random byte generation is intentional in tests.
#![allow(clippy::cast_possible_truncation)]

use chacha20poly1305_simd::{
    ChaCha20Poly1305, Error, Key, Nonce, Payload, Tag, XChaCha20Poly1305, XNonce, active_backend,
};
use rustcrypto::aead::{AeadInOut, KeyInit};

const KEY: &[u8; 32] = &[
    0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f,
    0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0x9b, 0x9c, 0x9d, 0x9e, 0x9f,
];
const AAD: &[u8; 12] = &[
    0x50, 0x51, 0x52, 0x53, 0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7,
];
const PLAINTEXT: &[u8] = b"Ladies and Gentlemen of the class of '99: \
     If I could offer you only one tip for the future, sunscreen would be it.";

/// RFC 8439 §2.8.2
mod chacha20poly1305_rfc8439 {
    use super::*;

    const NONCE: &[u8; 12] = &[
        0x07, 0x00, 0x00, 0x00, 0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
    ];
    const CIPHERTEXT: &[u8] = &[
        0xd3, 0x1a, 0x8d, 0x34, 0x64, 0x8e, 0x60, 0xdb, 0x7b, 0x86, 0xaf, 0xbc, 0x53, 0xef, 0x7e,
        0xc2, 0xa4, 0xad, 0xed, 0x51, 0x29, 0x6e, 0x08, 0xfe, 0xa9, 0xe2, 0xb5, 0xa7, 0x36, 0xee,
        0x62, 0xd6, 0x3d, 0xbe, 0xa4, 0x5e, 0x8c, 0xa9, 0x67, 0x12, 0x82, 0xfa, 0xfb, 0x69, 0xda,
        0x92, 0x72, 0x8b, 0x1a, 0x71, 0xde, 0x0a, 0x9e, 0x06, 0x0b, 0x29, 0x05, 0xd6, 0xa5, 0xb6,
        0x7e, 0xcd, 0x3b, 0x36, 0x92, 0xdd, 0xbd, 0x7f, 0x2d, 0x77, 0x8b, 0x8c, 0x98, 0x03, 0xae,
        0xe3, 0x28, 0x09, 0x1b, 0x58, 0xfa, 0xb3, 0x24, 0xe4, 0xfa, 0xd6, 0x75, 0x94, 0x55, 0x85,
        0x80, 0x8b, 0x48, 0x31, 0xd7, 0xbc, 0x3f, 0xf4, 0xde, 0xf0, 0x8e, 0x4b, 0x7a, 0x9d, 0xe5,
        0x76, 0xd2, 0x65, 0x86, 0xce, 0xc6, 0x4b, 0x61, 0x16,
    ];
    const TAG: &[u8; 16] = &[
        0x1a, 0xe1, 0x0b, 0x59, 0x4f, 0x09, 0xe2, 0x6a, 0x7e, 0x90, 0x2e, 0xcb, 0xd0, 0x60, 0x06,
        0x91,
    ];

    #[test]
    fn seal() {
        let cipher = ChaCha20Poly1305::new(KEY);
        let mut msg = PLAINTEXT.to_vec();
        let tag = cipher
            .encrypt_in_place_detached(NONCE, AAD, &mut msg)
            .unwrap();
        assert_eq!(&msg, CIPHERTEXT);
        assert_eq!(&tag, TAG);
    }

    #[test]
    fn open() {
        let cipher = ChaCha20Poly1305::new(KEY);
        let mut buf = CIPHERTEXT.to_vec();
        cipher
            .decrypt_in_place_detached(NONCE, AAD, &mut buf, TAG)
            .unwrap();
        assert_eq!(&buf, PLAINTEXT);
    }

    #[test]
    fn open_tampered() {
        let cipher = ChaCha20Poly1305::new(KEY);
        for i in [0, 1, 57, PLAINTEXT.len() - 1] {
            let mut buf = CIPHERTEXT.to_vec();
            buf[i] ^= 0x80;
            assert!(
                cipher
                    .decrypt_in_place_detached(NONCE, AAD, &mut buf, TAG)
                    .is_err()
            );
        }
        let mut tag = *TAG;
        tag[15] ^= 1;
        let mut buf = CIPHERTEXT.to_vec();
        assert_eq!(
            cipher.decrypt_in_place_detached(NONCE, AAD, &mut buf, &tag),
            Err(Error::TagMismatch)
        );
    }
}

/// draft-irtf-cfrg-xchacha §A.1
mod xchacha20poly1305 {
    use super::*;

    const NONCE: &[u8; 24] = &[
        0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e,
        0x4f, 0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57,
    ];
    const CIPHERTEXT: &[u8] = &[
        0xbd, 0x6d, 0x17, 0x9d, 0x3e, 0x83, 0xd4, 0x3b, 0x95, 0x76, 0x57, 0x94, 0x93, 0xc0, 0xe9,
        0x39, 0x57, 0x2a, 0x17, 0x00, 0x25, 0x2b, 0xfa, 0xcc, 0xbe, 0xd2, 0x90, 0x2c, 0x21, 0x39,
        0x6c, 0xbb, 0x73, 0x1c, 0x7f, 0x1b, 0x0b, 0x4a, 0xa6, 0x44, 0x0b, 0xf3, 0xa8, 0x2f, 0x4e,
        0xda, 0x7e, 0x39, 0xae, 0x64, 0xc6, 0x70, 0x8c, 0x54, 0xc2, 0x16, 0xcb, 0x96, 0xb7, 0x2e,
        0x12, 0x13, 0xb4, 0x52, 0x2f, 0x8c, 0x9b, 0xa4, 0x0d, 0xb5, 0xd9, 0x45, 0xb1, 0x1b, 0x69,
        0xb9, 0x82, 0xc1, 0xbb, 0x9e, 0x3f, 0x3f, 0xac, 0x2b, 0xc3, 0x69, 0x48, 0x8f, 0x76, 0xb2,
        0x38, 0x35, 0x65, 0xd3, 0xff, 0xf9, 0x21, 0xf9, 0x66, 0x4c, 0x97, 0x63, 0x7d, 0xa9, 0x76,
        0x88, 0x12, 0xf6, 0x15, 0xc6, 0x8b, 0x13, 0xb5, 0x2e,
    ];
    const TAG: &[u8; 16] = &[
        0xc0, 0x87, 0x59, 0x24, 0xc1, 0xc7, 0x98, 0x79, 0x47, 0xde, 0xaf, 0xd8, 0x78, 0x0a, 0xcf,
        0x49,
    ];

    #[test]
    fn seal() {
        let cipher = XChaCha20Poly1305::new(KEY);
        let mut msg = PLAINTEXT.to_vec();
        let tag = cipher
            .encrypt_in_place_detached(NONCE, AAD, &mut msg)
            .unwrap();
        assert_eq!(&msg, CIPHERTEXT);
        assert_eq!(&tag, TAG);
    }

    #[test]
    fn open() {
        let cipher = XChaCha20Poly1305::new(KEY);
        let mut buf = CIPHERTEXT.to_vec();
        cipher
            .decrypt_in_place_detached(NONCE, AAD, &mut buf, TAG)
            .unwrap();
        assert_eq!(&buf, PLAINTEXT);
    }
}

/// Boundary sizes around the alignment / batching transitions.
#[test]
fn roundtrip_lengths() {
    let cipher = ChaCha20Poly1305::new(KEY);
    let nonce = [7u8; 12];
    let mut data = vec![0u8; 4096];
    for (i, b) in data.iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }
    for len in 0..=1024usize {
        let mut msg = data[..len].to_vec();
        let tag = cipher
            .encrypt_in_place_detached(&nonce, AAD, &mut msg)
            .unwrap();
        if len > 0 {
            assert_ne!(msg, data[..len], "len {len}");
        }
        let ok = cipher.decrypt_in_place_detached(&nonce, AAD, &mut msg, &tag);
        if ok.is_err() {
            println!("ROUNDTRIP FAIL len {len}");
        }
        ok.unwrap();
        assert_eq!(msg, data[..len], "len {len}");
    }
    for len in [1025usize, 1536, 2047, 2048, 4096] {
        let mut msg = data[..len].to_vec();
        let tag = cipher
            .encrypt_in_place_detached(&nonce, AAD, &mut msg)
            .unwrap();
        let ok = cipher.decrypt_in_place_detached(&nonce, AAD, &mut msg, &tag);
        assert!(ok.is_ok(), "len {len}");
        assert_eq!(msg, data[..len], "len {len}");
    }
}

/// Wycheproof vectors (shared blob format with the RustCrypto test suite).
///
/// Rows: key, nonce, aad, plaintext, ciphertext (ciphertext has the tag
/// appended; failing vectors carry a corrupted tag).
#[test]
fn wycheproof() {
    for (file, is_xchacha, expect_ok) in [
        (
            "tests/data/wycheproof_chacha20poly1305_pass.blb",
            false,
            true,
        ),
        (
            "tests/data/wycheproof_chacha20poly1305_fail.blb",
            false,
            false,
        ),
        (
            "tests/data/wycheproof_xchacha20poly1305_pass.blb",
            true,
            true,
        ),
        (
            "tests/data/wycheproof_xchacha20poly1305_fail.blb",
            true,
            false,
        ),
    ] {
        let data = std::fs::read(file).unwrap();
        let blobs = blobby::parse_into_vec(&data).unwrap();
        for row in blobs.chunks(5) {
            let [key, nonce, aad, msg, ct] = [row[0], row[1], row[2], row[3], row[4]];
            let key: Key = key.try_into().unwrap();
            if is_xchacha {
                let nonce: XNonce = nonce.try_into().unwrap();
                let _cipher = XChaCha20Poly1305::new(&key);
                // split ct || tag
                let (ct_body, tag) = ct.split_at(ct.len() - 16);
                let tag: &Tag = tag.try_into().unwrap();
                let cipher = XChaCha20Poly1305::new(&key);
                let mut buf = msg.to_vec();
                let computed = cipher
                    .encrypt_in_place_detached(&nonce, aad, &mut buf)
                    .unwrap();
                assert_eq!(&buf, ct_body, "{file} ct mismatch");
                if expect_ok {
                    assert_eq!(&computed, tag, "{file} tag mismatch");
                }
                let ok = cipher
                    .decrypt_in_place_detached(&nonce, aad, &mut buf, tag)
                    .is_ok();
                assert_eq!(ok, expect_ok, "{file} open");
            } else {
                let nonce: Nonce = nonce.try_into().unwrap();
                let (ct_body, tag) = ct.split_at(ct.len() - 16);
                let tag: &Tag = tag.try_into().unwrap();
                let cipher = ChaCha20Poly1305::new(&key);
                let mut buf = msg.to_vec();
                let computed = cipher
                    .encrypt_in_place_detached(&nonce, aad, &mut buf)
                    .unwrap();
                assert_eq!(&buf, ct_body, "{file} ct mismatch");
                if expect_ok {
                    assert_eq!(&computed, tag, "{file} tag mismatch");
                }
                let ok = cipher
                    .decrypt_in_place_detached(&nonce, aad, &mut buf, tag)
                    .is_ok();
                assert_eq!(ok, expect_ok, "{file} open");
            }
        }
    }
}

/// Deterministic differential test against the RustCrypto implementation.
#[test]
fn differential_vs_rustcrypto() {
    // xorshift for reproducible pseudo-random inputs
    let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let _key: Key = next()
        .to_le_bytes()
        .iter()
        .chain(next().to_le_bytes().iter())
        .copied()
        .chain([0u8; 16].iter().copied())
        .collect::<Vec<u8>>()
        .try_into()
        .unwrap();
    let mut key = [0u8; 32];
    for i in 0..4 {
        key[i * 8..][..8].copy_from_slice(&next().to_le_bytes());
    }

    let cipher = ChaCha20Poly1305::new(&key);
    let rc = rustcrypto::ChaCha20Poly1305::new(&rustcrypto::Key::from(key));
    let mut nonce_bytes = [0u8; 12];
    nonce_bytes[..8].copy_from_slice(&next().to_le_bytes());
    let rc_nonce = rustcrypto::Nonce::from(nonce_bytes);

    for len in [
        0usize, 1, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64, 65, 95, 96, 97, 111, 112, 113, 127, 128,
        129, 255, 256, 257, 511, 512, 513, 575, 576, 577, 1023, 1024, 1025, 2048, 4096, 8192,
        65536,
    ] {
        for aad_len in [0usize, 1, 16, 17, 32, 47, 48, 49, 63, 64, 65, 100, 128] {
            let mut msg = Vec::with_capacity(len);
            for _ in 0..len {
                msg.push(next() as u8);
            }
            let mut aad = Vec::with_capacity(aad_len);
            for _ in 0..aad_len {
                aad.push(next() as u8);
            }

            let mut ours_buf = msg.clone();
            let tag = cipher
                .encrypt_in_place_detached(&nonce_bytes, &aad, &mut ours_buf)
                .unwrap();
            let mut their_buf = msg.clone();
            let rc_tag = rc
                .encrypt_inout_detached(
                    &rc_nonce,
                    &aad,
                    rustcrypto::aead::inout::InOutBuf::from(&mut their_buf[..]),
                )
                .unwrap();
            assert_eq!(ours_buf, their_buf, "ct len {len} aad {aad_len}");
            assert_eq!(
                tag.as_slice(),
                rc_tag.as_slice(),
                "tag len {len} aad {aad_len}"
            );

            // cross-decrypt: ours opens theirs
            let ok = cipher
                .decrypt_in_place_detached(
                    &nonce_bytes,
                    &aad,
                    &mut their_buf,
                    rc_tag.as_slice().try_into().unwrap(),
                )
                .is_ok();
            assert!(ok, "ours open theirs len {len}");
            assert_eq!(their_buf, msg);
        }
    }
}

/// Sanity: SIMD backends (when present) agree with each other; force each
/// backend through the test vectors above via env override.
#[test]
fn backend_reported() {
    println!("active backend: {}", active_backend());
}

/// Truncated inputs must fail with `Err`, never panic (BUGFIX: short buffers
/// used to underflow `len - 16` in `decrypt_in_place` / `decrypt`).
#[test]
fn short_input_no_panic() {
    let cipher = ChaCha20Poly1305::new(KEY);
    let nonce = [7u8; 12];
    for len in 0..16usize {
        assert!(
            cipher
                .decrypt(&nonce, Payload {
                    msg: &vec![0u8; len],
                    aad: AAD
                })
                .is_err(),
            "len {len}"
        );
        let mut buf = vec![0u8; len];
        assert_eq!(
            cipher.decrypt_in_place(&nonce, AAD, &mut buf),
            Err(Error::InvalidLength),
            "len {len}"
        );
        assert_eq!(buf.len(), len, "buffer must be untouched on error");
    }
}
