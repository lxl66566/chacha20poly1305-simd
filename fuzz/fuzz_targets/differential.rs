//! Differential fuzz target: this crate vs the RustCrypto `chacha20poly1305`
//! reference. For every input, both implementations must produce identical
//! ciphertext+tag, and both must agree on open success/failure and plaintext.

#![no_main]

use chacha20poly1305_simd::{ChaCha20Poly1305, Payload, XChaCha20Poly1305};
use libfuzzer_sys::fuzz_target;
use rustcrypto::aead::{Aead, AeadInOut, KeyInit};

#[derive(arbitrary::Arbitrary, Debug)]
struct Input {
    key: [u8; 32],
    nonce: [u8; 12],
    xnonce: [u8; 24],
    aad: Vec<u8>,
    msg: Vec<u8>,
    /// Fuzz-controlled offset selecting which byte of `aad || ct` to corrupt.
    corrupt: u32,
}

/// Flip one bit at a fuzz-controlled position across `lhs || rhs` (in place).
fn corrupt_split(corrupt: u32, lhs: &mut [u8], rhs: &mut [u8]) {
    let total = lhs.len() + rhs.len();
    if total > 0 {
        let pos = (corrupt as usize) % total;
        if pos < lhs.len() {
            lhs[pos] ^= 1;
        } else {
            rhs[pos - lhs.len()] ^= 1;
        }
    }
}

fn check(key: &[u8; 32], nonce: &[u8; 12], aad: &[u8], msg: &[u8], corrupt: u32) {
    let ours = ChaCha20Poly1305::new(*key);
    let theirs = rustcrypto::ChaCha20Poly1305::new(&rustcrypto::Key::from(*key));

    // seal: ciphertext and tag must match the reference exactly
    let mut ours_buf = msg.to_vec();
    let tag = ours
        .encrypt_in_place_detached(nonce, aad, &mut ours_buf)
        .unwrap();
    let mut their_buf = msg.to_vec();
    let their_tag = theirs
        .encrypt_inout_detached(
            &rustcrypto::Nonce::from(*nonce),
            aad,
            rustcrypto::aead::inout::InOutBuf::from(&mut their_buf[..]),
        )
        .unwrap();
    assert_eq!(ours_buf, their_buf, "ciphertext mismatch");
    assert_eq!(tag, their_tag.as_slice(), "tag mismatch");

    // open: valid tag must decrypt to the original message
    let mut dec = ours_buf.clone();
    ours.decrypt_in_place_detached(nonce, aad, &mut dec, &tag)
        .unwrap();
    assert_eq!(dec, msg, "roundtrip mismatch");

    // open: differential verdict on a corrupted (aad, ct) pair. Both
    // implementations must agree on accept/reject and, when accepted, on the
    // plaintext. The corruption position is fuzz-driven so every region
    // (aad / ct interior / boundary) gets explored.
    let mut bad_aad = aad.to_vec();
    let mut bad_ct = ours_buf.clone();
    corrupt_split(corrupt, &mut bad_aad, &mut bad_ct);
    let mut ours_dec = bad_ct.clone();
    let ours_ok = ours.decrypt_in_place_detached(nonce, &bad_aad, &mut ours_dec, &tag);
    let mut their_dec = bad_ct;
    let their_ok = theirs.decrypt_inout_detached(
        &rustcrypto::Nonce::from(*nonce),
        &bad_aad,
        rustcrypto::aead::inout::InOutBuf::from(&mut their_dec[..]),
        &rustcrypto::Tag::from(tag),
    );
    match (ours_ok, their_ok) {
        (Ok(()), Ok(())) => assert_eq!(ours_dec, their_dec, "open plaintext mismatch"),
        (Err(_), Err(_)) => {},
        (o, t) => panic!("open verdict mismatch: ours={o:?} theirs={t:?}"),
    }

    // open: every single-bit tag corruption must be rejected
    let mut bad_tag = tag;
    bad_tag[0] ^= 1;
    let mut dec = ours_buf;
    assert!(
        ours.decrypt_in_place_detached(nonce, aad, &mut dec, &bad_tag)
            .is_err()
    );
}

fuzz_target!(|input: Input| {
    check(
        &input.key,
        &input.nonce,
        &input.aad,
        &input.msg,
        input.corrupt,
    );

    // XChaCha20-Poly1305 path (HChaCha20 subkey derivation included)
    let ours_x = XChaCha20Poly1305::new(input.key);
    let theirs_x = rustcrypto::XChaCha20Poly1305::new(&rustcrypto::Key::from(input.key));
    let ct_ours = ours_x
        .encrypt(&input.xnonce, Payload {
            msg: &input.msg,
            aad: &input.aad,
        })
        .unwrap();
    let ct_theirs = theirs_x
        .encrypt(
            &rustcrypto::XNonce::from(input.xnonce),
            rustcrypto::aead::Payload {
                msg: &input.msg,
                aad: &input.aad,
            },
        )
        .unwrap();
    assert_eq!(ct_ours, ct_theirs, "xchacha ciphertext mismatch");
    assert_eq!(
        ours_x
            .decrypt(&input.xnonce, Payload {
                msg: &ct_ours,
                aad: &input.aad
            })
            .unwrap(),
        input.msg,
        "xchacha roundtrip mismatch"
    );

    // xchacha open verdict differential (fuzz-controlled corruption position)
    let pos = usize::try_from(input.corrupt).unwrap_or(0);
    let mut ct_bad = ct_ours.clone();
    let idx = pos % ct_bad.len().max(1);
    if let Some(byte) = ct_bad.get_mut(idx) {
        *byte ^= 1;
    }
    let ours_v = ours_x.decrypt(&input.xnonce, Payload {
        msg: &ct_bad,
        aad: &input.aad,
    });
    let theirs_v = theirs_x.decrypt(
        &rustcrypto::XNonce::from(input.xnonce),
        rustcrypto::aead::Payload {
            msg: &ct_bad,
            aad: &input.aad,
        },
    );
    match (ours_v, theirs_v) {
        (Ok(o), Ok(t)) => assert_eq!(o, t, "xchacha open plaintext mismatch"),
        (Err(_), Err(_)) => {},
        (o, t) => panic!("xchacha open verdict mismatch: ours={o:?} theirs={t:?}"),
    }
});
