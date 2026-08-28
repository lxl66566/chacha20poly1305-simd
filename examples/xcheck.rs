//! Seeded randomized differential cross-check against the RustCrypto
//! reference — same assertions as `fuzz/fuzz_targets/differential.rs`, but a
//! plain binary so it also runs under QEMU user emulation (libFuzzer's
//! mutation loop is not QEMU-safe).
//!
//! Run: `cargo run --release --example xcheck -- [iterations] [seed]`

// PRNG helpers deliberately truncate raw u64 output; values are bounded or
// only the low bits are wanted.
#![allow(clippy::cast_possible_truncation)]

use chacha20poly1305_simd::{ChaCha20Poly1305, Payload, XChaCha20Poly1305};
use rustcrypto::aead::{Aead, AeadInOut, KeyInit};

/// xorshift64* — deterministic across hosts so failures are reproducible.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn byte(&mut self) -> u8 {
        self.next() as u8
    }

    /// Uniform-ish value in `0..n` (result is < n, so it always fits usize).
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Lengths biased towards ChaCha block / Poly1305 alignment boundaries.
fn pick_len(r: &mut Rng) -> usize {
    const HOT: &[usize] = &[
        0, 1, 15, 16, 17, 31, 32, 33, 48, 49, 63, 64, 65, 95, 96, 97, 111, 112, 113, 127, 128, 129,
        191, 192, 193, 255, 256, 257, 319, 320, 321, 511, 512, 513, 1023, 1024, 1025,
    ];
    let hot = HOT[r.below(HOT.len())];
    if r.below(4) == 0 {
        // occasionally fully random up to 8 KiB
        r.below(8192)
    } else {
        // small jitter of -2..=2 around the boundary, clamped at 0
        let j = r.below(5);
        if j >= 2 {
            hot + (j - 2)
        } else {
            hot.saturating_sub(2 - j)
        }
    }
}

fn check(case: usize, r: &mut Rng) {
    let key: [u8; 32] = core::array::from_fn(|_| r.byte());
    let nonce: [u8; 12] = core::array::from_fn(|_| r.byte());
    let xnonce: [u8; 24] = core::array::from_fn(|_| r.byte());
    let aad: Vec<u8> = (0..pick_len(r)).map(|_| r.byte()).collect();
    let msg: Vec<u8> = (0..pick_len(r)).map(|_| r.byte()).collect();
    let corrupt = r.next() as u32;

    let ours = ChaCha20Poly1305::new(key);
    let theirs = rustcrypto::ChaCha20Poly1305::new(&rustcrypto::Key::from(key));

    let mut ours_buf = msg.clone();
    let tag = ours
        .encrypt_in_place_detached(&nonce, &aad, &mut ours_buf)
        .unwrap();
    let mut their_buf = msg.clone();
    let their_tag = theirs
        .encrypt_inout_detached(
            &rustcrypto::Nonce::from(nonce),
            &aad,
            rustcrypto::aead::inout::InOutBuf::from(&mut their_buf[..]),
        )
        .unwrap();
    assert_eq!(ours_buf, their_buf, "ct mismatch (case {case})");
    assert_eq!(tag, their_tag.as_slice(), "tag mismatch (case {case})");

    let mut dec = ours_buf.clone();
    ours.decrypt_in_place_detached(&nonce, &aad, &mut dec, &tag)
        .unwrap();
    assert_eq!(dec, msg, "roundtrip mismatch (case {case})");

    // differential verdict on a corrupted (aad, ct) pair
    let mut bad_aad = aad.clone();
    let mut bad_ct = ours_buf.clone();
    let total = bad_aad.len() + bad_ct.len();
    if total > 0 {
        let pos = corrupt as usize % total;
        if pos < bad_aad.len() {
            bad_aad[pos] ^= 1;
        } else {
            bad_ct[pos - bad_aad.len()] ^= 1;
        }
    }
    let mut ours_dec = bad_ct.clone();
    let ours_ok = ours.decrypt_in_place_detached(&nonce, &bad_aad, &mut ours_dec, &tag);
    let mut their_dec = bad_ct;
    let their_ok = theirs.decrypt_inout_detached(
        &rustcrypto::Nonce::from(nonce),
        &bad_aad,
        rustcrypto::aead::inout::InOutBuf::from(&mut their_dec[..]),
        &rustcrypto::Tag::from(tag),
    );
    match (ours_ok, their_ok) {
        (Ok(()), Ok(())) => assert_eq!(ours_dec, their_dec, "open pt mismatch (case {case})"),
        (Err(_), Err(_)) => {},
        (o, t) => panic!("open verdict mismatch (case {case}): ours={o:?} theirs={t:?}"),
    }

    // XChaCha20 path (HChaCha20 subkey included)
    let ours_x = XChaCha20Poly1305::new(key);
    let theirs_x = rustcrypto::XChaCha20Poly1305::new(&rustcrypto::Key::from(key));
    let ct_ours = ours_x
        .encrypt(&xnonce, Payload {
            msg: &msg,
            aad: &aad,
        })
        .unwrap();
    let ct_theirs = theirs_x
        .encrypt(
            &rustcrypto::XNonce::from(xnonce),
            rustcrypto::aead::Payload {
                msg: &msg,
                aad: &aad,
            },
        )
        .unwrap();
    assert_eq!(ct_ours, ct_theirs, "xchacha ct mismatch (case {case})");
    assert_eq!(
        ours_x
            .decrypt(&xnonce, Payload {
                msg: &ct_ours,
                aad: &aad
            })
            .unwrap(),
        msg,
        "xchacha roundtrip mismatch (case {case})"
    );
}

fn main() {
    let mut args = std::env::args().skip(1);
    let iterations: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(20_000);
    let seed: u64 = args
        .next()
        .and_then(|a| a.parse().ok())
        .unwrap_or(0x1234_5678_9abc_def0);
    println!(
        "backend: {}, iterations: {iterations}, seed: {seed:#x}",
        chacha20poly1305_simd::active_backend()
    );
    let mut r = Rng(seed);
    for case in 0..iterations {
        check(case, &mut r);
    }
    println!("ok: all {iterations} differential cases passed");
}
