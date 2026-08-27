//! One-shot profiling driver for chacha20poly1305-simd. Standalone package
//! under `perf/`; depends on the crate's `hotpath` feature, which turns the
//! `#[hotpath::measure]` probes planted on the public AEAD entry points and
//! the fused engine into real per-function recorders under a `HotpathGuard`.
//!
//! ```text
//! cargo run --release -p hotpath-profile          # (from this directory)
//!
//! # machine-readable output for A/B comparison:
//! HOTPATH_OUTPUT_FORMAT=json-pretty HOTPATH_OUTPUT_PATH=report.json \
//!   cargo run --release --manifest-path perf/hotpath-profile/Cargo.toml
//! ```

use std::hint::black_box;

use chacha20poly1305_simd::{ChaCha20Poly1305, XChaCha20Poly1305, active_backend};
use hotpath::{Format, HotpathGuardBuilder};

fn main() {
    let _guard = HotpathGuardBuilder::new("chacha20poly1305_profile")
        .percentiles(&[50.0, 90.0, 99.0])
        .format(Format::Table)
        .build();

    println!("backend: {}", active_backend());

    let key = [0x24u8; 32];
    let nonce = [0x42u8; 12];
    let xnonce = [0x42u8; 24];
    let aad = [0xaau8; 16];
    let cipher = ChaCha20Poly1305::new(&key);
    let xcipher = XChaCha20Poly1305::new(&key);

    // Small-message path (1 KiB): per-call fixed overhead dominates.
    let mut small = vec![0u8; 1024];
    for _ in 0..50_000 {
        let tag = cipher.encrypt_in_place_detached(black_box(&nonce), black_box(&aad), &mut small).unwrap();
        black_box(tag);
        cipher
            .decrypt_in_place_detached(black_box(&nonce), black_box(&aad), &mut small, black_box(&tag))
            .unwrap();
    }

    // Bulk path (1 MiB): the fused ChaCha20+Poly1305 loop dominates.
    let mut big = vec![0u8; 1 << 20];
    for _ in 0..200 {
        let tag = cipher.encrypt_in_place_detached(black_box(&nonce), black_box(&aad), &mut big).unwrap();
        black_box(tag);
        cipher
            .decrypt_in_place_detached(black_box(&nonce), black_box(&aad), &mut big, black_box(&tag))
            .unwrap();
    }

    // XChaCha key-derivation overhead.
    for _ in 0..10_000 {
        black_box(xcipher.encrypt_in_place_detached(black_box(&xnonce), black_box(&aad), &mut small).unwrap());
    }
}
