//! Throughput benchmark: this crate vs the RustCrypto `chacha20poly1305` fork
//! vs aws-lc-rs (AWS-LC, runtime auto-dispatch like OpenSSL).
//!
//! Run: `cargo bench --bench aead`
//!
//! Both sides can be pinned to a specific backend via RUSTFLAGS so that the
//! comparison is always backend-to-backend. Bench ids embed
//! the active backend labels, so runs with different pinning coexist under
//! `target/criterion` instead of overwriting each other:
//!
//! ```text
//! RUSTFLAGS='--cfg chacha20poly1305_backend="avx2" -Ctarget-feature=+avx2 \
//!            --cfg chacha20_backend="avx2"' cargo bench --bench aead
//! ```
//!
//! aws-lc-rs exposes no XChaCha20-Poly1305, so it only takes part in the
//! plain ChaCha20-Poly1305 seal/open groups.

use std::hint::black_box;

use awslc::aead::{Aad, CHACHA20_POLY1305, LessSafeKey, Nonce, UnboundKey};
use chacha20poly1305_simd::{ChaCha20Poly1305, XChaCha20Poly1305, active_backend};
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use rustcrypto::{
    ChaCha20Poly1305 as RcCipher, XChaCha20Poly1305 as RcXCipher,
    aead::{Aead, AeadInOut, KeyInit, Payload as RcPayload, inout::InOutBuf},
};

/// aws-lc-rs bench id: single series, CPU auto-dispatch (no cfg variants).
const AWSLC_ID: &str = "awslc";

fn awslc_key(key: &[u8; 32]) -> LessSafeKey {
    LessSafeKey::new(UnboundKey::new(&CHACHA20_POLY1305, key).expect("aws-lc key init"))
}

/// Label for our side: the forced backend name, or the probed one in auto mode.
fn ours_tag() -> String {
    active_backend().to_owned()
}

/// Label for the RustCrypto side: which `--cfg`-selected chacha20/poly1305
/// backends the reference build uses (`auto` = runtime autodetection).
fn rustcrypto_tag() -> String {
    let chacha = if cfg!(chacha20_backend = "soft") {
        "soft"
    } else if cfg!(chacha20_backend = "sse2") {
        "sse2"
    } else if cfg!(chacha20_backend = "avx2") {
        "avx2"
    } else if cfg!(chacha20_backend = "avx512") {
        "avx512"
    } else if cfg!(chacha20_avx512) {
        "auto(<=avx512)"
    } else {
        "auto(<=avx2)"
    };
    let poly = if cfg!(poly1305_backend = "soft") {
        "soft"
    } else {
        "auto(avx2|soft)"
    };
    format!("rc-{chacha}+poly-{poly}")
}

const SIZES: &[usize] = &[
    16,
    31,
    64,
    128,
    256,
    512,
    1024,
    4096,
    16 * 1024,
    65_536,
    262_144,
    1_048_576,
];

fn bench_seal(c: &mut Criterion) {
    let key = [0x24u8; 32];
    let nonce = [0x42u8; 12];
    let aad = [0xaau8; 16];
    let ours = ChaCha20Poly1305::new(key);
    let theirs = RcCipher::new(&rustcrypto::Key::from(key));
    let rnonce = rustcrypto::Nonce::from(nonce);

    let (ours_tag, rc_tag) = (ours_tag(), rustcrypto_tag());
    let awslc = awslc_key(&key);
    let mut group = c.benchmark_group("seal/chacha20poly1305");
    for &size in SIZES {
        let mut buf = vec![0u8; size];
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_function(format!("{ours_tag}/{size}"), |b| {
            b.iter(|| {
                black_box(ours.encrypt_in_place_detached(
                    black_box(&nonce),
                    black_box(&aad),
                    &mut buf,
                ))
            });
        });
        group.bench_function(format!("{rc_tag}/{size}"), |b| {
            b.iter_batched(
                || buf.clone(),
                |mut b| {
                    let _ = black_box(theirs.clone().encrypt_inout_detached(
                        black_box(&rnonce),
                        black_box(&aad),
                        InOutBuf::from(&mut b[..]),
                    ));
                },
                BatchSize::LargeInput,
            );
        });
        group.bench_function(format!("{AWSLC_ID}/{size}"), |b| {
            b.iter(|| {
                let _ = black_box(awslc.seal_in_place_separate_tag(
                    Nonce::assume_unique_for_key(nonce),
                    Aad::from(aad),
                    &mut buf,
                ));
            });
        });
    }
    group.finish();
}

fn bench_open(c: &mut Criterion) {
    let key = [0x24u8; 32];
    let nonce = [0x42u8; 12];
    let aad = [0xaau8; 16];
    let ours = ChaCha20Poly1305::new(key);
    let mut buf = vec![0u8; 1 << 20];
    let tag = ours
        .encrypt_in_place_detached(&nonce, &aad, &mut buf)
        .unwrap();

    let mut group = c.benchmark_group("open/chacha20poly1305");
    group.throughput(Throughput::Bytes(buf.len() as u64));
    group.bench_function(format!("{}/1MiB", ours_tag()), |b| {
        b.iter(|| {
            ours.decrypt_in_place_detached(
                black_box(&nonce),
                black_box(&aad),
                &mut buf,
                black_box(&tag),
            )
        });
    });
    let awslc = awslc_key(&key);
    group.bench_function(format!("{AWSLC_ID}/1MiB"), |b| {
        b.iter(|| {
            let _ = black_box(awslc.open_in_place_separate_tag(
                Nonce::assume_unique_for_key(nonce),
                Aad::from(aad),
                tag.as_ref(),
                &mut buf,
            ));
        });
    });
    group.finish();
}

fn bench_xchacha(c: &mut Criterion) {
    let key = [0x24u8; 32];
    let xnonce = [0x42u8; 24];
    let aad = [0xaau8; 16];
    let ours = XChaCha20Poly1305::new(key);
    let theirs = RcXCipher::new(&rustcrypto::Key::from(key));
    let rxnonce = rustcrypto::XNonce::from(xnonce);

    let (ours_tag, rc_tag) = (ours_tag(), rustcrypto_tag());
    let mut group = c.benchmark_group("seal/xchacha20poly1305");
    for &size in &[1024usize, 1 << 20] {
        let mut buf = vec![0u8; size];
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_function(format!("{ours_tag}/{size}"), |b| {
            b.iter(|| {
                black_box(ours.encrypt_in_place_detached(
                    black_box(&xnonce),
                    black_box(&aad),
                    &mut buf,
                ))
            });
        });
        group.bench_function(format!("{rc_tag}/{size}"), |b| {
            b.iter_batched(
                || buf.clone(),
                |b| {
                    let _ = black_box(theirs.encrypt(
                        black_box(&rxnonce),
                        black_box(RcPayload { msg: &b, aad: &aad }),
                    ));
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn bench_alloc_api(c: &mut Criterion) {
    // Real-world allocation level: Vec in, Vec out.
    let key = [0x24u8; 32];
    let nonce = [0x42u8; 12];
    let ours = ChaCha20Poly1305::new(key);
    let theirs = RcCipher::new(&rustcrypto::Key::from(key));
    let rnonce = rustcrypto::Nonce::from(nonce);
    let pt = vec![0u8; 1 << 20];

    let (ours_tag, rc_tag) = (ours_tag(), rustcrypto_tag());
    let mut group = c.benchmark_group("alloc/chacha20poly1305");
    group.throughput(Throughput::Bytes(pt.len() as u64));
    group.bench_function(format!("{ours_tag}/1MiB"), |b| {
        b.iter(|| black_box(ours.encrypt(black_box(&nonce), black_box(pt.as_slice()))));
    });
    group.bench_function(format!("{rc_tag}/1MiB"), |b| {
        b.iter(|| {
            black_box(theirs.encrypt(
                black_box(&rnonce),
                black_box(RcPayload { msg: &pt, aad: b"" }),
            ))
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_seal,
    bench_open,
    bench_xchacha,
    bench_alloc_api
);
criterion_main!(benches);
