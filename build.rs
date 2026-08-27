//! Compile-time backend selection, mirroring RustCrypto's RUSTFLAGS contract:
//!
//! ```text
//! RUSTFLAGS='--cfg chacha20poly1305_backend="avx2"' cargo build
//! ```
//!
//! The requested backend is validated against the target arch and the enabled
//! `-Ctarget-feature`s (a forced SIMD backend must only run on CPUs that
//! support it), then translated into simple cfg aliases that gate backend
//! module compilation throughout the crate:
//!
//! - `backend_<name>`   — backend `<name>` code is compiled
//! - `force_backend`    — exactly one backend is compiled and dispatch is a compile-time constant
//!   (no CPU probing at all)
//!
//! Without the cfg, every backend available for the target arch is compiled
//! and selected at runtime (`std`) or via `target_feature` (`no_std`).
//!
//! The AVX-512 backend additionally requires the (default-on) `avx512` crate feature.

use std::{env, process};

/// (name, required arch ("" = any), required target features).
///
/// Backends whose required arch is "" or equal to the target arch are all
/// compiled in auto mode; baseline ISA backends (sse2 on x86-64, neon on
/// aarch64) need no explicit target features.
const BACKENDS: &[(&str, &str, &[&str])] = &[
    ("soft", "", &[]),
    ("sse2", "x86_64", &[]),
    ("avx2", "x86_64", &["avx2"]),
    ("avx512", "x86_64", &["avx2", "avx512f", "avx512vl"]),
    ("neon", "aarch64", &[]),
];

const VALID: &str = "one of: soft, sse2, avx2, avx512, neon";

fn fail(msg: &str) -> ! {
    println!("cargo:error=chacha20poly1305-simd: {msg}");
    process::exit(1);
}

fn main() {
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let features: Vec<&str> = target_features.split(',').collect();
    let avx512_feature = env::var_os("CARGO_FEATURE_AVX512").is_some();

    // Declare every cfg we emit or read so `unexpected_cfgs` stays quiet.
    for &(name, ..) in BACKENDS {
        println!("cargo:rustc-check-cfg=cfg(backend_{name})");
    }
    println!("cargo:rustc-check-cfg=cfg(force_backend)");
    println!(
        "cargo:rustc-check-cfg=cfg(chacha20poly1305_backend, values(\"soft\", \"sse2\", \"avx2\", \
         \"avx512\", \"neon\"))"
    );
    // Upstream cfgs, read by benches/aead.rs to label the reference build.
    println!(
        "cargo:rustc-check-cfg=cfg(chacha20_backend, values(\"soft\", \"sse2\", \"avx2\", \
         \"avx512\"))"
    );
    println!("cargo:rustc-check-cfg=cfg(chacha20_avx512)");
    println!("cargo:rustc-check-cfg=cfg(poly1305_backend, values(\"soft\", \"avx2\"))");

    let forced = forced_backend();

    if let Some(name) = &forced {
        let (_, want_arch, want_feats) =
            BACKENDS
                .iter()
                .find(|(n, ..)| n == name)
                .unwrap_or_else(|| {
                    fail(&format!(
                        "invalid chacha20poly1305_backend {name:?} ({VALID})"
                    ))
                });
        if name == "avx512" && !avx512_feature {
            fail(
                "forced backend \"avx512\" requires the `avx512` crate feature (enabled by \
                 default)",
            );
        }
        if !want_arch.is_empty() && *want_arch != arch {
            fail(&format!(
                "forced backend {name:?} requires target arch {want_arch} (current: {arch})"
            ));
        }
        let missing: Vec<&str> = want_feats
            .iter()
            .filter(|f| !features.contains(f))
            .copied()
            .collect();
        if !missing.is_empty() {
            fail(&format!(
                "forced backend {name:?} requires target features {} (missing: {}; add e.g. \
                 RUSTFLAGS='-Ctarget-feature=+{}' or build with -Ctarget-cpu=native)",
                want_feats.join(","),
                missing.join(","),
                missing.join(","),
            ));
        }
        println!("cargo:rustc-cfg=force_backend");
    }

    for &(name, want_arch, _) in BACKENDS {
        let on = match &forced {
            // Forced: exactly the requested backend.
            Some(b) => b == name,
            // Auto: every backend reachable on this target arch; the AVX-512
            // kernels are additionally gated behind the `avx512` feature.
            None => {
                (want_arch.is_empty() || want_arch == arch) && (name != "avx512" || avx512_feature)
            },
        };
        if on {
            println!("cargo:rustc-cfg=backend_{name}");
        }
    }
}

/// Extract `chacha20poly1305_backend=<value>` from `CARGO_ENCODED_RUSTFLAGS`
/// (`\x1f`-separated rustc args). Accepts `--cfg k=v`, `--cfg=k=v` and the
/// two-arg `--cfg` `k=v` spellings.
fn forced_backend() -> Option<String> {
    let flags = env::var("CARGO_ENCODED_RUSTFLAGS").ok()?;
    let args: Vec<&str> = flags.split('\u{1f}').collect();
    let mut value: Option<String> = None;
    for (i, arg) in args.iter().enumerate() {
        let cfg = if let Some(rest) = arg.strip_prefix("--cfg=") {
            rest
        } else if *arg == "--cfg" {
            match args.get(i + 1) {
                Some(next) => next,
                None => continue,
            }
        } else {
            continue;
        };
        if let Some(v) = cfg.trim().strip_prefix("chacha20poly1305_backend=") {
            // Later flags win, mirroring rustc's own `--cfg` override order.
            value = Some(v.trim_matches('"').to_owned());
        }
    }
    value
}
