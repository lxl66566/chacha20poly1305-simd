//! Backend selection.
//!
//! Two modes (cfg aliases are computed and validated by `build.rs`):
//!
//! - *auto* (default): every backend reachable on the target arch is compiled; `std` builds probe
//!   the CPU once and cache the result in a single `AtomicU8` (steady state: one relaxed load per
//!   AEAD call), `no_std` builds fall back to compile-time `target_feature`s.
//! - *forced*: `RUSTFLAGS='--cfg chacha20poly1305_backend="<name>"'` compiles exactly one backend —
//!   the other SIMD kernels are removed from the binary and dispatch resolves to a compile-time
//!   constant with zero probing (RustCrypto-style contract; `build.rs` rejects arch /
//!   target-feature mismatches at compile time).
//!
//! # Adding a backend (e.g. aarch64 NEON)
//!
//! 1. Implement `chacha::<arch>` (kernels + `BATCH_BLOCKS`) and, if a faster MAC exists,
//!    `poly1305::<arch>` behind the arch gate.
//! 2. Register it in `build.rs` (`BACKENDS`), add a `Kind` variant plus a unique id, extend
//!    [`detect`] with the runtime feature probe, and wire the `seal`/`open` match arms.
//!
//! The fused engine in [`crate::aead`] is arch-agnostic; nothing else changes.

use crate::{Tag, chacha::State};

// `soft` is also compiled under `test`: the SIMD backend suites use it as the
// correctness reference.
#[cfg(backend_avx2)]
pub(crate) mod avx2;
#[cfg(backend_avx512)]
pub(crate) mod avx512;
#[cfg(backend_neon)]
pub(crate) mod neon;
#[cfg(any(backend_soft, test))]
pub(crate) mod soft;
#[cfg(backend_sse2)]
pub(crate) mod sse2;

/// Backend identifier. The `u8` representation is what gets cached in
/// `KIND` (auto mode); `0` is reserved for "not yet detected".
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub(crate) enum Kind {
    #[cfg(backend_soft)]
    Soft = 3,
    #[cfg(backend_avx2)]
    Avx2 = 1,
    #[cfg(backend_avx512)]
    Avx512 = 2,
    #[cfg(backend_neon)]
    Neon = 4,
    #[cfg(backend_sse2)]
    Sse2 = 5,
}

// ── Forced mode: `build.rs` guarantees exactly one of these exists ──

#[cfg(all(force_backend, backend_soft))]
const FORCED: Kind = Kind::Soft;
#[cfg(all(force_backend, backend_avx2))]
const FORCED: Kind = Kind::Avx2;
#[cfg(all(force_backend, backend_avx512))]
const FORCED: Kind = Kind::Avx512;
#[cfg(all(force_backend, backend_neon))]
const FORCED: Kind = Kind::Neon;
#[cfg(all(force_backend, backend_sse2))]
const FORCED: Kind = Kind::Sse2;

#[cfg(force_backend)]
#[inline]
fn current() -> Kind {
    FORCED
}

// ── Auto mode: probe once, cache in a single atomic ──

#[cfg(not(force_backend))]
static KIND: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);

#[cfg(not(force_backend))]
#[cfg(all(target_arch = "x86_64", feature = "std"))]
fn detect() -> Kind {
    if std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512vl")
        // The AVX-512 entry points enable `avx2` in `target_feature`; detect it
        // explicitly instead of relying on "all AVX-512 CPUs have AVX2".
        && std::arch::is_x86_feature_detected!("avx2")
    {
        Kind::Avx512
    } else if std::arch::is_x86_feature_detected!("avx2") {
        Kind::Avx2
    } else {
        // SSE2 is part of the x86-64 baseline ISA: always present.
        Kind::Sse2
    }
}

#[cfg(not(force_backend))]
#[cfg(all(target_arch = "aarch64", feature = "std"))]
fn detect() -> Kind {
    // NEON is part of the aarch64 baseline ISA.
    Kind::Neon
}

#[cfg(not(force_backend))]
#[cfg(not(any(
    all(target_arch = "x86_64", feature = "std"),
    all(target_arch = "aarch64", feature = "std")
)))]
fn detect() -> Kind {
    // No runtime CPU detection available (no_std): fall back to compile-time
    // target features, e.g. `-C target-feature=+avx2`. A binary built this way
    // may only run on CPUs with those features, so this stays sound.
    #[cfg(target_arch = "x86_64")]
    {
        if cfg!(target_feature = "avx512f")
            && cfg!(target_feature = "avx512vl")
            && cfg!(target_feature = "avx2")
        {
            return Kind::Avx512;
        }
        if cfg!(target_feature = "avx2") {
            return Kind::Avx2;
        }
        // SSE2 is part of the x86-64 baseline ISA: always sound to use.
        return Kind::Sse2;
    }
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is part of the aarch64 baseline (always present in practice;
        // the purely hypothetical no-NEON toolchain falls back to soft).
        if cfg!(target_feature = "neon") {
            return Kind::Neon;
        }
    }
    // Neither x86-64 nor aarch64 (or a hypothetical no-NEON aarch64 toolchain).
    #[cfg(not(target_arch = "x86_64"))]
    {
        return Kind::Soft;
    }
}

/// Map a cached `u8` back to a [`Kind`]; unknown ids (e.g. from a future
/// version) degrade to the always-available soft backend.
#[cfg(not(force_backend))]
#[inline]
fn kind_from_u8(k: u8) -> Kind {
    #[cfg(target_arch = "x86_64")]
    match k {
        1 => return Kind::Avx2,
        2 => return Kind::Avx512,
        5 => return Kind::Sse2,
        _ => {},
    }
    #[cfg(target_arch = "aarch64")]
    if k == 4 {
        return Kind::Neon;
    }
    let _ = k;
    Kind::Soft
}

#[cfg(not(force_backend))]
#[inline]
fn current() -> Kind {
    let mut k = KIND.load(core::sync::atomic::Ordering::Relaxed);
    if k == 0 {
        k = detect() as u8;
        KIND.store(k, core::sync::atomic::Ordering::Relaxed);
    }
    kind_from_u8(k)
}

#[inline]
pub(crate) fn with_backend<R>(f: impl FnOnce(Kind) -> R) -> R {
    f(current())
}

impl Kind {
    pub(crate) fn seal(self, state: &mut State, aad: &[u8], msg: &mut [u8], tag: &mut Tag) {
        match self {
            #[cfg(backend_soft)]
            Self::Soft => crate::aead::seal::<soft::SoftOps>(state, aad, msg, tag),
            #[cfg(backend_avx2)]
            Self::Avx2 => unsafe { avx2::seal(state, aad, msg, tag) },
            #[cfg(backend_avx512)]
            Self::Avx512 => unsafe { avx512::seal(state, aad, msg, tag) },
            #[cfg(backend_neon)]
            Self::Neon => unsafe { neon::seal(state, aad, msg, tag) },
            #[cfg(backend_sse2)]
            Self::Sse2 => unsafe { sse2::seal(state, aad, msg, tag) },
        }
    }

    pub(crate) fn open(self, state: &mut State, aad: &[u8], buf: &mut [u8], tag: &Tag) -> bool {
        match self {
            #[cfg(backend_soft)]
            Self::Soft => crate::aead::open::<soft::SoftOps>(state, aad, buf, tag),
            #[cfg(backend_avx2)]
            Self::Avx2 => unsafe { avx2::open(state, aad, buf, tag) },
            #[cfg(backend_avx512)]
            Self::Avx512 => unsafe { avx512::open(state, aad, buf, tag) },
            #[cfg(backend_neon)]
            Self::Neon => unsafe { neon::open(state, aad, buf, tag) },
            #[cfg(backend_sse2)]
            Self::Sse2 => unsafe { sse2::open(state, aad, buf, tag) },
        }
    }
}

/// Which backend is actually active (exposed for tests / diagnostics).
#[must_use]
pub fn active_backend() -> &'static str {
    match with_backend(|k| k) {
        #[cfg(backend_soft)]
        Kind::Soft => "soft",
        #[cfg(backend_avx2)]
        Kind::Avx2 => "avx2",
        #[cfg(backend_avx512)]
        Kind::Avx512 => "avx512",
        #[cfg(backend_neon)]
        Kind::Neon => "neon",
        #[cfg(backend_sse2)]
        Kind::Sse2 => "sse2",
    }
}
