//! AVX2 Poly1305 — Goll–Gueron batched algorithm.
//!
//! Four blocks are folded per iteration into a polynomial
//! `P = T0 + T1·x + T2·x² + T3·x³` (with `x = 2^130 ≡ 5`), advancing it as
//! `P ← R⁴·P + blocks`, which amortizes the carry propagation across four
//! blocks and keeps the multiplier dependency chain short.
//!
//! Ported from RustCrypto `poly1305` v0.9 (Apache-2.0 OR MIT), which derives
//! from Goll & Gueron's AVX2 C code ("Vectorization of Poly1305 Message
//! Authentication Code", 2015) as modified by Bhattacharyya & Sarkar — minus
//! the streaming `partial_block` machinery (our AEAD construction always
//! zero-pads segments, see the module contract in `super`).

// m_r_N / m_5r_N mirror the Goll–Gueron paper notation (r^k and 5·r^k rows).
#![allow(clippy::similar_names)]

use core::arch::x86_64::*;

use super::Backend;

const fn set02(x3: u8, x2: u8, x1: u8, x0: u8) -> i32 {
    ((x3 << 6) | (x2 << 4) | (x1 << 2) | x0) as i32
}

#[derive(Clone)]
pub(crate) struct Avx2Poly {
    k: AdditionKey,
    r1: PrecomputedMultiplier,
    r2: PrecomputedMultiplier,
    initialized: Option<Initialized>,
    // Up to 8 whole blocks are deferred: the streaming GG state (r³/r⁴
    // powers) is only built when a 9th block arrives or the bulk `absorb4`
    // path runs, so short messages never pay for it (finalize folds the
    // cached blocks pairwise through `r1`/`r2` alone). `num_cached` is a
    // multiple of 4 whenever the engine calls `absorb4` (see aead.rs).
    cached: [u8; 128],
    num_cached: usize,
}

#[derive(Copy, Clone)]
struct Initialized {
    p: Aligned4x130,
    m: SpacedMultiplier4x130,
    r4: PrecomputedMultiplier,
}

impl Backend for Avx2Poly {
    #[inline(always)]
    unsafe fn init(key: &[u8; 32]) -> Self {
        let (k, r1) = prepare_keys(key);
        let r2 = PrecomputedMultiplier::from((Aligned130(r1.a) * r1).reduce());
        Self {
            k,
            r1,
            r2,
            initialized: None,
            cached: [0; 128],
            num_cached: 0,
        }
    }

    #[inline(always)]
    unsafe fn absorb_block(&mut self, block: &[u8; 16]) {
        if self.num_cached == 8 {
            // Cache full: this stream is long enough to amortize the
            // streaming GG state — drain and continue streaming.
            self.drain_cache();
        }
        self.cached[self.num_cached * 16..][..16].copy_from_slice(block);
        self.num_cached += 1;
    }

    #[inline(always)]
    unsafe fn absorb4(&mut self, blocks: &[u8; 64]) {
        // The engine's alignment prologue guarantees the deferred cache
        // holds a multiple of 4 blocks (a 64-byte boundary).
        debug_assert_eq!(self.num_cached % 4, 0, "stream must be 64B-aligned");
        self.drain_cache();
        let loaded = Aligned4x130::from_bytes(blocks);
        self.advance(loaded);
    }

    #[inline(always)]
    fn pending_blocks(&self) -> usize {
        self.num_cached
    }

    // All fields are plain SIMD/integer data; one volatile write per field
    // keeps the scrub from being optimized out without needing `zeroize`
    // impls for `__m256i`.
    #[cfg(feature = "zeroize")]
    unsafe fn zeroize_secrets(&mut self) {
        unsafe {
            core::ptr::write_volatile(&raw mut self.k, core::mem::zeroed());
            core::ptr::write_volatile(&raw mut self.r1, core::mem::zeroed());
            core::ptr::write_volatile(&raw mut self.r2, core::mem::zeroed());
            // Only scrub the payload; zeroing the whole Option could create
            // an invalid discriminant.
            if let Some(init) = &mut self.initialized {
                core::ptr::write_volatile(&raw mut *init, core::mem::zeroed());
            }
        }
        zeroize::Zeroize::zeroize(&mut self.cached);
        self.num_cached = 0;
    }

    #[inline(always)]
    unsafe fn finalize_into(&mut self, out: &mut [u8; 16]) {
        debug_assert!(self.num_cached <= 8);
        // P ← M·P then fold the four coefficients: P = T_0 + T_1 + T_2 + T_3.
        // NOTE: written as a match instead of Option::map — the map closure
        // used to be outlined into a separate (non-target-feature) function,
        // turning every SIMD intrinsic inside into an out-of-line call.
        #[allow(clippy::manual_map)]
        let mut p = match self.initialized.take() {
            Some(inner) => Some((inner.p * inner.m).sum().reduce()),
            None => None,
        };

        // Deferred-cache drain at end of stream: sequential 2-block folds
        // p ← (p + B₀)·r² + B₁·r — no r³/r⁴ powers needed.
        while self.num_cached >= 2 {
            let mut c = Aligned2x130::from_blocks(self.cached[..32].try_into().unwrap());
            if let Some(prev) = p {
                c = c + prev;
            }
            p = Some(c.mul_and_sum(self.r1, self.r2).reduce());
            self.num_cached -= 2;
            // shift the remaining cached blocks down (indexes shift by 32B)
            if self.num_cached > 0 {
                let rem = self.num_cached * 16;
                self.cached.copy_within(32..32 + rem, 0);
            }
        }

        if self.num_cached == 1 {
            let mut c = Aligned130::from_block(self.cached[..16].try_into().unwrap());
            if let Some(prev) = p {
                c = c + prev;
            }
            p = Some((c * self.r1).reduce());
            self.num_cached = 0;
        }

        let tag_int = match p {
            Some(p) => self.k + p,
            None => self.k.into(),
        };
        tag_int.write(out);
    }
}

impl Avx2Poly {
    #[inline(always)]
    unsafe fn advance(&mut self, blocks: Aligned4x130) {
        if let Some(inner) = &mut self.initialized {
            // P ← R⁴·P + blocks
            inner.p = (&inner.p * inner.r4).reduce() + blocks;
        } else {
            let (m, r4) = SpacedMultiplier4x130::new(self.r1, self.r2);
            self.initialized = Some(Initialized { p: blocks, m, r4 });
        }
    }

    /// Fold the whole deferred cache into the streaming state, in order.
    /// `num_cached` must be a multiple of 4.
    #[inline(always)]
    unsafe fn drain_cache(&mut self) {
        debug_assert_eq!(self.num_cached % 4, 0);
        let mut i = 0;
        while i < self.num_cached {
            let loaded = Aligned4x130::from_bytes(self.cached[i * 16..][..64].try_into().unwrap());
            self.advance(loaded);
            i += 4;
        }
        self.num_cached = 0;
    }
}

/// Derives the addition key and clamped polynomial key.
#[inline(always)]
unsafe fn prepare_keys(key: &[u8; 32]) -> (AdditionKey, PrecomputedMultiplier) {
    // [k7, k6, k5, k4, k3, k2, k1, k0]
    let key256 = _mm256_loadu_si256(key.as_ptr().cast());

    // Addition key: [0, k7, 0, k6, 0, k5, 0, k4]
    let k = AdditionKey(_mm256_and_si256(
        _mm256_permutevar8x32_epi32(key256, _mm256_set_epi32(3, 7, 2, 6, 1, 5, 0, 4)),
        _mm256_set_epi32(0, -1, 0, -1, 0, -1, 0, -1),
    ));

    // R = key & 0xffffffc0ffffffc0ffffffc0fffffff in 26-bit-aligned limbs
    let r = Aligned130::new(_mm256_and_si256(
        key256,
        _mm256_set_epi32(
            0,
            0,
            0,
            0,
            0x0fff_fffc,
            0x0fff_fffc,
            0x0fff_fffc,
            0x0fff_ffff,
        ),
    ));

    (k, PrecomputedMultiplier::from(r))
}

/// A 130-bit integer aligned across five 26-bit limbs (32-bit words).
#[derive(Clone, Copy)]
struct Aligned130(__m256i);

impl Aligned130 {
    /// Align a 16-byte block at 26-bit boundaries, high bit set.
    #[inline(always)]
    unsafe fn from_block(block: &[u8; 16]) -> Self {
        Aligned130::new(_mm256_or_si256(
            _mm256_and_si256(
                _mm256_castsi128_si256(_mm_loadu_si128(block.as_ptr().cast())),
                _mm256_set_epi64x(0, 0, -1, -1),
            ),
            _mm256_set_epi64x(0, 1, 0, 0),
        ))
    }

    /// Split a 130-bit integer (32-bit word layout) into five 26-bit limbs.
    #[inline(always)]
    unsafe fn new(x: __m256i) -> Self {
        let xl = _mm256_sllv_epi32(x, _mm256_set_epi32(32, 32, 32, 24, 18, 12, 6, 0));
        let xh = _mm256_permutevar8x32_epi32(
            _mm256_srlv_epi32(x, _mm256_set_epi32(32, 32, 32, 2, 8, 14, 20, 26)),
            _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7),
        );
        Aligned130(_mm256_and_si256(
            _mm256_or_si256(xl, xh),
            _mm256_set_epi32(
                0,
                0,
                0,
                0x03ff_ffff,
                0x03ff_ffff,
                0x03ff_ffff,
                0x03ff_ffff,
                0x03ff_ffff,
            ),
        ))
    }
}

impl core::ops::Add<Aligned130> for Aligned130 {
    type Output = Aligned130;

    #[inline(always)]
    fn add(self, other: Aligned130) -> Aligned130 {
        // 26-bit limbs inside 32-bit words leave slack for unreduced adds
        unsafe { Aligned130(_mm256_add_epi32(self.0, other.0)) }
    }
}

/// Multiplier precomputed as `[5r4, 5r3, 5r2, r4, r3, r2, r1, r0]` plus a
/// broadcast `5r1` vector.
#[derive(Clone, Copy)]
struct PrecomputedMultiplier {
    a: __m256i,
    a_5: __m256i,
}

impl From<Aligned130> for PrecomputedMultiplier {
    #[inline(always)]
    fn from(r: Aligned130) -> Self {
        unsafe {
            let a_5 = _mm256_permutevar8x32_epi32(
                _mm256_add_epi32(r.0, _mm256_slli_epi32(r.0, 2)),
                _mm256_set_epi32(4, 3, 2, 1, 1, 1, 1, 1),
            );
            let a = _mm256_blend_epi32(r.0, a_5, 0b1110_0000);
            let a_5 = _mm256_permute2x128_si256(a_5, a_5, 0);
            PrecomputedMultiplier { a, a_5 }
        }
    }
}

impl core::ops::Mul<PrecomputedMultiplier> for Aligned130 {
    type Output = Unreduced130;

    /// 26-bit × 26-bit multiply with lazy reduction, switching to 64-bit
    /// accumulator lanes.
    #[inline(always)]
    fn mul(self, other: PrecomputedMultiplier) -> Unreduced130 {
        unsafe {
            let x = self.0;
            let y = other.a;
            let z = other.a_5;

            let v0 = _mm256_mul_epu32(
                _mm256_permutevar8x32_epi32(x, _mm256_set_epi64x(4, 3, 2, 1)),
                _mm256_permutevar8x32_epi32(y, _mm256_set_epi64x(7, 7, 7, 7)),
            );
            let v0 = _mm256_add_epi64(
                v0,
                _mm256_mul_epu32(
                    _mm256_permutevar8x32_epi32(x, _mm256_set_epi64x(3, 2, 1, 0)),
                    _mm256_broadcastd_epi32(_mm256_castsi256_si128(y)),
                ),
            );
            let v0 = _mm256_add_epi64(
                v0,
                _mm256_mul_epu32(
                    _mm256_permutevar8x32_epi32(x, _mm256_set_epi64x(1, 1, 3, 3)),
                    _mm256_permutevar8x32_epi32(y, _mm256_set_epi64x(2, 1, 6, 5)),
                ),
            );
            let v0 = _mm256_add_epi64(
                v0,
                _mm256_mul_epu32(
                    _mm256_permute4x64_epi64(x, set02(1, 0, 0, 2)),
                    _mm256_blend_epi32(
                        _mm256_permutevar8x32_epi32(y, _mm256_set_epi64x(1, 2, 1, 1)),
                        z,
                        0x03,
                    ),
                ),
            );
            let v0 = _mm256_add_epi64(
                v0,
                _mm256_mul_epu32(
                    _mm256_permute4x64_epi64(x, set02(0, 2, 2, 1)),
                    _mm256_permutevar8x32_epi32(y, _mm256_set_epi64x(3, 6, 5, 6)),
                ),
            );

            let v1 = _mm256_mul_epu32(
                _mm256_permutevar8x32_epi32(x, _mm256_set_epi64x(3, 2, 1, 0)),
                _mm256_permutevar8x32_epi32(y, _mm256_set_epi64x(1, 2, 3, 4)),
            );
            let v1 = _mm256_add_epi64(v1, _mm256_permute4x64_epi64(v1, set02(1, 0, 3, 2)));
            let v1 = _mm256_add_epi64(v1, _mm256_permute4x64_epi64(v1, set02(0, 0, 0, 1)));
            let v1 = _mm256_add_epi64(
                v1,
                _mm256_mul_epu32(_mm256_permute4x64_epi64(x, set02(0, 0, 0, 2)), y),
            );

            Unreduced130 { v0, v1 }
        }
    }
}

/// Unreduced 130-bit product in 64-bit limbs: `v1 = [_, _, _, t4]`,
/// `v0 = [t3, t2, t1, t0]`.
#[derive(Clone, Copy)]
struct Unreduced130 {
    v0: __m256i,
    v1: __m256i,
}

impl Unreduced130 {
    /// Reduce modulo 2^130−5, back to 32-bit lanes.
    #[inline(always)]
    fn reduce(self) -> Aligned130 {
        unsafe {
            let (red_1, red_0) = adc(self.v1, self.v0);
            let (red_1, red_0) = red(red_1, red_0);
            let (red_1, red_0) = adc(red_1, red_0);
            Aligned130(_mm256_blend_epi32(
                _mm256_permutevar8x32_epi32(red_0, _mm256_set_epi32(0, 6, 4, 0, 6, 4, 2, 0)),
                _mm256_permutevar8x32_epi32(red_1, _mm256_set_epi32(0, 6, 4, 0, 6, 4, 2, 0)),
                0x90,
            ))
        }
    }
}

/// Carry chain: fold limb overflows upward.
#[inline(always)]
unsafe fn adc(v1: __m256i, v0: __m256i) -> (__m256i, __m256i) {
    let v0 = _mm256_add_epi64(
        _mm256_and_si256(
            v0,
            _mm256_set_epi64x(-1, 0x03ff_ffff, 0x03ff_ffff, 0x03ff_ffff),
        ),
        _mm256_permute4x64_epi64(
            _mm256_srlv_epi64(v0, _mm256_set_epi64x(64, 26, 26, 26)),
            set02(2, 1, 0, 3),
        ),
    );
    let v1 = _mm256_add_epi64(
        v1,
        _mm256_permute4x64_epi64(_mm256_srli_epi64(v0, 26), set02(2, 1, 0, 3)),
    );
    let chain = _mm256_and_si256(v0, _mm256_set_epi64x(0x03ff_ffff, -1, -1, -1));
    (v1, chain)
}

/// Fold the ≥2^130 part back via ×5.
#[inline(always)]
unsafe fn red(v1: __m256i, v0: __m256i) -> (__m256i, __m256i) {
    let t = _mm256_srlv_epi64(v1, _mm256_set_epi64x(64, 64, 64, 26));
    let red_0 = _mm256_add_epi64(_mm256_add_epi64(v0, t), _mm256_slli_epi64(t, 2));
    let red_1 = _mm256_and_si256(v1, _mm256_set_epi64x(0, 0, 0, 0x03ff_ffff));
    (red_1, red_0)
}

/// Two 130-bit integers (see `mul_and_sum`).
struct Aligned2x130 {
    v0: Aligned130,
    v1: Aligned130,
}

impl Aligned2x130 {
    #[inline(always)]
    unsafe fn from_blocks(src: &[u8; 32]) -> Self {
        Aligned2x130 {
            v0: Aligned130::from_block(src[..16].try_into().unwrap()),
            v1: Aligned130::from_block(src[16..32].try_into().unwrap()),
        }
    }

    /// Multiply both lanes by their respective r powers and sum, in one
    /// fused multiply tree.
    #[inline(always)]
    unsafe fn mul_and_sum(
        self,
        r1: PrecomputedMultiplier,
        r2: PrecomputedMultiplier,
    ) -> Unreduced130 {
        let x = self;
        let r15 = r1.a_5;
        let r25 = r2.a_5;
        let r1 = r1.a;
        let r2 = r2.a;

        let mut v0 = _mm256_mul_epu32(
            _mm256_permutevar8x32_epi32(x.v0.0, _mm256_set_epi64x(4, 3, 2, 1)),
            _mm256_permutevar8x32_epi32(r2, _mm256_set1_epi64x(7)),
        );
        let mut v1 = _mm256_mul_epu32(
            _mm256_permutevar8x32_epi32(x.v1.0, _mm256_set_epi64x(4, 3, 2, 1)),
            _mm256_permutevar8x32_epi32(r1, _mm256_set1_epi64x(7)),
        );
        v0 = _mm256_add_epi64(
            v0,
            _mm256_mul_epu32(
                _mm256_permute4x64_epi64(x.v0.0, set02(0, 2, 2, 1)),
                _mm256_permutevar8x32_epi32(r2, _mm256_set_epi64x(3, 6, 5, 6)),
            ),
        );
        v1 = _mm256_add_epi64(
            v1,
            _mm256_mul_epu32(
                _mm256_permute4x64_epi64(x.v1.0, set02(0, 2, 2, 1)),
                _mm256_permutevar8x32_epi32(r1, _mm256_set_epi64x(3, 6, 5, 6)),
            ),
        );
        v0 = _mm256_add_epi64(
            v0,
            _mm256_mul_epu32(
                _mm256_permutevar8x32_epi32(x.v0.0, _mm256_set_epi64x(1, 1, 3, 3)),
                _mm256_permutevar8x32_epi32(r2, _mm256_set_epi64x(2, 1, 6, 5)),
            ),
        );
        v1 = _mm256_add_epi64(
            v1,
            _mm256_mul_epu32(
                _mm256_permutevar8x32_epi32(x.v1.0, _mm256_set_epi64x(1, 1, 3, 3)),
                _mm256_permutevar8x32_epi32(r1, _mm256_set_epi64x(2, 1, 6, 5)),
            ),
        );
        v0 = _mm256_add_epi64(
            v0,
            _mm256_mul_epu32(
                _mm256_permutevar8x32_epi32(x.v0.0, _mm256_set_epi64x(3, 2, 1, 0)),
                _mm256_broadcastd_epi32(_mm256_castsi256_si128(r2)),
            ),
        );
        v1 = _mm256_add_epi64(
            v1,
            _mm256_mul_epu32(
                _mm256_permutevar8x32_epi32(x.v1.0, _mm256_set_epi64x(3, 2, 1, 0)),
                _mm256_broadcastd_epi32(_mm256_castsi256_si128(r1)),
            ),
        );
        let mut t0 = _mm256_permute4x64_epi64(x.v0.0, set02(1, 0, 0, 2));
        let mut t1 = _mm256_permute4x64_epi64(x.v1.0, set02(1, 0, 0, 2));
        v0 = _mm256_add_epi64(
            v0,
            _mm256_mul_epu32(
                t0,
                _mm256_blend_epi32(
                    _mm256_permutevar8x32_epi32(r2, _mm256_set_epi64x(1, 2, 1, 1)),
                    r25,
                    0b0000_0011,
                ),
            ),
        );
        v1 = _mm256_add_epi64(
            v1,
            _mm256_mul_epu32(
                t1,
                _mm256_blend_epi32(
                    _mm256_permutevar8x32_epi32(r1, _mm256_set_epi64x(1, 2, 1, 1)),
                    r15,
                    0b0000_0011,
                ),
            ),
        );
        v0 = _mm256_add_epi64(v0, v1);
        t0 = _mm256_mul_epu32(t0, r2);
        t1 = _mm256_mul_epu32(t1, r1);
        v1 = _mm256_add_epi64(t0, t1);
        t0 = _mm256_mul_epu32(
            _mm256_permutevar8x32_epi32(x.v0.0, _mm256_set_epi64x(3, 2, 1, 0)),
            _mm256_permutevar8x32_epi32(r2, _mm256_set_epi64x(1, 2, 3, 4)),
        );
        t1 = _mm256_mul_epu32(
            _mm256_permutevar8x32_epi32(x.v1.0, _mm256_set_epi64x(3, 2, 1, 0)),
            _mm256_permutevar8x32_epi32(r1, _mm256_set_epi64x(1, 2, 3, 4)),
        );
        t0 = _mm256_add_epi64(t0, t1);
        t0 = _mm256_add_epi64(t0, _mm256_permute4x64_epi64(t0, set02(1, 0, 3, 2)));
        t0 = _mm256_add_epi64(t0, _mm256_permute4x64_epi64(t0, set02(2, 3, 0, 1)));
        v1 = _mm256_add_epi64(v1, t0);
        Unreduced130 { v0, v1 }
    }
}

impl core::ops::Add<Aligned130> for Aligned2x130 {
    type Output = Aligned2x130;

    #[inline(always)]
    fn add(self, other: Aligned130) -> Aligned2x130 {
        Aligned2x130 {
            v0: self.v0 + other,
            v1: self.v1,
        }
    }
}

/// Multiplier taking `(x3, x2, x1, x0)` to `(x3·R⁴, x2·R³, x1·R², x0·R)`.
#[derive(Copy, Clone)]
struct SpacedMultiplier4x130 {
    v0: __m256i,
    v1: __m256i,
    r1: PrecomputedMultiplier,
}

impl SpacedMultiplier4x130 {
    /// Returns `(multiplier, R⁴)` given `(R¹, R²)`.
    #[inline(always)]
    unsafe fn new(
        r1: PrecomputedMultiplier,
        r2: PrecomputedMultiplier,
    ) -> (Self, PrecomputedMultiplier) {
        let r3 = (Aligned130(r2.a) * r1).reduce();
        // r4 needs the raw reduced limb layout for the v1 blend below, so
        // keep the Aligned130 around and convert only for the return value.
        let r4 = (Aligned130(r2.a) * r2).reduce();
        let r4_pm = PrecomputedMultiplier::from(r4);

        // v0 = [r2_4, r2_3, r2_1, r3_4, r3_3, r3_2, r3_1, r3_0]
        let v0 = _mm256_blend_epi32(
            r3.0,
            _mm256_permutevar8x32_epi32(r2.a, _mm256_set_epi32(4, 3, 1, 0, 0, 0, 0, 0)),
            0b1110_0000,
        );
        // v1 = [r2_4, r2_2, r2_0, r4_4, r4_3, r4_2, r4_1, r4_0]
        let v1 = _mm256_blend_epi32(
            r4.0,
            _mm256_permutevar8x32_epi32(r2.a, _mm256_set_epi32(4, 2, 0, 0, 0, 0, 0, 0)),
            0b1110_0000,
        );

        (Self { v0, v1, r1 }, r4_pm)
    }
}

/// Four 130-bit integers across three YMM vectors.
#[derive(Clone, Copy)]
struct Aligned4x130 {
    v0: __m256i,
    v1: __m256i,
    v2: __m256i,
}

impl Aligned4x130 {
    /// Load 4 blocks (64 bytes), align to 26-bit limbs, set high bits.
    #[inline(always)]
    unsafe fn from_bytes(src: &[u8; 64]) -> Self {
        let blocks_01 = _mm256_loadu_si256(src[..32].as_ptr().cast());
        let blocks_23 = _mm256_loadu_si256(src[32..].as_ptr().cast());

        let mask_26 = _mm256_set1_epi32(0x03ff_ffff);
        let set_hibit = _mm256_set1_epi32(1 << 24);

        let a0 = _mm256_permute4x64_epi64(
            _mm256_unpackhi_epi64(blocks_01, blocks_23),
            set02(3, 1, 2, 0),
        );
        let a1 = _mm256_permute4x64_epi64(
            _mm256_unpacklo_epi64(blocks_01, blocks_23),
            set02(3, 1, 2, 0),
        );

        let v2 = _mm256_or_si256(_mm256_srli_epi64(a0, 40), set_hibit);
        let a2 = _mm256_or_si256(_mm256_srli_epi64(a1, 46), _mm256_slli_epi64(a0, 18));
        let v1 = _mm256_and_si256(
            _mm256_blend_epi32(_mm256_srli_epi64(a1, 26), a2, 0xaa),
            mask_26,
        );
        let v0 = _mm256_and_si256(
            _mm256_blend_epi32(a1, _mm256_slli_epi64(a2, 26), 0xaa),
            mask_26,
        );

        Aligned4x130 { v0, v1, v2 }
    }
}

impl core::ops::Add<Aligned4x130> for Aligned4x130 {
    type Output = Aligned4x130;

    #[inline(always)]
    fn add(self, other: Aligned4x130) -> Aligned4x130 {
        unsafe {
            Aligned4x130 {
                v0: _mm256_add_epi32(self.v0, other.v0),
                v1: _mm256_add_epi32(self.v1, other.v1),
                v2: _mm256_add_epi32(self.v2, other.v2),
            }
        }
    }
}

impl core::ops::Mul<PrecomputedMultiplier> for &Aligned4x130 {
    type Output = Unreduced4x130;

    /// The hot loop: four 130-bit multiplies by the same multiplier in one
    /// instruction tree (≈18 `vpmuludq` per 64 bytes).
    #[inline(always)]
    fn mul(self, other: PrecomputedMultiplier) -> Unreduced4x130 {
        unsafe {
            let mut x = *self;
            let y = other.a;
            let z = other.a_5;

            let ord = _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1);

            let mut t0 = _mm256_permute4x64_epi64(y, set02(0, 0, 0, 0));
            let mut t1 = _mm256_permute4x64_epi64(y, set02(1, 1, 1, 1));

            let mut v0 = _mm256_mul_epu32(x.v0, t0);
            let mut v1 = _mm256_mul_epu32(x.v1, t0);
            let mut v4 = _mm256_mul_epu32(x.v2, t0);
            let mut v2 = _mm256_mul_epu32(x.v0, t1);
            let mut v3 = _mm256_mul_epu32(x.v1, t1);

            t0 = _mm256_permutevar8x32_epi32(t0, ord);
            t1 = _mm256_permutevar8x32_epi32(t1, ord);
            v1 = _mm256_add_epi64(v1, _mm256_mul_epu32(x.v0, t0));
            v2 = _mm256_add_epi64(v2, _mm256_mul_epu32(x.v1, t0));
            v3 = _mm256_add_epi64(v3, _mm256_mul_epu32(x.v0, t1));
            v4 = _mm256_add_epi64(v4, _mm256_mul_epu32(x.v1, t1));

            let mut t2 = _mm256_permute4x64_epi64(y, set02(2, 2, 2, 2));

            v4 = _mm256_add_epi64(v4, _mm256_mul_epu32(x.v0, t2));

            x.v0 = _mm256_permutevar8x32_epi32(x.v0, ord);
            x.v1 = _mm256_permutevar8x32_epi32(x.v1, ord);
            t2 = _mm256_permutevar8x32_epi32(t2, ord);

            v0 = _mm256_add_epi64(v0, _mm256_mul_epu32(x.v1, t2));
            v1 = _mm256_add_epi64(v1, _mm256_mul_epu32(x.v2, t2));
            v3 = _mm256_add_epi64(v3, _mm256_mul_epu32(x.v0, t0));
            v4 = _mm256_add_epi64(v4, _mm256_mul_epu32(x.v1, t0));

            t0 = _mm256_permutevar8x32_epi32(t0, ord);
            t1 = _mm256_permutevar8x32_epi32(t1, ord);

            v2 = _mm256_add_epi64(v2, _mm256_mul_epu32(x.v0, t0));
            v3 = _mm256_add_epi64(v3, _mm256_mul_epu32(x.v1, t0));
            v4 = _mm256_add_epi64(v4, _mm256_mul_epu32(x.v0, t1));

            t0 = _mm256_permute4x64_epi64(y, set02(3, 3, 3, 3));

            v0 = _mm256_add_epi64(v0, _mm256_mul_epu32(x.v0, t0));
            v1 = _mm256_add_epi64(v1, _mm256_mul_epu32(x.v1, t0));
            v2 = _mm256_add_epi64(v2, _mm256_mul_epu32(x.v2, t0));

            t0 = _mm256_permutevar8x32_epi32(t0, ord);

            v1 = _mm256_add_epi64(v1, _mm256_mul_epu32(x.v0, t0));
            v2 = _mm256_add_epi64(v2, _mm256_mul_epu32(x.v1, t0));
            v3 = _mm256_add_epi64(v3, _mm256_mul_epu32(x.v2, t0));

            x.v1 = _mm256_permutevar8x32_epi32(x.v1, ord);

            v0 = _mm256_add_epi64(v0, _mm256_mul_epu32(x.v1, t0));
            v0 = _mm256_add_epi64(v0, _mm256_mul_epu32(x.v2, z));

            Unreduced4x130 { v0, v1, v2, v3, v4 }
        }
    }
}

impl core::ops::Mul<SpacedMultiplier4x130> for Aligned4x130 {
    type Output = Unreduced4x130;

    /// Finalization: multiply the four coefficients by R¹..R⁴ respectively.
    #[inline(always)]
    fn mul(self, m: SpacedMultiplier4x130) -> Unreduced4x130 {
        unsafe {
            let mut x = self;
            let r1 = m.r1.a;

            let v0 = _mm256_unpacklo_epi32(m.v0, m.v1);
            let v1 = _mm256_unpackhi_epi32(m.v0, m.v1);

            let ord = _mm256_set_epi32(1, 0, 6, 7, 2, 0, 3, 1);
            let m_r_0 = _mm256_blend_epi32(
                _mm256_permutevar8x32_epi32(r1, ord),
                _mm256_permutevar8x32_epi32(v0, ord),
                0b0011_1111,
            );
            let ord = _mm256_set_epi32(3, 2, 4, 5, 2, 0, 3, 1);
            let m_r_2 = _mm256_blend_epi32(
                _mm256_permutevar8x32_epi32(r1, ord),
                _mm256_permutevar8x32_epi32(v1, ord),
                0b0011_1111,
            );
            let ord = _mm256_set_epi32(1, 4, 6, 6, 2, 4, 3, 5);
            let m_r_4 = _mm256_blend_epi32(
                _mm256_blend_epi32(
                    _mm256_permutevar8x32_epi32(r1, ord),
                    _mm256_permutevar8x32_epi32(v1, ord),
                    0b0001_0000,
                ),
                _mm256_permutevar8x32_epi32(v0, ord),
                0b0010_1111,
            );

            let mut v0 = _mm256_mul_epu32(x.v0, m_r_0);
            let mut v1 = _mm256_mul_epu32(x.v1, m_r_0);
            let mut v2 = _mm256_mul_epu32(x.v0, m_r_2);
            let mut v3 = _mm256_mul_epu32(x.v1, m_r_2);
            let mut v4 = _mm256_mul_epu32(x.v0, m_r_4);

            let ord = _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1);
            let m_r_1 = _mm256_permutevar8x32_epi32(m_r_0, ord);
            let m_r_3 = _mm256_permutevar8x32_epi32(m_r_2, ord);

            v1 = _mm256_add_epi64(v1, _mm256_mul_epu32(x.v0, m_r_1));
            v2 = _mm256_add_epi64(v2, _mm256_mul_epu32(x.v1, m_r_1));
            v3 = _mm256_add_epi64(v3, _mm256_mul_epu32(x.v0, m_r_3));
            v4 = _mm256_add_epi64(v4, _mm256_mul_epu32(x.v1, m_r_3));
            v4 = _mm256_add_epi64(v4, _mm256_mul_epu32(x.v2, m_r_0));

            x.v0 = _mm256_permutevar8x32_epi32(x.v0, ord);

            v2 = _mm256_add_epi64(v2, _mm256_mul_epu32(x.v0, m_r_0));
            v3 = _mm256_add_epi64(v3, _mm256_mul_epu32(x.v0, m_r_1));
            v4 = _mm256_add_epi64(v4, _mm256_mul_epu32(x.v0, m_r_2));

            let m_5r_3 = _mm256_add_epi32(m_r_3, _mm256_slli_epi32(m_r_3, 2));
            let m_5r_4 = _mm256_add_epi32(m_r_4, _mm256_slli_epi32(m_r_4, 2));

            v0 = _mm256_add_epi64(v0, _mm256_mul_epu32(x.v0, m_5r_3));
            v0 = _mm256_add_epi64(v0, _mm256_mul_epu32(x.v1, m_5r_4));
            v1 = _mm256_add_epi64(v1, _mm256_mul_epu32(x.v0, m_5r_4));
            v2 = _mm256_add_epi64(v2, _mm256_mul_epu32(x.v2, m_5r_3));
            v3 = _mm256_add_epi64(v3, _mm256_mul_epu32(x.v2, m_5r_4));

            x.v1 = _mm256_permutevar8x32_epi32(x.v1, ord);

            v1 = _mm256_add_epi64(v1, _mm256_mul_epu32(x.v1, m_5r_3));
            v2 = _mm256_add_epi64(v2, _mm256_mul_epu32(x.v1, m_5r_4));
            v3 = _mm256_add_epi64(v3, _mm256_mul_epu32(x.v1, m_r_0));
            v4 = _mm256_add_epi64(v4, _mm256_mul_epu32(x.v1, m_r_1));

            let m_5r_1 = _mm256_permutevar8x32_epi32(m_5r_4, ord);
            let m_5r_2 = _mm256_permutevar8x32_epi32(m_5r_3, ord);

            v0 = _mm256_add_epi64(v0, _mm256_mul_epu32(x.v1, m_5r_2));
            v0 = _mm256_add_epi64(v0, _mm256_mul_epu32(x.v2, m_5r_1));
            v1 = _mm256_add_epi64(v1, _mm256_mul_epu32(x.v2, m_5r_2));

            Unreduced4x130 { v0, v1, v2, v3, v4 }
        }
    }
}

/// Unreduced four-wide product in 64-bit limbs.
#[derive(Clone)]
struct Unreduced4x130 {
    v0: __m256i,
    v1: __m256i,
    v2: __m256i,
    v3: __m256i,
    v4: __m256i,
}

/// Fold limb overflows upward (4-wide variant of [`adc`]).
// NOTE: free fns, not closures — LLVM used to outline the closures into
// non-target-feature symbols, turning the intrinsics inside into out-of-line
// calls on the hot absorb4 path.
#[inline(always)]
unsafe fn adc4(x1: __m256i, x0: __m256i) -> (__m256i, __m256i) {
    let mask_26 = _mm256_set1_epi64x(0x03ff_ffff);
    let y1 = _mm256_add_epi64(x1, _mm256_srli_epi64(x0, 26));
    let y0 = _mm256_and_si256(x0, mask_26);
    (y1, y0)
}

/// Fold the ≥2^130 part back via ×5 (4-wide variant of [`red`]).
#[inline(always)]
unsafe fn red4(x4: __m256i, x0: __m256i) -> (__m256i, __m256i) {
    let mask_26 = _mm256_set1_epi64x(0x03ff_ffff);
    let y0 = _mm256_add_epi64(
        x0,
        _mm256_mul_epu32(_mm256_srli_epi64(x4, 26), _mm256_set1_epi64x(5)),
    );
    let y4 = _mm256_and_si256(x4, mask_26);
    (y4, y0)
}

impl Unreduced4x130 {
    #[inline(always)]
    fn reduce(self) -> Aligned4x130 {
        unsafe {
            let x = self;

            let (red_1, red_0) = adc4(x.v1, x.v0);
            let (red_4, red_3) = adc4(x.v4, x.v3);
            let (red_2, red_1) = adc4(x.v2, red_1);
            let (red_4, red_0) = red4(red_4, red_0);
            let (red_3, red_2) = adc4(red_3, red_2);
            let (red_1, red_0) = adc4(red_1, red_0);
            let (red_4, red_3) = adc4(red_4, red_3);

            Aligned4x130 {
                v0: _mm256_blend_epi32(red_0, _mm256_slli_epi64(red_2, 32), 0b1010_1010),
                v1: _mm256_blend_epi32(red_1, _mm256_slli_epi64(red_3, 32), 0b1010_1010),
                v2: red_4,
            }
        }
    }

    /// Sum the four coefficients (used at finalization).
    #[inline(always)]
    fn sum(self) -> Unreduced130 {
        unsafe {
            let x = self;
            let v0 = _mm256_add_epi64(
                _mm256_unpackhi_epi64(x.v0, x.v1),
                _mm256_unpacklo_epi64(x.v0, x.v1),
            );
            let v1 = _mm256_add_epi64(
                _mm256_unpackhi_epi64(x.v2, x.v3),
                _mm256_unpacklo_epi64(x.v2, x.v3),
            );
            let v0 = _mm256_add_epi64(
                _mm256_inserti128_si256(v0, _mm256_castsi256_si128(v1), 1),
                _mm256_inserti128_si256(v1, _mm256_extracti128_si256(v0, 1), 0),
            );
            let v1 = _mm256_add_epi64(x.v4, _mm256_permute4x64_epi64(x.v4, set02(1, 0, 3, 2)));
            let v1 = _mm256_add_epi64(v1, _mm256_permute4x64_epi64(v1, set02(0, 0, 0, 1)));
            Unreduced130 { v0, v1 }
        }
    }
}

#[derive(Clone, Copy)]
struct AdditionKey(__m256i);

impl core::ops::Add<Aligned130> for AdditionKey {
    type Output = IntegerTag;

    /// `(x + k) mod 2^128` with full carry handling.
    #[inline(always)]
    fn add(self, x: Aligned130) -> IntegerTag {
        unsafe {
            #[inline(always)]
            unsafe fn propagate_carry(x: __m256i) -> __m256i {
                let t = _mm256_permutevar8x32_epi32(
                    _mm256_srli_epi32(x, 26),
                    _mm256_set_epi32(7, 7, 7, 3, 2, 1, 0, 4),
                );
                _mm256_add_epi32(
                    _mm256_add_epi32(
                        _mm256_and_si256(
                            x,
                            _mm256_set_epi32(
                                0,
                                0,
                                0,
                                0x03ff_ffff,
                                0x03ff_ffff,
                                0x03ff_ffff,
                                0x03ff_ffff,
                                0x03ff_ffff,
                            ),
                        ),
                        t,
                    ),
                    _mm256_permutevar8x32_epi32(
                        _mm256_slli_epi32(t, 2),
                        _mm256_set_epi32(7, 7, 7, 7, 7, 7, 7, 0),
                    ),
                )
            }

            #[inline(always)]
            unsafe fn propagate_carry_32(x: __m256i) -> __m256i {
                _mm256_add_epi64(
                    _mm256_and_si256(x, _mm256_set_epi32(0, -1, 0, -1, 0, -1, 0, -1)),
                    _mm256_permute4x64_epi64(
                        _mm256_and_si256(
                            _mm256_srli_epi64(x, 32),
                            _mm256_set_epi64x(0, -1, -1, -1),
                        ),
                        set02(2, 1, 0, 3),
                    ),
                )
            }

            let mut x = _mm256_and_si256(x.0, _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1));
            let k = self.0;

            // reduce to an integer below 2^130
            for _ in 0..5 {
                x = propagate_carry(x);
            }

            // compute x + (2^130 - 5)·(-1)... i.e. g = x + 5 - 2^130
            let mut g = _mm256_add_epi32(x, _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 5));
            for _ in 0..4 {
                g = propagate_carry(g);
            }
            let g = _mm256_sub_epi32(g, _mm256_set_epi32(0, 0, 0, 1 << 26, 0, 0, 0, 0));

            // select x if g overflowed, else g
            let mask = _mm256_permutevar8x32_epi32(
                _mm256_sub_epi32(_mm256_srli_epi32(g, 32 - 1), _mm256_set1_epi32(1)),
                _mm256_set1_epi32(4),
            );
            let x = _mm256_or_si256(
                _mm256_and_si256(x, _mm256_xor_si256(mask, _mm256_set1_epi32(-1))),
                _mm256_and_si256(g, mask),
            );

            // realign limbs back to 32-bit word boundaries
            let x = _mm256_or_si256(
                _mm256_srlv_epi32(x, _mm256_set_epi32(32, 32, 32, 32, 18, 12, 6, 0)),
                _mm256_permutevar8x32_epi32(
                    _mm256_sllv_epi32(x, _mm256_set_epi32(32, 32, 32, 8, 14, 20, 26, 32)),
                    _mm256_set_epi32(7, 7, 7, 7, 4, 3, 2, 1),
                ),
            );

            // add key in 64-bit lanes and propagate
            let mut x = _mm256_add_epi64(
                _mm256_permutevar8x32_epi32(x, _mm256_set_epi32(7, 3, 7, 2, 7, 1, 7, 0)),
                k,
            );

            for _ in 0..3 {
                x = propagate_carry_32(x);
            }

            let x = _mm256_permutevar8x32_epi32(x, _mm256_set_epi32(7, 7, 7, 7, 6, 4, 2, 0));
            IntegerTag(_mm256_castsi256_si128(x))
        }
    }
}

impl From<AdditionKey> for IntegerTag {
    #[inline(always)]
    fn from(k: AdditionKey) -> Self {
        unsafe {
            IntegerTag(_mm256_castsi256_si128(_mm256_permutevar8x32_epi32(
                k.0,
                _mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0),
            )))
        }
    }
}

struct IntegerTag(__m128i);

impl IntegerTag {
    #[inline(always)]
    fn write(self, tag: &mut [u8; 16]) {
        unsafe { _mm_storeu_si128(tag.as_mut_ptr().cast(), self.0) }
    }
}

#[cfg(test)]
mod tests {
    use super::Avx2Poly;

    // These tests call AVX2 kernels directly (bypassing runtime dispatch);
    // skip on CPUs without AVX2 instead of executing illegal instructions.
    macro_rules! skip_without_avx2 {
        () => {
            if !std::arch::is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

    #[test]
    fn t_matches_python_reference() {
        skip_without_avx2!();
        crate::poly1305::test_common::matches_python_reference::<Avx2Poly>();
    }
    #[test]
    fn t_rfc8439_mac_stream() {
        skip_without_avx2!();
        crate::poly1305::test_common::rfc8439_mac_stream::<Avx2Poly>();
    }
    #[test]
    fn t_segmentation_equivalence() {
        skip_without_avx2!();
        crate::poly1305::test_common::segmentation_equivalence::<Avx2Poly>();
    }
}

#[cfg(test)]
mod extra_tests {
    use super::Avx2Poly;

    #[test]
    fn t_cross_backend() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        crate::poly1305::test_common::cross_backend_consistency::<Avx2Poly>();
    }
}
