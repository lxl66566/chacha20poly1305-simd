//! AVX-512 IFMA Poly1305 — OpenSSL `vpmadd52` 8-block algorithm.
//!
//! Copyright 2016-2025 The OpenSSL Project Authors. All Rights Reserved.
//!
//! Licensed under the Apache License 2.0 (the "License"); you may not use
//! this file except in compliance with the License. You can obtain a copy
//! at <https://www.apache.org/licenses/LICENSE-2.0>
//!
//! Source: OpenSSL `crypto/poly1305/asm/poly1305-x86_64.pl` (VPMADD52
//! radix-2^44 path). This Rust port modifies the original: the streaming
//! [`Backend`] contract caches blocks until eight are pending so engine
//! `absorb4` pairs fuse into one 8-lane round; the `r^2..r^8` powers are
//! only computed when the first full group arrives; finalization collapses
//! the eight lanes with per-lane powers `r^(8-pos)` instead of OpenSSL's
//! 4x/2x/1x tail tiers; leftover blocks fold scalar-side with `r^1`.
//!
//! Radix 2^44 (limbs 44/44/42 bits in 64-bit lanes) with the
//! `vpmadd52{lu,hu}q` fused multiply-accumulators: one zmm holds eight
//! independent block accumulators, so 128 bytes fold per ~18-multiplies
//! round (the AVX2 Goll–Gueron path needs ~18 `vpmuludq` per 64 bytes).
//! Every lane advances by `r^8`; finalization collapses the lanes with
//! per-lane powers `r^(8-pos)` and folds the ≤ 8 leftover blocks one at a
//! time with `r^1` scalar-side (the role of OpenSSL's 4x/2x/1x tail tiers,
//! at negligible cost since it runs once per message).

use core::arch::x86_64::*;

use super::Backend;

const M44: u64 = (1 << 44) - 1;
const M42: u64 = (1 << 42) - 1;
/// AEAD blocks always carry the 2^128 high bit: bit 40 of limb 2.
const HIBIT: u64 = 1 << 40;

/// Block position within an 8-block group held by zmm lane ℓ after the
/// `vpunpck{l,h}qdq` transpose of the two 64-byte halves.
const LANE_POS: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];

/// Three 64-bit limbs at 44/44/42-bit boundaries.
type Limb3 = [u64; 3];

/// Clamped `r` in radix 2^44.
#[cfg_attr(not(debug_assertions), inline(always))]
fn clamp_r(key: &[u8; 16]) -> Limb3 {
    let lo = u64::from_le_bytes(key[..8].try_into().unwrap()) & 0x0fff_fffc_0fff_ffff;
    let hi = u64::from_le_bytes(key[8..].try_into().unwrap()) & 0x0fff_fffc_0fff_fffc;
    [lo & M44, ((lo >> 44) | (hi << 20)) & M44, hi >> 24]
}

/// Split a 16-byte block into radix-2^44 limbs, high bit set.
#[cfg_attr(not(debug_assertions), inline(always))]
fn block44(block: &[u8; 16]) -> Limb3 {
    let lo = u64::from_le_bytes(block[..8].try_into().unwrap());
    let hi = u64::from_le_bytes(block[8..].try_into().unwrap());
    [
        lo & M44,
        ((lo >> 44) | (hi << 20)) & M44,
        (hi >> 24) | HIBIT,
    ]
}

/// `a·r` mod (2^130−5) with lazy reduction; `s1`/`s2` are `20·r1`/`20·r2`
/// (2^132 ≡ 20). Scalar twin of the zmm [`mul_reduce`] round.
#[cfg_attr(not(debug_assertions), inline(always))]
fn mul44s(a: Limb3, r: Limb3, s1: u64, s2: u64) -> Limb3 {
    let (a0, a1, a2) = (u128::from(a[0]), u128::from(a[1]), u128::from(a[2]));
    let (r0, r1, r2) = (u128::from(r[0]), u128::from(r[1]), u128::from(r[2]));
    let d0 = a0 * r0 + a1 * u128::from(s2) + a2 * u128::from(s1);
    let d1 = a0 * r1 + a1 * r0 + a2 * u128::from(s2);
    let d2 = a0 * r2 + a1 * r1 + a2 * r0;
    let d1 = d1 + (d0 >> 44);
    let d0 = d0 & u128::from(M44);
    let d2 = d2 + (d1 >> 44);
    let d1 = d1 & u128::from(M44);
    // d2 covers bits 88..130; its overflow (≥ 2^130) wraps ×5.
    let d0 = d0 + 5 * (d2 >> 42);
    let d2 = d2 & u128::from(M42);
    let d1 = d1 + (d0 >> 44);
    let d0 = d0 & u128::from(M44);
    [d0 as u64, d1 as u64, d2 as u64]
}

#[cfg_attr(not(debug_assertions), inline(always))]
fn mul44(a: Limb3, r: Limb3) -> Limb3 {
    mul44s(a, r, 20 * r[1], 20 * r[2])
}

/// Full carry chain, conditional subtraction of p, `+ pad` mod 2^128, emit.
#[cfg_attr(not(debug_assertions), inline(always))]
fn finalize44(h: Limb3, pad: [u64; 2], out: &mut [u8; 16]) {
    let (mut h0, mut h1, mut h2) = (h[0], h[1], h[2]);
    h1 += h0 >> 44;
    h0 &= M44;
    h2 += h1 >> 44;
    h1 &= M44;
    h0 += 5 * (h2 >> 42);
    h2 &= M42;
    h1 += h0 >> 44;
    h0 &= M44;

    // g = h + 5 - 2^130; select g unless it borrowed (h < p).
    let g0 = h0 + 5;
    let c = g0 >> 44;
    let g0 = g0 & M44;
    let g1 = h1 + c;
    let c = g1 >> 44;
    let g1 = g1 & M44;
    let g2 = (h2 + c).wrapping_sub(1 << 42);
    // all-ones when h ≥ p (no borrow out of g2)
    let use_g = (g2 >> 63).wrapping_sub(1);
    h0 = (g0 & use_g) | (h0 & !use_g);
    h1 = (g1 & use_g) | (h1 & !use_g);
    h2 = ((g2 & M42) & use_g) | (h2 & !use_g);

    // value mod 2^128 in two 64-bit halves, then += pad
    let lo = h0 | (h1 << 44);
    let hi = (h1 >> 20) | (h2 << 24);
    let (lo, c) = lo.overflowing_add(pad[0]);
    let hi = hi.wrapping_add(pad[1]).wrapping_add(u64::from(c));
    out[..8].copy_from_slice(&lo.to_le_bytes());
    out[8..].copy_from_slice(&hi.to_le_bytes());
}

/// Transpose 128 bytes (blocks 0..=3 at `lo`, 4..=7 at `hi`) into
/// radix-2^44 limb vectors; lane ℓ holds block [`LANE_POS`]`[ℓ]`.
#[cfg_attr(not(debug_assertions), inline(always))]
pub(crate) unsafe fn load8(lo: *const u8, hi: *const u8) -> (__m512i, __m512i, __m512i) {
    unsafe {
        let z0 = _mm512_loadu_si512(lo.cast());
        let z1 = _mm512_loadu_si512(hi.cast());
        let lo_q = _mm512_unpacklo_epi64(z0, z1);
        let hi_q = _mm512_unpackhi_epi64(z0, z1);
        let m44 = _mm512_set1_epi64(M44 as i64);
        let t0 = _mm512_and_si512(lo_q, m44);
        let t1 = _mm512_and_si512(
            _mm512_or_si512(_mm512_srli_epi64::<44>(lo_q), _mm512_slli_epi64::<20>(hi_q)),
            m44,
        );
        let t2 = _mm512_or_si512(
            _mm512_srli_epi64::<24>(hi_q),
            _mm512_set1_epi64(HIBIT as i64),
        );
        (t0, t1, t2)
    }
}

/// `(h0,h1,h2) · r` lane-wise: 9 lo + 9 hi fused multiply-accumulates, then
/// the partial reduction (carry at 44/44/42, ≥2^130 wraps ×5). Per lane the
/// product is `Dlo + (Dhi << 52)`, so the carry out of limb j is
/// `(Dlo >> w) + (Dhi << (52 - w))`.
#[cfg_attr(not(debug_assertions), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn mul_reduce(
    h0: __m512i,
    h1: __m512i,
    h2: __m512i,
    r0: __m512i,
    r1: __m512i,
    r2: __m512i,
    s1: __m512i,
    s2: __m512i,
) -> (__m512i, __m512i, __m512i) {
    unsafe {
        let z = _mm512_setzero_si512();
        let d0lo = _mm512_madd52lo_epu64(z, h0, r0);
        let d0lo = _mm512_madd52lo_epu64(d0lo, h1, s2);
        let d0lo = _mm512_madd52lo_epu64(d0lo, h2, s1);
        let d1lo = _mm512_madd52lo_epu64(z, h0, r1);
        let d1lo = _mm512_madd52lo_epu64(d1lo, h1, r0);
        let d1lo = _mm512_madd52lo_epu64(d1lo, h2, s2);
        let d2lo = _mm512_madd52lo_epu64(z, h0, r2);
        let d2lo = _mm512_madd52lo_epu64(d2lo, h1, r1);
        let d2lo = _mm512_madd52lo_epu64(d2lo, h2, r0);
        let d0hi = _mm512_madd52hi_epu64(z, h0, r0);
        let d0hi = _mm512_madd52hi_epu64(d0hi, h1, s2);
        let d0hi = _mm512_madd52hi_epu64(d0hi, h2, s1);
        let d1hi = _mm512_madd52hi_epu64(z, h0, r1);
        let d1hi = _mm512_madd52hi_epu64(d1hi, h1, r0);
        let d1hi = _mm512_madd52hi_epu64(d1hi, h2, s2);
        let d2hi = _mm512_madd52hi_epu64(z, h0, r2);
        let d2hi = _mm512_madd52hi_epu64(d2hi, h1, r1);
        let d2hi = _mm512_madd52hi_epu64(d2hi, h2, r0);

        let m44 = _mm512_set1_epi64(M44 as i64);
        let d0hi = _mm512_add_epi64(_mm512_slli_epi64::<8>(d0hi), _mm512_srli_epi64::<44>(d0lo));
        let h0 = _mm512_and_si512(d0lo, m44);
        let d1lo = _mm512_add_epi64(d1lo, d0hi);
        let d1hi = _mm512_add_epi64(_mm512_slli_epi64::<8>(d1hi), _mm512_srli_epi64::<44>(d1lo));
        let h1 = _mm512_and_si512(d1lo, m44);
        let d2lo = _mm512_add_epi64(d2lo, d1hi);
        let d2hi = _mm512_add_epi64(_mm512_slli_epi64::<10>(d2hi), _mm512_srli_epi64::<42>(d2lo));
        let h2 = _mm512_and_si512(d2lo, _mm512_set1_epi64(M42 as i64));
        let h0 = _mm512_add_epi64(h0, d2hi);
        let h0 = _mm512_add_epi64(h0, _mm512_slli_epi64::<2>(d2hi));
        let h1 = _mm512_add_epi64(h1, _mm512_srli_epi64::<44>(h0));
        let h0 = _mm512_and_si512(h0, m44);
        (h0, h1, h2)
    }
}

pub(crate) struct IfmaPoly {
    pad: [u64; 2],
    r: Limb3,
    /// Eight block-lane accumulators (valid iff `has_h`). A zero H makes
    /// the first group's round a pure load (`mul_reduce(0) + T = T`), so
    /// the fused loops need no first-group special case.
    pub(crate) h0: __m512i,
    pub(crate) h1: __m512i,
    pub(crate) h2: __m512i,
    has_h: bool,
    /// Power table, COLUMN-major: `lane_cols[j][i]` = limb j of lane i's
    /// `r^(8-LANE_POS[i])` (valid iff `has_lanes`), built once when the first
    /// 8-block group arrives — short MACs that never fill a group pay no
    /// power computation at all. Columns map straight onto the finalize
    /// mul_reduce broadcasts (no transpose at the use site).
    lane_cols: [[u64; 8]; 3],
    has_lanes: bool,
    // Up to 8 whole blocks are deferred so `absorb4` pairs merge into one
    // 8-lane round.
    cached: [u8; 128],
    num_cached: usize,
}

impl IfmaPoly {
    /// Build the scalar power table in place. Dependency tree (depth 3, was
    /// a 7-deep chain): r8 depends on r2/r4, the odd powers hang off the
    /// tree in parallel — ~3 chained scalar multiplies of latency.
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn build_lanes(&mut self) {
        debug_assert_eq!(core::array::from_fn::<_, 8, _>(|i| 8 - LANE_POS[i]), [
            8, 4, 7, 3, 6, 2, 5, 1
        ]);
        let r = self.r;
        let r2 = mul44(r, r);
        let r4 = mul44(r2, r2);
        let r8 = mul44(r4, r4);
        let r3 = mul44(r2, r);
        let r5 = mul44(r4, r);
        let r6 = mul44(r4, r2);
        let r7 = mul44(r4, r3);
        macro_rules! put {
            ($i:expr, $v:expr) => {{
                let v = $v;
                self.lane_cols[0][$i] = v[0];
                self.lane_cols[1][$i] = v[1];
                self.lane_cols[2][$i] = v[2];
            }};
        }
        put!(0, r8);
        put!(1, r4);
        put!(2, r7);
        put!(3, r3);
        put!(4, r6);
        put!(5, r2);
        put!(6, r5);
        put!(7, r);
        self.has_lanes = true;
    }

    /// Fused-kernel support: ensure the streaming state exists (zero H +
    /// power table). Idempotent.
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(crate) unsafe fn ensure_stream(&mut self) {
        if !self.has_lanes {
            self.build_lanes();
        }
        if !self.has_h {
            self.h0 = _mm512_setzero_si512();
            self.h1 = _mm512_setzero_si512();
            self.h2 = _mm512_setzero_si512();
            self.has_h = true;
        }
    }

    /// r⁸ limbs for the round broadcasts, derived at the fused-kernel call
    /// sites (5 `vpbroadcastq`s; requires `ensure_stream` first).
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(crate) fn r8_limbs(&self) -> (u64, u64, u64) {
        (
            self.lane_cols[0][0],
            self.lane_cols[1][0],
            self.lane_cols[2][0],
        )
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    unsafe fn process_cached(&mut self) {
        debug_assert_eq!(self.num_cached, 8);
        let p = self.cached.as_ptr();
        unsafe { process8(self, p, p.add(64)) };
        self.num_cached = 0;
    }

    /// Bulk path: the whole run is processed with H and the r^8 round
    /// multipliers held in registers across rounds (per-64B `absorb4` would
    /// round-trip the zmm state through the stack every window).
    // `#[inline(never)]` + explicit `target_feature`: inlined into the fused
    // engine, this loop's body got jump-threaded into ~30 clones and the
    // poly state spilled to the stack on every round. As a standalone
    // function the loop stays a single tight copy; one call per 1024-byte
    // engine batch is noise. The feature list must include `avx512ifma`:
    // with only `avx512f` LLVM refuses to inline the madd52 intrinsic
    // shims and emits an out-of-line call per multiply.
    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline(never)]
    unsafe fn absorb_blocks_bulk(&mut self, mut blocks: &[u8]) {
        debug_assert_eq!(blocks.len() % 64, 0);
        debug_assert_eq!(self.num_cached % 4, 0);
        unsafe {
            if self.num_cached == 8 {
                self.process_cached();
            }
            // Pair a pending half-cache with the first window.
            if self.num_cached == 4 && !blocks.is_empty() {
                process8(self, self.cached.as_ptr(), blocks.as_ptr());
                blocks = &blocks[64..];
                self.num_cached = 0;
            }
            if blocks.len() >= 128 {
                let (mut h0, mut h1, mut h2);
                if self.has_h {
                    h0 = self.h0;
                    h1 = self.h1;
                    h2 = self.h2;
                } else {
                    // First group only loads (no r^8 multiply yet).
                    self.ensure_stream();
                    (h0, h1, h2) = load8(blocks.as_ptr(), blocks.as_ptr().add(64));
                    blocks = &blocks[128..];
                }
                if blocks.len() >= 128 {
                    // Hoist the round multipliers: 5 broadcasts.
                    let (ra, rb, rc) = self.r8_limbs();
                    let p = (
                        _mm512_set1_epi64(ra as i64),
                        _mm512_set1_epi64(rb as i64),
                        _mm512_set1_epi64(rc as i64),
                        _mm512_set1_epi64((20 * rb) as i64),
                        _mm512_set1_epi64((20 * rc) as i64),
                    );
                    let mut ptr = blocks.as_ptr();
                    let mut left = blocks.len();
                    while left >= 128 {
                        let (t0, t1, t2) = load8(ptr, ptr.add(64));
                        (h0, h1, h2) = mul_reduce(h0, h1, h2, p.0, p.1, p.2, p.3, p.4);
                        h0 = _mm512_add_epi64(h0, t0);
                        h1 = _mm512_add_epi64(h1, t1);
                        h2 = _mm512_add_epi64(h2, t2);
                        ptr = ptr.add(128);
                        left -= 128;
                    }
                    blocks = blocks.split_at(blocks.len() - left).1;
                }
                // Single writeback covers both paths: the freshly loaded
                // first group must survive even when no full round follows.
                self.h0 = h0;
                self.h1 = h1;
                self.h2 = h2;
            }
            // Cache the trailing half-window (0 or 64 bytes).
            if !blocks.is_empty() {
                debug_assert_eq!(blocks.len(), 64);
                self.cached[..64].copy_from_slice(blocks);
                self.num_cached = 4;
            }
        }
    }
}

/// Fold one full 8-block group (pointed to by `lo`/`hi`, 64 bytes each)
/// into the stream (`H ·= r^8; H += T` — the previous H is multiplied
/// BEFORE the new group joins, so the freshest group never picks up an
/// extra r^8; OpenSSL orders its loop the same way). Builds the key powers
/// on first use. Free fn + raw pointers: taking `&mut self` here forced a
/// 128-byte by-value copy of the block cache on every round.
#[cfg_attr(not(debug_assertions), inline(always))]
unsafe fn process8(p: &mut IfmaPoly, lo: *const u8, hi: *const u8) {
    unsafe {
        let (t0, t1, t2) = load8(lo, hi);
        if p.has_h {
            let (ra, rb, rc) = p.r8_limbs();
            let (h0, h1, h2) = mul_reduce(
                p.h0,
                p.h1,
                p.h2,
                _mm512_set1_epi64(ra as i64),
                _mm512_set1_epi64(rb as i64),
                _mm512_set1_epi64(rc as i64),
                _mm512_set1_epi64((20 * rb) as i64),
                _mm512_set1_epi64((20 * rc) as i64),
            );
            p.h0 = _mm512_add_epi64(h0, t0);
            p.h1 = _mm512_add_epi64(h1, t1);
            p.h2 = _mm512_add_epi64(h2, t2);
        } else {
            p.ensure_stream();
            p.h0 = t0;
            p.h1 = t1;
            p.h2 = t2;
        }
    }
}

impl Backend for IfmaPoly {
    #[cfg_attr(not(debug_assertions), inline(always))]
    unsafe fn init_at(out: *mut Self, key: &[u8; 32]) {
        unsafe {
            core::ptr::addr_of_mut!((*out).pad).write([
                u64::from_le_bytes(key[16..24].try_into().unwrap()),
                u64::from_le_bytes(key[24..32].try_into().unwrap()),
            ]);
            core::ptr::addr_of_mut!((*out).r).write(clamp_r(key[..16].try_into().unwrap()));
            core::ptr::addr_of_mut!((*out).h0).write(_mm512_setzero_si512());
            core::ptr::addr_of_mut!((*out).h1).write(_mm512_setzero_si512());
            core::ptr::addr_of_mut!((*out).h2).write(_mm512_setzero_si512());
            core::ptr::addr_of_mut!((*out).has_h).write(false);
            core::ptr::addr_of_mut!((*out).lane_cols).write([[0; 8]; 3]);
            core::ptr::addr_of_mut!((*out).has_lanes).write(false);
            core::ptr::addr_of_mut!((*out).cached).write([0; 128]);
            core::ptr::addr_of_mut!((*out).num_cached).write(0);
        }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    unsafe fn absorb_block(&mut self, block: &[u8; 16]) {
        if self.num_cached == 8 {
            unsafe { self.process_cached() };
        }
        // SAFETY: the drain above keeps `num_cached < 8`, so the 16-byte
        // window at `num_cached * 16` is inside `cached`. The checked slice
        // index kept a bounds/panic branch on this per-block hot path.
        unsafe {
            self.cached
                .as_mut_ptr()
                .add(self.num_cached * 16)
                .copy_from_nonoverlapping(block.as_ptr(), 16);
        }
        self.num_cached += 1;
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    unsafe fn absorb4(&mut self, blocks: &[u8; 64]) {
        debug_assert_eq!(self.num_cached % 4, 0, "stream must be 64B-aligned");
        if self.num_cached == 4 {
            // Fuse the cached half with the new one: one 8-lane round.
            unsafe { process8(self, self.cached.as_ptr(), blocks.as_ptr()) };
            self.num_cached = 0;
        } else {
            if self.num_cached == 8 {
                unsafe { self.process_cached() };
            }
            self.cached[..64].copy_from_slice(blocks);
            self.num_cached = 4;
        }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    unsafe fn absorb_blocks(&mut self, blocks: &[u8]) {
        // SAFETY: the avx512-ifma backend entry points guarantee
        // AVX-512F+IFMA.
        unsafe { self.absorb_blocks_bulk(blocks) }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn pending_blocks(&self) -> usize {
        self.num_cached
    }

    #[cfg(feature = "zeroize")]
    unsafe fn zeroize_secrets(&mut self) {
        unsafe {
            core::ptr::write_volatile(&raw mut self.pad, [0; 2]);
            core::ptr::write_volatile(&raw mut self.r, [0; 3]);
            core::ptr::write_volatile(&raw mut self.h0, _mm512_setzero_si512());
            core::ptr::write_volatile(&raw mut self.h1, _mm512_setzero_si512());
            core::ptr::write_volatile(&raw mut self.h2, _mm512_setzero_si512());
            core::ptr::write_volatile(&raw mut self.lane_cols, [[0; 8]; 3]);
        }
        self.has_h = false;
        self.has_lanes = false;
        zeroize::Zeroize::zeroize(&mut self.cached);
        self.num_cached = 0;
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    unsafe fn finalize_into(&mut self, out: &mut [u8; 16]) {
        debug_assert!(self.num_cached <= 8);
        let mut acc: Limb3 = [0; 3];
        if self.has_h {
            self.has_h = false;
            // Collapse the 8 lanes: H · per-lane r^(8-pos), then sum lanes.
            // The column-major table loads straight into the mul_reduce
            // broadcasts; 20·x = (x<<4)+(x<<2) (no scalar column build).
            let (c0, c1, c2) = unsafe {
                (
                    _mm512_loadu_si512(self.lane_cols[0].as_ptr().cast()),
                    _mm512_loadu_si512(self.lane_cols[1].as_ptr().cast()),
                    _mm512_loadu_si512(self.lane_cols[2].as_ptr().cast()),
                )
            };
            let (d1, d2) = unsafe {
                (
                    _mm512_add_epi64(_mm512_slli_epi64::<4>(c1), _mm512_slli_epi64::<2>(c1)),
                    _mm512_add_epi64(_mm512_slli_epi64::<4>(c2), _mm512_slli_epi64::<2>(c2)),
                )
            };
            let (h0, h1, h2) = unsafe { mul_reduce(self.h0, self.h1, self.h2, c0, c1, c2, d1, d2) };
            // Bit-identical to an unsigned horizontal sum (two's complement).
            unsafe {
                acc[0] = _mm512_reduce_add_epi64(h0) as u64;
                acc[1] = _mm512_reduce_add_epi64(h1) as u64;
                acc[2] = _mm512_reduce_add_epi64(h2) as u64;
            }
            acc[1] += acc[0] >> 44;
            acc[0] &= M44;
            acc[2] += acc[1] >> 44;
            acc[1] &= M44;
            acc[0] += 5 * (acc[2] >> 42);
            acc[2] &= M42;
        }
        // Leftover blocks (never a full group): scalar r^1 folds. Correct
        // with or without a stream: acc·r^n + Σ T_j·r^(n-j).
        let (s1, s2) = (20 * self.r[1], 20 * self.r[2]);
        for i in 0..self.num_cached {
            let b = block44(self.cached[i * 16..][..16].try_into().unwrap());
            acc = mul44s(
                [acc[0] + b[0], acc[1] + b[1], acc[2] + b[2]],
                self.r,
                s1,
                s2,
            );
        }
        self.num_cached = 0;
        finalize44(acc, self.pad, out);
    }
}

#[cfg(test)]
mod tests {
    use super::IfmaPoly;

    // These tests call AVX-512 IFMA kernels directly (bypassing runtime
    // dispatch); skip on CPUs without the features.
    macro_rules! skip_without_ifma {
        () => {
            if !(std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512vl")
                && std::arch::is_x86_feature_detected!("avx512ifma"))
            {
                return;
            }
        };
    }

    #[test]
    fn t_matches_python_reference() {
        skip_without_ifma!();
        crate::poly1305::test_common::matches_python_reference::<IfmaPoly>();
    }
    #[test]
    fn t_rfc8439_mac_stream() {
        skip_without_ifma!();
        crate::poly1305::test_common::rfc8439_mac_stream::<IfmaPoly>();
    }
    #[test]
    fn t_segmentation_equivalence() {
        skip_without_ifma!();
        crate::poly1305::test_common::segmentation_equivalence::<IfmaPoly>();
    }
    #[test]
    fn t_cross_backend() {
        skip_without_ifma!();
        crate::poly1305::test_common::cross_backend_consistency::<IfmaPoly>();
    }
}
