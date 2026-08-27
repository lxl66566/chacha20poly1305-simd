//! NEON Poly1305 — four parallel 26-bit-limb accumulators (one per block
//! position mod 4), advanced as `P_i ← P_i·R⁴ + b_i` per 4-block step.
//!
//! The multiply-first order leaves lane `i` (blocks ≡ i mod 4, lane 0 the
//! oldest) short of `R^(4-i)` at finalize, so lanes and newer straggler
//! blocks fold with a short scalar Horner chain plus one final multiply
//! (every block carries exactly one r, the newest included). The carry
//! propagation is the standard donna-32 lazy chain in u64 lanes
//! (`vmull_u32` products).
//!
//! Same streaming contract as the AVX2 backend: whole blocks are deferred in
//! a byte cache until a 4-block group (or the final fold) consumes them, so
//! short messages never build the R⁴ powers.

// Block loads use unaligned `vld1q_u32`, so the stricter pointer alignment
// is intentional.
#![allow(clippy::cast_ptr_alignment)]

use core::arch::aarch64::*;

use super::{
    Backend,
    soft::{finalize_limbs, mul_r, parse_key},
};

pub(crate) struct NeonPoly {
    k: [u32; 4],
    r: [u32; 5],
    s: [u32; 4],
    /// Lazily built on the first 4-block step: the R⁴ rows plus the lane
    /// accumulators. `None` keeps short messages (≤ 8 deferred blocks) free
    /// of the power computation.
    lanes: Option<Lanes>,
    // `num_cached` is a multiple of 4 whenever the engine calls `absorb4`
    // (see aead.rs); it can hold any value 0..=8 at finalize.
    cached: [u8; 128],
    num_cached: usize,
}

/// Live vector state: limbs live in u32x4 lanes (lane = block mod 4).
struct Lanes {
    h: [uint32x4_t; 5],
    /// `[r⁴_0..r⁴_4, 5r⁴_1..5r⁴_4]` broadcast across the block lanes.
    coefs: [uint32x4_t; 9],
}

/// A value spread across both 64-bit halves of two u64x2 registers — i.e.
/// five-limb data for four blocks with lanes 0,1 in `lo` and 2,3 in `hi`.
#[derive(Clone, Copy)]
struct X2 {
    lo: uint64x2_t,
    hi: uint64x2_t,
}

impl X2 {
    #[inline(always)]
    unsafe fn mul(a: uint32x4_t, c: uint32x4_t) -> Self {
        Self {
            lo: vmull_u32(vget_low_u32(a), vget_low_u32(c)),
            hi: vmull_high_u32(a, c),
        }
    }

    #[inline(always)]
    unsafe fn acc(&mut self, a: uint32x4_t, c: uint32x4_t) {
        self.lo = vaddq_u64(self.lo, vmull_u32(vget_low_u32(a), vget_low_u32(c)));
        self.hi = vaddq_u64(self.hi, vmull_high_u32(a, c));
    }

    #[inline(always)]
    unsafe fn add(self, o: Self) -> Self {
        Self {
            lo: vaddq_u64(self.lo, o.lo),
            hi: vaddq_u64(self.hi, o.hi),
        }
    }

    #[inline(always)]
    unsafe fn shr26(self) -> Self {
        Self {
            lo: vshrq_n_u64::<26>(self.lo),
            hi: vshrq_n_u64::<26>(self.hi),
        }
    }

    #[inline(always)]
    unsafe fn mask26(self) -> Self {
        let m = vdupq_n_u64(0x03ff_ffff);
        Self {
            lo: vandq_u64(self.lo, m),
            hi: vandq_u64(self.hi, m),
        }
    }

    /// ×5 as `(x << 2) + x` (no 64-bit lane multiply in NEON).
    #[inline(always)]
    unsafe fn mul5(self) -> Self {
        let s4 = Self {
            lo: vshlq_n_u64::<2>(self.lo),
            hi: vshlq_n_u64::<2>(self.hi),
        };
        self.add(s4)
    }

    #[inline(always)]
    unsafe fn narrow(self) -> uint32x4_t {
        vcombine_u32(vmovn_u64(self.lo), vmovn_u64(self.hi))
    }
}

/// One 4-block step: `h ← h·r⁴ + blocks` (multiply first — the newest blocks
/// must not carry this step's r⁴).
// Free fn with `&mut Lanes` (not a closure): keep the intrinsics inlinable
// on the hot absorb4 path, mirroring the AVX2 backend's note.
#[inline(always)]
unsafe fn step(l: &mut Lanes, blocks: &[uint32x4_t; 5]) {
    let h = l.h;
    let c = &l.coefs;
    // d_j = Σ_k h_k · (r⁴ or 5r⁴), mirroring the donna matrix in soft::mul_r.
    let mut d0 = X2::mul(h[0], c[0]);
    d0.acc(h[1], c[8]);
    d0.acc(h[2], c[7]);
    d0.acc(h[3], c[6]);
    d0.acc(h[4], c[5]);
    let mut d1 = X2::mul(h[0], c[1]);
    d1.acc(h[1], c[0]);
    d1.acc(h[2], c[8]);
    d1.acc(h[3], c[7]);
    d1.acc(h[4], c[6]);
    let mut d2 = X2::mul(h[0], c[2]);
    d2.acc(h[1], c[1]);
    d2.acc(h[2], c[0]);
    d2.acc(h[3], c[8]);
    d2.acc(h[4], c[7]);
    let mut d3 = X2::mul(h[0], c[3]);
    d3.acc(h[1], c[2]);
    d3.acc(h[2], c[1]);
    d3.acc(h[3], c[0]);
    d3.acc(h[4], c[8]);
    let mut d4 = X2::mul(h[0], c[4]);
    d4.acc(h[1], c[3]);
    d4.acc(h[2], c[2]);
    d4.acc(h[3], c[1]);
    d4.acc(h[4], c[0]);

    // lazy partial reduction (per u64 lane = per block)
    let c = d0.shr26();
    let h0 = d0.mask26();
    let d1 = d1.add(c);
    let c = d1.shr26();
    let h1 = d1.mask26();
    let d2 = d2.add(c);
    let c = d2.shr26();
    let h2 = d2.mask26();
    let d3 = d3.add(c);
    let c = d3.shr26();
    let h3 = d3.mask26();
    let d4 = d4.add(c);
    let c = d4.shr26();
    let h4 = d4.mask26();
    let h0 = h0.add(c.mul5());
    let c = h0.shr26();
    let h0 = h0.mask26();
    let h1 = h1.add(c);

    l.h = [
        vaddq_u32(h0.narrow(), blocks[0]),
        vaddq_u32(h1.narrow(), blocks[1]),
        vaddq_u32(h2.narrow(), blocks[2]),
        vaddq_u32(h3.narrow(), blocks[3]),
        vaddq_u32(h4.narrow(), blocks[4]),
    ];
}

/// Split 4 blocks (64 bytes) into 26-bit limbs, lane = block, high bit set.
#[inline(always)]
unsafe fn load4(blocks: &[u8; 64]) -> [uint32x4_t; 5] {
    let p = blocks.as_ptr().cast::<u32>();
    let b0 = vreinterpretq_u64_u32(vld1q_u32(p));
    let b1 = vreinterpretq_u64_u32(vld1q_u32(p.add(4)));
    let b2 = vreinterpretq_u64_u32(vld1q_u32(p.add(8)));
    let b3 = vreinterpretq_u64_u32(vld1q_u32(p.add(12)));
    // X = (w0 | w1 << 32), Y = (w2 | w3 << 32) per block, lanes = blocks.
    let x01 = vzip1q_u64(b0, b1);
    let x23 = vzip1q_u64(b2, b3);
    let y01 = vzip2q_u64(b0, b1);
    let y23 = vzip2q_u64(b2, b3);

    let m = vdupq_n_u64(0x03ff_ffff);
    let hibit = vdupq_n_u64(1 << 24);
    // 26-bit limbs from the u64 view (w3 sits at Y bits 32..64):
    // l2 = (w1>>20 | w2<<12), l3 = (Y >> 14) & M = w2>>14 | w3<<18,
    // l4 = w3>>8 | 2^128.
    let l2a = vandq_u64(vorrq_u64(vshrq_n_u64::<52>(x01), vshlq_n_u64::<12>(y01)), m);
    let l2b = vandq_u64(vorrq_u64(vshrq_n_u64::<52>(x23), vshlq_n_u64::<12>(y23)), m);
    let l4a = vorrq_u64(vshrq_n_u64::<40>(y01), hibit);
    let l4b = vorrq_u64(vshrq_n_u64::<40>(y23), hibit);

    [
        vcombine_u32(vmovn_u64(vandq_u64(x01, m)), vmovn_u64(vandq_u64(x23, m))),
        vcombine_u32(
            vmovn_u64(vandq_u64(vshrq_n_u64::<26>(x01), m)),
            vmovn_u64(vandq_u64(vshrq_n_u64::<26>(x23), m)),
        ),
        vcombine_u32(vmovn_u64(l2a), vmovn_u64(l2b)),
        vcombine_u32(
            vmovn_u64(vandq_u64(vshrq_n_u64::<14>(y01), m)),
            vmovn_u64(vandq_u64(vshrq_n_u64::<14>(y23), m)),
        ),
        vcombine_u32(vmovn_u64(l4a), vmovn_u64(l4b)),
    ]
}

/// Scalar 26-bit limbs of a single block (high bit set) for the cold paths.
#[inline(always)]
fn block_limbs(block: &[u8; 16]) -> [u32; 5] {
    let w = |i: usize| u32::from_le_bytes(block[i..i + 4].try_into().unwrap());
    [
        w(0) & 0x03ff_ffff,
        (w(3) >> 2) & 0x03ff_ffff,
        (w(6) >> 4) & 0x03ff_ffff,
        (w(9) >> 6) & 0x03ff_ffff,
        (w(12) >> 8) | (1 << 24),
    ]
}

/// Limb-wise add (slack is absorbed by the next `mul_r` / final carry).
#[inline(always)]
fn add_limbs(a: [u32; 5], b: [u32; 5]) -> [u32; 5] {
    let mut out = [0u32; 5];
    for i in 0..5 {
        out[i] = a[i].wrapping_add(b[i]);
    }
    out
}

impl NeonPoly {
    /// Ensure the R⁴ power rows exist, then run one step.
    #[inline(always)]
    unsafe fn step_init(&mut self, blocks: &[uint32x4_t; 5]) {
        if self.lanes.is_none() {
            let r = self.r;
            let s = self.s;
            let r2 = mul_r(r, &r, &s);
            let r3 = mul_r(r2, &r, &s);
            let r4 = mul_r(r3, &r, &s);
            let dup = |x: u32| vdupq_n_u32(x);
            let coefs = [
                dup(r4[0]),
                dup(r4[1]),
                dup(r4[2]),
                dup(r4[3]),
                dup(r4[4]),
                dup(r4[1] * 5),
                dup(r4[2] * 5),
                dup(r4[3] * 5),
                dup(r4[4] * 5),
            ];
            self.lanes = Some(Lanes {
                h: [vdupq_n_u32(0); 5],
                coefs,
            });
        }
        // SAFETY: caller guaranteed NEON (baseline on aarch64).
        unsafe { step(self.lanes.as_mut().unwrap_unchecked(), blocks) };
    }

    /// Fold the whole deferred cache into the lane state, in order.
    /// `num_cached` must be a multiple of 4.
    #[inline(always)]
    unsafe fn drain_cache(&mut self) {
        debug_assert_eq!(self.num_cached % 4, 0);
        let mut i = 0;
        while i < self.num_cached {
            let limbs = load4(self.cached[i * 16..][..64].try_into().unwrap());
            self.step_init(&limbs);
            i += 4;
        }
        self.num_cached = 0;
    }
}

impl Backend for NeonPoly {
    #[inline(always)]
    unsafe fn init(key: &[u8; 32]) -> Self {
        let (r, k) = parse_key(key);
        let s = [r[1] * 5, r[2] * 5, r[3] * 5, r[4] * 5];
        Self {
            k,
            r,
            s,
            lanes: None,
            cached: [0; 128],
            num_cached: 0,
        }
    }

    #[inline(always)]
    unsafe fn absorb_block(&mut self, block: &[u8; 16]) {
        if self.num_cached == 8 {
            // Cache full: this stream is long enough to amortize the R⁴
            // power setup — drain and continue streaming.
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
        self.step_init(&load4(blocks));
    }

    #[inline(always)]
    fn pending_blocks(&self) -> usize {
        self.num_cached
    }

    #[cfg(feature = "zeroize")]
    unsafe fn zeroize_secrets(&mut self) {
        unsafe {
            core::ptr::write_volatile(&raw mut self.k, core::mem::zeroed());
            core::ptr::write_volatile(&raw mut self.r, core::mem::zeroed());
            core::ptr::write_volatile(&raw mut self.s, core::mem::zeroed());
            // Only scrub the payload; zeroing the whole Option could create
            // an invalid discriminant.
            if let Some(lanes) = &mut self.lanes {
                core::ptr::write_volatile(&raw mut *lanes, core::mem::zeroed());
            }
        }
        zeroize::Zeroize::zeroize(&mut self.cached);
        self.num_cached = 0;
    }

    #[inline(always)]
    unsafe fn finalize_into(&mut self, out: &mut [u8; 16]) {
        debug_assert!(self.num_cached <= 8);
        // NOTE: if/else assignment instead of `match self.lanes.take()` — a
        // `match` tripped `single_match_else`, and `Option::map` used to
        // outline the closure on the AVX2 backend (see its comment).
        let t: [u32; 5] = if let Some(mut lanes) = self.lanes.take() {
            // Full 4-block groups straight into the lanes; 1-3 newer
            // stragglers fold after the lane Horner (their extra `r`
            // multiplies bump the lane weights correctly).
            let mut i = 0;
            while i + 4 <= self.num_cached {
                let limbs = load4(self.cached[i * 16..][..64].try_into().unwrap());
                step(&mut lanes, &limbs);
                i += 4;
            }
            // Spill lanes to scalars: lane i limb j = buf[j * 4 + i].
            let mut buf = [0u32; 20];
            for j in 0..5 {
                vst1q_u32(buf[j * 4..].as_mut_ptr(), lanes.h[j]);
            }
            let lane = |i: usize| [buf[i], buf[4 + i], buf[8 + i], buf[12 + i], buf[16 + i]];
            // Fold lanes (lane 0 = oldest) and newer stragglers, then one
            // final multiply: every block — the newest included — carries
            // exactly one r (the spec's Σ m_j·r^(n-j)).
            let mut t = lane(0);
            t = add_limbs(mul_r(t, &self.r, &self.s), lane(1));
            t = add_limbs(mul_r(t, &self.r, &self.s), lane(2));
            t = add_limbs(mul_r(t, &self.r, &self.s), lane(3));
            while i < self.num_cached {
                let m = block_limbs(self.cached[i * 16..][..16].try_into().unwrap());
                t = add_limbs(mul_r(t, &self.r, &self.s), m);
                i += 1;
            }
            mul_r(t, &self.r, &self.s)
        } else {
            // ≤ 8 blocks total: scalar donna fold (absorb semantics —
            // every block gets one more r, including the last).
            let mut t = [0; 5];
            for i in 0..self.num_cached {
                let m = block_limbs(self.cached[i * 16..][..16].try_into().unwrap());
                t = mul_r(add_limbs(t, m), &self.r, &self.s);
            }
            t
        };
        self.num_cached = 0;
        finalize_limbs(&t, &self.k, out);
    }
}

#[cfg(test)]
mod tests {
    use super::NeonPoly;

    // NEON is baseline on aarch64; no runtime skip needed.
    #[test]
    fn t_matches_python_reference() {
        crate::poly1305::test_common::matches_python_reference::<NeonPoly>();
    }
    #[test]
    fn t_rfc8439_mac_stream() {
        crate::poly1305::test_common::rfc8439_mac_stream::<NeonPoly>();
    }
    #[test]
    fn t_segmentation_equivalence() {
        crate::poly1305::test_common::segmentation_equivalence::<NeonPoly>();
    }
    #[test]
    fn t_cross_backend() {
        crate::poly1305::test_common::cross_backend_consistency::<NeonPoly>();
    }
}
