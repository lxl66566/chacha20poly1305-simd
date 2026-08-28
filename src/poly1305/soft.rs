//! Scalar Poly1305 (26-bit limb "donna" variant).
//!
//! Structure follows poly1305-donna as shipped in RustCrypto `poly1305`
//! (Apache-2.0 OR MIT), with the streaming wrapper replaced by our AEAD
//! block-always-high-bit contract.

use super::Backend;

/// Parse a one-time key into the clamped `r` (5×26-bit limbs) and the
/// addition key `s` (shared with the NEON backend).
pub(crate) fn parse_key(key: &[u8; 32]) -> ([u32; 5], [u32; 4]) {
    // r &= 0x0ffffffc0ffffffc0ffffffc0fffffff
    let r = [
        u32::from_le_bytes(key[0..4].try_into().unwrap()) & 0x03ff_ffff,
        (u32::from_le_bytes(key[3..7].try_into().unwrap()) >> 2) & 0x03ff_ff03,
        (u32::from_le_bytes(key[6..10].try_into().unwrap()) >> 4) & 0x03ff_c0ff,
        (u32::from_le_bytes(key[9..13].try_into().unwrap()) >> 6) & 0x03f0_3fff,
        (u32::from_le_bytes(key[12..16].try_into().unwrap()) >> 8) & 0x000f_ffff,
    ];
    let mut pad = [0u32; 4];
    for i in 0..4 {
        pad[i] = u32::from_le_bytes(key[16 + i * 4..][..4].try_into().unwrap());
    }
    (r, pad)
}

/// Multiply a 5×26-bit limb value by `r` and lazily reduce (the multiply
/// half of a donna block step; shared with the NEON backend's scalar folds).
/// `s` holds `5·r[1..5]`.
#[cfg_attr(not(debug_assertions), inline(always))]
pub(crate) fn mul_r(mut h: [u32; 5], r: &[u32; 5], s: &[u32; 4]) -> [u32; 5] {
    let d0 = u64::from(h[0]) * u64::from(r[0])
        + u64::from(h[1]) * u64::from(s[3])
        + u64::from(h[2]) * u64::from(s[2])
        + u64::from(h[3]) * u64::from(s[1])
        + u64::from(h[4]) * u64::from(s[0]);
    let mut d1 = u64::from(h[0]) * u64::from(r[1])
        + u64::from(h[1]) * u64::from(r[0])
        + u64::from(h[2]) * u64::from(s[3])
        + u64::from(h[3]) * u64::from(s[2])
        + u64::from(h[4]) * u64::from(s[1]);
    let mut d2 = u64::from(h[0]) * u64::from(r[2])
        + u64::from(h[1]) * u64::from(r[1])
        + u64::from(h[2]) * u64::from(r[0])
        + u64::from(h[3]) * u64::from(s[3])
        + u64::from(h[4]) * u64::from(s[2]);
    let mut d3 = u64::from(h[0]) * u64::from(r[3])
        + u64::from(h[1]) * u64::from(r[2])
        + u64::from(h[2]) * u64::from(r[1])
        + u64::from(h[3]) * u64::from(r[0])
        + u64::from(h[4]) * u64::from(s[3]);
    let mut d4 = u64::from(h[0]) * u64::from(r[4])
        + u64::from(h[1]) * u64::from(r[3])
        + u64::from(h[2]) * u64::from(r[2])
        + u64::from(h[3]) * u64::from(r[1])
        + u64::from(h[4]) * u64::from(r[0]);

    // lazy partial reduction
    let mut c = (d0 >> 26) as u32;
    h[0] = d0 as u32 & 0x03ff_ffff;
    d1 += u64::from(c);
    c = (d1 >> 26) as u32;
    h[1] = d1 as u32 & 0x03ff_ffff;
    d2 += u64::from(c);
    c = (d2 >> 26) as u32;
    h[2] = d2 as u32 & 0x03ff_ffff;
    d3 += u64::from(c);
    c = (d3 >> 26) as u32;
    h[3] = d3 as u32 & 0x03ff_ffff;
    d4 += u64::from(c);
    c = (d4 >> 26) as u32;
    h[4] = d4 as u32 & 0x03ff_ffff;
    h[0] += c * 5;
    c = h[0] >> 26;
    h[0] &= 0x03ff_ffff;
    h[1] += c;
    h
}

/// Final reduce a 5×26-bit limb value, add the pad key mod 2^128 and emit
/// the tag (shared with the NEON backend's scalar finalize).
pub(crate) fn finalize_limbs(h: &[u32; 5], pad: &[u32; 4], out: &mut [u8; 16]) {
    // fully carry h
    let mut h0 = h[0];
    let mut h1 = h[1];
    let mut h2 = h[2];
    let mut h3 = h[3];
    let mut h4 = h[4];

    let mut c = h1 >> 26;
    h1 &= 0x03ff_ffff;
    h2 += c;
    c = h2 >> 26;
    h2 &= 0x03ff_ffff;
    h3 += c;
    c = h3 >> 26;
    h3 &= 0x03ff_ffff;
    h4 += c;
    c = h4 >> 26;
    h4 &= 0x03ff_ffff;
    h0 += c * 5;
    c = h0 >> 26;
    h0 &= 0x03ff_ffff;
    h1 += c;

    // compute h + -p
    let mut g0 = h0.wrapping_add(5);
    c = g0 >> 26;
    g0 &= 0x03ff_ffff;
    let mut g1 = h1.wrapping_add(c);
    c = g1 >> 26;
    g1 &= 0x03ff_ffff;
    let mut g2 = h2.wrapping_add(c);
    c = g2 >> 26;
    g2 &= 0x03ff_ffff;
    let mut g3 = h3.wrapping_add(c);
    c = g3 >> 26;
    g3 &= 0x03ff_ffff;
    let g4 = h4.wrapping_add(c).wrapping_sub(1 << 26);

    // select h if h < p
    let mut mask = (g4 >> (32 - 1)).wrapping_sub(1);
    g0 &= mask;
    g1 &= mask;
    g2 &= mask;
    g3 &= mask;
    mask = !mask;
    h0 = (h0 & mask) | g0;
    h1 = (h1 & mask) | g1;
    h2 = (h2 & mask) | g2;
    h3 = (h3 & mask) | g3;

    // repack to 128 bits
    h0 |= h1 << 26;
    h1 = (h1 >> 6) | (h2 << 20);
    h2 = (h2 >> 12) | (h3 << 14);
    h3 = (h3 >> 18) | (h4 << 8);

    // h = (h + pad) % 2^128
    let mut f = u64::from(h0) + u64::from(pad[0]);
    let o0 = f as u32;
    f = u64::from(h1) + u64::from(pad[1]) + (f >> 32);
    let o1 = f as u32;
    f = u64::from(h2) + u64::from(pad[2]) + (f >> 32);
    let o2 = f as u32;
    f = u64::from(h3) + u64::from(pad[3]) + (f >> 32);
    let o3 = f as u32;

    out[0..4].copy_from_slice(&o0.to_le_bytes());
    out[4..8].copy_from_slice(&o1.to_le_bytes());
    out[8..12].copy_from_slice(&o2.to_le_bytes());
    out[12..16].copy_from_slice(&o3.to_le_bytes());
}

#[derive(Clone)]
pub(crate) struct SoftPoly {
    r: [u32; 5],
    h: [u32; 5],
    pad: [u32; 4],
}

impl Backend for SoftPoly {
    unsafe fn init(key: &[u8; 32]) -> Self {
        let (r, pad) = parse_key(key);
        Self { r, h: [0; 5], pad }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    unsafe fn absorb_block(&mut self, block: &[u8; 16]) {
        let r = self.r;
        let s1 = r[1] * 5;
        let s2 = r[2] * 5;
        let s3 = r[3] * 5;
        let s4 = r[4] * 5;

        // h += m (with the 2^128 high bit set on every AEAD block)
        let h0 = self.h[0] + (u32::from_le_bytes(block[0..4].try_into().unwrap()) & 0x03ff_ffff);
        let h1 =
            self.h[1] + ((u32::from_le_bytes(block[3..7].try_into().unwrap()) >> 2) & 0x03ff_ffff);
        let h2 =
            self.h[2] + ((u32::from_le_bytes(block[6..10].try_into().unwrap()) >> 4) & 0x03ff_ffff);
        let h3 =
            self.h[3] + ((u32::from_le_bytes(block[9..13].try_into().unwrap()) >> 6) & 0x03ff_ffff);
        let h4 =
            self.h[4] + ((u32::from_le_bytes(block[12..16].try_into().unwrap()) >> 8) | (1 << 24));

        // h *= r (shared with the NEON backend's scalar folds)
        self.h = mul_r([h0, h1, h2, h3, h4], &r, &[s1, s2, s3, s4]);
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    unsafe fn absorb4(&mut self, blocks: &[u8; 64]) {
        for block in blocks.as_chunks::<16>().0 {
            self.absorb_block(block);
        }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    fn pending_blocks(&self) -> usize {
        0
    }

    #[cfg(feature = "zeroize")]
    unsafe fn zeroize_secrets(&mut self) {
        use zeroize::Zeroize;
        self.r.zeroize();
        self.h.zeroize();
        self.pad.zeroize();
    }

    unsafe fn finalize_into(&mut self, out: &mut [u8; 16]) {
        finalize_limbs(&self.h, &self.pad, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_matches_python_reference() {
        crate::poly1305::test_common::matches_python_reference::<SoftPoly>();
    }
    #[test]
    fn t_rfc8439_mac_stream() {
        crate::poly1305::test_common::rfc8439_mac_stream::<SoftPoly>();
    }
    #[test]
    fn t_segmentation_equivalence() {
        crate::poly1305::test_common::segmentation_equivalence::<SoftPoly>();
    }
}
