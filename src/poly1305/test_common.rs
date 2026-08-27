//! Shared per-backend Poly1305 test drivers (test-only).

use alloc::vec::Vec;

use super::{Backend, Poly};

fn hex32(s: &str) -> [u8; 32] {
    hex_bytes(s).try_into().unwrap()
}

fn hex16(s: &str) -> [u8; 16] {
    hex_bytes(s).try_into().unwrap()
}

fn hex_bytes(s: &str) -> Vec<u8> {
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap())
        .collect()
}

/// Fixed vector generated with an independent Python bignum reference.
pub(crate) fn matches_python_reference<B: Backend>() {
    let key = hex32("390c8c7d7247342cd8100f2f6f770d65d670e58e0351d8ae8e4f6eac342fc231");
    let blocks = [
        hex16("b7b08716eb3fc12896b9622317749428"),
        hex16("7733c28ee8ba53bdb56b8824577d53ec"),
        hex16("c28a70a61c7510a1cd89216ca16cffca"),
        hex16("ea4987477e86dbccb97046fc2e18384e"),
        hex16("51d820c5c3ef80053a88ae3996de50e8"),
    ];
    let mut poly = unsafe { Poly::<B>::new(&key) };
    let mut all = [0u8; 80];
    for (i, b) in blocks.iter().enumerate() {
        all[i * 16..][..16].copy_from_slice(b);
    }
    unsafe { poly.update(&all) };
    let mut tag = [0u8; 16];
    unsafe { poly.finalize_into(&mut tag) };
    assert_eq!(tag, hex16("9f97ae7c79fce44470c2811eeaa5478b"));
}

/// RFC 8439 §2.8.2 full MAC stream fed in the engine's call pattern.
pub(crate) fn rfc8439_mac_stream<B: Backend>() {
    let polykey = hex32("7bac2b252db447af09b67a55a4e955840ae1d6731075d9eb2a9375783ed553ff");
    let aad = hex_bytes("50515253c0c1c2c3c4c5c6c7");
    let ct = hex_bytes(
        "d31a8d34648e60db7b86afbc53ef7ec2a4aded51296e08fea9e2b5a736ee62d6\
         3dbea45e8ca9671282fafb69da92728b1a71de0a9e060b2905d6a5b67ecd3b3\
         692ddbd7f2d778b8c9803aee328091b58fab324e4fad675945585808b4831d7\
         bc3ff4def08e4b7a9de576d26586cec64b6116",
    );
    let mut poly = unsafe { Poly::<B>::new(&polykey) };
    unsafe {
        poly.update(&aad);
        poly.update(&[0u8; 16][..4]);
        poly.update(&ct[..48]);
        poly.absorb4(ct[48..112].try_into().unwrap());
        poly.update(&ct[112..]);
        poly.update(&[0u8; 16][..14]);
        poly.update(&{
            let mut l = [0u8; 16];
            l[..8].copy_from_slice(&12u64.to_le_bytes());
            l[8..].copy_from_slice(&114u64.to_le_bytes());
            l
        });
        let mut tag = [0u8; 16];
        poly.finalize_into(&mut tag);
        assert_eq!(tag, hex16("1ae10b594f09e26a7e902ecbd0600691"));
    }
}

/// Varying segmentations of one stream (incl. absorb4) must agree.
pub(crate) fn segmentation_equivalence<B: Backend>() {
    let key = hex32("0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20");
    let mut data = Vec::new();
    let mut seed = 0x1234_5678u64;
    for _ in 0..1008 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        data.push(seed as u8);
    }
    let mut ref_tag = [0u8; 16];
    {
        let mut poly = unsafe { Poly::<B>::new(&key) };
        unsafe {
            poly.update(&data);
            poly.finalize_into(&mut ref_tag);
        }
    }
    let mut poly = unsafe { Poly::<B>::new(&key) };
    let mut off = 0usize;
    let mut n = 1usize;
    unsafe {
        while off < data.len() {
            let end = (off + n).min(data.len());
            if end - off == 64 && poly.pending_blocks() == 0 {
                poly.absorb4(data[off..end].try_into().unwrap());
            } else {
                poly.update(&data[off..end]);
            }
            off = end;
            n = (n * 3 + 7) % 100;
        }
        let mut tag = [0u8; 16];
        poly.finalize_into(&mut tag);
        assert_eq!(tag, ref_tag);
    }
}

/// Cross-check a backend against the scalar reference over many block
/// counts, exercising every finalize path (0..=8 cached blocks + drains).
pub(crate) fn cross_backend_consistency<B: Backend>() {
    let key = hex32("390c8c7d7247342cd8100f2f6f770d65d670e58e0351d8ae8e4f6eac342fc231");
    let mut data = Vec::new();
    let mut seed = 0xdead_beefu64;
    for _ in 0..40 * 16 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        data.push(seed as u8);
    }
    for n in 0..40usize {
        let stream = &data[..n * 16];
        let mut ref_poly = unsafe { Poly::<crate::poly1305::soft::SoftPoly>::new(&key) };
        let mut ref_tag = [0u8; 16];
        unsafe {
            ref_poly.update(stream);
            ref_poly.finalize_into(&mut ref_tag);
        }
        let mut poly = unsafe { Poly::<B>::new(&key) };
        let mut tag = [0u8; 16];
        unsafe {
            poly.update(stream);
            poly.finalize_into(&mut tag);
        }
        assert_eq!(tag, ref_tag, "block count {n}");
    }
}
