//! Scalar ChaCha20 backend — correctness reference and no_std fallback.

use super::{BLOCK, State};

/// 20 rounds of ChaCha, in place on the working state.
#[inline(always)]
pub(crate) fn rounds(x: &mut [u32; 16]) {
    for _ in 0..10 {
        column_round(x);
        diagonal_round(x);
    }
}

#[inline(always)]
fn column_round(x: &mut [u32; 16]) {
    quarter_round(x, 0, 4, 8, 12);
    quarter_round(x, 1, 5, 9, 13);
    quarter_round(x, 2, 6, 10, 14);
    quarter_round(x, 3, 7, 11, 15);
}

#[inline(always)]
fn diagonal_round(x: &mut [u32; 16]) {
    quarter_round(x, 0, 5, 10, 15);
    quarter_round(x, 1, 6, 11, 12);
    quarter_round(x, 2, 7, 8, 13);
    quarter_round(x, 3, 4, 9, 14);
}

#[inline(always)]
fn quarter_round(x: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    x[a] = x[a].wrapping_add(x[b]);
    x[d] = (x[d] ^ x[a]).rotate_left(16);
    x[c] = x[c].wrapping_add(x[d]);
    x[b] = (x[b] ^ x[c]).rotate_left(12);
    x[a] = x[a].wrapping_add(x[b]);
    x[d] = (x[d] ^ x[a]).rotate_left(8);
    x[c] = x[c].wrapping_add(x[d]);
    x[b] = (x[b] ^ x[c]).rotate_left(7);
}

/// Generate one keystream block for `state`'s current counter into `out`
/// (without advancing).
#[inline(always)]
pub(crate) unsafe fn gen_block(state: &State, out: &mut [u8; BLOCK]) {
    let mut x = state.words;
    rounds(&mut x);
    for i in 0..16 {
        out[i * 4..][..4].copy_from_slice(&x[i].wrapping_add(state.words[i]).to_le_bytes());
    }
}

/// XOR `buf` with the keystream starting at `state`'s counter, advancing one
/// counter per 64-byte block.
#[inline(always)]
pub(crate) unsafe fn xor(state: &mut State, buf: &mut [u8]) {
    let mut ks = [0u8; BLOCK];
    for chunk in buf.chunks_mut(BLOCK) {
        gen_block(state, &mut ks);
        for (b, k) in chunk.iter_mut().zip(ks) {
            *b ^= k;
        }
        state.advance(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chacha::{CONSTANTS, State};

    /// RFC 8439 §2.3.2 block function test vector.
    #[test]
    fn block_function() {
        let key: [u8; 32] = core::array::from_fn(|i| i as u8);
        let nonce: [u8; 12] = [0, 0, 0, 9, 0, 0, 0, 0x4a, 0, 0, 0, 0];
        let mut st = State::new_ietf(&key, &nonce);
        st.words[12] = 1;
        assert_eq!(st.words[..4], CONSTANTS);
        let mut out = [0u8; 64];
        unsafe { gen_block(&st, &mut out) };
        let expected: [u8; 64] = [
            0x10, 0xf1, 0xe7, 0xe4, 0xd1, 0x3b, 0x59, 0x15, 0x50, 0x0f, 0xdd, 0x1f, 0xa3, 0x20,
            0x71, 0xc4, 0xc7, 0xd1, 0xf4, 0xc7, 0x33, 0xc0, 0x68, 0x03, 0x04, 0x22, 0xaa, 0x9a,
            0xc3, 0xd4, 0x6c, 0x4e, 0xd2, 0x82, 0x64, 0x46, 0x07, 0x9f, 0xaa, 0x09, 0x14, 0xc2,
            0xd7, 0x05, 0xd9, 0x8b, 0x02, 0xa2, 0xb5, 0x12, 0x9c, 0xd1, 0xde, 0x16, 0x4e, 0xb9,
            0xcb, 0xd0, 0x83, 0xe8, 0xa2, 0x50, 0x3c, 0x4e,
        ];
        assert_eq!(out, expected);
    }
}
