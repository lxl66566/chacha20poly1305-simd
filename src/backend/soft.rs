//! Soft backend wiring: scalar ChaCha20 + scalar Poly1305.

use crate::{
    aead::Ops,
    chacha::{BLOCK, State},
};

pub(crate) struct SoftOps;

impl Ops for SoftOps {
    type Poly = crate::poly1305::soft::SoftPoly;

    const CHACHA_BATCH: usize = 1;

    #[inline(always)]
    unsafe fn gen_key_xor2(state: &mut State, key_out: &mut [u8; 32], b1: &mut [u8]) {
        debug_assert_eq!(b1.len(), BLOCK);
        crate::chacha::soft::gen_key_xor2(state, key_out, b1.try_into().unwrap());
    }

    #[inline(always)]
    unsafe fn gen_ks_small(state: &mut State, key_out: &mut [u8; 32], ks: &mut [u8]) {
        crate::chacha::soft::gen_ks_small(state, key_out, ks);
    }

    #[inline(always)]
    unsafe fn chacha_xor(state: &mut State, buf: &mut [u8]) {
        crate::chacha::soft::xor(state, buf);
    }

    #[inline(always)]
    unsafe fn chacha_xor_batch(state: &mut State, buf: &mut [u8]) {
        crate::chacha::soft::xor(state, buf);
    }
}
