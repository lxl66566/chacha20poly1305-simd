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
    unsafe fn gen_block(state: &State, out: &mut [u8; BLOCK]) {
        crate::chacha::soft::gen_block(state, out);
    }

    #[inline(always)]
    unsafe fn chacha_xor(state: &mut State, buf: &mut [u8]) {
        crate::chacha::soft::xor(state, buf);
    }

    #[inline(always)]
    unsafe fn chacha_xor_batch(state: &mut State, buf: &mut [u8]) {
        crate::chacha::soft::xor(state, buf);
    }

    #[inline(always)]
    unsafe fn xor_block1(state: &mut State, buf: &mut [u8]) {
        debug_assert_eq!(buf.len(), BLOCK);
        crate::chacha::soft::xor(state, buf);
    }
}
