//! AEAD engine: RFC 8439 §2.8 construction with a fused pipeline.
//!
//! The seal loop interleaves ChaCha20 keystream batches with 64-byte Poly1305
//! absorption steps so that the (multiply-latency-bound) MAC and the
//! (add/xor-throughput-bound) cipher overlap in the out-of-order window.

use crate::{
    chacha::{BLOCK, State},
    poly1305::Poly,
};

/// Opaque authentication failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Error;

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("chacha20poly1305: authentication failure")
    }
}

#[cfg(feature = "std")]
impl core::error::Error for Error {}

/// Per-backend primitive operations, monomorphized into the fused engine.
pub(crate) trait Ops {
    /// ChaCha20 blocks per bulk-loop batch (a power of two ≥ 1).
    const CHACHA_BATCH: usize;
    /// Poly1305 backend.
    type Poly: crate::poly1305::Backend;

    /// Generate one keystream block at the current counter, without advancing.
    unsafe fn gen_block(state: &State, out: &mut [u8; BLOCK]);
    /// XOR `buf` (any length) with the keystream, advancing the counter.
    /// Backends batch internally.
    unsafe fn chacha_xor(state: &mut State, buf: &mut [u8]);
    /// XOR exactly `CHACHA_BATCH * 64` bytes, advancing the counter.
    /// Hot loop path; may assume full batches.
    unsafe fn chacha_xor_batch(state: &mut State, buf: &mut [u8]);
    /// XOR exactly one 64-byte block (the engine prologue).
    unsafe fn xor_block1(state: &mut State, buf: &mut [u8]);
}

/// Largest message the IETF 32-bit counter can address (256 GiB - 64 KiB).
/// Enforced at the public API layer ([`crate::ChaCha20Poly1305`]); the engine
/// only keeps a `debug_assert`.
pub(crate) const MAX_LEN: usize = u32::MAX as usize * BLOCK;

/// Constant-time tag comparison.
#[inline]
fn ct_eq(a: &[u8; 16], b: &[u8; 16]) -> bool {
    let mut acc = 0u8;
    for i in 0..16 {
        acc |= a[i] ^ b[i];
    }
    acc == 0
}

/// Fused seal (encrypt + MAC). Writes the tag into `tag_out`.
// The engine must inline into the backend's #[target_feature] entry: in a
// feature-less context every SIMD intrinsic would degrade to an out-of-line
// call (observed as ~30 cycles/byte — see the note in backend/*).
#[cfg_attr(feature = "hotpath", hotpath::measure)]
#[inline(always)]
pub(crate) fn seal<O: Ops>(state: &mut State, aad: &[u8], msg: &mut [u8], tag_out: &mut [u8; 16]) {
    debug_assert!(
        msg.len() < MAX_LEN,
        "message exceeds ChaCha20 counter space"
    );
    // SAFETY: caller (backend entry) ensured the required target features.
    unsafe { process::<O, true>(state, aad, msg, tag_out) };
}

/// Fused open (MAC + decrypt). Returns `false` on tag mismatch; `buf`
/// contents are unspecified in that case.
#[cfg_attr(feature = "hotpath", hotpath::measure)]
#[inline(always)]
pub(crate) fn open<O: Ops>(state: &mut State, aad: &[u8], buf: &mut [u8], tag: &[u8; 16]) -> bool {
    debug_assert!(
        buf.len() < MAX_LEN,
        "message exceeds ChaCha20 counter space"
    );
    let mut computed = [0u8; 16];
    // SAFETY: see seal.
    unsafe { process::<O, false>(state, aad, buf, &mut computed) };
    ct_eq(&computed, tag)
}

/// Shared fused pipeline. `SEAL` selects the per-chunk phase order:
/// seal = xor then MAC (MAC eats the produced ciphertext), open = MAC then
/// xor (MAC eats the received ciphertext).
#[inline(always)]
unsafe fn process<O: Ops, const SEAL: bool>(
    state: &mut State,
    aad: &[u8],
    msg: &mut [u8],
    tag_out: &mut [u8; 16],
) {
    // SAFETY: callers (seal/open) received target-feature guarantees from the
    // backend entry points.
    unsafe {
        // ── Keystream block 0 → Poly1305 one-time key; its tail is discarded ──
        let mut ks0 = [0u8; BLOCK];
        O::gen_block(state, &mut ks0);
        let mut poly = Poly::<O::Poly>::new(ks0[..32].try_into().unwrap());
        crate::poly1305::clear_key(&mut ks0[..32]);
        state.advance(1);

        // ── MAC the (zero-padded) AAD ──
        poly.update(aad);
        let aad_pad = (16 - aad.len() % 16) % 16;
        poly.update(&[0u8; 16][..aad_pad]);

        // ── MAC/keystream interleave ──
        // The Poly1305 stream (AAD zero-padded) is generally NOT aligned with the
        // ChaCha block grid: the AAD shifts the 4-block absorb windows by
        // `s = (-aad_total) mod 64` bytes relative to message byte 0. The two
        // cursors below absorb that shift:
        //   `off`      — keystream cursor (always ChaCha-block aligned);
        //   `poly_off` — ciphertext cursor fed into Poly1305.
        // seal keeps poly at or behind the xor cursor; open keeps it at or ahead
        // (it must read unmodified ciphertext).
        let m = poly.pending_blocks(); // whole blocks buffered (≤ 8)
        // Prologue absorbs just enough message bytes to bring the poly stream
        // to a 64-byte boundary (`pending_blocks % 4 == 0`) before `absorb4`.
        let s = (4 - m % 4) % 4 * 16;
        let mut off;
        let mut poly_off;

        if msg.len() < BLOCK + s {
            // No bulk loop will run: xor (or authenticate, for open) the whole
            // partial-block message through the generic path.
            if !SEAL {
                poly.update(msg);
            }
            O::chacha_xor(state, msg);
            if SEAL {
                poly.update(msg);
            }
            off = msg.len();
            poly_off = msg.len();
        } else {
            // Prologue: keystream block 1 + the poly alignment window. When s > 0
            // the first 4-block window straddles blocks 1 and 2.
            if SEAL {
                O::xor_block1(state, &mut msg[..BLOCK]);
                poly.update(&msg[..s]);
                poly_off = s;
            } else {
                poly.update(&msg[..s]);
                poly.absorb4(msg[s..s + BLOCK].try_into().unwrap());
                O::xor_block1(state, &mut msg[..BLOCK]);
                poly_off = s + BLOCK;
            }
            off = BLOCK;
            // Fused bulk loop.
            let batch = O::CHACHA_BATCH * BLOCK;
            while msg.len() - off >= batch {
                if SEAL {
                    O::chacha_xor_batch(state, &mut msg[off..off + batch]);
                    off += batch;
                    while poly_off + BLOCK <= off {
                        poly.absorb4(msg[poly_off..poly_off + BLOCK].try_into().unwrap());
                        poly_off += BLOCK;
                    }
                } else {
                    // Open: poly must lead the xor cursor — every window it
                    // absorbs must still be unmodified ciphertext. Poly
                    // windows sit on an `s`-shifted grid that never aligns
                    // with the ChaCha 64-byte grid (s ≠ 0), so the xor cursor
                    // only ever advances up to the largest 64-byte boundary
                    // the poly has already absorbed past.
                    let ahead = (off + 2 * batch).min(msg.len());
                    while poly_off + BLOCK <= ahead {
                        poly.absorb4(msg[poly_off..poly_off + BLOCK].try_into().unwrap());
                        poly_off += BLOCK;
                    }
                    let xor_end = (off + batch).min(poly_off & !(BLOCK - 1));
                    if xor_end <= off {
                        // Defense in depth: unreachable while the loop
                        // condition holds (poly then leads the xor cursor by
                        // more than `batch` bytes), but if the invariant ever
                        // breaks the tail still finishes the message
                        // correctly.
                        break;
                    }
                    if xor_end - off == batch {
                        O::chacha_xor_batch(state, &mut msg[off..off + batch]);
                    } else {
                        O::chacha_xor(state, &mut msg[off..xor_end]);
                    }
                    off = xor_end;
                }
            }

            // Tail: xor / absorb the remainder (may be up to batch-1 bytes).
            if off < msg.len() {
                if SEAL {
                    O::chacha_xor(state, &mut msg[off..]);
                }
                poly.update(&msg[poly_off..]);
                if !SEAL {
                    O::chacha_xor(state, &mut msg[off..]);
                }
                off = msg.len();
                poly_off = msg.len();
            } else if poly_off < msg.len() {
                poly.update(&msg[poly_off..]);
                poly_off = msg.len();
            }
        }
        debug_assert_eq!(off, msg.len());
        debug_assert_eq!(poly_off, msg.len());

        // ── Zero-pad the ciphertext, absorb lengths, emit tag ──
        let ct_pad = (16 - msg.len() % 16) % 16;
        poly.update(&[0u8; 16][..ct_pad]);
        let mut lengths = [0u8; 16];
        lengths[..8].copy_from_slice(&(aad.len() as u64).to_le_bytes());
        lengths[8..].copy_from_slice(&(msg.len() as u64).to_le_bytes());
        poly.update(&lengths);
        poly.finalize_into(tag_out);
    }
}
