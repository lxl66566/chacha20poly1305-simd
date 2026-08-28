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

    /// Compute keystream blocks 0 and 1 in a single kernel invocation
    /// (OpenSSL fuses its TLS path the same way): block 0's first 32 bytes
    /// (the Poly1305 one-time key) go to `key_out`, block 1's keystream is
    /// XORed into `b1` (`b1.len() == BLOCK`; a zeroed buffer therefore
    /// yields the raw keystream). Advances the counter by 2.
    unsafe fn gen_key_xor2(state: &mut State, key_out: &mut [u8; 32], b1: &mut [u8]);
    /// Small-message fused op (the OpenSSL `chacha20_poly1305_tls_cipher`
    /// shape, message ≤ 3 * BLOCK): ONE kernel call computes block 0 (first
    /// 32 bytes → the one-time key in `key_out`) and the RAW keystream
    /// blocks covering the message, written to `ks`
    /// (`ks.len() == ceil(msg_len / BLOCK) * BLOCK`; 0 for an empty
    /// message). Advances the counter by `1 + ks.len() / BLOCK`. The engine
    /// XORs `ks` into the message at whichever pipeline stage the MAC
    /// ordering requires.
    unsafe fn gen_ks_small(state: &mut State, key_out: &mut [u8; 32], ks: &mut [u8]);
    /// XOR `buf` (any length) with the keystream, advancing the counter.
    /// Backends batch internally.
    unsafe fn chacha_xor(state: &mut State, buf: &mut [u8]);
    /// XOR exactly `CHACHA_BATCH * 64` bytes, advancing the counter.
    /// Hot loop path; may assume full batches.
    unsafe fn chacha_xor_batch(state: &mut State, buf: &mut [u8]);

    /// Seal bulk run: while full batches remain AND the MAC window cache is
    /// empty AND at least one batch of already-written ciphertext is
    /// pending, advance both cursors by whole batches. Returns the new
    /// `(off, poly_off)`. Default: per-batch unfused steps; overridden when
    /// the backend fuses cipher+MAC into one loop (measured: across call
    /// boundaries the latency-bound MAC gets ~zero OoO overlap with the
    /// cipher).
    #[cfg_attr(debug_assertions, inline)]
    #[cfg_attr(not(debug_assertions), inline(always))]
    unsafe fn seal_bulk(
        state: &mut State,
        msg: &mut [u8],
        mut off: usize,
        mut poly_off: usize,
        poly: &mut Poly<Self::Poly>,
    ) -> (usize, usize) {
        let batch = Self::CHACHA_BATCH * BLOCK;
        while msg.len() - off >= batch && poly.pending_blocks() == 0 && off - poly_off >= batch {
            unsafe {
                Self::chacha_xor_batch(state, &mut msg[off..off + batch]);
                poly.absorb_blocks(&msg[poly_off..poly_off + batch]);
            }
            off += batch;
            poly_off += batch;
        }
        (off, poly_off)
    }

    /// Whether `open_bulk` fuses MAC+cipher (default false → the engine's
    /// poly-leads loop runs unchanged).
    const FUSED_OPEN: bool = false;

    /// Fused open bulk run. Only called when `FUSED_OPEN` holds and the
    /// engine has normalized the MAC window cache. Default: no-op.
    #[cfg_attr(debug_assertions, inline)]
    #[cfg_attr(not(debug_assertions), inline(always))]
    unsafe fn open_bulk(
        state: &mut State,
        msg: &mut [u8],
        off: usize,
        poly_off: usize,
        poly: &mut Poly<Self::Poly>,
    ) -> (usize, usize) {
        let _ = (state, msg, poly);
        (off, poly_off)
    }
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

/// Largest message handled by the small-message fast path (the OpenSSL
/// `chacha20_poly1305_tls_cipher` XOR128 threshold: 3 blocks).
pub(crate) const SMALL_MAX: usize = 3 * BLOCK;

/// Init the Poly1305 state from the one-time key (scrubbed afterwards when
/// `zeroize` is on) and absorb the zero-padded AAD.
#[cfg_attr(debug_assertions, inline)]
#[cfg_attr(not(debug_assertions), inline(always))]
unsafe fn poly_with_aad<B: crate::poly1305::Backend>(key: &mut [u8; 32], aad: &[u8]) -> Poly<B> {
    let mut poly = Poly::<B>::new(key);
    crate::poly1305::clear_key(key);
    poly.update(aad);
    let aad_pad = (16 - aad.len() % 16) % 16;
    poly.update(&[0u8; 16][..aad_pad]);
    poly
}

/// Small-path MAC sweep: absorb `data` (the ≤ [`SMALL_MAX`]-byte ciphertext)
/// plus the trailing zero-pad and length block in one aligned pass — the
/// engine-side mirror of OpenSSL's contiguous `tohash` region
/// (`AAD‖CT‖pad‖len` assembled so one `Poly1305_Update` covers it). Brings
/// the stream to a 64-byte boundary, drains `absorb4` windows over the
/// message, then folds the assembled `[ct remainder ‖ zero pad][lengths]`
/// tail as whole blocks.
#[cfg_attr(debug_assertions, inline)]
#[cfg_attr(not(debug_assertions), inline(always))]
unsafe fn mac_small_sweep<B: crate::poly1305::Backend>(
    poly: &mut Poly<B>,
    data: &[u8],
    aad_len: usize,
) {
    debug_assert!(data.len() <= SMALL_MAX);
    let m = poly.pending_blocks();
    let mut off = 0usize;
    let s = (4 - m % 4) % 4 * 16;
    // BUGFIX: skip the 64-byte alignment update when the message is shorter
    // than the window remainder — clamping `s` to data.len() instead would
    // park a partial block in the wrapper's pending buffer that the tail
    // assembly below never zero-pads. No absorb4 window fits either way.
    if data.len() >= s {
        poly.update(&data[..s]);
        off = s;
        while data.len() - off >= BLOCK {
            poly.absorb4(data[off..off + BLOCK].try_into().unwrap());
            off += BLOCK;
        }
    }
    while data.len() - off >= 16 {
        poly.update(&data[off..off + 16]);
        off += 16;
    }
    // [ct remainder ‖ zero pad to 16] then [aad_len ‖ ct_len], at most two
    // assembled blocks — no separate pad / length update calls needed.
    let mut tail = [0u8; 32];
    tail[16..24].copy_from_slice(&(aad_len as u64).to_le_bytes());
    tail[24..32].copy_from_slice(&(data.len() as u64).to_le_bytes());
    let rem = data.len() - off;
    if rem > 0 {
        tail[..rem].copy_from_slice(&data[off..]);
        poly.update(&tail[..16]);
    }
    poly.update(&tail[16..32]);
}

/// Absorb the ciphertext zero-padding and the AAD/message lengths, then
/// emit the tag (shared by every path's epilogue).
#[cfg_attr(debug_assertions, inline)]
#[cfg_attr(not(debug_assertions), inline(always))]
unsafe fn finish_tag<B: crate::poly1305::Backend>(
    poly: &mut Poly<B>,
    aad_len: usize,
    ct_len: usize,
    out: &mut [u8; 16],
) {
    let ct_pad = (16 - ct_len % 16) % 16;
    poly.update(&[0u8; 16][..ct_pad]);
    let mut lengths = [0u8; 16];
    lengths[..8].copy_from_slice(&(aad_len as u64).to_le_bytes());
    lengths[8..].copy_from_slice(&(ct_len as u64).to_le_bytes());
    poly.update(&lengths);
    poly.finalize_into(out);
}

/// Fused seal (encrypt + MAC). Writes the tag into `tag_out`.
// The engine must inline into the backend's #[target_feature] entry: in a
// feature-less context every SIMD intrinsic would degrade to an out-of-line
// call (observed as ~30 cycles/byte — see the note in backend/*).
#[cfg_attr(feature = "hotpath", hotpath::measure)]
#[cfg_attr(debug_assertions, inline)]
#[cfg_attr(not(debug_assertions), inline(always))]
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
#[cfg_attr(debug_assertions, inline)]
#[cfg_attr(not(debug_assertions), inline(always))]
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
#[cfg_attr(debug_assertions, inline)]
#[cfg_attr(not(debug_assertions), inline(always))]
unsafe fn process<O: Ops, const SEAL: bool>(
    state: &mut State,
    aad: &[u8],
    msg: &mut [u8],
    tag_out: &mut [u8; 16],
) {
    // SAFETY: callers (seal/open) received target-feature guarantees from the
    // backend entry points.
    unsafe {
        // ── Tiny-message path (≤ 1 block): quad key gen + plain updates ──
        // Structurally the pre-fusion tiny branch: the sweep machinery of
        // the small path measurably hurts sub-block sizes (code layout
        // around the rounds loop), so keep the minimal shape. Measured
        // boundary: 64 bytes is also faster here than in the sweep path.
        if msg.len() <= BLOCK {
            let mut key32 = [0u8; 32];
            let mut ks1 = [0u8; BLOCK];
            O::gen_key_xor2(state, &mut key32, &mut ks1);
            let mut poly = poly_with_aad::<O::Poly>(&mut key32, aad);
            if !SEAL {
                poly.update(msg);
            }
            crate::chacha::xor_bytes(msg, &ks1[..msg.len()]);
            if SEAL {
                poly.update(msg);
            }
            finish_tag(&mut poly, aad.len(), msg.len(), tag_out);
            return;
        }

        // ── Small-message fast path (1..=3 blocks) ──
        // OpenSSL's `chacha20_poly1305_tls_cipher` shape: ONE kernel call
        // covers the one-time key AND the whole message; the MAC then runs
        // as a single sweep over the contiguous message buffer (mac_small_sweep).
        if msg.len() <= SMALL_MAX {
            let mut key32 = [0u8; 32];
            let mut ks = [0u8; SMALL_MAX];
            let n = msg.len().div_ceil(BLOCK) * BLOCK;
            O::gen_ks_small(state, &mut key32, &mut ks[..n]);
            let mut poly = poly_with_aad::<O::Poly>(&mut key32, aad);
            // The xor lands on whichever side of the MAC sweep the pipeline
            // stage requires: seal MACs the produced ciphertext (xor first),
            // open MACs the received ciphertext (xor after).
            if SEAL {
                crate::chacha::xor_bytes(msg, &ks[..msg.len()]);
                mac_small_sweep(&mut poly, msg, aad.len());
            } else {
                mac_small_sweep(&mut poly, msg, aad.len());
                crate::chacha::xor_bytes(msg, &ks[..msg.len()]);
            }
            poly.finalize_into(tag_out);
            return;
        }

        // ── Blocks 0 + 1 in ONE kernel call (bulk prologue) ──
        // Block 0's first 32 bytes are the Poly1305 one-time key, block 1 is
        // the first message keystream. A single quad serves both (the old
        // shape computed 3 blocks to use 2: one 2-block call for the key
        // discarding block 1, then a separate call for the message).
        // Seal XORs block 1 straight into the message; open parks the
        // keystream in `ks1` — the MAC must read pristine ciphertext before
        // the XOR destroys it, so it is applied after absorbing msg[..BLOCK].
        let mut key32 = [0u8; 32];
        let mut ks1 = [0u8; BLOCK];
        if SEAL {
            O::gen_key_xor2(state, &mut key32, &mut msg[..BLOCK]);
        } else {
            O::gen_key_xor2(state, &mut key32, &mut ks1);
        }
        let mut poly = poly_with_aad::<O::Poly>(&mut key32, aad);

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

        // (msg.len() > SMALL_MAX ≥ BLOCK + s always holds: no tiny branch.)
        // Prologue: the poly alignment window. Block 1 was consumed by
        // the fused call above; when s > 0 the first 4-block window
        // straddles blocks 1 and 2.
        if SEAL {
            // (block 1 was xored into msg by the fused call)
            poly.update(&msg[..s]);
            poly_off = s;
        } else {
            poly.update(&msg[..s]);
            poly.absorb4(msg[s..s + BLOCK].try_into().unwrap());
            crate::chacha::xor_bytes(&mut msg[..BLOCK], &ks1);
            poly_off = s + BLOCK;
        }
        off = BLOCK;
        // Fused bulk loop.
        let batch = O::CHACHA_BATCH * BLOCK;
        while msg.len() - off >= batch {
            if SEAL {
                // Ciphertext from earlier batches not yet absorbed.
                let avail = off - poly_off;
                if poly.pending_blocks() == 0 && avail >= batch {
                    // Fused bulk run: the MAC over earlier batches is
                    // interleaved with the cipher batch-by-batch.
                    let (no, np) = O::seal_bulk(state, msg, off, poly_off, &mut poly);
                    off = no;
                    poly_off = np;
                } else {
                    O::chacha_xor_batch(state, &mut msg[off..off + batch]);
                    off += batch;
                    // Absorb only up to the START of the batch just written:
                    // reading freshly-stored zmm lines immediately stalls on
                    // store→load forwarding; lagging one batch lets the
                    // stores retire first (OpenSSL's fused loop pipelines
                    // the same way).
                    let target = off - batch;
                    let base = (target - poly_off) & !(BLOCK - 1);
                    // Choose n so the MAC's window cache ends EMPTY —
                    // a later iteration can then take the fused step.
                    let n = if poly.pending_blocks() % 8 == 4 && base >= BLOCK {
                        ((base - BLOCK) & !(2 * BLOCK - 1)) + BLOCK
                    } else {
                        base & !(2 * BLOCK - 1)
                    };
                    if n > 0 {
                        poly.absorb_blocks(&msg[poly_off..poly_off + n]);
                        poly_off += n;
                    }
                }
            } else {
                // Open: poly must lead the xor cursor — every window it
                // absorbs must still be unmodified ciphertext. Poly
                // windows sit on an `s`-shifted grid that never aligns
                // with the ChaCha 64-byte grid (s ≠ 0), so the xor cursor
                // only ever advances up to the largest 64-byte boundary
                // the poly has already absorbed past.
                if O::FUSED_OPEN {
                    // Normalize the MAC window cache: absorb one extra
                    // 64-byte window (still pristine ciphertext — the xor
                    // cursor is behind it) so the wide batching aligns,
                    // then run the fused bulk loop.
                    if poly.pending_blocks() % 8 == 4 && poly_off + BLOCK <= msg.len() {
                        poly.absorb_blocks(&msg[poly_off..poly_off + BLOCK]);
                        poly_off += BLOCK;
                    }
                    if poly.pending_blocks() == 0
                        && msg.len() - off >= batch
                        && msg.len() - poly_off >= batch
                    {
                        let (no, np) = O::open_bulk(state, msg, off, poly_off, &mut poly);
                        off = no;
                        poly_off = np;
                        continue;
                    }
                }
                // Poly-leads step (the whole body for non-fused backends).
                let ahead = (off + 2 * batch).min(msg.len());
                if poly_off < ahead {
                    let n = (ahead - poly_off) & !(BLOCK - 1);
                    if n > 0 {
                        poly.absorb_blocks(&msg[poly_off..poly_off + n]);
                        poly_off += n;
                    }
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
        // First drain the bulk windows the pipelined seal absorb deferred
        // (poly_off may lag `off` by up to one batch).
        if poly_off < off {
            let n = (off - poly_off) & !(BLOCK - 1);
            if n > 0 {
                poly.absorb_blocks(&msg[poly_off..poly_off + n]);
                poly_off += n;
            }
        }
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
        debug_assert_eq!(off, msg.len());
        debug_assert_eq!(poly_off, msg.len());

        // ── Zero-pad the ciphertext, absorb lengths, emit tag ──
        finish_tag(&mut poly, aad.len(), msg.len(), tag_out);
    }
}
