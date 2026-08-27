# chacha20poly1305-simd

Pure Rust implementation of [ChaCha20-Poly1305](https://tools.ietf.org/html/rfc8439) (RFC 8439) and XChaCha20-Poly1305 AEAD, with hand-written SIMD for x86_64 / aarch64:

- AVX-512 / AVX2 / SSE2 / NEON / scalar backends, runtime-detected
- `no_std` compatible, optional `zeroize`

> [!NOTE]
> Disclaimer: this project was developed with AI assistance and has not undergone a security audit

## Usage

```rust
use chacha20poly1305_simd::{ChaCha20Poly1305, Key, Nonce, Tag};

let key = [0x42u8; 32];
let nonce = [7u8; 12];
let cipher = ChaCha20Poly1305::new(&key);

let mut buffer = *b"hello world";
let tag: Tag = cipher.encrypt_in_place_detached(&nonce, b"aad", &mut buffer);
assert_ne!(&buffer, b"hello world");

cipher
    .decrypt_in_place_detached(&nonce, b"aad", &mut buffer, &tag)
    .unwrap();
assert_eq!(&buffer, b"hello world");
```

Allocating variants append / verify a 16-byte tag at the end of the ciphertext, matching the upstream wire format:

```rust
use chacha20poly1305_simd::{XChaCha20Poly1305, XNonce};

let cipher = XChaCha20Poly1305::new(&[1u8; 32]);
let nonce = [2u8; 24];
let ct = cipher.encrypt(&nonce, b"header", b"secret payload");
assert_eq!(ct.len(), b"secret payload".len() + 16);
assert_eq!(cipher.decrypt(&nonce, b"header", &ct).unwrap(), b"secret payload");
```

## Features

| feature   | default | description                                        |
| --------- | ------- | -------------------------------------------------- |
| `std`     | ✓       | runtime CPU detection (otherwise scalar backend)   |
| `alloc`   | ✓       | allocating API                                     |
| `zeroize` | –       | zeroize keys and intermediate secrets on drop      |
| `hotpath` | –       | [hotpath](https://crates.io/crates/hotpath) probes |

## Backend selection

Runtime detection is the default (x86-64: avx512 → avx2 → sse2; aarch64: neon). A backend can also be forced at compile time via RUSTFLAGS (forcing a SIMD backend is a promise that the target CPU supports that instruction set; other backend kernels are no longer compiled into the binary):

```sh
RUSTFLAGS='--cfg chacha20poly1305_backend="avx2" -Ctarget-feature=+avx2' cargo build --release
```

| backend  | target  | required target features   |
| -------- | ------- | -------------------------- |
| `soft`   | any     | –                          |
| `sse2`   | x86-64  | – (baseline ISA)           |
| `avx2`   | x86-64  | `+avx2`                    |
| `avx512` | x86-64  | `+avx2,+avx512f,+avx512vl` |
| `neon`   | aarch64 | – (baseline ISA)           |

## Performance

> Test environment: AMD Ryzen 9 7945HX (Zen 4) · `cargo bench --bench aead` (seal)  
> Baseline: RustCrypto `chacha20poly1305` 0.11. Both sides were benchmarked with backends pinned to the same instruction-set tier under identical `RUSTFLAGS`.

### x86_64

The table below lists this implementation's throughput and the speedup over RustCrypto in the same configuration:

| message | Scalar (soft) | speedup |       SSE2 | speedup |       AVX2 | speedup |    AVX-512 | speedup |
| ------: | ------------: | :-----: | ---------: | :-----: | ---------: | :-----: | ---------: | :-----: |
|    16 B |      58 MiB/s |  2.1×   |   68 MiB/s |  2.1×   |   61 MiB/s |  2.6×   |   56 MiB/s |  1.3×   |
|    64 B |     214 MiB/s |  2.1×   |  284 MiB/s |  2.3×   |  268 MiB/s |  3.0×   |  270 MiB/s |  1.7×   |
|   256 B |     356 MiB/s |  2.1×   |  496 MiB/s |  1.4×   |  614 MiB/s |  2.1×   |  633 MiB/s |  1.4×   |
|   1 KiB |     413 MiB/s |  2.0×   |  802 MiB/s |  1.6×   | 1.19 GiB/s |  2.7×   | 1.27 GiB/s |  1.9×   |
|   4 KiB |     431 MiB/s |  2.0×   |  945 MiB/s |  1.6×   | 1.98 GiB/s |  3.8×   | 2.23 GiB/s |  2.9×   |
|  64 KiB |     435 MiB/s |  2.0×   | 1000 MiB/s |  1.7×   | 2.49 GiB/s |  4.4×   | 2.91 GiB/s |  3.5×   |
|   1 MiB |     434 MiB/s |  1.9×   | 1007 MiB/s |  1.7×   | 2.52 GiB/s |  4.5×   | 2.96 GiB/s |  3.6×   |

RustCrypto's `poly1305` only supports soft/AVX2, so the benchmark combinations are:

- Scalar: ChaCha(soft) + Poly(soft)
- SSE2: ChaCha(sse2) + Poly(soft)
- AVX2: ChaCha(avx2) + Poly(avx2)
- AVX-512: ChaCha(avx-512) + Poly(avx2)

Core performance optimizations: fused ChaCha and Poly1305 pipeline (single pass, avoiding the multi-pass memory traffic of RustCrypto's keystream-then-MAC approach); an additional 4-block batch on the SSE2 tier.

### AArch64

> Measured under QEMU TCG emulation; the ratios roughly reflect the difference in instructions-per-byte / computational efficiency:

| message              | 64 B    | 256 B   | 1 KiB   | 4 KiB    | 64 KiB   | 1 MiB    |
| :------------------- | :------ | :------ | :------ | :------- | :------- | :------- |
| this impl throughput | 38 MB/s | 46 MB/s | 89 MB/s | 105 MB/s | 100 MB/s | 112 MB/s |
| speedup              | 8.9×    | 9.3×    | 3.6×    | 3.4×     | 2.9×     | 2.0×     |

On aarch64, RustCrypto only enables NEON for ChaCha; Poly1305 remains a scalar soft implementation.

## Verification

- RFC 8439 and XChaCha official test vectors (`cargo test`)
- Differential fuzzing against the RustCrypto implementation (`fuzz/run.sh`), asserting byte-exact ciphertext/tag equality and identical open verdicts. On x86_64 (soft/avx2/avx512 in parallel): a cumulative ≥ 211 million executions across the three backends (80.8M/72.2M/58.9M), 0 crashes.
- aarch64 cross-validation under QEMU user mode: randomized differential testing via `cargo run --release --example xcheck -- [iterations] [seed]`; the NEON backend passed 500 thousand differential cases under QEMU

## License

MIT OR Apache-2.0
