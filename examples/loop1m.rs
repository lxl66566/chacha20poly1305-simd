use std::hint::black_box;
use chacha20poly1305_simd::ChaCha20Poly1305;
fn main() {
    let key = [0x24u8; 32];
    let nonce = [0x42u8; 12];
    let aad = [0xaau8; 16];
    let c = ChaCha20Poly1305::new(&key);
    let mut buf = vec![0u8; 1 << 20];
    let t0 = std::time::Instant::now();
    let mut n = 0u64;
    while t0.elapsed().as_secs() < 6 {
        let tag = c.encrypt_in_place_detached(black_box(&nonce), black_box(&aad), black_box(&mut buf)).unwrap();
        black_box(tag);
        n += 1;
    }
    eprintln!("{:.3} ns/B", t0.elapsed().as_secs_f64() / n as f64 / 1048576.0 * 1e9);
}
