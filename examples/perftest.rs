use chacha20poly1305_simd::{ChaCha20Poly1305, active_backend};

fn main() {
    println!("backend: {}", active_backend());
    let cipher = ChaCha20Poly1305::new([0x42u8; 32]);
    let nonce = [7u8; 12];
    let mut buf = vec![0u8; 1024];
    // warm
    for _ in 0..1000 {
        let _t = cipher
            .encrypt_in_place_detached(&nonce, b"aad123456789012", &mut buf)
            .unwrap();
    }
    let n = 200_000;
    let start = std::time::Instant::now();
    for _ in 0..n {
        let _t = cipher
            .encrypt_in_place_detached(&nonce, b"aad123456789012", &mut buf)
            .unwrap();
    }
    let d = start.elapsed();
    #[allow(clippy::cast_precision_loss)]
    let ns = d.as_nanos() as f64;
    println!(
        "seal 1KiB: {:.1} ns/op ({:.0} MiB/s)",
        ns / f64::from(n),
        1024.0 * f64::from(n) / d.as_secs_f64() / 1_048_576.0
    );
}
