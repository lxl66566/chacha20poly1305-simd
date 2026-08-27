// OpenSSL EVP ChaCha20-Poly1305 throughput benchmark.
// Build: cc -O2 -I<openssl>/include openssl_bench.c <openssl>/libcrypto.a -lpthread -ldl -o openssl_bench
//
// Per iteration: full AEAD seal (ctx re-init with key+nonce, AAD, encrypt,
// finalize, read tag) over an in-place buffer, matching the in-place detached
// API semantics of the Rust benchmark. Throughput = size / median ns per iter.
#include <openssl/evp.h>
#include <openssl/crypto.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void bench_size(EVP_CIPHER_CTX *ctx, const EVP_CIPHER *cipher,
                       const unsigned char *key, const unsigned char *nonce,
                       const unsigned char *aad, size_t aad_len,
                       unsigned char *buf, size_t n) {
    // Warmup: ~50ms
    double t0 = now_ns();
    unsigned long warmup = 0;
    while (now_ns() - t0 < 50e6) {
        int outl = 0, finl = 0;
        unsigned char tag[16];
        EVP_EncryptInit_ex(ctx, cipher, NULL, key, nonce);
        EVP_EncryptUpdate(ctx, NULL, &outl, aad, (int)aad_len);
        EVP_EncryptUpdate(ctx, buf, &outl, buf, (int)n);
        EVP_EncryptFinal_ex(ctx, buf, &finl);
        EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_GET_TAG, 16, tag);
        warmup++;
    }
    // Measurement: 7 trials x 200ms, take the median per-iter time.
    double samples[7];
    for (int t = 0; t < 7; t++) {
        unsigned long iters = warmup * 4;
        double start = now_ns();
        for (unsigned long i = 0; i < iters; i++) {
            int outl = 0, finl = 0;
            unsigned char tag[16];
            EVP_EncryptInit_ex(ctx, cipher, NULL, key, nonce);
            EVP_EncryptUpdate(ctx, NULL, &outl, aad, (int)aad_len);
            EVP_EncryptUpdate(ctx, buf, &outl, buf, (int)n);
            EVP_EncryptFinal_ex(ctx, buf, &finl);
            EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_GET_TAG, 16, tag);
        }
        samples[t] = (now_ns() - start) / (double)iters;
    }
    qsort(samples, 7, sizeof(double), cmp_d);
    printf("%zu,%.1f\n", n, samples[3]);
}

int main(void) {
    static const size_t sizes[] = {16, 31, 64, 128, 256, 512, 1024, 4096,
                                   16384, 65536, 262144, 1048576};
    unsigned char key[32], nonce[12], aad[16];
    memset(key, 0x24, sizeof(key));
    memset(nonce, 0x42, sizeof(nonce));
    memset(aad, 0xaa, sizeof(aad));

    const EVP_CIPHER *cipher = EVP_chacha20_poly1305();
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx || !cipher) return 1;

    fprintf(stderr, "%s\n", OpenSSL_version(OPENSSL_VERSION));
    printf("size,ns_per_iter\n");
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
        unsigned char *buf = malloc(sizes[i]);
        memset(buf, 0, sizes[i]);
        bench_size(ctx, cipher, key, nonce, aad, sizeof(aad), buf, sizes[i]);
        free(buf);
    }
    EVP_CIPHER_CTX_free(ctx);
    return 0;
}
