/**
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#ifndef _picohash_h_
#define _picohash_h_

#include <assert.h>
#include <inttypes.h>
#include <string.h>

#ifdef _WIN32
/* assume Windows is little endian */
#elif defined __BIG_ENDIAN__
#define _PICOHASH_BIG_ENDIAN
#elif defined __LITTLE_ENDIAN__
/* override */
#elif defined __BYTE_ORDER
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define _PICOHASH_BIG_ENDIAN
#endif
#else               // ! defined __LITTLE_ENDIAN__
#include <endian.h> // machine/endian.h
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define _PICOHASH_BIG_ENDIAN
#endif
#endif

#define PICOHASH_MD5_BLOCK_LENGTH 64
#define PICOHASH_MD5_DIGEST_LENGTH 16

typedef struct {
    uint_fast32_t lo, hi;
    uint_fast32_t a, b, c, d;
    unsigned char buffer[64];
    uint_fast32_t block[PICOHASH_MD5_DIGEST_LENGTH];
} _picohash_md5_ctx_t;

static void _picohash_md5_init(_picohash_md5_ctx_t *ctx);
static void _picohash_md5_update(_picohash_md5_ctx_t *ctx, const void *data, size_t size);
static void _picohash_md5_final(_picohash_md5_ctx_t *ctx, void *digest);

#define PICOHASH_SHA1_BLOCK_LENGTH 64
#define PICOHASH_SHA1_DIGEST_LENGTH 20

typedef struct {
    uint32_t buffer[PICOHASH_SHA1_BLOCK_LENGTH / 4];
    uint32_t state[PICOHASH_SHA1_DIGEST_LENGTH / 4];
    uint64_t byteCount;
    uint8_t bufferOffset;
} _picohash_sha1_ctx_t;

static void _picohash_sha1_init(_picohash_sha1_ctx_t *ctx);
static void _picohash_sha1_update(_picohash_sha1_ctx_t *ctx, const void *input, size_t len);
static void _picohash_sha1_final(_picohash_sha1_ctx_t *ctx, void *digest);

#define PICOHASH_SHA256_BLOCK_LENGTH 64
#define PICOHASH_SHA256_DIGEST_LENGTH 32
#define PICOHASH_SHA224_BLOCK_LENGTH PICOHASH_SHA256_BLOCK_LENGTH
#define PICOHASH_SHA224_DIGEST_LENGTH 28

typedef struct {
    uint64_t length;
    uint32_t state[PICOHASH_SHA256_DIGEST_LENGTH / 4];
    uint32_t curlen;
    unsigned char buf[PICOHASH_SHA256_BLOCK_LENGTH];
} _picohash_sha256_ctx_t;

static void _picohash_sha256_init(_picohash_sha256_ctx_t *ctx);
static void _picohash_sha256_update(_picohash_sha256_ctx_t *ctx, const void *data, size_t len);
static void _picohash_sha256_final(_picohash_sha256_ctx_t *ctx, void *digest);
static void _picohash_sha224_init(_picohash_sha256_ctx_t *ctx);
static void _picohash_sha224_final(_picohash_sha256_ctx_t *ctx, void *digest);

#define PICOHASH_MAX_BLOCK_LENGTH 64
#define PICOHASH_MAX_DIGEST_LENGTH 32

typedef struct {
    union {
        _picohash_md5_ctx_t _md5;
        _picohash_sha1_ctx_t _sha1;
        _picohash_sha256_ctx_t _sha256;
    };
    size_t block_length;
    size_t digest_length;
    void (*_reset)(void *ctx);
    void (*_update)(void *ctx, const void *input, size_t len);
    void (*_final)(void *ctx, void *digest);
    struct {
        unsigned char key[PICOHASH_MAX_BLOCK_LENGTH];
        void (*hash_reset)(void *ctx);
        void (*hash_final)(void *ctx, void *digest);
    } _hmac;
} picohash_ctx_t;

static void picohash_init_md5(picohash_ctx_t *ctx);
static void picohash_init_sha1(picohash_ctx_t *ctx);
static void picohash_init_sha224(picohash_ctx_t *ctx);
static void picohash_init_sha256(picohash_ctx_t *ctx);
static void picohash_update(picohash_ctx_t *ctx, const void *input, size_t len);
static void picohash_final(picohash_ctx_t *ctx, void *digest);
static void picohash_reset(picohash_ctx_t *ctx);

static void picohash_init_hmac(picohash_ctx_t *ctx, void (*initf)(picohash_ctx_t *), const void *key, size_t key_len);

/* following are private definitions */

/*
 * The basic MD5 functions.
 *
 * F is optimized compared to its RFC 1321 definition just like in Colin
 * Plumb's implementation.
 */
#define _PICOHASH_MD5_F(x, y, z) ((z) ^ ((x) & ((y) ^ (z))))
#define _PICOHASH_MD5_G(x, y, z) ((y) ^ ((z) & ((x) ^ (y))))
#define _PICOHASH_MD5_H(x, y, z) ((x) ^ (y) ^ (z))
#define _PICOHASH_MD5_I(x, y, z) ((y) ^ ((x) | ~(z)))

/*
 * The MD5 transformation for all four rounds.
 */
#define _PICOHASH_MD5_STEP(f, a, b, c, d, x, t, s)                                                                                 \
    (a) += f((b), (c), (d)) + (x) + (t);                                                                                           \
    (a) = (((a) << (s)) | (((a)&0xffffffff) >> (32 - (s))));                                                                       \
    (a) += (b);

/*
 * SET reads 4 input bytes in little-endian byte order and stores them
 * in a properly aligned word in host byte order.
 *
 * Paul-Louis Ageneau: Removed optimization for little-endian architectures
 * as it resulted in incorrect behavior when compiling with gcc optimizations.
 */
#define _PICOHASH_MD5_SET(n)                                                                                                       \
    (ctx->block[(n)] = (uint_fast32_t)ptr[(n)*4] | ((uint_fast32_t)ptr[(n)*4 + 1] << 8) | ((uint_fast32_t)ptr[(n)*4 + 2] << 16) |  \
                       ((uint_fast32_t)ptr[(n)*4 + 3] << 24))
#define _PICOHASH_MD5_GET(n) (ctx->block[(n)])

/*
 * This processes one or more 64-byte data blocks, but does NOT update
 * the bit counters.  There're no alignment requirements.
 */
static const void *_picohash_md5_body(_picohash_md5_ctx_t *ctx, const void *data, size_t size)
{
    const unsigned char *ptr;
    uint_fast32_t a, b, c, d;
    uint_fast32_t saved_a, saved_b, saved_c, saved_d;

    ptr = data;

    a = ctx->a;
    b = ctx->b;
    c = ctx->c;
    d = ctx->d;

    do {
        saved_a = a;
        saved_b = b;
        saved_c = c;
        saved_d = d;

        /* Round 1 */
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_F, a, b, c, d, _PICOHASH_MD5_SET(0), 0xd76aa478, 7)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_F, d, a, b, c, _PICOHASH_MD5_SET(1), 0xe8c7b756, 12)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_F, c, d, a, b, _PICOHASH_MD5_SET(2), 0x242070db, 17)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_F, b, c, d, a, _PICOHASH_MD5_SET(3), 0xc1bdceee, 22)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_F, a, b, c, d, _PICOHASH_MD5_SET(4), 0xf57c0faf, 7)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_F, d, a, b, c, _PICOHASH_MD5_SET(5), 0x4787c62a, 12)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_F, c, d, a, b, _PICOHASH_MD5_SET(6), 0xa8304613, 17)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_F, b, c, d, a, _PICOHASH_MD5_SET(7), 0xfd469501, 22)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_F, a, b, c, d, _PICOHASH_MD5_SET(8), 0x698098d8, 7)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_F, d, a, b, c, _PICOHASH_MD5_SET(9), 0x8b44f7af, 12)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_F, c, d, a, b, _PICOHASH_MD5_SET(10), 0xffff5bb1, 17)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_F, b, c, d, a, _PICOHASH_MD5_SET(11), 0x895cd7be, 22)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_F, a, b, c, d, _PICOHASH_MD5_SET(12), 0x6b901122, 7)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_F, d, a, b, c, _PICOHASH_MD5_SET(13), 0xfd987193, 12)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_F, c, d, a, b, _PICOHASH_MD5_SET(14), 0xa679438e, 17)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_F, b, c, d, a, _PICOHASH_MD5_SET(15), 0x49b40821, 22)

        /* Round 2 */
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_G, a, b, c, d, _PICOHASH_MD5_GET(1), 0xf61e2562, 5)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_G, d, a, b, c, _PICOHASH_MD5_GET(6), 0xc040b340, 9)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_G, c, d, a, b, _PICOHASH_MD5_GET(11), 0x265e5a51, 14)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_G, b, c, d, a, _PICOHASH_MD5_GET(0), 0xe9b6c7aa, 20)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_G, a, b, c, d, _PICOHASH_MD5_GET(5), 0xd62f105d, 5)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_G, d, a, b, c, _PICOHASH_MD5_GET(10), 0x02441453, 9)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_G, c, d, a, b, _PICOHASH_MD5_GET(15), 0xd8a1e681, 14)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_G, b, c, d, a, _PICOHASH_MD5_GET(4), 0xe7d3fbc8, 20)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_G, a, b, c, d, _PICOHASH_MD5_GET(9), 0x21e1cde6, 5)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_G, d, a, b, c, _PICOHASH_MD5_GET(14), 0xc33707d6, 9)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_G, c, d, a, b, _PICOHASH_MD5_GET(3), 0xf4d50d87, 14)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_G, b, c, d, a, _PICOHASH_MD5_GET(8), 0x455a14ed, 20)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_G, a, b, c, d, _PICOHASH_MD5_GET(13), 0xa9e3e905, 5)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_G, d, a, b, c, _PICOHASH_MD5_GET(2), 0xfcefa3f8, 9)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_G, c, d, a, b, _PICOHASH_MD5_GET(7), 0x676f02d9, 14)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_G, b, c, d, a, _PICOHASH_MD5_GET(12), 0x8d2a4c8a, 20)

        /* Round 3 */
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_H, a, b, c, d, _PICOHASH_MD5_GET(5), 0xfffa3942, 4)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_H, d, a, b, c, _PICOHASH_MD5_GET(8), 0x8771f681, 11)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_H, c, d, a, b, _PICOHASH_MD5_GET(11), 0x6d9d6122, 16)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_H, b, c, d, a, _PICOHASH_MD5_GET(14), 0xfde5380c, 23)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_H, a, b, c, d, _PICOHASH_MD5_GET(1), 0xa4beea44, 4)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_H, d, a, b, c, _PICOHASH_MD5_GET(4), 0x4bdecfa9, 11)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_H, c, d, a, b, _PICOHASH_MD5_GET(7), 0xf6bb4b60, 16)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_H, b, c, d, a, _PICOHASH_MD5_GET(10), 0xbebfbc70, 23)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_H, a, b, c, d, _PICOHASH_MD5_GET(13), 0x289b7ec6, 4)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_H, d, a, b, c, _PICOHASH_MD5_GET(0), 0xeaa127fa, 11)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_H, c, d, a, b, _PICOHASH_MD5_GET(3), 0xd4ef3085, 16)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_H, b, c, d, a, _PICOHASH_MD5_GET(6), 0x04881d05, 23)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_H, a, b, c, d, _PICOHASH_MD5_GET(9), 0xd9d4d039, 4)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_H, d, a, b, c, _PICOHASH_MD5_GET(12), 0xe6db99e5, 11)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_H, c, d, a, b, _PICOHASH_MD5_GET(15), 0x1fa27cf8, 16)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_H, b, c, d, a, _PICOHASH_MD5_GET(2), 0xc4ac5665, 23)

        /* Round 4 */
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_I, a, b, c, d, _PICOHASH_MD5_GET(0), 0xf4292244, 6)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_I, d, a, b, c, _PICOHASH_MD5_GET(7), 0x432aff97, 10)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_I, c, d, a, b, _PICOHASH_MD5_GET(14), 0xab9423a7, 15)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_I, b, c, d, a, _PICOHASH_MD5_GET(5), 0xfc93a039, 21)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_I, a, b, c, d, _PICOHASH_MD5_GET(12), 0x655b59c3, 6)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_I, d, a, b, c, _PICOHASH_MD5_GET(3), 0x8f0ccc92, 10)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_I, c, d, a, b, _PICOHASH_MD5_GET(10), 0xffeff47d, 15)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_I, b, c, d, a, _PICOHASH_MD5_GET(1), 0x85845dd1, 21)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_I, a, b, c, d, _PICOHASH_MD5_GET(8), 0x6fa87e4f, 6)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_I, d, a, b, c, _PICOHASH_MD5_GET(15), 0xfe2ce6e0, 10)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_I, c, d, a, b, _PICOHASH_MD5_GET(6), 0xa3014314, 15)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_I, b, c, d, a, _PICOHASH_MD5_GET(13), 0x4e0811a1, 21)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_I, a, b, c, d, _PICOHASH_MD5_GET(4), 0xf7537e82, 6)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_I, d, a, b, c, _PICOHASH_MD5_GET(11), 0xbd3af235, 10)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_I, c, d, a, b, _PICOHASH_MD5_GET(2), 0x2ad7d2bb, 15)
        _PICOHASH_MD5_STEP(_PICOHASH_MD5_I, b, c, d, a, _PICOHASH_MD5_GET(9), 0xeb86d391, 21)

        a += saved_a;
        b += saved_b;
        c += saved_c;
        d += saved_d;

        ptr += 64;
    } while (size -= 64);

    ctx->a = a;
    ctx->b = b;
    ctx->c = c;
    ctx->d = d;

    return ptr;
}

inline void _picohash_md5_init(_picohash_md5_ctx_t *ctx)
{
    ctx->a = 0x67452301;
    ctx->b = 0xefcdab89;
    ctx->c = 0x98badcfe;
    ctx->d = 0x10325476;

    ctx->lo = 0;
    ctx->hi = 0;
}

inline void _picohash_md5_update(_picohash_md5_ctx_t *ctx, const void *data, size_t size)
{
    uint_fast32_t saved_lo;
    unsigned long used, free;

    saved_lo = ctx->lo;
    if ((ctx->lo = (saved_lo + size) & 0x1fffffff) < saved_lo)
        ctx->hi++;
    ctx->hi += (uint_fast32_t)(size >> 29);

    used = saved_lo & 0x3f;

    if (used) {
        free = 64 - used;

        if (size < free) {
            memcpy(&ctx->buffer[used], data, size);
            return;
        }

        memcpy(&ctx->buffer[used], data, free);
        data = (const unsigned char *)data + free;
        size -= free;
        _picohash_md5_body(ctx, ctx->buffer, 64);
    }

    if (size >= 64) {
        data = _picohash_md5_body(ctx, data, size & ~(unsigned long)0x3f);
        size &= 0x3f;
    }

    memcpy(ctx->buffer, data, size);
}

inline void _picohash_md5_final(_picohash_md5_ctx_t *ctx, void *_digest)
{
    unsigned char *digest = _digest;
    unsigned long used, free;

    used = ctx->lo & 0x3f;

    ctx->buffer[used++] = 0x80;

    free = 64 - used;

    if (free < 8) {
        memset(&ctx->buffer[used], 0, free);
        _picohash_md5_body(ctx, ctx->buffer, 64);
        used = 0;
        free = 64;
    }

    memset(&ctx->buffer[used], 0, free - 8);

    ctx->lo <<= 3;
    ctx->buffer[56] = ctx->lo;
    ctx->buffer[57] = ctx->lo >> 8;
    ctx->buffer[58] = ctx->lo >> 16;
    ctx->buffer[59] = ctx->lo >> 24;
    ctx->buffer[60] = ctx->hi;
    ctx->buffer[61] = ctx->hi >> 8;
    ctx->buffer[62] = ctx->hi >> 16;
    ctx->buffer[63] = ctx->hi >> 24;

    _picohash_md5_body(ctx, ctx->buffer, 64);

    digest[0] = ctx->a;
    digest[1] = ctx->a >> 8;
    digest[2] = ctx->a >> 16;
    digest[3] = ctx->a >> 24;
    digest[4] = ctx->b;
    digest[5] = ctx->b >> 8;
    digest[6] = ctx->b >> 16;
    digest[7] = ctx->b >> 24;
    digest[8] = ctx->c;
    digest[9] = ctx->c >> 8;
    digest[10] = ctx->c >> 16;
    digest[11] = ctx->c >> 24;
    digest[12] = ctx->d;
    digest[13] = ctx->d >> 8;
    digest[14] = ctx->d >> 16;
    digest[15] = ctx->d >> 24;

    memset(ctx, 0, sizeof(*ctx));
}

#define _PICOHASH_SHA1_K0 0x5a827999
#define _PICOHASH_SHA1_K20 0x6ed9eba1
#define _PICOHASH_SHA1_K40 0x8f1bbcdc
#define _PICOHASH_SHA1_K60 0xca62c1d6

static inline uint32_t _picohash_sha1_rol32(uint32_t number, uint8_t bits)
{
    return ((number << bits) | (number >> (32 - bits)));
}

static inline void _picohash_sha1_hash_block(_picohash_sha1_ctx_t *s)
{
    uint8_t i;
    uint32_t a, b, c, d, e, t;

    a = s->state[0];
    b = s->state[1];
    c = s->state[2];
    d = s->state[3];
    e = s->state[4];
    for (i = 0; i < 80; i++) {
        if (i >= 16) {
            t = s->buffer[(i + 13) & 15] ^ s->buffer[(i + 8) & 15] ^ s->buffer[(i + 2) & 15] ^ s->buffer[i & 15];
            s->buffer[i & 15] = _picohash_sha1_rol32(t, 1);
        }
        if (i < 20) {
            t = (d ^ (b & (c ^ d))) + _PICOHASH_SHA1_K0;
        } else if (i < 40) {
            t = (b ^ c ^ d) + _PICOHASH_SHA1_K20;
        } else if (i < 60) {
            t = ((b & c) | (d & (b | c))) + _PICOHASH_SHA1_K40;
        } else {
            t = (b ^ c ^ d) + _PICOHASH_SHA1_K60;
        }
        t += _picohash_sha1_rol32(a, 5) + e + s->buffer[i & 15];
        e = d;
        d = c;
        c = _picohash_sha1_rol32(b, 30);
        b = a;
        a = t;
    }
    s->state[0] += a;
    s->state[1] += b;
    s->state[2] += c;
    s->state[3] += d;
    s->state[4] += e;
}

static inline void _picohash_sha1_add_uncounted(_picohash_sha1_ctx_t *s, uint8_t data)
{
    uint8_t *const b = (uint8_t *)s->buffer;
#ifdef _PICOHASH_BIG_ENDIAN
    b[s->bufferOffset] = data;
#else
    b[s->bufferOffset ^ 3] = data;
#endif
    s->bufferOffset++;
    if (s->bufferOffset == PICOHASH_SHA1_BLOCK_LENGTH) {
        _picohash_sha1_hash_block(s);
        s->bufferOffset = 0;
    }
}

inline void _picohash_sha1_init(_picohash_sha1_ctx_t *s)
{
    s->state[0] = 0x67452301;
    s->state[1] = 0xefcdab89;
    s->state[2] = 0x98badcfe;
    s->state[3] = 0x10325476;
    s->state[4] = 0xc3d2e1f0;
    s->byteCount = 0;
    s->bufferOffset = 0;
}

inline void _picohash_sha1_update(_picohash_sha1_ctx_t *s, const void *_data, size_t len)
{
    const uint8_t *data = _data;
    for (; len != 0; --len) {
        ++s->byteCount;
        _picohash_sha1_add_uncounted(s, *data++);
    }
}

inline void _picohash_sha1_final(_picohash_sha1_ctx_t *s, void *digest)
{
    // Pad with 0x80 followed by 0x00 until the end of the block
    _picohash_sha1_add_uncounted(s, 0x80);
    while (s->bufferOffset != 56)
        _picohash_sha1_add_uncounted(s, 0x00);

    // Append length in the last 8 bytes
    _picohash_sha1_add_uncounted(s, (uint8_t)(s->byteCount >> 53)); // Shifting to multiply by 8
    _picohash_sha1_add_uncounted(s, (uint8_t)(s->byteCount >> 45)); // as SHA-1 supports bitstreams
    _picohash_sha1_add_uncounted(s, (uint8_t)(s->byteCount >> 37)); // as well as byte.
    _picohash_sha1_add_uncounted(s, (uint8_t)(s->byteCount >> 29));
    _picohash_sha1_add_uncounted(s, (uint8_t)(s->byteCount >> 21));
    _picohash_sha1_add_uncounted(s, (uint8_t)(s->byteCount >> 13));
    _picohash_sha1_add_uncounted(s, (uint8_t)(s->byteCount >> 5));
    _picohash_sha1_add_uncounted(s, (uint8_t)(s->byteCount << 3));

#ifndef SHA_BIG_ENDIAN
    { // Swap byte order back
        int i;
        for (i = 0; i < 5; i++) {
            s->state[i] = (((s->state[i]) << 24) & 0xff000000) | (((s->state[i]) << 8) & 0x00ff0000) |
                          (((s->state[i]) >> 8) & 0x0000ff00) | (((s->state[i]) >> 24) & 0x000000ff);
        }
    }
#endif

    memcpy(digest, s->state, sizeof(s->state));
}

#define _picohash_sha256_ch(x, y, z) (z ^ (x & (y ^ z)))
#define _picohash_sha256_maj(x, y, z) (((x | y) & z) | (x & y))
#define _picohash_sha256_s(x, y)                                                                                                   \
    (((((uint32_t)(x)&0xFFFFFFFFUL) >> (uint32_t)((y)&31)) | ((uint32_t)(x) << (uint32_t)(32 - ((y)&31)))) & 0xFFFFFFFFUL)
#define _picohash_sha256_r(x, n) (((x)&0xFFFFFFFFUL) >> (n))
#define _picohash_sha256_sigma0(x) (_picohash_sha256_s(x, 2) ^ _picohash_sha256_s(x, 13) ^ _picohash_sha256_s(x, 22))
#define _picohash_sha256_sigma1(x) (_picohash_sha256_s(x, 6) ^ _picohash_sha256_s(x, 11) ^ _picohash_sha256_s(x, 25))
#define _picohash_sha256_gamma0(x) (_picohash_sha256_s(x, 7) ^ _picohash_sha256_s(x, 18) ^ _picohash_sha256_r(x, 3))
#define _picohash_sha256_gamma1(x) (_picohash_sha256_s(x, 17) ^ _picohash_sha256_s(x, 19) ^ _picohash_sha256_r(x, 10))
#define _picohash_sha256_rnd(a, b, c, d, e, f, g, h, i)                                                                            \
    t0 = h + _picohash_sha256_sigma1(e) + _picohash_sha256_ch(e, f, g) + K[i] + W[i];                                              \
    t1 = _picohash_sha256_sigma0(a) + _picohash_sha256_maj(a, b, c);                                                               \
    d += t0;                                                                                                                       \
    h = t0 + t1;

static inline void _picohash_sha256_compress(_picohash_sha256_ctx_t *ctx, unsigned char *buf)
{
    static const uint32_t K[64] = {
        0x428a2f98UL, 0x71374491UL, 0xb5c0fbcfUL, 0xe9b5dba5UL, 0x3956c25bUL, 0x59f111f1UL, 0x923f82a4UL, 0xab1c5ed5UL,
        0xd807aa98UL, 0x12835b01UL, 0x243185beUL, 0x550c7dc3UL, 0x72be5d74UL, 0x80deb1feUL, 0x9bdc06a7UL, 0xc19bf174UL,
        0xe49b69c1UL, 0xefbe4786UL, 0x0fc19dc6UL, 0x240ca1ccUL, 0x2de92c6fUL, 0x4a7484aaUL, 0x5cb0a9dcUL, 0x76f988daUL,
        0x983e5152UL, 0xa831c66dUL, 0xb00327c8UL, 0xbf597fc7UL, 0xc6e00bf3UL, 0xd5a79147UL, 0x06ca6351UL, 0x14292967UL,
        0x27b70a85UL, 0x2e1b2138UL, 0x4d2c6dfcUL, 0x53380d13UL, 0x650a7354UL, 0x766a0abbUL, 0x81c2c92eUL, 0x92722c85UL,
        0xa2bfe8a1UL, 0xa81a664bUL, 0xc24b8b70UL, 0xc76c51a3UL, 0xd192e819UL, 0xd6990624UL, 0xf40e3585UL, 0x106aa070UL,
        0x19a4c116UL, 0x1e376c08UL, 0x2748774cUL, 0x34b0bcb5UL, 0x391c0cb3UL, 0x4ed8aa4aUL, 0x5b9cca4fUL, 0x682e6ff3UL,
        0x748f82eeUL, 0x78a5636fUL, 0x84c87814UL, 0x8cc70208UL, 0x90befffaUL, 0xa4506cebUL, 0xbef9a3f7UL, 0xc67178f2UL};
    uint32_t S[8], W[64], t, t0, t1;
    int i;

    /* copy state into S */
    for (i = 0; i < 8; i++)
        S[i] = ctx->state[i];

    /* copy the state into 512-bits into W[0..15] */
    for (i = 0; i < 16; i++)
        W[i] =
            (uint32_t)buf[4 * i] << 24 | (uint32_t)buf[4 * i + 1] << 16 | (uint32_t)buf[4 * i + 2] << 8 | (uint32_t)buf[4 * i + 3];

    /* fill W[16..63] */
    for (i = 16; i < 64; i++)
        W[i] = _picohash_sha256_gamma1(W[i - 2]) + W[i - 7] + _picohash_sha256_gamma0(W[i - 15]) + W[i - 16];

    /* Compress */
    for (i = 0; i < 64; ++i) {
        _picohash_sha256_rnd(S[0], S[1], S[2], S[3], S[4], S[5], S[6], S[7], i);
        t = S[7];
        S[7] = S[6];
        S[6] = S[5];
        S[5] = S[4];
        S[4] = S[3];
        S[3] = S[2];
        S[2] = S[1];
        S[1] = S[0];
        S[0] = t;
    }

    /* feedback */
    for (i = 0; i < 8; i++)
        ctx->state[i] = ctx->state[i] + S[i];
}

static inline void _picohash_sha256_do_final(_picohash_sha256_ctx_t *ctx, void *digest, size_t len)
{
    unsigned char *out = digest;
    size_t i;

    /* increase the length of the message */
    ctx->length += ctx->curlen * 8;

    /* append the '1' bit */
    ctx->buf[ctx->curlen++] = (unsigned char)0x80;

    /* if the length is currently above 56 bytes we append zeros
     * then compress.  Then we can fall back to padding zeros and length
     * encoding like normal.
     */
    if (ctx->curlen > 56) {
        while (ctx->curlen < 64) {
            ctx->buf[ctx->curlen++] = (unsigned char)0;
        }
        _picohash_sha256_compress(ctx, ctx->buf);
        ctx->curlen = 0;
    }

    /* pad upto 56 bytes of zeroes */
    while (ctx->curlen < 56) {
        ctx->buf[ctx->curlen++] = (unsigned char)0;
    }

    /* store length */
    for (i = 0; i != 8; ++i)
        ctx->buf[56 + i] = (unsigned char)(ctx->length >> (56 - 8 * i));
    _picohash_sha256_compress(ctx, ctx->buf);

    /* copy output */
    for (i = 0; i != len / 4; ++i) {
        out[i * 4] = ctx->state[i] >> 24;
        out[i * 4 + 1] = ctx->state[i] >> 16;
        out[i * 4 + 2] = ctx->state[i] >> 8;
        out[i * 4 + 3] = ctx->state[i];
    }
}

inline void _picohash_sha256_init(_picohash_sha256_ctx_t *ctx)
{
    ctx->curlen = 0;
    ctx->length = 0;
    ctx->state[0] = 0x6A09E667UL;
    ctx->state[1] = 0xBB67AE85UL;
    ctx->state[2] = 0x3C6EF372UL;
    ctx->state[3] = 0xA54FF53AUL;
    ctx->state[4] = 0x510E527FUL;
    ctx->state[5] = 0x9B05688CUL;
    ctx->state[6] = 0x1F83D9ABUL;
    ctx->state[7] = 0x5BE0CD19UL;
}

inline void _picohash_sha256_update(_picohash_sha256_ctx_t *ctx, const void *data, size_t len)
{
    const unsigned char *in = data;
    size_t n;

    while (len > 0) {
        if (ctx->curlen == 0 && len >= PICOHASH_SHA256_BLOCK_LENGTH) {
            _picohash_sha256_compress(ctx, (unsigned char *)in);
            ctx->length += PICOHASH_SHA256_BLOCK_LENGTH * 8;
            in += PICOHASH_SHA256_BLOCK_LENGTH;
            len -= PICOHASH_SHA256_BLOCK_LENGTH;
        } else {
            n = PICOHASH_SHA256_BLOCK_LENGTH - ctx->curlen;
            if (n > len)
                n = len;
            memcpy(ctx->buf + ctx->curlen, in, n);
            ctx->curlen += (uint32_t)n;
            in += n;
            len -= n;
            if (ctx->curlen == 64) {
                _picohash_sha256_compress(ctx, ctx->buf);
                ctx->length += 8 * PICOHASH_SHA256_BLOCK_LENGTH;
                ctx->curlen = 0;
            }
        }
    }
}

inline void _picohash_sha256_final(_picohash_sha256_ctx_t *ctx, void *digest)
{
    _picohash_sha256_do_final(ctx, digest, PICOHASH_SHA256_DIGEST_LENGTH);
}

inline void _picohash_sha224_init(_picohash_sha256_ctx_t *ctx)
{
    ctx->curlen = 0;
    ctx->length = 0;
    ctx->state[0] = 0xc1059ed8UL;
    ctx->state[1] = 0x367cd507UL;
    ctx->state[2] = 0x3070dd17UL;
    ctx->state[3] = 0xf70e5939UL;
    ctx->state[4] = 0xffc00b31UL;
    ctx->state[5] = 0x68581511UL;
    ctx->state[6] = 0x64f98fa7UL;
    ctx->state[7] = 0xbefa4fa4UL;
}

inline void _picohash_sha224_final(_picohash_sha256_ctx_t *ctx, void *digest)
{
    _picohash_sha256_do_final(ctx, digest, PICOHASH_SHA224_DIGEST_LENGTH);
}

inline void picohash_init_md5(picohash_ctx_t *ctx)
{
    ctx->block_length = PICOHASH_MD5_BLOCK_LENGTH;
    ctx->digest_length = PICOHASH_MD5_DIGEST_LENGTH;
    ctx->_reset = (void *)_picohash_md5_init;
    ctx->_update = (void *)_picohash_md5_update;
    ctx->_final = (void *)_picohash_md5_final;

    _picohash_md5_init(&ctx->_md5);
}

inline void picohash_init_sha1(picohash_ctx_t *ctx)
{
    ctx->block_length = PICOHASH_SHA1_BLOCK_LENGTH;
    ctx->digest_length = PICOHASH_SHA1_DIGEST_LENGTH;
    ctx->_reset = (void *)_picohash_sha1_init;
    ctx->_update = (void *)_picohash_sha1_update;
    ctx->_final = (void *)_picohash_sha1_final;
    _picohash_sha1_init(&ctx->_sha1);
}

inline void picohash_init_sha224(picohash_ctx_t *ctx)
{
    ctx->block_length = PICOHASH_SHA224_BLOCK_LENGTH;
    ctx->digest_length = PICOHASH_SHA224_DIGEST_LENGTH;
    ctx->_reset = (void *)_picohash_sha224_init;
    ctx->_update = (void *)_picohash_sha256_update;
    ctx->_final = (void *)_picohash_sha224_final;
    _picohash_sha224_init(&ctx->_sha256);
}

inline void picohash_init_sha256(picohash_ctx_t *ctx)
{
    ctx->block_length = PICOHASH_SHA256_BLOCK_LENGTH;
    ctx->digest_length = PICOHASH_SHA256_DIGEST_LENGTH;
    ctx->_reset = (void *)_picohash_sha256_init;
    ctx->_update = (void *)_picohash_sha256_update;
    ctx->_final = (void *)_picohash_sha256_final;
    _picohash_sha256_init(&ctx->_sha256);
}

inline void picohash_update(picohash_ctx_t *ctx, const void *input, size_t len)
{
    ctx->_update(ctx, input, len);
}

inline void picohash_final(picohash_ctx_t *ctx, void *digest)
{
    ctx->_final(ctx, digest);
}

inline void picohash_reset(picohash_ctx_t *ctx)
{
    ctx->_reset(ctx);
}

static inline void _picohash_hmac_apply_key(picohash_ctx_t *ctx, unsigned char delta)
{
    size_t i;
    for (i = 0; i != ctx->block_length; ++i)
        ctx->_hmac.key[i] ^= delta;
    picohash_update(ctx, ctx->_hmac.key, ctx->block_length);
    for (i = 0; i != ctx->block_length; ++i)
        ctx->_hmac.key[i] ^= delta;
}

static void _picohash_hmac_final(picohash_ctx_t *ctx, void *digest)
{
    unsigned char inner_digest[PICOHASH_MAX_DIGEST_LENGTH];

    ctx->_hmac.hash_final(ctx, inner_digest);

    ctx->_hmac.hash_reset(ctx);
    _picohash_hmac_apply_key(ctx, 0x5c);
    picohash_update(ctx, inner_digest, ctx->digest_length);
    memset(inner_digest, 0, ctx->digest_length);

    ctx->_hmac.hash_final(ctx, digest);
}

static inline void _picohash_hmac_reset(picohash_ctx_t *ctx)
{
    ctx->_hmac.hash_reset(ctx);
    _picohash_hmac_apply_key(ctx, 0x36);
}

inline void picohash_init_hmac(picohash_ctx_t *ctx, void (*initf)(picohash_ctx_t *), const void *key, size_t key_len)
{
    initf(ctx);

    memset(ctx->_hmac.key, 0, ctx->block_length);
    if (key_len > ctx->block_length) {
        /* hash the key if it is too long */
        picohash_update(ctx, key, key_len);
        picohash_final(ctx, ctx->_hmac.key);
        ctx->_hmac.hash_reset(ctx);
    } else {
        memcpy(ctx->_hmac.key, key, key_len);
    }

    /* replace reset and final function */
    ctx->_hmac.hash_reset = ctx->_reset;
    ctx->_hmac.hash_final = ctx->_final;
    ctx->_reset = (void *)_picohash_hmac_reset;
    ctx->_final = (void *)_picohash_hmac_final;

    /* start calculating the inner hash */
    _picohash_hmac_apply_key(ctx, 0x36);
}

#endif
