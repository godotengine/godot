/*
 *  FIPS-202 compliant SHA3 implementation
 *
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */
/*
 *  The SHA-3 Secure Hash Standard was published by NIST in 2015.
 *
 *  https://nvlpubs.nist.gov/nistpubs/fips/nist.fips.202.pdf
 */

#include "common.h"

#if defined(MBEDTLS_SHA3_C)

#include "mbedtls/sha3.h"
#include "mbedtls/platform_util.h"
#include "mbedtls/error.h"

#include <string.h>

#if defined(MBEDTLS_SELF_TEST)
#include "mbedtls/platform.h"
#endif /* MBEDTLS_SELF_TEST */

#define XOR_BYTE 0x6

typedef struct mbedtls_sha3_family_functions {
    mbedtls_sha3_id id;

    uint16_t r;
    uint16_t olen;
}
mbedtls_sha3_family_functions;

/*
 * List of supported SHA-3 families
 */
static mbedtls_sha3_family_functions sha3_families[] = {
    { MBEDTLS_SHA3_224,      1152, 224 },
    { MBEDTLS_SHA3_256,      1088, 256 },
    { MBEDTLS_SHA3_384,       832, 384 },
    { MBEDTLS_SHA3_512,       576, 512 },
    { MBEDTLS_SHA3_NONE, 0, 0 }
};

static const uint64_t rc[24] = {
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000,
    0x000000000000808b, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
};

static const uint8_t rho[24] = {
    1, 62, 28, 27, 36, 44,  6, 55, 20,
    3, 10, 43, 25, 39, 41, 45, 15,
    21,  8, 18,  2, 61, 56, 14
};

static const uint8_t pi[24] = {
    10,  7, 11, 17, 18, 3,  5, 16,  8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22,  9,  6, 1,
};

#define ROT64(x, y) (((x) << (y)) | ((x) >> (64U - (y))))
#define ABSORB(ctx, idx, v) do { ctx->state[(idx) >> 3] ^= ((uint64_t) (v)) << (((idx) & 0x7) << 3); \
} while (0)
#define SQUEEZE(ctx, idx) ((uint8_t) (ctx->state[(idx) >> 3] >> (((idx) & 0x7) << 3)))
#define SWAP(x, y) do { uint64_t tmp = (x); (x) = (y); (y) = tmp; } while (0)

/* The permutation function.  */
static void keccak_f1600(mbedtls_sha3_context *ctx)
{
    uint64_t lane[5];
    uint64_t *s = ctx->state;
    int i;

    for (int round = 0; round < 24; round++) {
        uint64_t t;

        /* Theta */
        lane[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
        lane[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
        lane[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
        lane[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
        lane[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

        t = lane[4] ^ ROT64(lane[1], 1);
        s[0] ^= t; s[5] ^= t; s[10] ^= t; s[15] ^= t; s[20] ^= t;

        t = lane[0] ^ ROT64(lane[2], 1);
        s[1] ^= t; s[6] ^= t; s[11] ^= t; s[16] ^= t; s[21] ^= t;

        t = lane[1] ^ ROT64(lane[3], 1);
        s[2] ^= t; s[7] ^= t; s[12] ^= t; s[17] ^= t; s[22] ^= t;

        t = lane[2] ^ ROT64(lane[4], 1);
        s[3] ^= t; s[8] ^= t; s[13] ^= t; s[18] ^= t; s[23] ^= t;

        t = lane[3] ^ ROT64(lane[0], 1);
        s[4] ^= t; s[9] ^= t; s[14] ^= t; s[19] ^= t; s[24] ^= t;

        /* Rho */
        for (i = 1; i < 25; i++) {
            s[i] = ROT64(s[i], rho[i-1]);
        }

        /* Pi */
        t = s[1];
        for (i = 0; i < 24; i++) {
            SWAP(s[pi[i]], t);
        }

        /* Chi */
        lane[0] = s[0]; lane[1] = s[1]; lane[2] = s[2]; lane[3] = s[3]; lane[4] = s[4];
        s[0] ^= (~lane[1]) & lane[2];
        s[1] ^= (~lane[2]) & lane[3];
        s[2] ^= (~lane[3]) & lane[4];
        s[3] ^= (~lane[4]) & lane[0];
        s[4] ^= (~lane[0]) & lane[1];

        lane[0] = s[5]; lane[1] = s[6]; lane[2] = s[7]; lane[3] = s[8]; lane[4] = s[9];
        s[5] ^= (~lane[1]) & lane[2];
        s[6] ^= (~lane[2]) & lane[3];
        s[7] ^= (~lane[3]) & lane[4];
        s[8] ^= (~lane[4]) & lane[0];
        s[9] ^= (~lane[0]) & lane[1];

        lane[0] = s[10]; lane[1] = s[11]; lane[2] = s[12]; lane[3] = s[13]; lane[4] = s[14];
        s[10] ^= (~lane[1]) & lane[2];
        s[11] ^= (~lane[2]) & lane[3];
        s[12] ^= (~lane[3]) & lane[4];
        s[13] ^= (~lane[4]) & lane[0];
        s[14] ^= (~lane[0]) & lane[1];

        lane[0] = s[15]; lane[1] = s[16]; lane[2] = s[17]; lane[3] = s[18]; lane[4] = s[19];
        s[15] ^= (~lane[1]) & lane[2];
        s[16] ^= (~lane[2]) & lane[3];
        s[17] ^= (~lane[3]) & lane[4];
        s[18] ^= (~lane[4]) & lane[0];
        s[19] ^= (~lane[0]) & lane[1];

        lane[0] = s[20]; lane[1] = s[21]; lane[2] = s[22]; lane[3] = s[23]; lane[4] = s[24];
        s[20] ^= (~lane[1]) & lane[2];
        s[21] ^= (~lane[2]) & lane[3];
        s[22] ^= (~lane[3]) & lane[4];
        s[23] ^= (~lane[4]) & lane[0];
        s[24] ^= (~lane[0]) & lane[1];

        /* Iota */
        s[0] ^= rc[round];
    }
}

void mbedtls_sha3_init(mbedtls_sha3_context *ctx)
{
    memset(ctx, 0, sizeof(mbedtls_sha3_context));
}

void mbedtls_sha3_free(mbedtls_sha3_context *ctx)
{
    if (ctx == NULL) {
        return;
    }

    mbedtls_platform_zeroize(ctx, sizeof(mbedtls_sha3_context));
}

void mbedtls_sha3_clone(mbedtls_sha3_context *dst,
                        const mbedtls_sha3_context *src)
{
    *dst = *src;
}

/*
 * SHA-3 context setup
 */
int mbedtls_sha3_starts(mbedtls_sha3_context *ctx, mbedtls_sha3_id id)
{
    mbedtls_sha3_family_functions *p = NULL;

    for (p = sha3_families; p->id != MBEDTLS_SHA3_NONE; p++) {
        if (p->id == id) {
            break;
        }
    }

    if (p->id == MBEDTLS_SHA3_NONE) {
        return MBEDTLS_ERR_SHA3_BAD_INPUT_DATA;
    }

    ctx->olen = p->olen / 8;
    ctx->max_block_size = p->r / 8;

    memset(ctx->state, 0, sizeof(ctx->state));
    ctx->index = 0;

    return 0;
}

/*
 * SHA-3 process buffer
 */
int mbedtls_sha3_update(mbedtls_sha3_context *ctx,
                        const uint8_t *input,
                        size_t ilen)
{
    if (ilen >= 8) {
        // 8-byte align index
        int align_bytes = 8 - (ctx->index % 8);
        if (align_bytes) {
            for (; align_bytes > 0; align_bytes--) {
                ABSORB(ctx, ctx->index, *input++);
                ilen--;
                ctx->index++;
            }
            if ((ctx->index = ctx->index % ctx->max_block_size) == 0) {
                keccak_f1600(ctx);
            }
        }

        // process input in 8-byte chunks
        while (ilen >= 8) {
            ABSORB(ctx, ctx->index, MBEDTLS_GET_UINT64_LE(input, 0));
            input += 8;
            ilen -= 8;
            if ((ctx->index = (ctx->index + 8) % ctx->max_block_size) == 0) {
                keccak_f1600(ctx);
            }
        }
    }

    // handle remaining bytes
    while (ilen-- > 0) {
        ABSORB(ctx, ctx->index, *input++);
        if ((ctx->index = (ctx->index + 1) % ctx->max_block_size) == 0) {
            keccak_f1600(ctx);
        }
    }

    return 0;
}

int mbedtls_sha3_finish(mbedtls_sha3_context *ctx,
                        uint8_t *output, size_t olen)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;

    /* Catch SHA-3 families, with fixed output length */
    if (ctx->olen > 0) {
        if (ctx->olen > olen) {
            ret = MBEDTLS_ERR_SHA3_BAD_INPUT_DATA;
            goto exit;
        }
        olen = ctx->olen;
    }

    ABSORB(ctx, ctx->index, XOR_BYTE);
    ABSORB(ctx, ctx->max_block_size - 1, 0x80);
    keccak_f1600(ctx);
    ctx->index = 0;

    while (olen-- > 0) {
        *output++ = SQUEEZE(ctx, ctx->index);

        if ((ctx->index = (ctx->index + 1) % ctx->max_block_size) == 0) {
            keccak_f1600(ctx);
        }
    }

    ret = 0;

exit:
    mbedtls_sha3_free(ctx);
    return ret;
}

/*
 * output = SHA-3( input buffer )
 */
int mbedtls_sha3(mbedtls_sha3_id id, const uint8_t *input,
                 size_t ilen, uint8_t *output, size_t olen)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    mbedtls_sha3_context ctx;

    mbedtls_sha3_init(&ctx);

    /* Sanity checks are performed in every mbedtls_sha3_xxx() */
    if ((ret = mbedtls_sha3_starts(&ctx, id)) != 0) {
        goto exit;
    }

    if ((ret = mbedtls_sha3_update(&ctx, input, ilen)) != 0) {
        goto exit;
    }

    if ((ret = mbedtls_sha3_finish(&ctx, output, olen)) != 0) {
        goto exit;
    }

exit:
    mbedtls_sha3_free(&ctx);

    return ret;
}

/**************** Self-tests ****************/

#if defined(MBEDTLS_SELF_TEST)

static const unsigned char test_data[2][4] =
{
    "",
    "abc",
};

static const size_t test_data_len[2] =
{
    0, /* "" */
    3  /* "abc" */
};

static const unsigned char test_hash_sha3_224[2][28] =
{
    { /* "" */
        0x6B, 0x4E, 0x03, 0x42, 0x36, 0x67, 0xDB, 0xB7,
        0x3B, 0x6E, 0x15, 0x45, 0x4F, 0x0E, 0xB1, 0xAB,
        0xD4, 0x59, 0x7F, 0x9A, 0x1B, 0x07, 0x8E, 0x3F,
        0x5B, 0x5A, 0x6B, 0xC7
    },
    { /* "abc" */
        0xE6, 0x42, 0x82, 0x4C, 0x3F, 0x8C, 0xF2, 0x4A,
        0xD0, 0x92, 0x34, 0xEE, 0x7D, 0x3C, 0x76, 0x6F,
        0xC9, 0xA3, 0xA5, 0x16, 0x8D, 0x0C, 0x94, 0xAD,
        0x73, 0xB4, 0x6F, 0xDF
    }
};

static const unsigned char test_hash_sha3_256[2][32] =
{
    { /* "" */
        0xA7, 0xFF, 0xC6, 0xF8, 0xBF, 0x1E, 0xD7, 0x66,
        0x51, 0xC1, 0x47, 0x56, 0xA0, 0x61, 0xD6, 0x62,
        0xF5, 0x80, 0xFF, 0x4D, 0xE4, 0x3B, 0x49, 0xFA,
        0x82, 0xD8, 0x0A, 0x4B, 0x80, 0xF8, 0x43, 0x4A
    },
    { /* "abc" */
        0x3A, 0x98, 0x5D, 0xA7, 0x4F, 0xE2, 0x25, 0xB2,
        0x04, 0x5C, 0x17, 0x2D, 0x6B, 0xD3, 0x90, 0xBD,
        0x85, 0x5F, 0x08, 0x6E, 0x3E, 0x9D, 0x52, 0x5B,
        0x46, 0xBF, 0xE2, 0x45, 0x11, 0x43, 0x15, 0x32
    }
};

static const unsigned char test_hash_sha3_384[2][48] =
{
    { /* "" */
        0x0C, 0x63, 0xA7, 0x5B, 0x84, 0x5E, 0x4F, 0x7D,
        0x01, 0x10, 0x7D, 0x85, 0x2E, 0x4C, 0x24, 0x85,
        0xC5, 0x1A, 0x50, 0xAA, 0xAA, 0x94, 0xFC, 0x61,
        0x99, 0x5E, 0x71, 0xBB, 0xEE, 0x98, 0x3A, 0x2A,
        0xC3, 0x71, 0x38, 0x31, 0x26, 0x4A, 0xDB, 0x47,
        0xFB, 0x6B, 0xD1, 0xE0, 0x58, 0xD5, 0xF0, 0x04
    },
    { /* "abc" */
        0xEC, 0x01, 0x49, 0x82, 0x88, 0x51, 0x6F, 0xC9,
        0x26, 0x45, 0x9F, 0x58, 0xE2, 0xC6, 0xAD, 0x8D,
        0xF9, 0xB4, 0x73, 0xCB, 0x0F, 0xC0, 0x8C, 0x25,
        0x96, 0xDA, 0x7C, 0xF0, 0xE4, 0x9B, 0xE4, 0xB2,
        0x98, 0xD8, 0x8C, 0xEA, 0x92, 0x7A, 0xC7, 0xF5,
        0x39, 0xF1, 0xED, 0xF2, 0x28, 0x37, 0x6D, 0x25
    }
};

static const unsigned char test_hash_sha3_512[2][64] =
{
    { /* "" */
        0xA6, 0x9F, 0x73, 0xCC, 0xA2, 0x3A, 0x9A, 0xC5,
        0xC8, 0xB5, 0x67, 0xDC, 0x18, 0x5A, 0x75, 0x6E,
        0x97, 0xC9, 0x82, 0x16, 0x4F, 0xE2, 0x58, 0x59,
        0xE0, 0xD1, 0xDC, 0xC1, 0x47, 0x5C, 0x80, 0xA6,
        0x15, 0xB2, 0x12, 0x3A, 0xF1, 0xF5, 0xF9, 0x4C,
        0x11, 0xE3, 0xE9, 0x40, 0x2C, 0x3A, 0xC5, 0x58,
        0xF5, 0x00, 0x19, 0x9D, 0x95, 0xB6, 0xD3, 0xE3,
        0x01, 0x75, 0x85, 0x86, 0x28, 0x1D, 0xCD, 0x26
    },
    { /* "abc" */
        0xB7, 0x51, 0x85, 0x0B, 0x1A, 0x57, 0x16, 0x8A,
        0x56, 0x93, 0xCD, 0x92, 0x4B, 0x6B, 0x09, 0x6E,
        0x08, 0xF6, 0x21, 0x82, 0x74, 0x44, 0xF7, 0x0D,
        0x88, 0x4F, 0x5D, 0x02, 0x40, 0xD2, 0x71, 0x2E,
        0x10, 0xE1, 0x16, 0xE9, 0x19, 0x2A, 0xF3, 0xC9,
        0x1A, 0x7E, 0xC5, 0x76, 0x47, 0xE3, 0x93, 0x40,
        0x57, 0x34, 0x0B, 0x4C, 0xF4, 0x08, 0xD5, 0xA5,
        0x65, 0x92, 0xF8, 0x27, 0x4E, 0xEC, 0x53, 0xF0
    }
};

static const unsigned char long_kat_hash_sha3_224[28] =
{
    0xD6, 0x93, 0x35, 0xB9, 0x33, 0x25, 0x19, 0x2E,
    0x51, 0x6A, 0x91, 0x2E, 0x6D, 0x19, 0xA1, 0x5C,
    0xB5, 0x1C, 0x6E, 0xD5, 0xC1, 0x52, 0x43, 0xE7,
    0xA7, 0xFD, 0x65, 0x3C
};

static const unsigned char long_kat_hash_sha3_256[32] =
{
    0x5C, 0x88, 0x75, 0xAE, 0x47, 0x4A, 0x36, 0x34,
    0xBA, 0x4F, 0xD5, 0x5E, 0xC8, 0x5B, 0xFF, 0xD6,
    0x61, 0xF3, 0x2A, 0xCA, 0x75, 0xC6, 0xD6, 0x99,
    0xD0, 0xCD, 0xCB, 0x6C, 0x11, 0x58, 0x91, 0xC1
};

static const unsigned char long_kat_hash_sha3_384[48] =
{
    0xEE, 0xE9, 0xE2, 0x4D, 0x78, 0xC1, 0x85, 0x53,
    0x37, 0x98, 0x34, 0x51, 0xDF, 0x97, 0xC8, 0xAD,
    0x9E, 0xED, 0xF2, 0x56, 0xC6, 0x33, 0x4F, 0x8E,
    0x94, 0x8D, 0x25, 0x2D, 0x5E, 0x0E, 0x76, 0x84,
    0x7A, 0xA0, 0x77, 0x4D, 0xDB, 0x90, 0xA8, 0x42,
    0x19, 0x0D, 0x2C, 0x55, 0x8B, 0x4B, 0x83, 0x40
};

static const unsigned char long_kat_hash_sha3_512[64] =
{
    0x3C, 0x3A, 0x87, 0x6D, 0xA1, 0x40, 0x34, 0xAB,
    0x60, 0x62, 0x7C, 0x07, 0x7B, 0xB9, 0x8F, 0x7E,
    0x12, 0x0A, 0x2A, 0x53, 0x70, 0x21, 0x2D, 0xFF,
    0xB3, 0x38, 0x5A, 0x18, 0xD4, 0xF3, 0x88, 0x59,
    0xED, 0x31, 0x1D, 0x0A, 0x9D, 0x51, 0x41, 0xCE,
    0x9C, 0xC5, 0xC6, 0x6E, 0xE6, 0x89, 0xB2, 0x66,
    0xA8, 0xAA, 0x18, 0xAC, 0xE8, 0x28, 0x2A, 0x0E,
    0x0D, 0xB5, 0x96, 0xC9, 0x0B, 0x0A, 0x7B, 0x87
};

static int mbedtls_sha3_kat_test(int verbose,
                                 const char *type_name,
                                 mbedtls_sha3_id id,
                                 int test_num)
{
    uint8_t hash[64];
    int result;

    result = mbedtls_sha3(id,
                          test_data[test_num], test_data_len[test_num],
                          hash, sizeof(hash));
    if (result != 0) {
        if (verbose != 0) {
            mbedtls_printf("  %s test %d error code: %d\n",
                           type_name, test_num, result);
        }

        return result;
    }

    switch (id) {
        case MBEDTLS_SHA3_224:
            result = memcmp(hash, test_hash_sha3_224[test_num], 28);
            break;
        case MBEDTLS_SHA3_256:
            result = memcmp(hash, test_hash_sha3_256[test_num], 32);
            break;
        case MBEDTLS_SHA3_384:
            result = memcmp(hash, test_hash_sha3_384[test_num], 48);
            break;
        case MBEDTLS_SHA3_512:
            result = memcmp(hash, test_hash_sha3_512[test_num], 64);
            break;
        default:
            break;
    }

    if (0 != result) {
        if (verbose != 0) {
            mbedtls_printf("  %s test %d failed\n", type_name, test_num);
        }

        return -1;
    }

    if (verbose != 0) {
        mbedtls_printf("  %s test %d passed\n", type_name, test_num);
    }

    return 0;
}

static int mbedtls_sha3_long_kat_test(int verbose,
                                      const char *type_name,
                                      mbedtls_sha3_id id)
{
    mbedtls_sha3_context ctx;
    unsigned char buffer[1000];
    unsigned char hash[64];
    int result = 0;

    memset(buffer, 'a', 1000);

    if (verbose != 0) {
        mbedtls_printf("  %s long KAT test ", type_name);
    }

    mbedtls_sha3_init(&ctx);

    result = mbedtls_sha3_starts(&ctx, id);
    if (result != 0) {
        if (verbose != 0) {
            mbedtls_printf("setup failed\n ");
        }
    }

    /* Process 1,000,000 (one million) 'a' characters */
    for (int i = 0; i < 1000; i++) {
        result = mbedtls_sha3_update(&ctx, buffer, 1000);
        if (result != 0) {
            if (verbose != 0) {
                mbedtls_printf("update error code: %i\n", result);
            }

            goto cleanup;
        }
    }

    result = mbedtls_sha3_finish(&ctx, hash, sizeof(hash));
    if (result != 0) {
        if (verbose != 0) {
            mbedtls_printf("finish error code: %d\n", result);
        }

        goto cleanup;
    }

    switch (id) {
        case MBEDTLS_SHA3_224:
            result = memcmp(hash, long_kat_hash_sha3_224, 28);
            break;
        case MBEDTLS_SHA3_256:
            result = memcmp(hash, long_kat_hash_sha3_256, 32);
            break;
        case MBEDTLS_SHA3_384:
            result = memcmp(hash, long_kat_hash_sha3_384, 48);
            break;
        case MBEDTLS_SHA3_512:
            result = memcmp(hash, long_kat_hash_sha3_512, 64);
            break;
        default:
            break;
    }

    if (result != 0) {
        if (verbose != 0) {
            mbedtls_printf("failed\n");
        }
    }

    if (verbose != 0) {
        mbedtls_printf("passed\n");
    }

cleanup:
    mbedtls_sha3_free(&ctx);
    return result;
}

int mbedtls_sha3_self_test(int verbose)
{
    int i;

    /* SHA-3 Known Answer Tests (KAT) */
    for (i = 0; i < 2; i++) {
        if (0 != mbedtls_sha3_kat_test(verbose,
                                       "SHA3-224", MBEDTLS_SHA3_224, i)) {
            return 1;
        }

        if (0 != mbedtls_sha3_kat_test(verbose,
                                       "SHA3-256", MBEDTLS_SHA3_256, i)) {
            return 1;
        }

        if (0 != mbedtls_sha3_kat_test(verbose,
                                       "SHA3-384", MBEDTLS_SHA3_384, i)) {
            return 1;
        }

        if (0 != mbedtls_sha3_kat_test(verbose,
                                       "SHA3-512", MBEDTLS_SHA3_512, i)) {
            return 1;
        }
    }

    /* SHA-3 long KAT tests */
    if (0 != mbedtls_sha3_long_kat_test(verbose,
                                        "SHA3-224", MBEDTLS_SHA3_224)) {
        return 1;
    }

    if (0 != mbedtls_sha3_long_kat_test(verbose,
                                        "SHA3-256", MBEDTLS_SHA3_256)) {
        return 1;
    }

    if (0 != mbedtls_sha3_long_kat_test(verbose,
                                        "SHA3-384", MBEDTLS_SHA3_384)) {
        return 1;
    }

    if (0 != mbedtls_sha3_long_kat_test(verbose,
                                        "SHA3-512", MBEDTLS_SHA3_512)) {
        return 1;
    }

    if (verbose != 0) {
        mbedtls_printf("\n");
    }

    return 0;
}
#endif /* MBEDTLS_SELF_TEST */

#endif /* MBEDTLS_SHA3_C */
