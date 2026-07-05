/**
 * \file chacha20_neon.c
 *
 * \brief Neon implementation of ChaCha20
 *
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */

#include "tf_psa_crypto_common.h"
#include "chacha20_neon.h"

#if defined(MBEDTLS_CHACHA20_C) && (MBEDTLS_CHACHA20_NEON_MULTIBLOCK != 0)

#include "mbedtls/private/chacha20.h"
#include "mbedtls/platform_util.h"
#include "mbedtls/private/error_common.h"

#include <stddef.h>
#include <string.h>

#include "mbedtls/platform.h"

// Tested on all combinations of Armv7 arm/thumb2; Armv8 arm/thumb2/aarch64; Armv8 aarch64_be on
// clang 14, gcc 11, and some more recent versions.

typedef struct {
    uint32x4_t a, b, c, d;
} chacha20_neon_regs_t;

// Define rotate-left operations that rotate within each 32-bit element in a 128-bit vector.
static inline uint32x4_t chacha20_neon_vrotlq_16_u32(uint32x4_t v)
{
    return vreinterpretq_u32_u16(vrev32q_u16(vreinterpretq_u16_u32(v)));
}

static inline uint32x4_t chacha20_neon_vrotlq_12_u32(uint32x4_t v)
{
    uint32x4_t x = vshlq_n_u32(v, 12);
    return vsriq_n_u32(x, v, 20);
}

static inline uint32x4_t chacha20_neon_vrotlq_8_u32(uint32x4_t v)
{
    uint32x4_t result;
#if defined(MBEDTLS_ARCH_IS_ARM64)
    // This implementation is slightly faster, but only supported on 64-bit Arm
    // Table look-up which results in an 8-bit rotate-left within each 32-bit element
    const uint8_t    idx_rotl8[16] = { 3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14 };
    const uint8x16_t vrotl8_tbl = vld1q_u8(idx_rotl8);
    result = vreinterpretq_u32_u8(vqtbl1q_u8(vreinterpretq_u8_u32(v), vrotl8_tbl));
#else
    uint32x4_t a = vshlq_n_u32(v, 8);
    result = vsriq_n_u32(a, v, 24);
#endif
    return result;
}

static inline uint32x4_t chacha20_neon_vrotlq_7_u32(uint32x4_t v)
{
    uint32x4_t x = vshlq_n_u32(v, 7);
    return vsriq_n_u32(x, v, 25);
}

// Increment the 32-bit element within v that corresponds to the ChaCha20 counter
static inline uint32x4_t chacha20_neon_inc_counter(uint32x4_t v)
{
    /* { 1, 0, 0, 0 } */
    uint32x4_t counter_increment = vcombine_u32(vcreate_u32(1), vdup_n_u32(0));
    return vaddq_u32(v, counter_increment);
}

static inline chacha20_neon_regs_t chacha20_neon_singlepass(chacha20_neon_regs_t r)
{
    for (unsigned i = 0; i < 2; i++) {
        r.a = vaddq_u32(r.a, r.b);                    // r.a += b
        r.d = veorq_u32(r.d, r.a);                    // r.d ^= a
        r.d = chacha20_neon_vrotlq_16_u32(r.d);       // r.d <<<= 16

        r.c = vaddq_u32(r.c, r.d);                    // r.c += d
        r.b = veorq_u32(r.b, r.c);                    // r.b ^= c
        r.b = chacha20_neon_vrotlq_12_u32(r.b);       // r.b <<<= 12

        r.a = vaddq_u32(r.a, r.b);                    // r.a += b
        r.d = veorq_u32(r.d, r.a);                    // r.d ^= a
        r.d = chacha20_neon_vrotlq_8_u32(r.d);        // r.d <<<= 8

        r.c = vaddq_u32(r.c, r.d);                    // r.c += d
        r.b = veorq_u32(r.b, r.c);                    // r.b ^= c
        r.b = chacha20_neon_vrotlq_7_u32(r.b);        // r.b <<<= 7

        // re-order b, c and d for the diagonal rounds
        r.c = vextq_u32(r.c, r.c, 2);
        if (i == 0) {
            r.b = vextq_u32(r.b, r.b, 1);
            r.d = vextq_u32(r.d, r.d, 3);
        } else {
            // restore element order in b, c, d
            r.b = vextq_u32(r.b, r.b, 3);
            r.d = vextq_u32(r.d, r.d, 1);
        }
    }

    return r;
}

static inline void chacha20_neon_finish_block(chacha20_neon_regs_t r,
                                              chacha20_neon_regs_t r_original,
                                              uint8_t **output,
                                              const uint8_t **input)
{
    const uint8_t *i = *input;
    uint8_t       *o = *output;

    r.a = vaddq_u32(r.a, r_original.a);
    r.b = vaddq_u32(r.b, r_original.b);
    r.c = vaddq_u32(r.c, r_original.c);
    r.d = vaddq_u32(r.d, r_original.d);

    vst1q_u8(o + 0,  veorq_u8(vld1q_u8(i + 0),  vreinterpretq_u8_u32(r.a)));
    vst1q_u8(o + 16, veorq_u8(vld1q_u8(i + 16), vreinterpretq_u8_u32(r.b)));
    vst1q_u8(o + 32, veorq_u8(vld1q_u8(i + 32), vreinterpretq_u8_u32(r.c)));
    vst1q_u8(o + 48, veorq_u8(vld1q_u8(i + 48), vreinterpretq_u8_u32(r.d)));

    *input  = i + MBEDTLS_CHACHA20_BLOCK_SIZE_BYTES;
    *output = o + MBEDTLS_CHACHA20_BLOCK_SIZE_BYTES;
}

// Prevent gcc from rolling up the (manually unrolled) interleaved block loops
MBEDTLS_OPTIMIZE_FOR_PERFORMANCE
static inline uint32x4_t chacha20_neon_blocks(chacha20_neon_regs_t r_original,
                                              uint8_t *output,
                                              const uint8_t *input,
                                              size_t blocks)
{
    // Assuming 32 regs, with 4 for original values plus 4 for scratch, with 4 regs per block,
    // we should be able to process up to 24/4 = 6 blocks simultaneously.
    // Testing confirms that perf indeed increases with more blocks, and then falls off after 6.

    for (;;) {
        chacha20_neon_regs_t r[6];

        // It's essential to unroll these loops to benefit from interleaving multiple blocks.
        // If MBEDTLS_CHACHA20_NEON_MULTIBLOCK < 6, gcc and clang will optimise away the unused bits
        r[0] = r_original;
        r[1] = r_original;
        r[2] = r_original;
        r[3] = r_original;
        r[4] = r_original;
        r[5] = r_original;
        r[1].d = chacha20_neon_inc_counter(r[0].d);
        r[2].d = chacha20_neon_inc_counter(r[1].d);
        r[3].d = chacha20_neon_inc_counter(r[2].d);
        r[4].d = chacha20_neon_inc_counter(r[3].d);
        r[5].d = chacha20_neon_inc_counter(r[4].d);

        for (unsigned i = 0; i < 10; i++) {
            r[0] = chacha20_neon_singlepass(r[0]);
            r[1] = chacha20_neon_singlepass(r[1]);
            r[2] = chacha20_neon_singlepass(r[2]);
            r[3] = chacha20_neon_singlepass(r[3]);
            r[4] = chacha20_neon_singlepass(r[4]);
            r[5] = chacha20_neon_singlepass(r[5]);
        }

        chacha20_neon_finish_block(r[0], r_original, &output, &input);
        r_original.d = chacha20_neon_inc_counter(r_original.d);
        if (--blocks == 0) {
            return r_original.d;
        }
#if MBEDTLS_CHACHA20_NEON_MULTIBLOCK >= 2
        chacha20_neon_finish_block(r[1], r_original, &output, &input);
        r_original.d = chacha20_neon_inc_counter(r_original.d);
        if (--blocks == 0) {
            return r_original.d;
        }
#endif
#if MBEDTLS_CHACHA20_NEON_MULTIBLOCK >= 3
        chacha20_neon_finish_block(r[2], r_original, &output, &input);
        r_original.d = chacha20_neon_inc_counter(r_original.d);
        if (--blocks == 0) {
            return r_original.d;
        }
#endif
#if MBEDTLS_CHACHA20_NEON_MULTIBLOCK >= 4
        chacha20_neon_finish_block(r[3], r_original, &output, &input);
        r_original.d = chacha20_neon_inc_counter(r_original.d);
        if (--blocks == 0) {
            return r_original.d;
        }
#endif
#if MBEDTLS_CHACHA20_NEON_MULTIBLOCK >= 5
        chacha20_neon_finish_block(r[4], r_original, &output, &input);
        r_original.d = chacha20_neon_inc_counter(r_original.d);
        if (--blocks == 0) {
            return r_original.d;
        }
#endif
#if MBEDTLS_CHACHA20_NEON_MULTIBLOCK >= 6
        chacha20_neon_finish_block(r[5], r_original, &output, &input);
        r_original.d = chacha20_neon_inc_counter(r_original.d);
        if (--blocks == 0) {
            return r_original.d;
        }
#endif
    }
}

int mbedtls_chacha20_update(mbedtls_chacha20_context *ctx,
                            size_t size,
                            const unsigned char *input,
                            unsigned char *output)
{
    size_t offset = 0U;

    /* Use leftover keystream bytes, if available */
    while (ctx->keystream_bytes_used < MBEDTLS_CHACHA20_BLOCK_SIZE_BYTES && size > 0) {
        output[offset] = input[offset]
                         ^ ctx->keystream8[ctx->keystream_bytes_used];

        ctx->keystream_bytes_used++;
        offset++;
        size--;
    }

    /* Load state into NEON registers */
    chacha20_neon_regs_t state;
    state.a = vld1q_u32(&ctx->state[0]);
    state.b = vld1q_u32(&ctx->state[4]);
    state.c = vld1q_u32(&ctx->state[8]);
    state.d = vld1q_u32(&ctx->state[12]);

    /* Process full blocks */
    if (size >= MBEDTLS_CHACHA20_BLOCK_SIZE_BYTES) {
        size_t blocks = size / MBEDTLS_CHACHA20_BLOCK_SIZE_BYTES;
        state.d = chacha20_neon_blocks(state, output + offset, input + offset, blocks);

        offset += MBEDTLS_CHACHA20_BLOCK_SIZE_BYTES * blocks;
        size   -= MBEDTLS_CHACHA20_BLOCK_SIZE_BYTES * blocks;
    }

    /* Last (partial) block */
    if (size > 0U) {
        /* Generate new keystream block and increment counter */
        memset(ctx->keystream8, 0, MBEDTLS_CHACHA20_BLOCK_SIZE_BYTES);
        state.d = chacha20_neon_blocks(state, ctx->keystream8, ctx->keystream8, 1);

        mbedtls_xor_no_simd(output + offset, input + offset, ctx->keystream8, size);

        ctx->keystream_bytes_used = size;
    }

    /* Capture state */
    vst1q_u32(&ctx->state[12], state.d);

    return 0;
}

#endif /* defined(MBEDTLS_CHACHA20_C) && (MBEDTLS_CHACHA20_NEON_MULTIBLOCK != 0) */
