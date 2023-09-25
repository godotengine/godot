/*
 *  Armv8-A Cryptographic Extension support functions for Aarch64
 *
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0
 *
 *  Licensed under the Apache License, Version 2.0 (the "License"); you may
 *  not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#if defined(__aarch64__) && !defined(__ARM_FEATURE_CRYPTO) && \
    defined(__clang__) && __clang_major__ >= 4
/* TODO: Re-consider above after https://reviews.llvm.org/D131064 merged.
 *
 * The intrinsic declaration are guarded by predefined ACLE macros in clang:
 * these are normally only enabled by the -march option on the command line.
 * By defining the macros ourselves we gain access to those declarations without
 * requiring -march on the command line.
 *
 * `arm_neon.h` could be included by any header file, so we put these defines
 * at the top of this file, before any includes.
 */
#define __ARM_FEATURE_CRYPTO 1
/* See: https://arm-software.github.io/acle/main/acle.html#cryptographic-extensions
 *
 * `__ARM_FEATURE_CRYPTO` is deprecated, but we need to continue to specify it
 * for older compilers.
 */
#define __ARM_FEATURE_AES    1
#define MBEDTLS_ENABLE_ARM_CRYPTO_EXTENSIONS_COMPILER_FLAG
#endif

#include <string.h>
#include "common.h"

#if defined(MBEDTLS_AESCE_C)

#include "aesce.h"

#if defined(MBEDTLS_HAVE_ARM64)

#if !defined(__ARM_FEATURE_AES) || defined(MBEDTLS_ENABLE_ARM_CRYPTO_EXTENSIONS_COMPILER_FLAG)
#   if defined(__clang__)
#       if __clang_major__ < 4
#           error "A more recent Clang is required for MBEDTLS_AESCE_C"
#       endif
#       pragma clang attribute push (__attribute__((target("crypto"))), apply_to=function)
#       define MBEDTLS_POP_TARGET_PRAGMA
#   elif defined(__GNUC__)
#       if __GNUC__ < 6
#           error "A more recent GCC is required for MBEDTLS_AESCE_C"
#       endif
#       pragma GCC push_options
#       pragma GCC target ("arch=armv8-a+crypto")
#       define MBEDTLS_POP_TARGET_PRAGMA
#   else
#       error "Only GCC and Clang supported for MBEDTLS_AESCE_C"
#   endif
#endif /* !__ARM_FEATURE_AES || MBEDTLS_ENABLE_ARM_CRYPTO_EXTENSIONS_COMPILER_FLAG */

#include <arm_neon.h>

#if defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

/*
 * AES instruction support detection routine
 */
int mbedtls_aesce_has_support(void)
{
#if defined(__linux__)
    unsigned long auxval = getauxval(AT_HWCAP);
    return (auxval & (HWCAP_ASIMD | HWCAP_AES)) ==
           (HWCAP_ASIMD | HWCAP_AES);
#else
    /* Assume AES instructions are supported. */
    return 1;
#endif
}

static uint8x16_t aesce_encrypt_block(uint8x16_t block,
                                      unsigned char *keys,
                                      int rounds)
{
    for (int i = 0; i < rounds - 1; i++) {
        /* AES AddRoundKey, SubBytes, ShiftRows (in this order).
         * AddRoundKey adds the round key for the previous round. */
        block = vaeseq_u8(block, vld1q_u8(keys + i * 16));
        /* AES mix columns */
        block = vaesmcq_u8(block);
    }

    /* AES AddRoundKey for the previous round.
     * SubBytes, ShiftRows for the final round.  */
    block = vaeseq_u8(block, vld1q_u8(keys + (rounds -1) * 16));

    /* Final round: no MixColumns */

    /* Final AddRoundKey */
    block = veorq_u8(block, vld1q_u8(keys + rounds  * 16));

    return block;
}

static uint8x16_t aesce_decrypt_block(uint8x16_t block,
                                      unsigned char *keys,
                                      int rounds)
{

    for (int i = 0; i < rounds - 1; i++) {
        /* AES AddRoundKey, SubBytes, ShiftRows */
        block = vaesdq_u8(block, vld1q_u8(keys + i * 16));
        /* AES inverse MixColumns for the next round.
         *
         * This means that we switch the order of the inverse AddRoundKey and
         * inverse MixColumns operations. We have to do this as AddRoundKey is
         * done in an atomic instruction together with the inverses of SubBytes
         * and ShiftRows.
         *
         * It works because MixColumns is a linear operation over GF(2^8) and
         * AddRoundKey is an exclusive or, which is equivalent to addition over
         * GF(2^8). (The inverse of MixColumns needs to be applied to the
         * affected round keys separately which has been done when the
         * decryption round keys were calculated.) */
        block = vaesimcq_u8(block);
    }

    /* The inverses of AES AddRoundKey, SubBytes, ShiftRows finishing up the
     * last full round. */
    block = vaesdq_u8(block, vld1q_u8(keys + (rounds - 1) * 16));

    /* Inverse AddRoundKey for inverting the initial round key addition. */
    block = veorq_u8(block, vld1q_u8(keys + rounds * 16));

    return block;
}

/*
 * AES-ECB block en(de)cryption
 */
int mbedtls_aesce_crypt_ecb(mbedtls_aes_context *ctx,
                            int mode,
                            const unsigned char input[16],
                            unsigned char output[16])
{
    uint8x16_t block = vld1q_u8(&input[0]);
    unsigned char *keys = (unsigned char *) (ctx->buf + ctx->rk_offset);

    if (mode == MBEDTLS_AES_ENCRYPT) {
        block = aesce_encrypt_block(block, keys, ctx->nr);
    } else {
        block = aesce_decrypt_block(block, keys, ctx->nr);
    }
    vst1q_u8(&output[0], block);

    return 0;
}

/*
 * Compute decryption round keys from encryption round keys
 */
void mbedtls_aesce_inverse_key(unsigned char *invkey,
                               const unsigned char *fwdkey,
                               int nr)
{
    int i, j;
    j = nr;
    vst1q_u8(invkey, vld1q_u8(fwdkey + j * 16));
    for (i = 1, j--; j > 0; i++, j--) {
        vst1q_u8(invkey + i * 16,
                 vaesimcq_u8(vld1q_u8(fwdkey + j * 16)));
    }
    vst1q_u8(invkey + i * 16, vld1q_u8(fwdkey + j * 16));

}

static inline uint32_t aes_rot_word(uint32_t word)
{
    return (word << (32 - 8)) | (word >> 8);
}

static inline uint32_t aes_sub_word(uint32_t in)
{
    uint8x16_t v = vreinterpretq_u8_u32(vdupq_n_u32(in));
    uint8x16_t zero = vdupq_n_u8(0);

    /* vaeseq_u8 does both SubBytes and ShiftRows. Taking the first row yields
     * the correct result as ShiftRows doesn't change the first row. */
    v = vaeseq_u8(zero, v);
    return vgetq_lane_u32(vreinterpretq_u32_u8(v), 0);
}

/*
 * Key expansion function
 */
static void aesce_setkey_enc(unsigned char *rk,
                             const unsigned char *key,
                             const size_t key_bit_length)
{
    static uint8_t const rcon[] = { 0x01, 0x02, 0x04, 0x08, 0x10,
                                    0x20, 0x40, 0x80, 0x1b, 0x36 };
    /* See https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197.pdf
     *   - Section 5, Nr = Nk + 6
     *   - Section 5.2, the length of round keys is Nb*(Nr+1)
     */
    const uint32_t key_len_in_words = key_bit_length / 32;  /* Nk */
    const size_t round_key_len_in_words = 4;                /* Nb */
    const size_t rounds_needed = key_len_in_words + 6;      /* Nr */
    const size_t round_keys_len_in_words =
        round_key_len_in_words * (rounds_needed + 1);       /* Nb*(Nr+1) */
    const uint32_t *rko_end = (uint32_t *) rk + round_keys_len_in_words;

    memcpy(rk, key, key_len_in_words * 4);

    for (uint32_t *rki = (uint32_t *) rk;
         rki + key_len_in_words < rko_end;
         rki += key_len_in_words) {

        size_t iteration = (rki - (uint32_t *) rk) / key_len_in_words;
        uint32_t *rko;
        rko = rki + key_len_in_words;
        rko[0] = aes_rot_word(aes_sub_word(rki[key_len_in_words - 1]));
        rko[0] ^= rcon[iteration] ^ rki[0];
        rko[1] = rko[0] ^ rki[1];
        rko[2] = rko[1] ^ rki[2];
        rko[3] = rko[2] ^ rki[3];
        if (rko + key_len_in_words > rko_end) {
            /* Do not write overflow words.*/
            continue;
        }
        switch (key_bit_length) {
            case 128:
                break;
            case 192:
                rko[4] = rko[3] ^ rki[4];
                rko[5] = rko[4] ^ rki[5];
                break;
            case 256:
                rko[4] = aes_sub_word(rko[3]) ^ rki[4];
                rko[5] = rko[4] ^ rki[5];
                rko[6] = rko[5] ^ rki[6];
                rko[7] = rko[6] ^ rki[7];
                break;
        }
    }
}

/*
 * Key expansion, wrapper
 */
int mbedtls_aesce_setkey_enc(unsigned char *rk,
                             const unsigned char *key,
                             size_t bits)
{
    switch (bits) {
        case 128:
        case 192:
        case 256:
            aesce_setkey_enc(rk, key, bits);
            break;
        default:
            return MBEDTLS_ERR_AES_INVALID_KEY_LENGTH;
    }

    return 0;
}

#if defined(MBEDTLS_GCM_C)

#if !defined(__clang__) && defined(__GNUC__) && __GNUC__ == 5
/* Some intrinsics are not available for GCC 5.X. */
#define vreinterpretq_p64_u8(a) ((poly64x2_t) a)
#define vreinterpretq_u8_p128(a) ((uint8x16_t) a)
static inline poly64_t vget_low_p64(poly64x2_t __a)
{
    uint64x2_t tmp = (uint64x2_t) (__a);
    uint64x1_t lo = vcreate_u64(vgetq_lane_u64(tmp, 0));
    return (poly64_t) (lo);
}
#endif /* !__clang__ && __GNUC__ && __GNUC__ == 5*/

/* vmull_p64/vmull_high_p64 wrappers.
 *
 * Older compilers miss some intrinsic functions for `poly*_t`. We use
 * uint8x16_t and uint8x16x3_t as input/output parameters.
 */
static inline uint8x16_t pmull_low(uint8x16_t a, uint8x16_t b)
{
    return vreinterpretq_u8_p128(
        vmull_p64(
            (poly64_t) vget_low_p64(vreinterpretq_p64_u8(a)),
            (poly64_t) vget_low_p64(vreinterpretq_p64_u8(b))));
}

static inline uint8x16_t pmull_high(uint8x16_t a, uint8x16_t b)
{
    return vreinterpretq_u8_p128(
        vmull_high_p64(vreinterpretq_p64_u8(a),
                       vreinterpretq_p64_u8(b)));
}

/* GHASH does 128b polynomial multiplication on block in GF(2^128) defined by
 * `x^128 + x^7 + x^2 + x + 1`.
 *
 * Arm64 only has 64b->128b polynomial multipliers, we need to do 4 64b
 * multiplies to generate a 128b.
 *
 * `poly_mult_128` executes polynomial multiplication and outputs 256b that
 * represented by 3 128b due to code size optimization.
 *
 * Output layout:
 * |            |             |             |
 * |------------|-------------|-------------|
 * | ret.val[0] | h3:h2:00:00 | high   128b |
 * | ret.val[1] |   :m2:m1:00 | middle 128b |
 * | ret.val[2] |   :  :l1:l0 | low    128b |
 */
static inline uint8x16x3_t poly_mult_128(uint8x16_t a, uint8x16_t b)
{
    uint8x16x3_t ret;
    uint8x16_t h, m, l; /* retval high/middle/low */
    uint8x16_t c, d, e;

    h = pmull_high(a, b);                       /* h3:h2:00:00 = a1*b1 */
    l = pmull_low(a, b);                        /*   :  :l1:l0 = a0*b0 */
    c = vextq_u8(b, b, 8);                      /*      :c1:c0 = b0:b1 */
    d = pmull_high(a, c);                       /*   :d2:d1:00 = a1*b0 */
    e = pmull_low(a, c);                        /*   :e2:e1:00 = a0*b1 */
    m = veorq_u8(d, e);                         /*   :m2:m1:00 = d + e */

    ret.val[0] = h;
    ret.val[1] = m;
    ret.val[2] = l;
    return ret;
}

/*
 * Modulo reduction.
 *
 * See: https://www.researchgate.net/publication/285612706_Implementing_GCM_on_ARMv8
 *
 * Section 4.3
 *
 * Modular reduction is slightly more complex. Write the GCM modulus as f(z) =
 * z^128 +r(z), where r(z) = z^7+z^2+z+ 1. The well known approach is to
 * consider that z^128 ≡r(z) (mod z^128 +r(z)), allowing us to write the 256-bit
 * operand to be reduced as a(z) = h(z)z^128 +l(z)≡h(z)r(z) + l(z). That is, we
 * simply multiply the higher part of the operand by r(z) and add it to l(z). If
 * the result is still larger than 128 bits, we reduce again.
 */
static inline uint8x16_t poly_mult_reduce(uint8x16x3_t input)
{
    uint8x16_t const ZERO = vdupq_n_u8(0);
    /* use 'asm' as an optimisation barrier to prevent loading MODULO from memory */
    uint64x2_t r = vreinterpretq_u64_u8(vdupq_n_u8(0x87));
    asm ("" : "+w" (r));
    uint8x16_t const MODULO = vreinterpretq_u8_u64(vshrq_n_u64(r, 64 - 8));
    uint8x16_t h, m, l; /* input high/middle/low 128b */
    uint8x16_t c, d, e, f, g, n, o;
    h = input.val[0];            /* h3:h2:00:00                          */
    m = input.val[1];            /*   :m2:m1:00                          */
    l = input.val[2];            /*   :  :l1:l0                          */
    c = pmull_high(h, MODULO);   /*   :c2:c1:00 = reduction of h3        */
    d = pmull_low(h, MODULO);    /*   :  :d1:d0 = reduction of h2        */
    e = veorq_u8(c, m);          /*   :e2:e1:00 = m2:m1:00 + c2:c1:00    */
    f = pmull_high(e, MODULO);   /*   :  :f1:f0 = reduction of e2        */
    g = vextq_u8(ZERO, e, 8);    /*   :  :g1:00 = e1:00                  */
    n = veorq_u8(d, l);          /*   :  :n1:n0 = d1:d0 + l1:l0          */
    o = veorq_u8(n, f);          /*       o1:o0 = f1:f0 + n1:n0          */
    return veorq_u8(o, g);       /*             = o1:o0 + g1:00          */
}

/*
 * GCM multiplication: c = a times b in GF(2^128)
 */
void mbedtls_aesce_gcm_mult(unsigned char c[16],
                            const unsigned char a[16],
                            const unsigned char b[16])
{
    uint8x16_t va, vb, vc;
    va = vrbitq_u8(vld1q_u8(&a[0]));
    vb = vrbitq_u8(vld1q_u8(&b[0]));
    vc = vrbitq_u8(poly_mult_reduce(poly_mult_128(va, vb)));
    vst1q_u8(&c[0], vc);
}

#endif /* MBEDTLS_GCM_C */

#if defined(MBEDTLS_POP_TARGET_PRAGMA)
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#undef MBEDTLS_POP_TARGET_PRAGMA
#endif

#endif /* MBEDTLS_HAVE_ARM64 */

#endif /* MBEDTLS_AESCE_C */
