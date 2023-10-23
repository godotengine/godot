/**
 * \file aesni.h
 *
 * \brief AES-NI for hardware AES acceleration on some Intel processors
 *
 * \warning These functions are only for internal use by other library
 *          functions; you must not call them directly.
 */
/*
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
#ifndef MBEDTLS_AESNI_H
#define MBEDTLS_AESNI_H

#if !defined(MBEDTLS_CONFIG_FILE)
#include "mbedtls/config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#include "mbedtls/aes.h"

#define MBEDTLS_AESNI_AES      0x02000000u
#define MBEDTLS_AESNI_CLMUL    0x00000002u

#if !defined(MBEDTLS_HAVE_X86_64) && \
    (defined(__amd64__) || defined(__x86_64__) || \
    defined(_M_X64) || defined(_M_AMD64)) && \
    !defined(_M_ARM64EC)
#define MBEDTLS_HAVE_X86_64
#endif

#if !defined(MBEDTLS_HAVE_X86) && \
    (defined(__i386__) || defined(_M_IX86))
#define MBEDTLS_HAVE_X86
#endif

#if defined(MBEDTLS_AESNI_C) && \
    (defined(MBEDTLS_HAVE_X86_64) || defined(MBEDTLS_HAVE_X86))

/* Can we do AESNI with intrinsics?
 * (Only implemented with certain compilers, only for certain targets.)
 *
 * NOTE: MBEDTLS_AESNI_HAVE_INTRINSICS and MBEDTLS_AESNI_HAVE_CODE are internal
 *       macros that may change in future releases.
 */
#undef MBEDTLS_AESNI_HAVE_INTRINSICS
#if defined(_MSC_VER)
/* Visual Studio supports AESNI intrinsics since VS 2008 SP1. We only support
 * VS 2013 and up for other reasons anyway, so no need to check the version. */
#define MBEDTLS_AESNI_HAVE_INTRINSICS
#endif
/* GCC-like compilers: currently, we only support intrinsics if the requisite
 * target flag is enabled when building the library (e.g. `gcc -mpclmul -msse2`
 * or `clang -maes -mpclmul`). */
#if defined(__GNUC__) && defined(__AES__) && defined(__PCLMUL__)
#define MBEDTLS_AESNI_HAVE_INTRINSICS
#endif

/* Choose the implementation of AESNI, if one is available. */
#undef MBEDTLS_AESNI_HAVE_CODE
/* To minimize disruption when releasing the intrinsics-based implementation,
 * favor the assembly-based implementation if it's available. We intend to
 * revise this in a later release of Mbed TLS 3.x. In the long run, we will
 * likely remove the assembly implementation. */
#if defined(MBEDTLS_HAVE_ASM) && \
    defined(__GNUC__) && defined(MBEDTLS_HAVE_X86_64)
/* Can we do AESNI with inline assembly?
 * (Only implemented with gas syntax, only for 64-bit.)
 */
#define MBEDTLS_AESNI_HAVE_CODE 1 // via assembly
#elif defined(MBEDTLS_AESNI_HAVE_INTRINSICS)
#define MBEDTLS_AESNI_HAVE_CODE 2 // via intrinsics
#endif

#if defined(MBEDTLS_AESNI_HAVE_CODE)

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief          Internal function to detect the AES-NI feature in CPUs.
 *
 * \note           This function is only for internal use by other library
 *                 functions; you must not call it directly.
 *
 * \param what     The feature to detect
 *                 (MBEDTLS_AESNI_AES or MBEDTLS_AESNI_CLMUL)
 *
 * \return         1 if CPU has support for the feature, 0 otherwise
 */
int mbedtls_aesni_has_support(unsigned int what);

/**
 * \brief          Internal AES-NI AES-ECB block encryption and decryption
 *
 * \note           This function is only for internal use by other library
 *                 functions; you must not call it directly.
 *
 * \param ctx      AES context
 * \param mode     MBEDTLS_AES_ENCRYPT or MBEDTLS_AES_DECRYPT
 * \param input    16-byte input block
 * \param output   16-byte output block
 *
 * \return         0 on success (cannot fail)
 */
int mbedtls_aesni_crypt_ecb(mbedtls_aes_context *ctx,
                            int mode,
                            const unsigned char input[16],
                            unsigned char output[16]);

/**
 * \brief          Internal GCM multiplication: c = a * b in GF(2^128)
 *
 * \note           This function is only for internal use by other library
 *                 functions; you must not call it directly.
 *
 * \param c        Result
 * \param a        First operand
 * \param b        Second operand
 *
 * \note           Both operands and result are bit strings interpreted as
 *                 elements of GF(2^128) as per the GCM spec.
 */
void mbedtls_aesni_gcm_mult(unsigned char c[16],
                            const unsigned char a[16],
                            const unsigned char b[16]);

/**
 * \brief           Internal round key inversion. This function computes
 *                  decryption round keys from the encryption round keys.
 *
 * \note            This function is only for internal use by other library
 *                  functions; you must not call it directly.
 *
 * \param invkey    Round keys for the equivalent inverse cipher
 * \param fwdkey    Original round keys (for encryption)
 * \param nr        Number of rounds (that is, number of round keys minus one)
 */
void mbedtls_aesni_inverse_key(unsigned char *invkey,
                               const unsigned char *fwdkey,
                               int nr);

/**
 * \brief           Internal key expansion for encryption
 *
 * \note            This function is only for internal use by other library
 *                  functions; you must not call it directly.
 *
 * \param rk        Destination buffer where the round keys are written
 * \param key       Encryption key
 * \param bits      Key size in bits (must be 128, 192 or 256)
 *
 * \return          0 if successful, or MBEDTLS_ERR_AES_INVALID_KEY_LENGTH
 */
int mbedtls_aesni_setkey_enc(unsigned char *rk,
                             const unsigned char *key,
                             size_t bits);

#ifdef __cplusplus
}
#endif

#endif /* MBEDTLS_AESNI_HAVE_CODE */
#endif /* MBEDTLS_AESNI_C && (MBEDTLS_HAVE_X86_64 || MBEDTLS_HAVE_X86) */

#endif /* MBEDTLS_AESNI_H */
