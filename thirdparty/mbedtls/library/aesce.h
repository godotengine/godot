/**
 * \file aesce.h
 *
 * \brief Support hardware AES acceleration on Armv8-A processors with
 *        the Armv8-A Cryptographic Extension in AArch64 execution state.
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
#ifndef MBEDTLS_AESCE_H
#define MBEDTLS_AESCE_H

#include "mbedtls/build_info.h"

#include "mbedtls/aes.h"


#if defined(MBEDTLS_HAVE_ASM) && defined(__GNUC__) && \
    defined(__aarch64__) && !defined(MBEDTLS_HAVE_ARM64)
#define MBEDTLS_HAVE_ARM64
#endif

#if defined(MBEDTLS_HAVE_ARM64)

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief          Internal function to detect the crypto extension in CPUs.
 *
 * \return         1 if CPU has support for the feature, 0 otherwise
 */
int mbedtls_aesce_has_support(void);

/**
 * \brief          Internal AES-ECB block encryption and decryption
 *
 * \param ctx      AES context
 * \param mode     MBEDTLS_AES_ENCRYPT or MBEDTLS_AES_DECRYPT
 * \param input    16-byte input block
 * \param output   16-byte output block
 *
 * \return         0 on success (cannot fail)
 */
int mbedtls_aesce_crypt_ecb(mbedtls_aes_context *ctx,
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
void mbedtls_aesce_gcm_mult(unsigned char c[16],
                            const unsigned char a[16],
                            const unsigned char b[16]);


/**
 * \brief           Internal round key inversion. This function computes
 *                  decryption round keys from the encryption round keys.
 *
 * \param invkey    Round keys for the equivalent inverse cipher
 * \param fwdkey    Original round keys (for encryption)
 * \param nr        Number of rounds (that is, number of round keys minus one)
 */
void mbedtls_aesce_inverse_key(unsigned char *invkey,
                               const unsigned char *fwdkey,
                               int nr);

/**
 * \brief           Internal key expansion for encryption
 *
 * \param rk        Destination buffer where the round keys are written
 * \param key       Encryption key
 * \param bits      Key size in bits (must be 128, 192 or 256)
 *
 * \return          0 if successful, or MBEDTLS_ERR_AES_INVALID_KEY_LENGTH
 */
int mbedtls_aesce_setkey_enc(unsigned char *rk,
                             const unsigned char *key,
                             size_t bits);

#ifdef __cplusplus
}
#endif

#endif /* MBEDTLS_HAVE_ARM64 */

#endif /* MBEDTLS_AESCE_H */
