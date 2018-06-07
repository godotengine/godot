/**
 * \file camellia.h
 *
 * \brief Camellia block cipher
 */
/*
 *  Copyright (C) 2006-2015, ARM Limited, All Rights Reserved
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
 *
 *  This file is part of mbed TLS (https://tls.mbed.org)
 */
#ifndef MBEDTLS_CAMELLIA_H
#define MBEDTLS_CAMELLIA_H

#if !defined(MBEDTLS_CONFIG_FILE)
#include "config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#include <stddef.h>
#include <stdint.h>

#define MBEDTLS_CAMELLIA_ENCRYPT     1
#define MBEDTLS_CAMELLIA_DECRYPT     0

#define MBEDTLS_ERR_CAMELLIA_INVALID_KEY_LENGTH           -0x0024  /**< Invalid key length. */
#define MBEDTLS_ERR_CAMELLIA_INVALID_INPUT_LENGTH         -0x0026  /**< Invalid data input length. */
#define MBEDTLS_ERR_CAMELLIA_HW_ACCEL_FAILED              -0x0027  /**< Camellia hardware accelerator failed. */

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(MBEDTLS_CAMELLIA_ALT)
// Regular implementation
//

/**
 * \brief          CAMELLIA context structure
 */
typedef struct
{
    int nr;                     /*!<  number of rounds  */
    uint32_t rk[68];            /*!<  CAMELLIA round keys    */
}
mbedtls_camellia_context;

#else  /* MBEDTLS_CAMELLIA_ALT */
#include "camellia_alt.h"
#endif /* MBEDTLS_CAMELLIA_ALT */

/**
 * \brief          Initialize CAMELLIA context
 *
 * \param ctx      CAMELLIA context to be initialized
 */
void mbedtls_camellia_init( mbedtls_camellia_context *ctx );

/**
 * \brief          Clear CAMELLIA context
 *
 * \param ctx      CAMELLIA context to be cleared
 */
void mbedtls_camellia_free( mbedtls_camellia_context *ctx );

/**
 * \brief          CAMELLIA key schedule (encryption)
 *
 * \param ctx      CAMELLIA context to be initialized
 * \param key      encryption key
 * \param keybits  must be 128, 192 or 256
 *
 * \return         0 if successful, or MBEDTLS_ERR_CAMELLIA_INVALID_KEY_LENGTH
 */
int mbedtls_camellia_setkey_enc( mbedtls_camellia_context *ctx, const unsigned char *key,
                         unsigned int keybits );

/**
 * \brief          CAMELLIA key schedule (decryption)
 *
 * \param ctx      CAMELLIA context to be initialized
 * \param key      decryption key
 * \param keybits  must be 128, 192 or 256
 *
 * \return         0 if successful, or MBEDTLS_ERR_CAMELLIA_INVALID_KEY_LENGTH
 */
int mbedtls_camellia_setkey_dec( mbedtls_camellia_context *ctx, const unsigned char *key,
                         unsigned int keybits );

/**
 * \brief          CAMELLIA-ECB block encryption/decryption
 *
 * \param ctx      CAMELLIA context
 * \param mode     MBEDTLS_CAMELLIA_ENCRYPT or MBEDTLS_CAMELLIA_DECRYPT
 * \param input    16-byte input block
 * \param output   16-byte output block
 *
 * \return         0 if successful
 */
int mbedtls_camellia_crypt_ecb( mbedtls_camellia_context *ctx,
                    int mode,
                    const unsigned char input[16],
                    unsigned char output[16] );

#if defined(MBEDTLS_CIPHER_MODE_CBC)
/**
 * \brief          CAMELLIA-CBC buffer encryption/decryption
 *                 Length should be a multiple of the block
 *                 size (16 bytes)
 *
 * \note           Upon exit, the content of the IV is updated so that you can
 *                 call the function same function again on the following
 *                 block(s) of data and get the same result as if it was
 *                 encrypted in one call. This allows a "streaming" usage.
 *                 If on the other hand you need to retain the contents of the
 *                 IV, you should either save it manually or use the cipher
 *                 module instead.
 *
 * \param ctx      CAMELLIA context
 * \param mode     MBEDTLS_CAMELLIA_ENCRYPT or MBEDTLS_CAMELLIA_DECRYPT
 * \param length   length of the input data
 * \param iv       initialization vector (updated after use)
 * \param input    buffer holding the input data
 * \param output   buffer holding the output data
 *
 * \return         0 if successful, or
 *                 MBEDTLS_ERR_CAMELLIA_INVALID_INPUT_LENGTH
 */
int mbedtls_camellia_crypt_cbc( mbedtls_camellia_context *ctx,
                    int mode,
                    size_t length,
                    unsigned char iv[16],
                    const unsigned char *input,
                    unsigned char *output );
#endif /* MBEDTLS_CIPHER_MODE_CBC */

#if defined(MBEDTLS_CIPHER_MODE_CFB)
/**
 * \brief          CAMELLIA-CFB128 buffer encryption/decryption
 *
 * Note: Due to the nature of CFB you should use the same key schedule for
 * both encryption and decryption. So a context initialized with
 * mbedtls_camellia_setkey_enc() for both MBEDTLS_CAMELLIA_ENCRYPT and CAMELLIE_DECRYPT.
 *
 * \note           Upon exit, the content of the IV is updated so that you can
 *                 call the function same function again on the following
 *                 block(s) of data and get the same result as if it was
 *                 encrypted in one call. This allows a "streaming" usage.
 *                 If on the other hand you need to retain the contents of the
 *                 IV, you should either save it manually or use the cipher
 *                 module instead.
 *
 * \param ctx      CAMELLIA context
 * \param mode     MBEDTLS_CAMELLIA_ENCRYPT or MBEDTLS_CAMELLIA_DECRYPT
 * \param length   length of the input data
 * \param iv_off   offset in IV (updated after use)
 * \param iv       initialization vector (updated after use)
 * \param input    buffer holding the input data
 * \param output   buffer holding the output data
 *
 * \return         0 if successful, or
 *                 MBEDTLS_ERR_CAMELLIA_INVALID_INPUT_LENGTH
 */
int mbedtls_camellia_crypt_cfb128( mbedtls_camellia_context *ctx,
                       int mode,
                       size_t length,
                       size_t *iv_off,
                       unsigned char iv[16],
                       const unsigned char *input,
                       unsigned char *output );
#endif /* MBEDTLS_CIPHER_MODE_CFB */

#if defined(MBEDTLS_CIPHER_MODE_CTR)
/**
 * \brief               CAMELLIA-CTR buffer encryption/decryption
 *
 * Note: Due to the nature of CTR you should use the same key schedule for
 * both encryption and decryption. So a context initialized with
 * mbedtls_camellia_setkey_enc() for both MBEDTLS_CAMELLIA_ENCRYPT and MBEDTLS_CAMELLIA_DECRYPT.
 *
 * \warning    You must never reuse a nonce value with the same key. Doing so
 *             would void the encryption for the two messages encrypted with
 *             the same nonce and key.
 *
 *             There are two common strategies for managing nonces with CTR:
 *
 *             1. You can handle everything as a single message processed over
 *             successive calls to this function. In that case, you want to
 *             set \p nonce_counter and \p nc_off to 0 for the first call, and
 *             then preserve the values of \p nonce_counter, \p nc_off and \p
 *             stream_block across calls to this function as they will be
 *             updated by this function.
 *
 *             With this strategy, you must not encrypt more than 2**128
 *             blocks of data with the same key.
 *
 *             2. You can encrypt separate messages by dividing the \p
 *             nonce_counter buffer in two areas: the first one used for a
 *             per-message nonce, handled by yourself, and the second one
 *             updated by this function internally.
 *
 *             For example, you might reserve the first 12 bytes for the
 *             per-message nonce, and the last 4 bytes for internal use. In that
 *             case, before calling this function on a new message you need to
 *             set the first 12 bytes of \p nonce_counter to your chosen nonce
 *             value, the last 4 to 0, and \p nc_off to 0 (which will cause \p
 *             stream_block to be ignored). That way, you can encrypt at most
 *             2**96 messages of up to 2**32 blocks each with the same key.
 *
 *             The per-message nonce (or information sufficient to reconstruct
 *             it) needs to be communicated with the ciphertext and must be unique.
 *             The recommended way to ensure uniqueness is to use a message
 *             counter. An alternative is to generate random nonces, but this
 *             limits the number of messages that can be securely encrypted:
 *             for example, with 96-bit random nonces, you should not encrypt
 *             more than 2**32 messages with the same key.
 *
 *             Note that for both stategies, sizes are measured in blocks and
 *             that a CAMELLIA block is 16 bytes.
 *
 * \warning    Upon return, \p stream_block contains sensitive data. Its
 *             content must not be written to insecure storage and should be
 *             securely discarded as soon as it's no longer needed.
 *
 * \param ctx           CAMELLIA context
 * \param length        The length of the data
 * \param nc_off        The offset in the current stream_block (for resuming
 *                      within current cipher stream). The offset pointer to
 *                      should be 0 at the start of a stream.
 * \param nonce_counter The 128-bit nonce and counter.
 * \param stream_block  The saved stream-block for resuming. Is overwritten
 *                      by the function.
 * \param input         The input data stream
 * \param output        The output data stream
 *
 * \return         0 if successful
 */
int mbedtls_camellia_crypt_ctr( mbedtls_camellia_context *ctx,
                       size_t length,
                       size_t *nc_off,
                       unsigned char nonce_counter[16],
                       unsigned char stream_block[16],
                       const unsigned char *input,
                       unsigned char *output );
#endif /* MBEDTLS_CIPHER_MODE_CTR */

/**
 * \brief          Checkup routine
 *
 * \return         0 if successful, or 1 if the test failed
 */
int mbedtls_camellia_self_test( int verbose );

#ifdef __cplusplus
}
#endif

#endif /* camellia.h */
