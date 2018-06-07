/**
 * \file blowfish.h
 *
 * \brief Blowfish block cipher
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
#ifndef MBEDTLS_BLOWFISH_H
#define MBEDTLS_BLOWFISH_H

#if !defined(MBEDTLS_CONFIG_FILE)
#include "config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#include <stddef.h>
#include <stdint.h>

#define MBEDTLS_BLOWFISH_ENCRYPT     1
#define MBEDTLS_BLOWFISH_DECRYPT     0
#define MBEDTLS_BLOWFISH_MAX_KEY_BITS     448
#define MBEDTLS_BLOWFISH_MIN_KEY_BITS     32
#define MBEDTLS_BLOWFISH_ROUNDS      16         /**< Rounds to use. When increasing this value, make sure to extend the initialisation vectors */
#define MBEDTLS_BLOWFISH_BLOCKSIZE   8          /* Blowfish uses 64 bit blocks */

#define MBEDTLS_ERR_BLOWFISH_INVALID_KEY_LENGTH                -0x0016  /**< Invalid key length. */
#define MBEDTLS_ERR_BLOWFISH_HW_ACCEL_FAILED                   -0x0017  /**< Blowfish hardware accelerator failed. */
#define MBEDTLS_ERR_BLOWFISH_INVALID_INPUT_LENGTH              -0x0018  /**< Invalid data input length. */

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(MBEDTLS_BLOWFISH_ALT)
// Regular implementation
//

/**
 * \brief          Blowfish context structure
 */
typedef struct
{
    uint32_t P[MBEDTLS_BLOWFISH_ROUNDS + 2];    /*!<  Blowfish round keys    */
    uint32_t S[4][256];                 /*!<  key dependent S-boxes  */
}
mbedtls_blowfish_context;

#else  /* MBEDTLS_BLOWFISH_ALT */
#include "blowfish_alt.h"
#endif /* MBEDTLS_BLOWFISH_ALT */

/**
 * \brief          Initialize Blowfish context
 *
 * \param ctx      Blowfish context to be initialized
 */
void mbedtls_blowfish_init( mbedtls_blowfish_context *ctx );

/**
 * \brief          Clear Blowfish context
 *
 * \param ctx      Blowfish context to be cleared
 */
void mbedtls_blowfish_free( mbedtls_blowfish_context *ctx );

/**
 * \brief          Blowfish key schedule
 *
 * \param ctx      Blowfish context to be initialized
 * \param key      encryption key
 * \param keybits  must be between 32 and 448 bits
 *
 * \return         0 if successful, or MBEDTLS_ERR_BLOWFISH_INVALID_KEY_LENGTH
 */
int mbedtls_blowfish_setkey( mbedtls_blowfish_context *ctx, const unsigned char *key,
                     unsigned int keybits );

/**
 * \brief          Blowfish-ECB block encryption/decryption
 *
 * \param ctx      Blowfish context
 * \param mode     MBEDTLS_BLOWFISH_ENCRYPT or MBEDTLS_BLOWFISH_DECRYPT
 * \param input    8-byte input block
 * \param output   8-byte output block
 *
 * \return         0 if successful
 */
int mbedtls_blowfish_crypt_ecb( mbedtls_blowfish_context *ctx,
                        int mode,
                        const unsigned char input[MBEDTLS_BLOWFISH_BLOCKSIZE],
                        unsigned char output[MBEDTLS_BLOWFISH_BLOCKSIZE] );

#if defined(MBEDTLS_CIPHER_MODE_CBC)
/**
 * \brief          Blowfish-CBC buffer encryption/decryption
 *                 Length should be a multiple of the block
 *                 size (8 bytes)
 *
 * \note           Upon exit, the content of the IV is updated so that you can
 *                 call the function same function again on the following
 *                 block(s) of data and get the same result as if it was
 *                 encrypted in one call. This allows a "streaming" usage.
 *                 If on the other hand you need to retain the contents of the
 *                 IV, you should either save it manually or use the cipher
 *                 module instead.
 *
 * \param ctx      Blowfish context
 * \param mode     MBEDTLS_BLOWFISH_ENCRYPT or MBEDTLS_BLOWFISH_DECRYPT
 * \param length   length of the input data
 * \param iv       initialization vector (updated after use)
 * \param input    buffer holding the input data
 * \param output   buffer holding the output data
 *
 * \return         0 if successful, or
 *                 MBEDTLS_ERR_BLOWFISH_INVALID_INPUT_LENGTH
 */
int mbedtls_blowfish_crypt_cbc( mbedtls_blowfish_context *ctx,
                        int mode,
                        size_t length,
                        unsigned char iv[MBEDTLS_BLOWFISH_BLOCKSIZE],
                        const unsigned char *input,
                        unsigned char *output );
#endif /* MBEDTLS_CIPHER_MODE_CBC */

#if defined(MBEDTLS_CIPHER_MODE_CFB)
/**
 * \brief          Blowfish CFB buffer encryption/decryption.
 *
 * \note           Upon exit, the content of the IV is updated so that you can
 *                 call the function same function again on the following
 *                 block(s) of data and get the same result as if it was
 *                 encrypted in one call. This allows a "streaming" usage.
 *                 If on the other hand you need to retain the contents of the
 *                 IV, you should either save it manually or use the cipher
 *                 module instead.
 *
 * \param ctx      Blowfish context
 * \param mode     MBEDTLS_BLOWFISH_ENCRYPT or MBEDTLS_BLOWFISH_DECRYPT
 * \param length   length of the input data
 * \param iv_off   offset in IV (updated after use)
 * \param iv       initialization vector (updated after use)
 * \param input    buffer holding the input data
 * \param output   buffer holding the output data
 *
 * \return         0 if successful
 */
int mbedtls_blowfish_crypt_cfb64( mbedtls_blowfish_context *ctx,
                          int mode,
                          size_t length,
                          size_t *iv_off,
                          unsigned char iv[MBEDTLS_BLOWFISH_BLOCKSIZE],
                          const unsigned char *input,
                          unsigned char *output );
#endif /*MBEDTLS_CIPHER_MODE_CFB */

#if defined(MBEDTLS_CIPHER_MODE_CTR)
/**
 * \brief               Blowfish-CTR buffer encryption/decryption
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
 *             With this strategy, you must not encrypt more than 2**64
 *             blocks of data with the same key.
 *
 *             2. You can encrypt separate messages by dividing the \p
 *             nonce_counter buffer in two areas: the first one used for a
 *             per-message nonce, handled by yourself, and the second one
 *             updated by this function internally.
 *
 *             For example, you might reserve the first 4 bytes for the
 *             per-message nonce, and the last 4 bytes for internal use. In that
 *             case, before calling this function on a new message you need to
 *             set the first 4 bytes of \p nonce_counter to your chosen nonce
 *             value, the last 4 to 0, and \p nc_off to 0 (which will cause \p
 *             stream_block to be ignored). That way, you can encrypt at most
 *             2**32 messages of up to 2**32 blocks each with the same key.
 *
 *             The per-message nonce (or information sufficient to reconstruct
 *             it) needs to be communicated with the ciphertext and must be unique.
 *             The recommended way to ensure uniqueness is to use a message
 *             counter.
 *
 *             Note that for both stategies, sizes are measured in blocks and
 *             that a Blowfish block is 8 bytes.
 *
 * \warning    Upon return, \p stream_block contains sensitive data. Its
 *             content must not be written to insecure storage and should be
 *             securely discarded as soon as it's no longer needed.
 *
 * \param ctx           Blowfish context
 * \param length        The length of the data
 * \param nc_off        The offset in the current stream_block (for resuming
 *                      within current cipher stream). The offset pointer to
 *                      should be 0 at the start of a stream.
 * \param nonce_counter The 64-bit nonce and counter.
 * \param stream_block  The saved stream-block for resuming. Is overwritten
 *                      by the function.
 * \param input         The input data stream
 * \param output        The output data stream
 *
 * \return         0 if successful
 */
int mbedtls_blowfish_crypt_ctr( mbedtls_blowfish_context *ctx,
                        size_t length,
                        size_t *nc_off,
                        unsigned char nonce_counter[MBEDTLS_BLOWFISH_BLOCKSIZE],
                        unsigned char stream_block[MBEDTLS_BLOWFISH_BLOCKSIZE],
                        const unsigned char *input,
                        unsigned char *output );
#endif /* MBEDTLS_CIPHER_MODE_CTR */

#ifdef __cplusplus
}
#endif

#endif /* blowfish.h */
