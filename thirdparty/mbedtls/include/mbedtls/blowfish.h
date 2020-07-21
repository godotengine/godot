/**
 * \file blowfish.h
 *
 * \brief Blowfish block cipher
 */
/*
 *  Copyright (C) 2006-2015, ARM Limited, All Rights Reserved
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 *
 *  This file is provided under the Apache License 2.0, or the
 *  GNU General Public License v2.0 or later.
 *
 *  **********
 *  Apache License 2.0:
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
 *  **********
 *
 *  **********
 *  GNU General Public License v2.0 or later:
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 *  **********
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

#include "platform_util.h"

#define MBEDTLS_BLOWFISH_ENCRYPT     1
#define MBEDTLS_BLOWFISH_DECRYPT     0
#define MBEDTLS_BLOWFISH_MAX_KEY_BITS     448
#define MBEDTLS_BLOWFISH_MIN_KEY_BITS     32
#define MBEDTLS_BLOWFISH_ROUNDS      16         /**< Rounds to use. When increasing this value, make sure to extend the initialisation vectors */
#define MBEDTLS_BLOWFISH_BLOCKSIZE   8          /* Blowfish uses 64 bit blocks */

#if !defined(MBEDTLS_DEPRECATED_REMOVED)
#define MBEDTLS_ERR_BLOWFISH_INVALID_KEY_LENGTH   MBEDTLS_DEPRECATED_NUMERIC_CONSTANT( -0x0016 )
#endif /* !MBEDTLS_DEPRECATED_REMOVED */
#define MBEDTLS_ERR_BLOWFISH_BAD_INPUT_DATA -0x0016 /**< Bad input data. */

#define MBEDTLS_ERR_BLOWFISH_INVALID_INPUT_LENGTH -0x0018 /**< Invalid data input length. */

/* MBEDTLS_ERR_BLOWFISH_HW_ACCEL_FAILED is deprecated and should not be used.
 */
#define MBEDTLS_ERR_BLOWFISH_HW_ACCEL_FAILED                   -0x0017  /**< Blowfish hardware accelerator failed. */

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(MBEDTLS_BLOWFISH_ALT)
// Regular implementation
//

/**
 * \brief          Blowfish context structure
 */
typedef struct mbedtls_blowfish_context
{
    uint32_t P[MBEDTLS_BLOWFISH_ROUNDS + 2];    /*!<  Blowfish round keys    */
    uint32_t S[4][256];                 /*!<  key dependent S-boxes  */
}
mbedtls_blowfish_context;

#else  /* MBEDTLS_BLOWFISH_ALT */
#include "blowfish_alt.h"
#endif /* MBEDTLS_BLOWFISH_ALT */

/**
 * \brief          Initialize a Blowfish context.
 *
 * \param ctx      The Blowfish context to be initialized.
 *                 This must not be \c NULL.
 */
void mbedtls_blowfish_init( mbedtls_blowfish_context *ctx );

/**
 * \brief          Clear a Blowfish context.
 *
 * \param ctx      The Blowfish context to be cleared.
 *                 This may be \c NULL, in which case this function
 *                 returns immediately. If it is not \c NULL, it must
 *                 point to an initialized Blowfish context.
 */
void mbedtls_blowfish_free( mbedtls_blowfish_context *ctx );

/**
 * \brief          Perform a Blowfish key schedule operation.
 *
 * \param ctx      The Blowfish context to perform the key schedule on.
 * \param key      The encryption key. This must be a readable buffer of
 *                 length \p keybits Bits.
 * \param keybits  The length of \p key in Bits. This must be between
 *                 \c 32 and \c 448 and a multiple of \c 8.
 *
 * \return         \c 0 if successful.
 * \return         A negative error code on failure.
 */
int mbedtls_blowfish_setkey( mbedtls_blowfish_context *ctx, const unsigned char *key,
                     unsigned int keybits );

/**
 * \brief          Perform a Blowfish-ECB block encryption/decryption operation.
 *
 * \param ctx      The Blowfish context to use. This must be initialized
 *                 and bound to a key.
 * \param mode     The mode of operation. Possible values are
 *                 #MBEDTLS_BLOWFISH_ENCRYPT for encryption, or
 *                 #MBEDTLS_BLOWFISH_DECRYPT for decryption.
 * \param input    The input block. This must be a readable buffer
 *                 of size \c 8 Bytes.
 * \param output   The output block. This must be a writable buffer
 *                 of size \c 8 Bytes.
 *
 * \return         \c 0 if successful.
 * \return         A negative error code on failure.
 */
int mbedtls_blowfish_crypt_ecb( mbedtls_blowfish_context *ctx,
                        int mode,
                        const unsigned char input[MBEDTLS_BLOWFISH_BLOCKSIZE],
                        unsigned char output[MBEDTLS_BLOWFISH_BLOCKSIZE] );

#if defined(MBEDTLS_CIPHER_MODE_CBC)
/**
 * \brief          Perform a Blowfish-CBC buffer encryption/decryption operation.
 *
 * \note           Upon exit, the content of the IV is updated so that you can
 *                 call the function same function again on the following
 *                 block(s) of data and get the same result as if it was
 *                 encrypted in one call. This allows a "streaming" usage.
 *                 If on the other hand you need to retain the contents of the
 *                 IV, you should either save it manually or use the cipher
 *                 module instead.
 *
 * \param ctx      The Blowfish context to use. This must be initialized
 *                 and bound to a key.
 * \param mode     The mode of operation. Possible values are
 *                 #MBEDTLS_BLOWFISH_ENCRYPT for encryption, or
 *                 #MBEDTLS_BLOWFISH_DECRYPT for decryption.
 * \param length   The length of the input data in Bytes. This must be
 *                 multiple of \c 8.
 * \param iv       The initialization vector. This must be a read/write buffer
 *                 of length \c 8 Bytes. It is updated by this function.
 * \param input    The input data. This must be a readable buffer of length
 *                 \p length Bytes.
 * \param output   The output data. This must be a writable buffer of length
 *                 \p length Bytes.
 *
 * \return         \c 0 if successful.
 * \return         A negative error code on failure.
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
 * \brief          Perform a Blowfish CFB buffer encryption/decryption operation.
 *
 * \note           Upon exit, the content of the IV is updated so that you can
 *                 call the function same function again on the following
 *                 block(s) of data and get the same result as if it was
 *                 encrypted in one call. This allows a "streaming" usage.
 *                 If on the other hand you need to retain the contents of the
 *                 IV, you should either save it manually or use the cipher
 *                 module instead.
 *
 * \param ctx      The Blowfish context to use. This must be initialized
 *                 and bound to a key.
 * \param mode     The mode of operation. Possible values are
 *                 #MBEDTLS_BLOWFISH_ENCRYPT for encryption, or
 *                 #MBEDTLS_BLOWFISH_DECRYPT for decryption.
 * \param length   The length of the input data in Bytes.
 * \param iv_off   The offset in the initialiation vector.
 *                 The value pointed to must be smaller than \c 8 Bytes.
 *                 It is updated by this function to support the aforementioned
 *                 streaming usage.
 * \param iv       The initialization vector. This must be a read/write buffer
 *                 of size \c 8 Bytes. It is updated after use.
 * \param input    The input data. This must be a readable buffer of length
 *                 \p length Bytes.
 * \param output   The output data. This must be a writable buffer of length
 *                 \p length Bytes.
 *
 * \return         \c 0 if successful.
 * \return         A negative error code on failure.
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
 * \brief      Perform a Blowfish-CTR buffer encryption/decryption operation.
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
 * \param ctx           The Blowfish context to use. This must be initialized
 *                      and bound to a key.
 * \param length        The length of the input data in Bytes.
 * \param nc_off        The offset in the current stream_block (for resuming
 *                      within current cipher stream). The offset pointer
 *                      should be \c 0 at the start of a stream and must be
 *                      smaller than \c 8. It is updated by this function.
 * \param nonce_counter The 64-bit nonce and counter. This must point to a
 *                      read/write buffer of length \c 8 Bytes.
 * \param stream_block  The saved stream-block for resuming. This must point to
 *                      a read/write buffer of length \c 8 Bytes.
 * \param input         The input data. This must be a readable buffer of
 *                      length \p length Bytes.
 * \param output        The output data. This must be a writable buffer of
 *                      length \p length Bytes.
 *
 * \return              \c 0 if successful.
 * \return              A negative error code on failure.
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
