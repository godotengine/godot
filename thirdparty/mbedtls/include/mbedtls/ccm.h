/**
 * \file ccm.h
 *
 * \brief CCM combines Counter mode encryption with CBC-MAC authentication
 *        for 128-bit block ciphers.
 *
 * Input to CCM includes the following elements:
 * <ul><li>Payload - data that is both authenticated and encrypted.</li>
 * <li>Associated data (Adata) - data that is authenticated but not
 * encrypted, For example, a header.</li>
 * <li>Nonce - A unique value that is assigned to the payload and the
 * associated data.</li></ul>
 *
 */
/*
 *  Copyright (C) 2006-2018, Arm Limited (or its affiliates), All Rights Reserved
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
 *  This file is part of Mbed TLS (https://tls.mbed.org)
 */

#ifndef MBEDTLS_CCM_H
#define MBEDTLS_CCM_H

#include "cipher.h"

#define MBEDTLS_ERR_CCM_BAD_INPUT       -0x000D /**< Bad input parameters to the function. */
#define MBEDTLS_ERR_CCM_AUTH_FAILED     -0x000F /**< Authenticated decryption failed. */
#define MBEDTLS_ERR_CCM_HW_ACCEL_FAILED -0x0011 /**< CCM hardware accelerator failed. */

#if !defined(MBEDTLS_CCM_ALT)
// Regular implementation
//

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief    The CCM context-type definition. The CCM context is passed
 *           to the APIs called.
 */
typedef struct {
    mbedtls_cipher_context_t cipher_ctx;    /*!< The cipher context used. */
}
mbedtls_ccm_context;

/**
 * \brief           This function initializes the specified CCM context,
 *                  to make references valid, and prepare the context
 *                  for mbedtls_ccm_setkey() or mbedtls_ccm_free().
 *
 * \param ctx       The CCM context to initialize.
 */
void mbedtls_ccm_init( mbedtls_ccm_context *ctx );

/**
 * \brief           This function initializes the CCM context set in the
 *                  \p ctx parameter and sets the encryption key.
 *
 * \param ctx       The CCM context to initialize.
 * \param cipher    The 128-bit block cipher to use.
 * \param key       The encryption key.
 * \param keybits   The key size in bits. This must be acceptable by the cipher.
 *
 * \return          \c 0 on success, or a cipher-specific error code.
 */
int mbedtls_ccm_setkey( mbedtls_ccm_context *ctx,
                        mbedtls_cipher_id_t cipher,
                        const unsigned char *key,
                        unsigned int keybits );

/**
 * \brief   This function releases and clears the specified CCM context
 *          and underlying cipher sub-context.
 *
 * \param ctx       The CCM context to clear.
 */
void mbedtls_ccm_free( mbedtls_ccm_context *ctx );

/**
 * \brief           This function encrypts a buffer using CCM.
 *
 * \param ctx       The CCM context to use for encryption.
 * \param length    The length of the input data in Bytes.
 * \param iv        Initialization vector (nonce).
 * \param iv_len    The length of the IV in Bytes: 7, 8, 9, 10, 11, 12, or 13.
 * \param add       The additional data field.
 * \param add_len   The length of additional data in Bytes.
 *                  Must be less than 2^16 - 2^8.
 * \param input     The buffer holding the input data.
 * \param output    The buffer holding the output data.
 *                  Must be at least \p length Bytes wide.
 * \param tag       The buffer holding the tag.
 * \param tag_len   The length of the tag to generate in Bytes:
 *                  4, 6, 8, 10, 14 or 16.
 *
 * \note            The tag is written to a separate buffer. To concatenate
 *                  the \p tag with the \p output, as done in <em>RFC-3610:
 *                  Counter with CBC-MAC (CCM)</em>, use
 *                  \p tag = \p output + \p length, and make sure that the
 *                  output buffer is at least \p length + \p tag_len wide.
 *
 * \return          \c 0 on success.
 */
int mbedtls_ccm_encrypt_and_tag( mbedtls_ccm_context *ctx, size_t length,
                         const unsigned char *iv, size_t iv_len,
                         const unsigned char *add, size_t add_len,
                         const unsigned char *input, unsigned char *output,
                         unsigned char *tag, size_t tag_len );

/**
 * \brief           This function performs a CCM authenticated decryption of a
 *                  buffer.
 *
 * \param ctx       The CCM context to use for decryption.
 * \param length    The length of the input data in Bytes.
 * \param iv        Initialization vector.
 * \param iv_len    The length of the IV in Bytes: 7, 8, 9, 10, 11, 12, or 13.
 * \param add       The additional data field.
 * \param add_len   The length of additional data in Bytes.
 * \param input     The buffer holding the input data.
 * \param output    The buffer holding the output data.
 * \param tag       The buffer holding the tag.
 * \param tag_len   The length of the tag in Bytes.
 *
 * \return          0 if successful and authenticated, or
 *                  #MBEDTLS_ERR_CCM_AUTH_FAILED if the tag does not match.
 */
int mbedtls_ccm_auth_decrypt( mbedtls_ccm_context *ctx, size_t length,
                      const unsigned char *iv, size_t iv_len,
                      const unsigned char *add, size_t add_len,
                      const unsigned char *input, unsigned char *output,
                      const unsigned char *tag, size_t tag_len );

#ifdef __cplusplus
}
#endif

#else  /* MBEDTLS_CCM_ALT */
#include "ccm_alt.h"
#endif /* MBEDTLS_CCM_ALT */

#ifdef __cplusplus
extern "C" {
#endif

#if defined(MBEDTLS_SELF_TEST) && defined(MBEDTLS_AES_C)
/**
 * \brief          The CCM checkup routine.
 *
 * \return         \c 0 on success, or \c 1 on failure.
 */
int mbedtls_ccm_self_test( int verbose );
#endif /* MBEDTLS_SELF_TEST && MBEDTLS_AES_C */

#ifdef __cplusplus
}
#endif

#endif /* MBEDTLS_CCM_H */
