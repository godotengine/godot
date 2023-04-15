/**
 * \file gcm.h
 *
 * \brief This file contains GCM definitions and functions.
 *
 * The Galois/Counter Mode (GCM) for 128-bit block ciphers is defined
 * in <em>D. McGrew, J. Viega, The Galois/Counter Mode of Operation
 * (GCM), Natl. Inst. Stand. Technol.</em>
 *
 * For more information on GCM, see <em>NIST SP 800-38D: Recommendation for
 * Block Cipher Modes of Operation: Galois/Counter Mode (GCM) and GMAC</em>.
 *
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

#ifndef MBEDTLS_GCM_H
#define MBEDTLS_GCM_H

#if !defined(MBEDTLS_CONFIG_FILE)
#include "mbedtls/config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#include "mbedtls/cipher.h"

#include <stdint.h>

#define MBEDTLS_GCM_ENCRYPT     1
#define MBEDTLS_GCM_DECRYPT     0

/** Authenticated decryption failed. */
#define MBEDTLS_ERR_GCM_AUTH_FAILED                       -0x0012

/* MBEDTLS_ERR_GCM_HW_ACCEL_FAILED is deprecated and should not be used. */
/** GCM hardware accelerator failed. */
#define MBEDTLS_ERR_GCM_HW_ACCEL_FAILED                   -0x0013

/** Bad input parameters to function. */
#define MBEDTLS_ERR_GCM_BAD_INPUT                         -0x0014

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(MBEDTLS_GCM_ALT)

/**
 * \brief          The GCM context structure.
 */
typedef struct mbedtls_gcm_context
{
    mbedtls_cipher_context_t cipher_ctx;  /*!< The cipher context used. */
    uint64_t HL[16];                      /*!< Precalculated HTable low. */
    uint64_t HH[16];                      /*!< Precalculated HTable high. */
    uint64_t len;                         /*!< The total length of the encrypted data. */
    uint64_t add_len;                     /*!< The total length of the additional data. */
    unsigned char base_ectr[16];          /*!< The first ECTR for tag. */
    unsigned char y[16];                  /*!< The Y working value. */
    unsigned char buf[16];                /*!< The buf working value. */
    int mode;                             /*!< The operation to perform:
                                               #MBEDTLS_GCM_ENCRYPT or
                                               #MBEDTLS_GCM_DECRYPT. */
}
mbedtls_gcm_context;

#else  /* !MBEDTLS_GCM_ALT */
#include "gcm_alt.h"
#endif /* !MBEDTLS_GCM_ALT */

/**
 * \brief           This function initializes the specified GCM context,
 *                  to make references valid, and prepares the context
 *                  for mbedtls_gcm_setkey() or mbedtls_gcm_free().
 *
 *                  The function does not bind the GCM context to a particular
 *                  cipher, nor set the key. For this purpose, use
 *                  mbedtls_gcm_setkey().
 *
 * \param ctx       The GCM context to initialize. This must not be \c NULL.
 */
void mbedtls_gcm_init( mbedtls_gcm_context *ctx );

/**
 * \brief           This function associates a GCM context with a
 *                  cipher algorithm and a key.
 *
 * \param ctx       The GCM context. This must be initialized.
 * \param cipher    The 128-bit block cipher to use.
 * \param key       The encryption key. This must be a readable buffer of at
 *                  least \p keybits bits.
 * \param keybits   The key size in bits. Valid options are:
 *                  <ul><li>128 bits</li>
 *                  <li>192 bits</li>
 *                  <li>256 bits</li></ul>
 *
 * \return          \c 0 on success.
 * \return          A cipher-specific error code on failure.
 */
int mbedtls_gcm_setkey( mbedtls_gcm_context *ctx,
                        mbedtls_cipher_id_t cipher,
                        const unsigned char *key,
                        unsigned int keybits );

/**
 * \brief           This function performs GCM encryption or decryption of a buffer.
 *
 * \note            For encryption, the output buffer can be the same as the
 *                  input buffer. For decryption, the output buffer cannot be
 *                  the same as input buffer. If the buffers overlap, the output
 *                  buffer must trail at least 8 Bytes behind the input buffer.
 *
 * \warning         When this function performs a decryption, it outputs the
 *                  authentication tag and does not verify that the data is
 *                  authentic. You should use this function to perform encryption
 *                  only. For decryption, use mbedtls_gcm_auth_decrypt() instead.
 *
 * \param ctx       The GCM context to use for encryption or decryption. This
 *                  must be initialized.
 * \param mode      The operation to perform:
 *                  - #MBEDTLS_GCM_ENCRYPT to perform authenticated encryption.
 *                    The ciphertext is written to \p output and the
 *                    authentication tag is written to \p tag.
 *                  - #MBEDTLS_GCM_DECRYPT to perform decryption.
 *                    The plaintext is written to \p output and the
 *                    authentication tag is written to \p tag.
 *                    Note that this mode is not recommended, because it does
 *                    not verify the authenticity of the data. For this reason,
 *                    you should use mbedtls_gcm_auth_decrypt() instead of
 *                    calling this function in decryption mode.
 * \param length    The length of the input data, which is equal to the length
 *                  of the output data.
 * \param iv        The initialization vector. This must be a readable buffer of
 *                  at least \p iv_len Bytes.
 * \param iv_len    The length of the IV.
 * \param add       The buffer holding the additional data. This must be of at
 *                  least that size in Bytes.
 * \param add_len   The length of the additional data.
 * \param input     The buffer holding the input data. If \p length is greater
 *                  than zero, this must be a readable buffer of at least that
 *                  size in Bytes.
 * \param output    The buffer for holding the output data. If \p length is greater
 *                  than zero, this must be a writable buffer of at least that
 *                  size in Bytes.
 * \param tag_len   The length of the tag to generate.
 * \param tag       The buffer for holding the tag. This must be a writable
 *                  buffer of at least \p tag_len Bytes.
 *
 * \return          \c 0 if the encryption or decryption was performed
 *                  successfully. Note that in #MBEDTLS_GCM_DECRYPT mode,
 *                  this does not indicate that the data is authentic.
 * \return          #MBEDTLS_ERR_GCM_BAD_INPUT if the lengths or pointers are
 *                  not valid or a cipher-specific error code if the encryption
 *                  or decryption failed.
 */
int mbedtls_gcm_crypt_and_tag( mbedtls_gcm_context *ctx,
                       int mode,
                       size_t length,
                       const unsigned char *iv,
                       size_t iv_len,
                       const unsigned char *add,
                       size_t add_len,
                       const unsigned char *input,
                       unsigned char *output,
                       size_t tag_len,
                       unsigned char *tag );

/**
 * \brief           This function performs a GCM authenticated decryption of a
 *                  buffer.
 *
 * \note            For decryption, the output buffer cannot be the same as
 *                  input buffer. If the buffers overlap, the output buffer
 *                  must trail at least 8 Bytes behind the input buffer.
 *
 * \param ctx       The GCM context. This must be initialized.
 * \param length    The length of the ciphertext to decrypt, which is also
 *                  the length of the decrypted plaintext.
 * \param iv        The initialization vector. This must be a readable buffer
 *                  of at least \p iv_len Bytes.
 * \param iv_len    The length of the IV.
 * \param add       The buffer holding the additional data. This must be of at
 *                  least that size in Bytes.
 * \param add_len   The length of the additional data.
 * \param tag       The buffer holding the tag to verify. This must be a
 *                  readable buffer of at least \p tag_len Bytes.
 * \param tag_len   The length of the tag to verify.
 * \param input     The buffer holding the ciphertext. If \p length is greater
 *                  than zero, this must be a readable buffer of at least that
 *                  size.
 * \param output    The buffer for holding the decrypted plaintext. If \p length
 *                  is greater than zero, this must be a writable buffer of at
 *                  least that size.
 *
 * \return          \c 0 if successful and authenticated.
 * \return          #MBEDTLS_ERR_GCM_AUTH_FAILED if the tag does not match.
 * \return          #MBEDTLS_ERR_GCM_BAD_INPUT if the lengths or pointers are
 *                  not valid or a cipher-specific error code if the decryption
 *                  failed.
 */
int mbedtls_gcm_auth_decrypt( mbedtls_gcm_context *ctx,
                      size_t length,
                      const unsigned char *iv,
                      size_t iv_len,
                      const unsigned char *add,
                      size_t add_len,
                      const unsigned char *tag,
                      size_t tag_len,
                      const unsigned char *input,
                      unsigned char *output );

/**
 * \brief           This function starts a GCM encryption or decryption
 *                  operation.
 *
 * \param ctx       The GCM context. This must be initialized.
 * \param mode      The operation to perform: #MBEDTLS_GCM_ENCRYPT or
 *                  #MBEDTLS_GCM_DECRYPT.
 * \param iv        The initialization vector. This must be a readable buffer of
 *                  at least \p iv_len Bytes.
 * \param iv_len    The length of the IV.
 * \param add       The buffer holding the additional data, or \c NULL
 *                  if \p add_len is \c 0.
 * \param add_len   The length of the additional data. If \c 0,
 *                  \p add may be \c NULL.
 *
 * \return          \c 0 on success.
 */
int mbedtls_gcm_starts( mbedtls_gcm_context *ctx,
                int mode,
                const unsigned char *iv,
                size_t iv_len,
                const unsigned char *add,
                size_t add_len );

/**
 * \brief           This function feeds an input buffer into an ongoing GCM
 *                  encryption or decryption operation.
 *
 *    `             The function expects input to be a multiple of 16
 *                  Bytes. Only the last call before calling
 *                  mbedtls_gcm_finish() can be less than 16 Bytes.
 *
 * \note            For decryption, the output buffer cannot be the same as
 *                  input buffer. If the buffers overlap, the output buffer
 *                  must trail at least 8 Bytes behind the input buffer.
 *
 * \param ctx       The GCM context. This must be initialized.
 * \param length    The length of the input data. This must be a multiple of
 *                  16 except in the last call before mbedtls_gcm_finish().
 * \param input     The buffer holding the input data. If \p length is greater
 *                  than zero, this must be a readable buffer of at least that
 *                  size in Bytes.
 * \param output    The buffer for holding the output data. If \p length is
 *                  greater than zero, this must be a writable buffer of at
 *                  least that size in Bytes.
 *
 * \return         \c 0 on success.
 * \return         #MBEDTLS_ERR_GCM_BAD_INPUT on failure.
 */
int mbedtls_gcm_update( mbedtls_gcm_context *ctx,
                size_t length,
                const unsigned char *input,
                unsigned char *output );

/**
 * \brief           This function finishes the GCM operation and generates
 *                  the authentication tag.
 *
 *                  It wraps up the GCM stream, and generates the
 *                  tag. The tag can have a maximum length of 16 Bytes.
 *
 * \param ctx       The GCM context. This must be initialized.
 * \param tag       The buffer for holding the tag. This must be a writable
 *                  buffer of at least \p tag_len Bytes.
 * \param tag_len   The length of the tag to generate. This must be at least
 *                  four.
 *
 * \return          \c 0 on success.
 * \return          #MBEDTLS_ERR_GCM_BAD_INPUT on failure.
 */
int mbedtls_gcm_finish( mbedtls_gcm_context *ctx,
                unsigned char *tag,
                size_t tag_len );

/**
 * \brief           This function clears a GCM context and the underlying
 *                  cipher sub-context.
 *
 * \param ctx       The GCM context to clear. If this is \c NULL, the call has
 *                  no effect. Otherwise, this must be initialized.
 */
void mbedtls_gcm_free( mbedtls_gcm_context *ctx );

#if defined(MBEDTLS_SELF_TEST)

/**
 * \brief          The GCM checkup routine.
 *
 * \return         \c 0 on success.
 * \return         \c 1 on failure.
 */
int mbedtls_gcm_self_test( int verbose );

#endif /* MBEDTLS_SELF_TEST */

#ifdef __cplusplus
}
#endif


#endif /* gcm.h */
