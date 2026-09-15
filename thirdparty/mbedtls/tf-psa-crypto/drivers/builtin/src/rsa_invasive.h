/**
 * \file rsa_invasive.h
 *
 * \brief Function declarations for invasive testing of built-in RSA.
 */
/**
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */

#ifndef TF_PSA_CRYPTO_RSA_INVASIVE_H
#define TF_PSA_CRYPTO_RSA_INVASIVE_H

#if defined(MBEDTLS_TEST_HOOKS)

#if defined(MBEDTLS_PKCS1_V15) && defined(MBEDTLS_RSA_C)
/** This function performs the unpadding part of a PKCS#1 v1.5 decryption
 *  operation (EME-PKCS1-v1_5 decoding).
 *
 * \note The return value from this function is a sensitive value
 *       (this is unusual). #PSA_ERROR_BUFFER_TOO_SMALL shouldn't happen
 *       in a well-written application, but 0 vs #PSA_ERROR_INVALID_PADDING
 *       is often a situation that an attacker can provoke and leaking which
 *       one is the result is precisely the information the attacker wants.
 *
 * \param input          The input buffer which is the payload inside PKCS#1v1.5
 *                       encryption padding, called the "encoded message EM"
 *                       by the terminology.
 * \param ilen           The length of the payload in the \p input buffer.
 * \param output         The buffer for the payload, called "message M" by the
 *                       PKCS#1 terminology. This must be a writable buffer of
 *                       length \p output_max_len bytes.
 * \param olen           The address at which to store the length of
 *                       the payload. This must not be \c NULL.
 * \param output_max_len The length in bytes of the output buffer \p output.
 *
 * \return      \c 0 on success.
 * \return      #PSA_ERROR_BUFFER_TOO_SMALL
 *              The output buffer is too small for the unpadded payload.
 * \return      #PSA_ERROR_INVALID_PADDING
 *              The input doesn't contain properly formatted padding.
 */
MBEDTLS_STATIC_TESTABLE int mbedtls_ct_rsaes_pkcs1_v15_unpadding(
    unsigned char *input, size_t ilen,
    unsigned char *output, size_t output_max_len, size_t *olen);
#endif /* MBEDTLS_PKCS1_V15 && MBEDTLS_RSA_C */

/* mbedtls_rsa_private() is not fully constant-time yet, so we can't do
 * end-to-end constant-time testing by marking external inputs as secret.
 *
 * The next best thing we can do is mark its output as a secret, when we expect
 * further processing to be constant-time, including error handling. */
extern void (*mbedtls_rsa_cf_secret)(const void *ptr, size_t size);

#endif /* MBEDTLS_TEST_HOOKS */

#endif /* TF_PSA_CRYPTO_RSA_INVASIVE_H */
