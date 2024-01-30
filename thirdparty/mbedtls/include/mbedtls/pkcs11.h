/**
 * \file pkcs11.h
 *
 * \brief Wrapper for PKCS#11 library libpkcs11-helper
 *
 * \author Adriaan de Jong <dejong@fox-it.com>
 */
/*
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */
#ifndef MBEDTLS_PKCS11_H
#define MBEDTLS_PKCS11_H

#if !defined(MBEDTLS_CONFIG_FILE)
#include "mbedtls/config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#if defined(MBEDTLS_PKCS11_C)

#include "mbedtls/x509_crt.h"

#include <pkcs11-helper-1.0/pkcs11h-certificate.h>

#if (defined(__ARMCC_VERSION) || defined(_MSC_VER)) && \
    !defined(inline) && !defined(__cplusplus)
#define inline __inline
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if defined(MBEDTLS_DEPRECATED_REMOVED)

/**
 * Context for PKCS #11 private keys.
 */
typedef struct mbedtls_pkcs11_context {
    pkcs11h_certificate_t pkcs11h_cert;
    int len;
} mbedtls_pkcs11_context;

#if defined(MBEDTLS_DEPRECATED_WARNING)
#define MBEDTLS_DEPRECATED      __attribute__((deprecated))
#else
#define MBEDTLS_DEPRECATED
#endif

/**
 * Initialize a mbedtls_pkcs11_context.
 * (Just making memory references valid.)
 *
 * \deprecated          This function is deprecated and will be removed in a
 *                      future version of the library.
 */
MBEDTLS_DEPRECATED void mbedtls_pkcs11_init(mbedtls_pkcs11_context *ctx);

/**
 * Fill in a Mbed TLS certificate, based on the given PKCS11 helper certificate.
 *
 * \deprecated          This function is deprecated and will be removed in a
 *                      future version of the library.
 *
 * \param cert          X.509 certificate to fill
 * \param pkcs11h_cert  PKCS #11 helper certificate
 *
 * \return              0 on success.
 */
MBEDTLS_DEPRECATED int mbedtls_pkcs11_x509_cert_bind(mbedtls_x509_crt *cert,
                                                     pkcs11h_certificate_t pkcs11h_cert);

/**
 * Set up a mbedtls_pkcs11_context storing the given certificate. Note that the
 * mbedtls_pkcs11_context will take over control of the certificate, freeing it when
 * done.
 *
 * \deprecated          This function is deprecated and will be removed in a
 *                      future version of the library.
 *
 * \param priv_key      Private key structure to fill.
 * \param pkcs11_cert   PKCS #11 helper certificate
 *
 * \return              0 on success
 */
MBEDTLS_DEPRECATED int mbedtls_pkcs11_priv_key_bind(
    mbedtls_pkcs11_context *priv_key,
    pkcs11h_certificate_t pkcs11_cert);

/**
 * Free the contents of the given private key context. Note that the structure
 * itself is not freed.
 *
 * \deprecated          This function is deprecated and will be removed in a
 *                      future version of the library.
 *
 * \param priv_key      Private key structure to cleanup
 */
MBEDTLS_DEPRECATED void mbedtls_pkcs11_priv_key_free(
    mbedtls_pkcs11_context *priv_key);

/**
 * \brief          Do an RSA private key decrypt, then remove the message
 *                 padding
 *
 * \deprecated     This function is deprecated and will be removed in a future
 *                 version of the library.
 *
 * \param ctx      PKCS #11 context
 * \param mode     must be MBEDTLS_RSA_PRIVATE, for compatibility with rsa.c's signature
 * \param input    buffer holding the encrypted data
 * \param output   buffer that will hold the plaintext
 * \param olen     will contain the plaintext length
 * \param output_max_len    maximum length of the output buffer
 *
 * \return         0 if successful, or an MBEDTLS_ERR_RSA_XXX error code
 *
 * \note           The output buffer must be as large as the size
 *                 of ctx->N (eg. 128 bytes if RSA-1024 is used) otherwise
 *                 an error is thrown.
 */
MBEDTLS_DEPRECATED int mbedtls_pkcs11_decrypt(mbedtls_pkcs11_context *ctx,
                                              int mode, size_t *olen,
                                              const unsigned char *input,
                                              unsigned char *output,
                                              size_t output_max_len);

/**
 * \brief          Do a private RSA to sign a message digest
 *
 * \deprecated     This function is deprecated and will be removed in a future
 *                 version of the library.
 *
 * \param ctx      PKCS #11 context
 * \param mode     must be MBEDTLS_RSA_PRIVATE, for compatibility with rsa.c's signature
 * \param md_alg   a MBEDTLS_MD_XXX (use MBEDTLS_MD_NONE for signing raw data)
 * \param hashlen  message digest length (for MBEDTLS_MD_NONE only)
 * \param hash     buffer holding the message digest
 * \param sig      buffer that will hold the ciphertext
 *
 * \return         0 if the signing operation was successful,
 *                 or an MBEDTLS_ERR_RSA_XXX error code
 *
 * \note           The "sig" buffer must be as large as the size
 *                 of ctx->N (eg. 128 bytes if RSA-1024 is used).
 */
MBEDTLS_DEPRECATED int mbedtls_pkcs11_sign(mbedtls_pkcs11_context *ctx,
                                           int mode,
                                           mbedtls_md_type_t md_alg,
                                           unsigned int hashlen,
                                           const unsigned char *hash,
                                           unsigned char *sig);

/**
 * SSL/TLS wrappers for PKCS#11 functions
 *
 * \deprecated     This function is deprecated and will be removed in a future
 *                 version of the library.
 */
MBEDTLS_DEPRECATED static inline int mbedtls_ssl_pkcs11_decrypt(void *ctx,
                                                                int mode,
                                                                size_t *olen,
                                                                const unsigned char *input,
                                                                unsigned char *output,
                                                                size_t output_max_len)
{
    return mbedtls_pkcs11_decrypt((mbedtls_pkcs11_context *) ctx, mode, olen, input, output,
                                  output_max_len);
}

/**
 * \brief          This function signs a message digest using RSA.
 *
 * \deprecated     This function is deprecated and will be removed in a future
 *                 version of the library.
 *
 * \param ctx      The PKCS #11 context.
 * \param f_rng    The RNG function. This parameter is unused.
 * \param p_rng    The RNG context. This parameter is unused.
 * \param mode     The operation to run. This must be set to
 *                 MBEDTLS_RSA_PRIVATE, for compatibility with rsa.c's
 *                 signature.
 * \param md_alg   The message digest algorithm. One of the MBEDTLS_MD_XXX
 *                 must be passed to this function and MBEDTLS_MD_NONE can be
 *                 used for signing raw data.
 * \param hashlen  The message digest length (for MBEDTLS_MD_NONE only).
 * \param hash     The buffer holding the message digest.
 * \param sig      The buffer that will hold the ciphertext.
 *
 * \return         \c 0 if the signing operation was successful.
 * \return         A non-zero error code on failure.
 *
 * \note           The \p sig buffer must be as large as the size of
 *                 <code>ctx->N</code>. For example, 128 bytes if RSA-1024 is
 *                 used.
 */
MBEDTLS_DEPRECATED static inline int mbedtls_ssl_pkcs11_sign(void *ctx,
                                                             int (*f_rng)(void *,
                                                                          unsigned char *,
                                                                          size_t),
                                                             void *p_rng,
                                                             int mode,
                                                             mbedtls_md_type_t md_alg,
                                                             unsigned int hashlen,
                                                             const unsigned char *hash,
                                                             unsigned char *sig)
{
    ((void) f_rng);
    ((void) p_rng);
    return mbedtls_pkcs11_sign((mbedtls_pkcs11_context *) ctx, mode, md_alg,
                               hashlen, hash, sig);
}

/**
 * This function gets the length of the private key.
 *
 * \deprecated     This function is deprecated and will be removed in a future
 *                 version of the library.
 *
 * \param ctx      The PKCS #11 context.
 *
 * \return         The length of the private key.
 */
MBEDTLS_DEPRECATED static inline size_t mbedtls_ssl_pkcs11_key_len(void *ctx)
{
    return ((mbedtls_pkcs11_context *) ctx)->len;
}

#undef MBEDTLS_DEPRECATED

#endif /* MBEDTLS_DEPRECATED_REMOVED */

#ifdef __cplusplus
}
#endif

#endif /* MBEDTLS_PKCS11_C */

#endif /* MBEDTLS_PKCS11_H */
