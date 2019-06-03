/**
 * \file pkcs11.h
 *
 * \brief Wrapper for PKCS#11 library libpkcs11-helper
 *
 * \author Adriaan de Jong <dejong@fox-it.com>
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
#ifndef MBEDTLS_PKCS11_H
#define MBEDTLS_PKCS11_H

#if !defined(MBEDTLS_CONFIG_FILE)
#include "config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#if defined(MBEDTLS_PKCS11_C)

#include "x509_crt.h"

#include <pkcs11-helper-1.0/pkcs11h-certificate.h>

#if ( defined(__ARMCC_VERSION) || defined(_MSC_VER) ) && \
    !defined(inline) && !defined(__cplusplus)
#define inline __inline
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Context for PKCS #11 private keys.
 */
typedef struct mbedtls_pkcs11_context
{
        pkcs11h_certificate_t pkcs11h_cert;
        int len;
} mbedtls_pkcs11_context;

/**
 * Initialize a mbedtls_pkcs11_context.
 * (Just making memory references valid.)
 */
void mbedtls_pkcs11_init( mbedtls_pkcs11_context *ctx );

/**
 * Fill in a mbed TLS certificate, based on the given PKCS11 helper certificate.
 *
 * \param cert          X.509 certificate to fill
 * \param pkcs11h_cert  PKCS #11 helper certificate
 *
 * \return              0 on success.
 */
int mbedtls_pkcs11_x509_cert_bind( mbedtls_x509_crt *cert, pkcs11h_certificate_t pkcs11h_cert );

/**
 * Set up a mbedtls_pkcs11_context storing the given certificate. Note that the
 * mbedtls_pkcs11_context will take over control of the certificate, freeing it when
 * done.
 *
 * \param priv_key      Private key structure to fill.
 * \param pkcs11_cert   PKCS #11 helper certificate
 *
 * \return              0 on success
 */
int mbedtls_pkcs11_priv_key_bind( mbedtls_pkcs11_context *priv_key,
        pkcs11h_certificate_t pkcs11_cert );

/**
 * Free the contents of the given private key context. Note that the structure
 * itself is not freed.
 *
 * \param priv_key      Private key structure to cleanup
 */
void mbedtls_pkcs11_priv_key_free( mbedtls_pkcs11_context *priv_key );

/**
 * \brief          Do an RSA private key decrypt, then remove the message
 *                 padding
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
int mbedtls_pkcs11_decrypt( mbedtls_pkcs11_context *ctx,
                       int mode, size_t *olen,
                       const unsigned char *input,
                       unsigned char *output,
                       size_t output_max_len );

/**
 * \brief          Do a private RSA to sign a message digest
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
int mbedtls_pkcs11_sign( mbedtls_pkcs11_context *ctx,
                    int mode,
                    mbedtls_md_type_t md_alg,
                    unsigned int hashlen,
                    const unsigned char *hash,
                    unsigned char *sig );

/**
 * SSL/TLS wrappers for PKCS#11 functions
 */
static inline int mbedtls_ssl_pkcs11_decrypt( void *ctx, int mode, size_t *olen,
                        const unsigned char *input, unsigned char *output,
                        size_t output_max_len )
{
    return mbedtls_pkcs11_decrypt( (mbedtls_pkcs11_context *) ctx, mode, olen, input, output,
                           output_max_len );
}

static inline int mbedtls_ssl_pkcs11_sign( void *ctx,
                     int (*f_rng)(void *, unsigned char *, size_t), void *p_rng,
                     int mode, mbedtls_md_type_t md_alg, unsigned int hashlen,
                     const unsigned char *hash, unsigned char *sig )
{
    ((void) f_rng);
    ((void) p_rng);
    return mbedtls_pkcs11_sign( (mbedtls_pkcs11_context *) ctx, mode, md_alg,
                        hashlen, hash, sig );
}

static inline size_t mbedtls_ssl_pkcs11_key_len( void *ctx )
{
    return ( (mbedtls_pkcs11_context *) ctx )->len;
}

#ifdef __cplusplus
}
#endif

#endif /* MBEDTLS_PKCS11_C */

#endif /* MBEDTLS_PKCS11_H */
