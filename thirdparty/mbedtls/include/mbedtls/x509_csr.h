/**
 * \file x509_csr.h
 *
 * \brief X.509 certificate signing request parsing and writing
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
#ifndef MBEDTLS_X509_CSR_H
#define MBEDTLS_X509_CSR_H

#if !defined(MBEDTLS_CONFIG_FILE)
#include "mbedtls/config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#include "mbedtls/x509.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup x509_module
 * \{ */

/**
 * \name Structures and functions for X.509 Certificate Signing Requests (CSR)
 * \{
 */

/**
 * Certificate Signing Request (CSR) structure.
 */
typedef struct mbedtls_x509_csr {
    mbedtls_x509_buf raw;           /**< The raw CSR data (DER). */
    mbedtls_x509_buf cri;           /**< The raw CertificateRequestInfo body (DER). */

    int version;            /**< CSR version (1=v1). */

    mbedtls_x509_buf  subject_raw;  /**< The raw subject data (DER). */
    mbedtls_x509_name subject;      /**< The parsed subject data (named information object). */

    mbedtls_pk_context pk;          /**< Container for the public key context. */

    mbedtls_x509_buf sig_oid;
    mbedtls_x509_buf sig;
    mbedtls_md_type_t sig_md;       /**< Internal representation of the MD algorithm of the signature algorithm, e.g. MBEDTLS_MD_SHA256 */
    mbedtls_pk_type_t sig_pk;       /**< Internal representation of the Public Key algorithm of the signature algorithm, e.g. MBEDTLS_PK_RSA */
    void *sig_opts;         /**< Signature options to be passed to mbedtls_pk_verify_ext(), e.g. for RSASSA-PSS */
}
mbedtls_x509_csr;

/**
 * Container for writing a CSR
 */
typedef struct mbedtls_x509write_csr {
    mbedtls_pk_context *key;
    mbedtls_asn1_named_data *subject;
    mbedtls_md_type_t md_alg;
    mbedtls_asn1_named_data *extensions;
}
mbedtls_x509write_csr;

#if defined(MBEDTLS_X509_CSR_PARSE_C)
/**
 * \brief          Load a Certificate Signing Request (CSR) in DER format
 *
 * \note           CSR attributes (if any) are currently silently ignored.
 *
 * \note           If #MBEDTLS_USE_PSA_CRYPTO is enabled, the PSA crypto
 *                 subsystem must have been initialized by calling
 *                 psa_crypto_init() before calling this function.
 *
 * \param csr      CSR context to fill
 * \param buf      buffer holding the CRL data
 * \param buflen   size of the buffer
 *
 * \return         0 if successful, or a specific X509 error code
 */
int mbedtls_x509_csr_parse_der(mbedtls_x509_csr *csr,
                               const unsigned char *buf, size_t buflen);

/**
 * \brief          Load a Certificate Signing Request (CSR), DER or PEM format
 *
 * \note           See notes for \c mbedtls_x509_csr_parse_der()
 *
 * \note           If #MBEDTLS_USE_PSA_CRYPTO is enabled, the PSA crypto
 *                 subsystem must have been initialized by calling
 *                 psa_crypto_init() before calling this function.
 *
 * \param csr      CSR context to fill
 * \param buf      buffer holding the CRL data
 * \param buflen   size of the buffer
 *                 (including the terminating null byte for PEM data)
 *
 * \return         0 if successful, or a specific X509 or PEM error code
 */
int mbedtls_x509_csr_parse(mbedtls_x509_csr *csr, const unsigned char *buf, size_t buflen);

#if defined(MBEDTLS_FS_IO)
/**
 * \brief          Load a Certificate Signing Request (CSR)
 *
 * \note           See notes for \c mbedtls_x509_csr_parse()
 *
 * \param csr      CSR context to fill
 * \param path     filename to read the CSR from
 *
 * \return         0 if successful, or a specific X509 or PEM error code
 */
int mbedtls_x509_csr_parse_file(mbedtls_x509_csr *csr, const char *path);
#endif /* MBEDTLS_FS_IO */

/**
 * \brief          Returns an informational string about the
 *                 CSR.
 *
 * \param buf      Buffer to write to
 * \param size     Maximum size of buffer
 * \param prefix   A line prefix
 * \param csr      The X509 CSR to represent
 *
 * \return         The length of the string written (not including the
 *                 terminated nul byte), or a negative error code.
 */
int mbedtls_x509_csr_info(char *buf, size_t size, const char *prefix,
                          const mbedtls_x509_csr *csr);

/**
 * \brief          Initialize a CSR
 *
 * \param csr      CSR to initialize
 */
void mbedtls_x509_csr_init(mbedtls_x509_csr *csr);

/**
 * \brief          Unallocate all CSR data
 *
 * \param csr      CSR to free
 */
void mbedtls_x509_csr_free(mbedtls_x509_csr *csr);
#endif /* MBEDTLS_X509_CSR_PARSE_C */

/** \} name Structures and functions for X.509 Certificate Signing Requests (CSR) */

#if defined(MBEDTLS_X509_CSR_WRITE_C)
/**
 * \brief           Initialize a CSR context
 *
 * \param ctx       CSR context to initialize
 */
void mbedtls_x509write_csr_init(mbedtls_x509write_csr *ctx);

/**
 * \brief           Set the subject name for a CSR
 *                  Subject names should contain a comma-separated list
 *                  of OID types and values:
 *                  e.g. "C=UK,O=ARM,CN=mbed TLS Server 1"
 *
 * \param ctx           CSR context to use
 * \param subject_name  subject name to set
 *
 * \return          0 if subject name was parsed successfully, or
 *                  a specific error code
 */
int mbedtls_x509write_csr_set_subject_name(mbedtls_x509write_csr *ctx,
                                           const char *subject_name);

/**
 * \brief           Set the key for a CSR (public key will be included,
 *                  private key used to sign the CSR when writing it)
 *
 * \param ctx       CSR context to use
 * \param key       Asymmetric key to include
 */
void mbedtls_x509write_csr_set_key(mbedtls_x509write_csr *ctx, mbedtls_pk_context *key);

/**
 * \brief           Set the MD algorithm to use for the signature
 *                  (e.g. MBEDTLS_MD_SHA1)
 *
 * \param ctx       CSR context to use
 * \param md_alg    MD algorithm to use
 */
void mbedtls_x509write_csr_set_md_alg(mbedtls_x509write_csr *ctx, mbedtls_md_type_t md_alg);

/**
 * \brief           Set the Key Usage Extension flags
 *                  (e.g. MBEDTLS_X509_KU_DIGITAL_SIGNATURE | MBEDTLS_X509_KU_KEY_CERT_SIGN)
 *
 * \param ctx       CSR context to use
 * \param key_usage key usage flags to set
 *
 * \return          0 if successful, or MBEDTLS_ERR_X509_ALLOC_FAILED
 *
 * \note            The <code>decipherOnly</code> flag from the Key Usage
 *                  extension is represented by bit 8 (i.e.
 *                  <code>0x8000</code>), which cannot typically be represented
 *                  in an unsigned char. Therefore, the flag
 *                  <code>decipherOnly</code> (i.e.
 *                  #MBEDTLS_X509_KU_DECIPHER_ONLY) cannot be set using this
 *                  function.
 */
int mbedtls_x509write_csr_set_key_usage(mbedtls_x509write_csr *ctx, unsigned char key_usage);

/**
 * \brief           Set the Netscape Cert Type flags
 *                  (e.g. MBEDTLS_X509_NS_CERT_TYPE_SSL_CLIENT | MBEDTLS_X509_NS_CERT_TYPE_EMAIL)
 *
 * \param ctx           CSR context to use
 * \param ns_cert_type  Netscape Cert Type flags to set
 *
 * \return          0 if successful, or MBEDTLS_ERR_X509_ALLOC_FAILED
 */
int mbedtls_x509write_csr_set_ns_cert_type(mbedtls_x509write_csr *ctx,
                                           unsigned char ns_cert_type);

/**
 * \brief           Generic function to add to or replace an extension in the
 *                  CSR
 *
 * \param ctx       CSR context to use
 * \param oid       OID of the extension
 * \param oid_len   length of the OID
 * \param val       value of the extension OCTET STRING
 * \param val_len   length of the value data
 *
 * \return          0 if successful, or a MBEDTLS_ERR_X509_ALLOC_FAILED
 */
int mbedtls_x509write_csr_set_extension(mbedtls_x509write_csr *ctx,
                                        const char *oid, size_t oid_len,
                                        const unsigned char *val, size_t val_len);

/**
 * \brief           Free the contents of a CSR context
 *
 * \param ctx       CSR context to free
 */
void mbedtls_x509write_csr_free(mbedtls_x509write_csr *ctx);

/**
 * \brief           Write a CSR (Certificate Signing Request) to a
 *                  DER structure
 *                  Note: data is written at the end of the buffer! Use the
 *                        return value to determine where you should start
 *                        using the buffer
 *
 * \param ctx       CSR to write away
 * \param buf       buffer to write to
 * \param size      size of the buffer
 * \param f_rng     RNG function (for signature, see note)
 * \param p_rng     RNG parameter
 *
 * \return          length of data written if successful, or a specific
 *                  error code
 *
 * \note            f_rng may be NULL if RSA is used for signature and the
 *                  signature is made offline (otherwise f_rng is desirable
 *                  for countermeasures against timing attacks).
 *                  ECDSA signatures always require a non-NULL f_rng.
 */
int mbedtls_x509write_csr_der(mbedtls_x509write_csr *ctx, unsigned char *buf, size_t size,
                              int (*f_rng)(void *, unsigned char *, size_t),
                              void *p_rng);

#if defined(MBEDTLS_PEM_WRITE_C)
/**
 * \brief           Write a CSR (Certificate Signing Request) to a
 *                  PEM string
 *
 * \param ctx       CSR to write away
 * \param buf       buffer to write to
 * \param size      size of the buffer
 * \param f_rng     RNG function (for signature, see note)
 * \param p_rng     RNG parameter
 *
 * \return          0 if successful, or a specific error code
 *
 * \note            f_rng may be NULL if RSA is used for signature and the
 *                  signature is made offline (otherwise f_rng is desirable
 *                  for countermeasures against timing attacks).
 *                  ECDSA signatures always require a non-NULL f_rng.
 */
int mbedtls_x509write_csr_pem(mbedtls_x509write_csr *ctx, unsigned char *buf, size_t size,
                              int (*f_rng)(void *, unsigned char *, size_t),
                              void *p_rng);
#endif /* MBEDTLS_PEM_WRITE_C */
#endif /* MBEDTLS_X509_CSR_WRITE_C */

/** \} addtogroup x509_module */

#ifdef __cplusplus
}
#endif

#endif /* mbedtls_x509_csr.h */
