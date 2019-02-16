/**
 * \file x509_crt.h
 *
 * \brief X.509 certificate parsing and writing
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
#ifndef MBEDTLS_X509_CRT_H
#define MBEDTLS_X509_CRT_H

#if !defined(MBEDTLS_CONFIG_FILE)
#include "config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#include "x509.h"
#include "x509_crl.h"

/**
 * \addtogroup x509_module
 * \{
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \name Structures and functions for parsing and writing X.509 certificates
 * \{
 */

/**
 * Container for an X.509 certificate. The certificate may be chained.
 */
typedef struct mbedtls_x509_crt
{
    mbedtls_x509_buf raw;               /**< The raw certificate data (DER). */
    mbedtls_x509_buf tbs;               /**< The raw certificate body (DER). The part that is To Be Signed. */

    int version;                /**< The X.509 version. (1=v1, 2=v2, 3=v3) */
    mbedtls_x509_buf serial;            /**< Unique id for certificate issued by a specific CA. */
    mbedtls_x509_buf sig_oid;           /**< Signature algorithm, e.g. sha1RSA */

    mbedtls_x509_buf issuer_raw;        /**< The raw issuer data (DER). Used for quick comparison. */
    mbedtls_x509_buf subject_raw;       /**< The raw subject data (DER). Used for quick comparison. */

    mbedtls_x509_name issuer;           /**< The parsed issuer data (named information object). */
    mbedtls_x509_name subject;          /**< The parsed subject data (named information object). */

    mbedtls_x509_time valid_from;       /**< Start time of certificate validity. */
    mbedtls_x509_time valid_to;         /**< End time of certificate validity. */

    mbedtls_pk_context pk;              /**< Container for the public key context. */

    mbedtls_x509_buf issuer_id;         /**< Optional X.509 v2/v3 issuer unique identifier. */
    mbedtls_x509_buf subject_id;        /**< Optional X.509 v2/v3 subject unique identifier. */
    mbedtls_x509_buf v3_ext;            /**< Optional X.509 v3 extensions.  */
    mbedtls_x509_sequence subject_alt_names;    /**< Optional list of Subject Alternative Names (Only dNSName supported). */

    int ext_types;              /**< Bit string containing detected and parsed extensions */
    int ca_istrue;              /**< Optional Basic Constraint extension value: 1 if this certificate belongs to a CA, 0 otherwise. */
    int max_pathlen;            /**< Optional Basic Constraint extension value: The maximum path length to the root certificate. Path length is 1 higher than RFC 5280 'meaning', so 1+ */

    unsigned int key_usage;     /**< Optional key usage extension value: See the values in x509.h */

    mbedtls_x509_sequence ext_key_usage; /**< Optional list of extended key usage OIDs. */

    unsigned char ns_cert_type; /**< Optional Netscape certificate type extension value: See the values in x509.h */

    mbedtls_x509_buf sig;               /**< Signature: hash of the tbs part signed with the private key. */
    mbedtls_md_type_t sig_md;           /**< Internal representation of the MD algorithm of the signature algorithm, e.g. MBEDTLS_MD_SHA256 */
    mbedtls_pk_type_t sig_pk;           /**< Internal representation of the Public Key algorithm of the signature algorithm, e.g. MBEDTLS_PK_RSA */
    void *sig_opts;             /**< Signature options to be passed to mbedtls_pk_verify_ext(), e.g. for RSASSA-PSS */

    struct mbedtls_x509_crt *next;     /**< Next certificate in the CA-chain. */
}
mbedtls_x509_crt;

/**
 * Build flag from an algorithm/curve identifier (pk, md, ecp)
 * Since 0 is always XXX_NONE, ignore it.
 */
#define MBEDTLS_X509_ID_FLAG( id )   ( 1 << ( id - 1 ) )

/**
 * Security profile for certificate verification.
 *
 * All lists are bitfields, built by ORing flags from MBEDTLS_X509_ID_FLAG().
 */
typedef struct mbedtls_x509_crt_profile
{
    uint32_t allowed_mds;       /**< MDs for signatures         */
    uint32_t allowed_pks;       /**< PK algs for signatures     */
    uint32_t allowed_curves;    /**< Elliptic curves for ECDSA  */
    uint32_t rsa_min_bitlen;    /**< Minimum size for RSA keys  */
}
mbedtls_x509_crt_profile;

#define MBEDTLS_X509_CRT_VERSION_1              0
#define MBEDTLS_X509_CRT_VERSION_2              1
#define MBEDTLS_X509_CRT_VERSION_3              2

#define MBEDTLS_X509_RFC5280_MAX_SERIAL_LEN 32
#define MBEDTLS_X509_RFC5280_UTC_TIME_LEN   15

#if !defined( MBEDTLS_X509_MAX_FILE_PATH_LEN )
#define MBEDTLS_X509_MAX_FILE_PATH_LEN 512
#endif

/**
 * Container for writing a certificate (CRT)
 */
typedef struct mbedtls_x509write_cert
{
    int version;
    mbedtls_mpi serial;
    mbedtls_pk_context *subject_key;
    mbedtls_pk_context *issuer_key;
    mbedtls_asn1_named_data *subject;
    mbedtls_asn1_named_data *issuer;
    mbedtls_md_type_t md_alg;
    char not_before[MBEDTLS_X509_RFC5280_UTC_TIME_LEN + 1];
    char not_after[MBEDTLS_X509_RFC5280_UTC_TIME_LEN + 1];
    mbedtls_asn1_named_data *extensions;
}
mbedtls_x509write_cert;

/**
 * Item in a verification chain: cert and flags for it
 */
typedef struct {
    mbedtls_x509_crt *crt;
    uint32_t flags;
} mbedtls_x509_crt_verify_chain_item;

/**
 * Max size of verification chain: end-entity + intermediates + trusted root
 */
#define MBEDTLS_X509_MAX_VERIFY_CHAIN_SIZE  ( MBEDTLS_X509_MAX_INTERMEDIATE_CA + 2 )

/**
 * Verification chain as built by \c mbedtls_crt_verify_chain()
 */
typedef struct
{
    mbedtls_x509_crt_verify_chain_item items[MBEDTLS_X509_MAX_VERIFY_CHAIN_SIZE];
    unsigned len;
} mbedtls_x509_crt_verify_chain;

#if defined(MBEDTLS_ECDSA_C) && defined(MBEDTLS_ECP_RESTARTABLE)

/**
 * \brief       Context for resuming X.509 verify operations
 */
typedef struct
{
    /* for check_signature() */
    mbedtls_pk_restart_ctx pk;

    /* for find_parent_in() */
    mbedtls_x509_crt *parent; /* non-null iff parent_in in progress */
    mbedtls_x509_crt *fallback_parent;
    int fallback_signature_is_good;

    /* for find_parent() */
    int parent_is_trusted; /* -1 if find_parent is not in progress */

    /* for verify_chain() */
    enum {
        x509_crt_rs_none,
        x509_crt_rs_find_parent,
    } in_progress;  /* none if no operation is in progress */
    int self_cnt;
    mbedtls_x509_crt_verify_chain ver_chain;

} mbedtls_x509_crt_restart_ctx;

#else /* MBEDTLS_ECDSA_C && MBEDTLS_ECP_RESTARTABLE */

/* Now we can declare functions that take a pointer to that */
typedef void mbedtls_x509_crt_restart_ctx;

#endif /* MBEDTLS_ECDSA_C && MBEDTLS_ECP_RESTARTABLE */

#if defined(MBEDTLS_X509_CRT_PARSE_C)
/**
 * Default security profile. Should provide a good balance between security
 * and compatibility with current deployments.
 */
extern const mbedtls_x509_crt_profile mbedtls_x509_crt_profile_default;

/**
 * Expected next default profile. Recommended for new deployments.
 * Currently targets a 128-bit security level, except for RSA-2048.
 */
extern const mbedtls_x509_crt_profile mbedtls_x509_crt_profile_next;

/**
 * NSA Suite B profile.
 */
extern const mbedtls_x509_crt_profile mbedtls_x509_crt_profile_suiteb;

/**
 * \brief          Parse a single DER formatted certificate and add it
 *                 to the chained list.
 *
 * \param chain    points to the start of the chain
 * \param buf      buffer holding the certificate DER data
 * \param buflen   size of the buffer
 *
 * \return         0 if successful, or a specific X509 or PEM error code
 */
int mbedtls_x509_crt_parse_der( mbedtls_x509_crt *chain, const unsigned char *buf,
                        size_t buflen );

/**
 * \brief          Parse one DER-encoded or one or more concatenated PEM-encoded
 *                 certificates and add them to the chained list.
 *
 *                 For CRTs in PEM encoding, the function parses permissively:
 *                 if at least one certificate can be parsed, the function
 *                 returns the number of certificates for which parsing failed
 *                 (hence \c 0 if all certificates were parsed successfully).
 *                 If no certificate could be parsed, the function returns
 *                 the first (negative) error encountered during parsing.
 *
 *                 PEM encoded certificates may be interleaved by other data
 *                 such as human readable descriptions of their content, as
 *                 long as the certificates are enclosed in the PEM specific
 *                 '-----{BEGIN/END} CERTIFICATE-----' delimiters.
 *
 * \param chain    The chain to which to add the parsed certificates.
 * \param buf      The buffer holding the certificate data in PEM or DER format.
 *                 For certificates in PEM encoding, this may be a concatenation
 *                 of multiple certificates; for DER encoding, the buffer must
 *                 comprise exactly one certificate.
 * \param buflen   The size of \p buf, including the terminating \c NULL byte
 *                 in case of PEM encoded data.
 *
 * \return         \c 0 if all certificates were parsed successfully.
 * \return         The (positive) number of certificates that couldn't
 *                 be parsed if parsing was partly successful (see above).
 * \return         A negative X509 or PEM error code otherwise.
 *
 */
int mbedtls_x509_crt_parse( mbedtls_x509_crt *chain, const unsigned char *buf, size_t buflen );

#if defined(MBEDTLS_FS_IO)
/**
 * \brief          Load one or more certificates and add them
 *                 to the chained list. Parses permissively. If some
 *                 certificates can be parsed, the result is the number
 *                 of failed certificates it encountered. If none complete
 *                 correctly, the first error is returned.
 *
 * \param chain    points to the start of the chain
 * \param path     filename to read the certificates from
 *
 * \return         0 if all certificates parsed successfully, a positive number
 *                 if partly successful or a specific X509 or PEM error code
 */
int mbedtls_x509_crt_parse_file( mbedtls_x509_crt *chain, const char *path );

/**
 * \brief          Load one or more certificate files from a path and add them
 *                 to the chained list. Parses permissively. If some
 *                 certificates can be parsed, the result is the number
 *                 of failed certificates it encountered. If none complete
 *                 correctly, the first error is returned.
 *
 * \param chain    points to the start of the chain
 * \param path     directory / folder to read the certificate files from
 *
 * \return         0 if all certificates parsed successfully, a positive number
 *                 if partly successful or a specific X509 or PEM error code
 */
int mbedtls_x509_crt_parse_path( mbedtls_x509_crt *chain, const char *path );
#endif /* MBEDTLS_FS_IO */

/**
 * \brief          Returns an informational string about the
 *                 certificate.
 *
 * \param buf      Buffer to write to
 * \param size     Maximum size of buffer
 * \param prefix   A line prefix
 * \param crt      The X509 certificate to represent
 *
 * \return         The length of the string written (not including the
 *                 terminated nul byte), or a negative error code.
 */
int mbedtls_x509_crt_info( char *buf, size_t size, const char *prefix,
                   const mbedtls_x509_crt *crt );

/**
 * \brief          Returns an informational string about the
 *                 verification status of a certificate.
 *
 * \param buf      Buffer to write to
 * \param size     Maximum size of buffer
 * \param prefix   A line prefix
 * \param flags    Verification flags created by mbedtls_x509_crt_verify()
 *
 * \return         The length of the string written (not including the
 *                 terminated nul byte), or a negative error code.
 */
int mbedtls_x509_crt_verify_info( char *buf, size_t size, const char *prefix,
                          uint32_t flags );

/**
 * \brief          Verify the certificate signature
 *
 *                 The verify callback is a user-supplied callback that
 *                 can clear / modify / add flags for a certificate. If set,
 *                 the verification callback is called for each
 *                 certificate in the chain (from the trust-ca down to the
 *                 presented crt). The parameters for the callback are:
 *                 (void *parameter, mbedtls_x509_crt *crt, int certificate_depth,
 *                 int *flags). With the flags representing current flags for
 *                 that specific certificate and the certificate depth from
 *                 the bottom (Peer cert depth = 0).
 *
 *                 All flags left after returning from the callback
 *                 are also returned to the application. The function should
 *                 return 0 for anything (including invalid certificates)
 *                 other than fatal error, as a non-zero return code
 *                 immediately aborts the verification process. For fatal
 *                 errors, a specific error code should be used (different
 *                 from MBEDTLS_ERR_X509_CERT_VERIFY_FAILED which should not
 *                 be returned at this point), or MBEDTLS_ERR_X509_FATAL_ERROR
 *                 can be used if no better code is available.
 *
 * \note           In case verification failed, the results can be displayed
 *                 using \c mbedtls_x509_crt_verify_info()
 *
 * \note           Same as \c mbedtls_x509_crt_verify_with_profile() with the
 *                 default security profile.
 *
 * \note           It is your responsibility to provide up-to-date CRLs for
 *                 all trusted CAs. If no CRL is provided for the CA that was
 *                 used to sign the certificate, CRL verification is skipped
 *                 silently, that is *without* setting any flag.
 *
 * \note           The \c trust_ca list can contain two types of certificates:
 *                 (1) those of trusted root CAs, so that certificates
 *                 chaining up to those CAs will be trusted, and (2)
 *                 self-signed end-entity certificates to be trusted (for
 *                 specific peers you know) - in that case, the self-signed
 *                 certificate doesn't need to have the CA bit set.
 *
 * \param crt      a certificate (chain) to be verified
 * \param trust_ca the list of trusted CAs (see note above)
 * \param ca_crl   the list of CRLs for trusted CAs (see note above)
 * \param cn       expected Common Name (can be set to
 *                 NULL if the CN must not be verified)
 * \param flags    result of the verification
 * \param f_vrfy   verification function
 * \param p_vrfy   verification parameter
 *
 * \return         0 (and flags set to 0) if the chain was verified and valid,
 *                 MBEDTLS_ERR_X509_CERT_VERIFY_FAILED if the chain was verified
 *                 but found to be invalid, in which case *flags will have one
 *                 or more MBEDTLS_X509_BADCERT_XXX or MBEDTLS_X509_BADCRL_XXX
 *                 flags set, or another error (and flags set to 0xffffffff)
 *                 in case of a fatal error encountered during the
 *                 verification process.
 */
int mbedtls_x509_crt_verify( mbedtls_x509_crt *crt,
                     mbedtls_x509_crt *trust_ca,
                     mbedtls_x509_crl *ca_crl,
                     const char *cn, uint32_t *flags,
                     int (*f_vrfy)(void *, mbedtls_x509_crt *, int, uint32_t *),
                     void *p_vrfy );

/**
 * \brief          Verify the certificate signature according to profile
 *
 * \note           Same as \c mbedtls_x509_crt_verify(), but with explicit
 *                 security profile.
 *
 * \note           The restrictions on keys (RSA minimum size, allowed curves
 *                 for ECDSA) apply to all certificates: trusted root,
 *                 intermediate CAs if any, and end entity certificate.
 *
 * \param crt      a certificate (chain) to be verified
 * \param trust_ca the list of trusted CAs
 * \param ca_crl   the list of CRLs for trusted CAs
 * \param profile  security profile for verification
 * \param cn       expected Common Name (can be set to
 *                 NULL if the CN must not be verified)
 * \param flags    result of the verification
 * \param f_vrfy   verification function
 * \param p_vrfy   verification parameter
 *
 * \return         0 if successful or MBEDTLS_ERR_X509_CERT_VERIFY_FAILED
 *                 in which case *flags will have one or more
 *                 MBEDTLS_X509_BADCERT_XXX or MBEDTLS_X509_BADCRL_XXX flags
 *                 set,
 *                 or another error in case of a fatal error encountered
 *                 during the verification process.
 */
int mbedtls_x509_crt_verify_with_profile( mbedtls_x509_crt *crt,
                     mbedtls_x509_crt *trust_ca,
                     mbedtls_x509_crl *ca_crl,
                     const mbedtls_x509_crt_profile *profile,
                     const char *cn, uint32_t *flags,
                     int (*f_vrfy)(void *, mbedtls_x509_crt *, int, uint32_t *),
                     void *p_vrfy );

/**
 * \brief          Restartable version of \c mbedtls_crt_verify_with_profile()
 *
 * \note           Performs the same job as \c mbedtls_crt_verify_with_profile()
 *                 but can return early and restart according to the limit
 *                 set with \c mbedtls_ecp_set_max_ops() to reduce blocking.
 *
 * \param crt      a certificate (chain) to be verified
 * \param trust_ca the list of trusted CAs
 * \param ca_crl   the list of CRLs for trusted CAs
 * \param profile  security profile for verification
 * \param cn       expected Common Name (can be set to
 *                 NULL if the CN must not be verified)
 * \param flags    result of the verification
 * \param f_vrfy   verification function
 * \param p_vrfy   verification parameter
 * \param rs_ctx   restart context (NULL to disable restart)
 *
 * \return         See \c mbedtls_crt_verify_with_profile(), or
 * \return         #MBEDTLS_ERR_ECP_IN_PROGRESS if maximum number of
 *                 operations was reached: see \c mbedtls_ecp_set_max_ops().
 */
int mbedtls_x509_crt_verify_restartable( mbedtls_x509_crt *crt,
                     mbedtls_x509_crt *trust_ca,
                     mbedtls_x509_crl *ca_crl,
                     const mbedtls_x509_crt_profile *profile,
                     const char *cn, uint32_t *flags,
                     int (*f_vrfy)(void *, mbedtls_x509_crt *, int, uint32_t *),
                     void *p_vrfy,
                     mbedtls_x509_crt_restart_ctx *rs_ctx );

#if defined(MBEDTLS_X509_CHECK_KEY_USAGE)
/**
 * \brief          Check usage of certificate against keyUsage extension.
 *
 * \param crt      Leaf certificate used.
 * \param usage    Intended usage(s) (eg MBEDTLS_X509_KU_KEY_ENCIPHERMENT
 *                 before using the certificate to perform an RSA key
 *                 exchange).
 *
 * \note           Except for decipherOnly and encipherOnly, a bit set in the
 *                 usage argument means this bit MUST be set in the
 *                 certificate. For decipherOnly and encipherOnly, it means
 *                 that bit MAY be set.
 *
 * \return         0 is these uses of the certificate are allowed,
 *                 MBEDTLS_ERR_X509_BAD_INPUT_DATA if the keyUsage extension
 *                 is present but does not match the usage argument.
 *
 * \note           You should only call this function on leaf certificates, on
 *                 (intermediate) CAs the keyUsage extension is automatically
 *                 checked by \c mbedtls_x509_crt_verify().
 */
int mbedtls_x509_crt_check_key_usage( const mbedtls_x509_crt *crt,
                                      unsigned int usage );
#endif /* MBEDTLS_X509_CHECK_KEY_USAGE) */

#if defined(MBEDTLS_X509_CHECK_EXTENDED_KEY_USAGE)
/**
 * \brief           Check usage of certificate against extendedKeyUsage.
 *
 * \param crt       Leaf certificate used.
 * \param usage_oid Intended usage (eg MBEDTLS_OID_SERVER_AUTH or
 *                  MBEDTLS_OID_CLIENT_AUTH).
 * \param usage_len Length of usage_oid (eg given by MBEDTLS_OID_SIZE()).
 *
 * \return          0 if this use of the certificate is allowed,
 *                  MBEDTLS_ERR_X509_BAD_INPUT_DATA if not.
 *
 * \note            Usually only makes sense on leaf certificates.
 */
int mbedtls_x509_crt_check_extended_key_usage( const mbedtls_x509_crt *crt,
                                               const char *usage_oid,
                                               size_t usage_len );
#endif /* MBEDTLS_X509_CHECK_EXTENDED_KEY_USAGE */

#if defined(MBEDTLS_X509_CRL_PARSE_C)
/**
 * \brief          Verify the certificate revocation status
 *
 * \param crt      a certificate to be verified
 * \param crl      the CRL to verify against
 *
 * \return         1 if the certificate is revoked, 0 otherwise
 *
 */
int mbedtls_x509_crt_is_revoked( const mbedtls_x509_crt *crt, const mbedtls_x509_crl *crl );
#endif /* MBEDTLS_X509_CRL_PARSE_C */

/**
 * \brief          Initialize a certificate (chain)
 *
 * \param crt      Certificate chain to initialize
 */
void mbedtls_x509_crt_init( mbedtls_x509_crt *crt );

/**
 * \brief          Unallocate all certificate data
 *
 * \param crt      Certificate chain to free
 */
void mbedtls_x509_crt_free( mbedtls_x509_crt *crt );

#if defined(MBEDTLS_ECDSA_C) && defined(MBEDTLS_ECP_RESTARTABLE)
/**
 * \brief           Initialize a restart context
 */
void mbedtls_x509_crt_restart_init( mbedtls_x509_crt_restart_ctx *ctx );

/**
 * \brief           Free the components of a restart context
 */
void mbedtls_x509_crt_restart_free( mbedtls_x509_crt_restart_ctx *ctx );
#endif /* MBEDTLS_ECDSA_C && MBEDTLS_ECP_RESTARTABLE */
#endif /* MBEDTLS_X509_CRT_PARSE_C */

/* \} name */
/* \} addtogroup x509_module */

#if defined(MBEDTLS_X509_CRT_WRITE_C)
/**
 * \brief           Initialize a CRT writing context
 *
 * \param ctx       CRT context to initialize
 */
void mbedtls_x509write_crt_init( mbedtls_x509write_cert *ctx );

/**
 * \brief           Set the verion for a Certificate
 *                  Default: MBEDTLS_X509_CRT_VERSION_3
 *
 * \param ctx       CRT context to use
 * \param version   version to set (MBEDTLS_X509_CRT_VERSION_1, MBEDTLS_X509_CRT_VERSION_2 or
 *                                  MBEDTLS_X509_CRT_VERSION_3)
 */
void mbedtls_x509write_crt_set_version( mbedtls_x509write_cert *ctx, int version );

/**
 * \brief           Set the serial number for a Certificate.
 *
 * \param ctx       CRT context to use
 * \param serial    serial number to set
 *
 * \return          0 if successful
 */
int mbedtls_x509write_crt_set_serial( mbedtls_x509write_cert *ctx, const mbedtls_mpi *serial );

/**
 * \brief           Set the validity period for a Certificate
 *                  Timestamps should be in string format for UTC timezone
 *                  i.e. "YYYYMMDDhhmmss"
 *                  e.g. "20131231235959" for December 31st 2013
 *                       at 23:59:59
 *
 * \param ctx       CRT context to use
 * \param not_before    not_before timestamp
 * \param not_after     not_after timestamp
 *
 * \return          0 if timestamp was parsed successfully, or
 *                  a specific error code
 */
int mbedtls_x509write_crt_set_validity( mbedtls_x509write_cert *ctx, const char *not_before,
                                const char *not_after );

/**
 * \brief           Set the issuer name for a Certificate
 *                  Issuer names should contain a comma-separated list
 *                  of OID types and values:
 *                  e.g. "C=UK,O=ARM,CN=mbed TLS CA"
 *
 * \param ctx           CRT context to use
 * \param issuer_name   issuer name to set
 *
 * \return          0 if issuer name was parsed successfully, or
 *                  a specific error code
 */
int mbedtls_x509write_crt_set_issuer_name( mbedtls_x509write_cert *ctx,
                                   const char *issuer_name );

/**
 * \brief           Set the subject name for a Certificate
 *                  Subject names should contain a comma-separated list
 *                  of OID types and values:
 *                  e.g. "C=UK,O=ARM,CN=mbed TLS Server 1"
 *
 * \param ctx           CRT context to use
 * \param subject_name  subject name to set
 *
 * \return          0 if subject name was parsed successfully, or
 *                  a specific error code
 */
int mbedtls_x509write_crt_set_subject_name( mbedtls_x509write_cert *ctx,
                                    const char *subject_name );

/**
 * \brief           Set the subject public key for the certificate
 *
 * \param ctx       CRT context to use
 * \param key       public key to include
 */
void mbedtls_x509write_crt_set_subject_key( mbedtls_x509write_cert *ctx, mbedtls_pk_context *key );

/**
 * \brief           Set the issuer key used for signing the certificate
 *
 * \param ctx       CRT context to use
 * \param key       private key to sign with
 */
void mbedtls_x509write_crt_set_issuer_key( mbedtls_x509write_cert *ctx, mbedtls_pk_context *key );

/**
 * \brief           Set the MD algorithm to use for the signature
 *                  (e.g. MBEDTLS_MD_SHA1)
 *
 * \param ctx       CRT context to use
 * \param md_alg    MD algorithm to use
 */
void mbedtls_x509write_crt_set_md_alg( mbedtls_x509write_cert *ctx, mbedtls_md_type_t md_alg );

/**
 * \brief           Generic function to add to or replace an extension in the
 *                  CRT
 *
 * \param ctx       CRT context to use
 * \param oid       OID of the extension
 * \param oid_len   length of the OID
 * \param critical  if the extension is critical (per the RFC's definition)
 * \param val       value of the extension OCTET STRING
 * \param val_len   length of the value data
 *
 * \return          0 if successful, or a MBEDTLS_ERR_X509_ALLOC_FAILED
 */
int mbedtls_x509write_crt_set_extension( mbedtls_x509write_cert *ctx,
                                 const char *oid, size_t oid_len,
                                 int critical,
                                 const unsigned char *val, size_t val_len );

/**
 * \brief           Set the basicConstraints extension for a CRT
 *
 * \param ctx       CRT context to use
 * \param is_ca     is this a CA certificate
 * \param max_pathlen   maximum length of certificate chains below this
 *                      certificate (only for CA certificates, -1 is
 *                      inlimited)
 *
 * \return          0 if successful, or a MBEDTLS_ERR_X509_ALLOC_FAILED
 */
int mbedtls_x509write_crt_set_basic_constraints( mbedtls_x509write_cert *ctx,
                                         int is_ca, int max_pathlen );

#if defined(MBEDTLS_SHA1_C)
/**
 * \brief           Set the subjectKeyIdentifier extension for a CRT
 *                  Requires that mbedtls_x509write_crt_set_subject_key() has been
 *                  called before
 *
 * \param ctx       CRT context to use
 *
 * \return          0 if successful, or a MBEDTLS_ERR_X509_ALLOC_FAILED
 */
int mbedtls_x509write_crt_set_subject_key_identifier( mbedtls_x509write_cert *ctx );

/**
 * \brief           Set the authorityKeyIdentifier extension for a CRT
 *                  Requires that mbedtls_x509write_crt_set_issuer_key() has been
 *                  called before
 *
 * \param ctx       CRT context to use
 *
 * \return          0 if successful, or a MBEDTLS_ERR_X509_ALLOC_FAILED
 */
int mbedtls_x509write_crt_set_authority_key_identifier( mbedtls_x509write_cert *ctx );
#endif /* MBEDTLS_SHA1_C */

/**
 * \brief           Set the Key Usage Extension flags
 *                  (e.g. MBEDTLS_X509_KU_DIGITAL_SIGNATURE | MBEDTLS_X509_KU_KEY_CERT_SIGN)
 *
 * \param ctx       CRT context to use
 * \param key_usage key usage flags to set
 *
 * \return          0 if successful, or MBEDTLS_ERR_X509_ALLOC_FAILED
 */
int mbedtls_x509write_crt_set_key_usage( mbedtls_x509write_cert *ctx,
                                         unsigned int key_usage );

/**
 * \brief           Set the Netscape Cert Type flags
 *                  (e.g. MBEDTLS_X509_NS_CERT_TYPE_SSL_CLIENT | MBEDTLS_X509_NS_CERT_TYPE_EMAIL)
 *
 * \param ctx           CRT context to use
 * \param ns_cert_type  Netscape Cert Type flags to set
 *
 * \return          0 if successful, or MBEDTLS_ERR_X509_ALLOC_FAILED
 */
int mbedtls_x509write_crt_set_ns_cert_type( mbedtls_x509write_cert *ctx,
                                    unsigned char ns_cert_type );

/**
 * \brief           Free the contents of a CRT write context
 *
 * \param ctx       CRT context to free
 */
void mbedtls_x509write_crt_free( mbedtls_x509write_cert *ctx );

/**
 * \brief           Write a built up certificate to a X509 DER structure
 *                  Note: data is written at the end of the buffer! Use the
 *                        return value to determine where you should start
 *                        using the buffer
 *
 * \param ctx       certificate to write away
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
int mbedtls_x509write_crt_der( mbedtls_x509write_cert *ctx, unsigned char *buf, size_t size,
                       int (*f_rng)(void *, unsigned char *, size_t),
                       void *p_rng );

#if defined(MBEDTLS_PEM_WRITE_C)
/**
 * \brief           Write a built up certificate to a X509 PEM string
 *
 * \param ctx       certificate to write away
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
int mbedtls_x509write_crt_pem( mbedtls_x509write_cert *ctx, unsigned char *buf, size_t size,
                       int (*f_rng)(void *, unsigned char *, size_t),
                       void *p_rng );
#endif /* MBEDTLS_PEM_WRITE_C */
#endif /* MBEDTLS_X509_CRT_WRITE_C */

#ifdef __cplusplus
}
#endif

#endif /* mbedtls_x509_crt.h */
