/**
 * \file pk_private.h
 *
 * \brief Private Public Key abstraction layer
 */
/*
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */

#ifndef MBEDTLS_PRIVATE_PK_PRIVATE_H
#define MBEDTLS_PRIVATE_PK_PRIVATE_H

#include <mbedtls/pk.h>

#if defined(MBEDTLS_DECLARE_PRIVATE_IDENTIFIERS)

/**
 * \brief          Public key types
 */
typedef enum {
    MBEDTLS_PK_NONE = MBEDTLS_PK_SIGALG_NONE,
    MBEDTLS_PK_RSA = MBEDTLS_PK_SIGALG_RSA_PKCS1V15,
    MBEDTLS_PK_RSASSA_PSS = MBEDTLS_PK_SIGALG_RSA_PSS,
    MBEDTLS_PK_ECDSA = MBEDTLS_PK_SIGALG_ECDSA,
    MBEDTLS_PK_ECKEY,
    MBEDTLS_PK_ECKEY_DH,
    MBEDTLS_PK_OPAQUE,
} mbedtls_pk_type_t;

/**
 * \brief           Return information associated with the given PK type
 *
 * \param pk_type   PK type to search for.
 *
 * \return          The PK info associated with the type or NULL if not found.
 */
const mbedtls_pk_info_t *mbedtls_pk_info_from_type(mbedtls_pk_type_t pk_type);

/**
 * \brief           Initialize a PK context with the information given
 *                  and allocates the type-specific PK subcontext.
 *
 * \param ctx       Context to initialize. It must not have been set
 *                  up yet (type #MBEDTLS_PK_NONE).
 * \param info      Information to use
 *
 * \return          0 on success,
 *                  MBEDTLS_ERR_PK_BAD_INPUT_DATA on invalid input,
 *                  MBEDTLS_ERR_PK_ALLOC_FAILED on allocation failure.
 */
int mbedtls_pk_setup(mbedtls_pk_context *ctx, const mbedtls_pk_info_t *info);

/**
 * \brief           Get the length in bytes of the underlying key
 *
 * \param ctx       The context to query. It must have been initialized.
 *
 * \return          Key length in bytes, or 0 on error
 */
static inline size_t mbedtls_pk_get_len(const mbedtls_pk_context *ctx)
{
    return (mbedtls_pk_get_bitlen(ctx) + 7) / 8;
}

/**
 * \brief           Tell if a context can do the operation given by type
 *
 * \param ctx       The context to query. It must have been initialized.
 * \param type      The desired type.
 *
 * \return          1 if the context can do operations on the given type.
 * \return          0 if the context cannot do the operations on the given
 *                  type. This is always the case for a context that has
 *                  been initialized but not set up, or that has been
 *                  cleared with mbedtls_pk_free().
 */
int mbedtls_pk_can_do(const mbedtls_pk_context *ctx, mbedtls_pk_type_t type);

/**
 * \brief           Tell if context can do the operation given by PSA algorithm
 *
 * \param ctx       The context to query. It must have been initialized.
 * \param alg       PSA algorithm to check against, the following are allowed:
 *                  PSA_ALG_RSA_PKCS1V15_SIGN(hash),
 *                  PSA_ALG_RSA_PSS(hash),
 *                  PSA_ALG_RSA_PKCS1V15_CRYPT,
 *                  PSA_ALG_ECDSA(hash),
 *                  PSA_ALG_ECDH, where hash is a specific hash.
 * \param usage     PSA usage flag to check against, must be composed of:
 *                  PSA_KEY_USAGE_SIGN_HASH
 *                  PSA_KEY_USAGE_DECRYPT
 *                  PSA_KEY_USAGE_DERIVE.
 *                  Context key must match all passed usage flags.
 *
 * \warning         Since the set of allowed algorithms and usage flags may be
 *                  expanded in the future, the return value \c 0 should not
 *                  be taken in account for non-allowed algorithms and usage
 *                  flags.
 *
 * \return          1 if the context can do operations on the given type.
 * \return          0 if the context cannot do the operations on the given
 *                  type, for non-allowed algorithms and usage flags, or
 *                  for a context that has been initialized but not set up
 *                  or that has been cleared with mbedtls_pk_free().
 */
int mbedtls_pk_can_do_ext(const mbedtls_pk_context *ctx, psa_algorithm_t alg,
                          psa_key_usage_t usage);

/**
 * \brief           Get the key type
 *
 * \param ctx       The PK context to use. It must have been initialized.
 *
 * \return          Type on success.
 * \return          #MBEDTLS_PK_NONE for a context that has not been set up.
 */
mbedtls_pk_type_t mbedtls_pk_get_type(const mbedtls_pk_context *ctx);

#if defined(MBEDTLS_PK_PARSE_C)
/**
 * \brief           Parse a SubjectPublicKeyInfo DER structure
 *
 * \param p         the position in the ASN.1 data
 * \param end       end of the buffer
 * \param pk        The PK context to fill. It must have been initialized
 *                  but not set up.
 *
 * \return          0 if successful, or a specific PK error code
 */
int mbedtls_pk_parse_subpubkey(unsigned char **p, const unsigned char *end,
                               mbedtls_pk_context *pk);
#endif /* MBEDTLS_PK_PARSE_C */

#if defined(MBEDTLS_PK_WRITE_C)
/**
 * \brief           Write a subjectPublicKey to ASN.1 data
 *                  Note: function works backwards in data buffer
 *
 * \param p         reference to current position pointer
 * \param start     start of the buffer (for bounds-checking)
 * \param key       PK context which must contain a valid public or private key.
 *
 * \return          the length written or a negative error code
 */
int mbedtls_pk_write_pubkey(unsigned char **p, unsigned char *start,
                            const mbedtls_pk_context *key);
#endif /* MBEDTLS_PK_WRITE_C */
#endif /* MBEDTLS_DECLARE_PRIVATE_IDENTIFIERS */
#endif /* MBEDTLS_PRIVATE_PK_PRIVATE_H */
