/**
 * \file pk_internal.h
 *
 * \brief Public Key abstraction layer: internal (i.e. library only) functions
 *        and definitions.
 */
/*
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */
#ifndef MBEDTLS_PK_INTERNAL_H
#define MBEDTLS_PK_INTERNAL_H

#include "mbedtls/pk.h"

#if defined(MBEDTLS_PK_HAVE_ECC_KEYS)
#include "mbedtls/ecp.h"
#endif

#if defined(MBEDTLS_USE_PSA_CRYPTO)
#include "psa/crypto.h"
#endif

#if defined(MBEDTLS_PSA_CRYPTO_C)
#include "psa_util_internal.h"
#define PSA_PK_TO_MBEDTLS_ERR(status) psa_pk_status_to_mbedtls(status)
#define PSA_PK_RSA_TO_MBEDTLS_ERR(status) PSA_TO_MBEDTLS_ERR_LIST(status,     \
                                                                  psa_to_pk_rsa_errors,            \
                                                                  psa_pk_status_to_mbedtls)
#define PSA_PK_ECDSA_TO_MBEDTLS_ERR(status) PSA_TO_MBEDTLS_ERR_LIST(status,   \
                                                                    psa_to_pk_ecdsa_errors,        \
                                                                    psa_pk_status_to_mbedtls)
#endif

#if defined(MBEDTLS_PK_HAVE_ECC_KEYS)
/**
 * Public function mbedtls_pk_ec() can be used to get direct access to the
 * wrapped ecp_keypair structure pointed to the pk_ctx. However this is not
 * ideal because it bypasses the PK module on the control of its internal
 * structure (pk_context) fields.
 * For backward compatibility we keep mbedtls_pk_ec() when ECP_C is defined, but
 * we provide 2 very similar functions when only ECP_LIGHT is enabled and not
 * ECP_C.
 * These variants embed the "ro" or "rw" keywords in their name to make the
 * usage of the returned pointer explicit. Of course the returned value is
 * const or non-const accordingly.
 */
static inline const mbedtls_ecp_keypair *mbedtls_pk_ec_ro(const mbedtls_pk_context pk)
{
    switch (mbedtls_pk_get_type(&pk)) {
        case MBEDTLS_PK_ECKEY:
        case MBEDTLS_PK_ECKEY_DH:
        case MBEDTLS_PK_ECDSA:
            return (const mbedtls_ecp_keypair *) (pk).MBEDTLS_PRIVATE(pk_ctx);
        default:
            return NULL;
    }
}

static inline mbedtls_ecp_keypair *mbedtls_pk_ec_rw(const mbedtls_pk_context pk)
{
    switch (mbedtls_pk_get_type(&pk)) {
        case MBEDTLS_PK_ECKEY:
        case MBEDTLS_PK_ECKEY_DH:
        case MBEDTLS_PK_ECDSA:
            return (mbedtls_ecp_keypair *) (pk).MBEDTLS_PRIVATE(pk_ctx);
        default:
            return NULL;
    }
}

static inline mbedtls_ecp_group_id mbedtls_pk_get_group_id(const mbedtls_pk_context *pk)
{
    mbedtls_ecp_group_id id;

#if defined(MBEDTLS_USE_PSA_CRYPTO)
    if (mbedtls_pk_get_type(pk) == MBEDTLS_PK_OPAQUE) {
        psa_key_attributes_t opaque_attrs = PSA_KEY_ATTRIBUTES_INIT;
        psa_key_type_t opaque_key_type;
        psa_ecc_family_t curve;

        if (psa_get_key_attributes(pk->priv_id, &opaque_attrs) != PSA_SUCCESS) {
            return MBEDTLS_ECP_DP_NONE;
        }
        opaque_key_type = psa_get_key_type(&opaque_attrs);
        curve = PSA_KEY_TYPE_ECC_GET_FAMILY(opaque_key_type);
        id = mbedtls_ecc_group_of_psa(curve, psa_get_key_bits(&opaque_attrs), 0);
        psa_reset_key_attributes(&opaque_attrs);
    } else
#endif /* MBEDTLS_USE_PSA_CRYPTO */
    {
#if defined(MBEDTLS_PK_USE_PSA_EC_DATA)
        id = mbedtls_ecc_group_of_psa(pk->ec_family, pk->ec_bits, 0);
#else /* MBEDTLS_PK_USE_PSA_EC_DATA */
        id = mbedtls_pk_ec_ro(*pk)->grp.id;
#endif /* MBEDTLS_PK_USE_PSA_EC_DATA */
    }

    return id;
}

/* Helper for Montgomery curves */
#if defined(MBEDTLS_ECP_HAVE_CURVE25519) || defined(MBEDTLS_ECP_HAVE_CURVE448)
#define MBEDTLS_PK_HAVE_RFC8410_CURVES
#endif /* MBEDTLS_ECP_HAVE_CURVE25519 || MBEDTLS_ECP_DP_CURVE448 */
#endif /* MBEDTLS_PK_HAVE_ECC_KEYS */

#if defined(MBEDTLS_TEST_HOOKS)

MBEDTLS_STATIC_TESTABLE int mbedtls_pk_parse_key_pkcs8_encrypted_der(
    mbedtls_pk_context *pk,
    unsigned char *key, size_t keylen,
    const unsigned char *pwd, size_t pwdlen,
    int (*f_rng)(void *, unsigned char *, size_t), void *p_rng);

#endif

#endif /* MBEDTLS_PK_INTERNAL_H */
