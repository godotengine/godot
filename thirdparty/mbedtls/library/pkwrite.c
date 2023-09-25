/*
 *  Public Key layer for writing key files and structures
 *
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */

#include "common.h"

#if defined(MBEDTLS_PK_WRITE_C)

#include "mbedtls/pk.h"
#include "mbedtls/asn1write.h"
#include "mbedtls/oid.h"
#include "mbedtls/platform_util.h"
#include "mbedtls/error.h"
#include "pk_internal.h"

#include <string.h>

#if defined(MBEDTLS_RSA_C)
#include "mbedtls/rsa.h"
#endif
#if defined(MBEDTLS_ECP_C)
#include "mbedtls/bignum.h"
#include "mbedtls/ecp.h"
#include "mbedtls/platform_util.h"
#endif
#if defined(MBEDTLS_PK_HAVE_ECC_KEYS)
#include "pk_internal.h"
#endif
#if defined(MBEDTLS_RSA_C) || defined(MBEDTLS_PK_HAVE_ECC_KEYS)
#include "pkwrite.h"
#endif
#if defined(MBEDTLS_ECDSA_C)
#include "mbedtls/ecdsa.h"
#endif
#if defined(MBEDTLS_PEM_WRITE_C)
#include "mbedtls/pem.h"
#endif

#if defined(MBEDTLS_USE_PSA_CRYPTO)
#include "psa/crypto.h"
#include "psa_util_internal.h"
#endif
#include "mbedtls/platform.h"

/* Helper for Montgomery curves */
#if defined(MBEDTLS_PK_HAVE_ECC_KEYS)
#if defined(MBEDTLS_PK_HAVE_RFC8410_CURVES)
static inline int mbedtls_pk_is_rfc8410(const mbedtls_pk_context *pk)
{
    mbedtls_ecp_group_id id = mbedtls_pk_get_group_id(pk);

#if defined(MBEDTLS_ECP_HAVE_CURVE25519)
    if (id == MBEDTLS_ECP_DP_CURVE25519) {
        return 1;
    }
#endif
#if defined(MBEDTLS_ECP_HAVE_CURVE448)
    if (id == MBEDTLS_ECP_DP_CURVE448) {
        return 1;
    }
#endif
    return 0;
}

#if defined(MBEDTLS_USE_PSA_CRYPTO) && defined(MBEDTLS_PEM_WRITE_C)
/* It is assumed that the input key is opaque */
static psa_ecc_family_t pk_get_opaque_ec_family(const mbedtls_pk_context *pk)
{
    psa_ecc_family_t ec_family = 0;
    psa_key_attributes_t key_attrs = PSA_KEY_ATTRIBUTES_INIT;

    if (psa_get_key_attributes(pk->priv_id, &key_attrs) != PSA_SUCCESS) {
        return 0;
    }
    ec_family = PSA_KEY_TYPE_ECC_GET_FAMILY(psa_get_key_type(&key_attrs));
    psa_reset_key_attributes(&key_attrs);

    return ec_family;
}
#endif /* MBETLS_USE_PSA_CRYPTO && MBEDTLS_PEM_WRITE_C */
#endif /* MBEDTLS_PK_HAVE_RFC8410_CURVES */
#endif /* MBEDTLS_PK_HAVE_ECC_KEYS */

#if defined(MBEDTLS_USE_PSA_CRYPTO)
/* It is assumed that the input key is opaque */
static psa_key_type_t pk_get_opaque_key_type(const mbedtls_pk_context *pk)
{
    psa_key_attributes_t opaque_attrs = PSA_KEY_ATTRIBUTES_INIT;
    psa_key_type_t opaque_key_type;

    if (psa_get_key_attributes(pk->priv_id, &opaque_attrs) != PSA_SUCCESS) {
        return 0;
    }
    opaque_key_type = psa_get_key_type(&opaque_attrs);
    psa_reset_key_attributes(&opaque_attrs);

    return opaque_key_type;
}
#endif /* MBETLS_USE_PSA_CRYPTO */

#if defined(MBEDTLS_RSA_C)
/*
 *  RSAPublicKey ::= SEQUENCE {
 *      modulus           INTEGER,  -- n
 *      publicExponent    INTEGER   -- e
 *  }
 */
static int pk_write_rsa_pubkey(unsigned char **p, unsigned char *start,
                               const mbedtls_pk_context *pk)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    size_t len = 0;
    mbedtls_mpi T;
    mbedtls_rsa_context *rsa = mbedtls_pk_rsa(*pk);

    mbedtls_mpi_init(&T);

    /* Export E */
    if ((ret = mbedtls_rsa_export(rsa, NULL, NULL, NULL, NULL, &T)) != 0 ||
        (ret = mbedtls_asn1_write_mpi(p, start, &T)) < 0) {
        goto end_of_export;
    }
    len += ret;

    /* Export N */
    if ((ret = mbedtls_rsa_export(rsa, &T, NULL, NULL, NULL, NULL)) != 0 ||
        (ret = mbedtls_asn1_write_mpi(p, start, &T)) < 0) {
        goto end_of_export;
    }
    len += ret;

end_of_export:

    mbedtls_mpi_free(&T);
    if (ret < 0) {
        return ret;
    }

    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_len(p, start, len));
    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_tag(p, start, MBEDTLS_ASN1_CONSTRUCTED |
                                                     MBEDTLS_ASN1_SEQUENCE));

    return (int) len;
}
#endif /* MBEDTLS_RSA_C */

#if defined(MBEDTLS_PK_HAVE_ECC_KEYS)
#if defined(MBEDTLS_PK_USE_PSA_EC_DATA)
static int pk_write_ec_pubkey(unsigned char **p, unsigned char *start,
                              const mbedtls_pk_context *pk)
{
    size_t len = 0;
    uint8_t buf[PSA_EXPORT_PUBLIC_KEY_MAX_SIZE];

    if (mbedtls_pk_get_type(pk) == MBEDTLS_PK_OPAQUE) {
        if (psa_export_public_key(pk->priv_id, buf, sizeof(buf), &len) != PSA_SUCCESS) {
            return MBEDTLS_ERR_PK_BAD_INPUT_DATA;
        }
    } else {
        len = pk->pub_raw_len;
        memcpy(buf, pk->pub_raw, len);
    }

    if (*p < start || (size_t) (*p - start) < len) {
        return MBEDTLS_ERR_ASN1_BUF_TOO_SMALL;
    }

    *p -= len;
    memcpy(*p, buf, len);

    return (int) len;
}
#else /* MBEDTLS_PK_USE_PSA_EC_DATA */
static int pk_write_ec_pubkey(unsigned char **p, unsigned char *start,
                              const mbedtls_pk_context *pk)
{
    size_t len = 0;
#if defined(MBEDTLS_USE_PSA_CRYPTO)
    uint8_t buf[PSA_EXPORT_PUBLIC_KEY_MAX_SIZE];
#else
    unsigned char buf[MBEDTLS_ECP_MAX_PT_LEN];
#endif /* MBEDTLS_USE_PSA_CRYPTO */
    mbedtls_ecp_keypair *ec = mbedtls_pk_ec(*pk);
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;

#if defined(MBEDTLS_USE_PSA_CRYPTO)
    if (mbedtls_pk_get_type(pk) == MBEDTLS_PK_OPAQUE) {
        if (psa_export_public_key(pk->priv_id, buf, sizeof(buf), &len) != PSA_SUCCESS) {
            return MBEDTLS_ERR_PK_BAD_INPUT_DATA;
        }
        *p -= len;
        memcpy(*p, buf, len);
        return (int) len;
    } else
#endif /* MBEDTLS_USE_PSA_CRYPTO */
    {
        if ((ret = mbedtls_ecp_point_write_binary(&ec->grp, &ec->Q,
                                                  MBEDTLS_ECP_PF_UNCOMPRESSED,
                                                  &len, buf, sizeof(buf))) != 0) {
            return ret;
        }
    }

    if (*p < start || (size_t) (*p - start) < len) {
        return MBEDTLS_ERR_ASN1_BUF_TOO_SMALL;
    }

    *p -= len;
    memcpy(*p, buf, len);

    return (int) len;
}
#endif /* MBEDTLS_PK_USE_PSA_EC_DATA */

/*
 * ECParameters ::= CHOICE {
 *   namedCurve         OBJECT IDENTIFIER
 * }
 */
static int pk_write_ec_param(unsigned char **p, unsigned char *start,
                             mbedtls_ecp_group_id grp_id)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    size_t len = 0;
    const char *oid;
    size_t oid_len;

    if ((ret = mbedtls_oid_get_oid_by_ec_grp(grp_id, &oid, &oid_len)) != 0) {
        return ret;
    }

    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_oid(p, start, oid, oid_len));

    return (int) len;
}

/*
 * privateKey  OCTET STRING -- always of length ceil(log2(n)/8)
 */
#if defined(MBEDTLS_PK_USE_PSA_EC_DATA)
static int pk_write_ec_private(unsigned char **p, unsigned char *start,
                               const mbedtls_pk_context *pk)
{
    size_t byte_length;
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    unsigned char tmp[MBEDTLS_PSA_MAX_EC_KEY_PAIR_LENGTH];
    psa_status_t status;

    if (mbedtls_pk_get_type(pk) == MBEDTLS_PK_OPAQUE) {
        status = psa_export_key(pk->priv_id, tmp, sizeof(tmp), &byte_length);
        if (status != PSA_SUCCESS) {
            ret = PSA_PK_ECDSA_TO_MBEDTLS_ERR(status);
            return ret;
        }
    } else {
        status = psa_export_key(pk->priv_id, tmp, sizeof(tmp), &byte_length);
        if (status != PSA_SUCCESS) {
            ret = PSA_PK_ECDSA_TO_MBEDTLS_ERR(status);
            goto exit;
        }
    }

    ret = mbedtls_asn1_write_octet_string(p, start, tmp, byte_length);
exit:
    mbedtls_platform_zeroize(tmp, sizeof(tmp));
    return ret;
}
#else /* MBEDTLS_PK_USE_PSA_EC_DATA */
static int pk_write_ec_private(unsigned char **p, unsigned char *start,
                               const mbedtls_pk_context *pk)
{
    size_t byte_length;
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
#if defined(MBEDTLS_USE_PSA_CRYPTO)
    unsigned char tmp[MBEDTLS_PSA_MAX_EC_KEY_PAIR_LENGTH];
    psa_status_t status;
#else
    unsigned char tmp[MBEDTLS_ECP_MAX_BYTES];
#endif /* MBEDTLS_USE_PSA_CRYPTO */

#if defined(MBEDTLS_USE_PSA_CRYPTO)
    if (mbedtls_pk_get_type(pk) == MBEDTLS_PK_OPAQUE) {
        status = psa_export_key(pk->priv_id, tmp, sizeof(tmp), &byte_length);
        if (status != PSA_SUCCESS) {
            ret = PSA_PK_ECDSA_TO_MBEDTLS_ERR(status);
            return ret;
        }
    } else
#endif /* MBEDTLS_USE_PSA_CRYPTO */
    {
        mbedtls_ecp_keypair *ec = mbedtls_pk_ec_rw(*pk);
        byte_length = (ec->grp.pbits + 7) / 8;

        ret = mbedtls_ecp_write_key(ec, tmp, byte_length);
        if (ret != 0) {
            goto exit;
        }
    }
    ret = mbedtls_asn1_write_octet_string(p, start, tmp, byte_length);
exit:
    mbedtls_platform_zeroize(tmp, sizeof(tmp));
    return ret;
}
#endif /* MBEDTLS_PK_USE_PSA_EC_DATA */
#endif /* MBEDTLS_PK_HAVE_ECC_KEYS */

#if defined(MBEDTLS_USE_PSA_CRYPTO)
static int pk_write_opaque_pubkey(unsigned char **p, unsigned char *start,
                                  const mbedtls_pk_context *pk)
{
    size_t buffer_size;
    size_t len = 0;

    if (*p < start) {
        return MBEDTLS_ERR_PK_BAD_INPUT_DATA;
    }

    buffer_size = (size_t) (*p - start);
    if (psa_export_public_key(pk->priv_id, start, buffer_size,
                              &len) != PSA_SUCCESS) {
        return MBEDTLS_ERR_PK_BAD_INPUT_DATA;
    }

    *p -= len;
    memmove(*p, start, len);

    return (int) len;
}
#endif /* MBEDTLS_USE_PSA_CRYPTO */

int mbedtls_pk_write_pubkey(unsigned char **p, unsigned char *start,
                            const mbedtls_pk_context *key)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    size_t len = 0;

#if defined(MBEDTLS_RSA_C)
    if (mbedtls_pk_get_type(key) == MBEDTLS_PK_RSA) {
        MBEDTLS_ASN1_CHK_ADD(len, pk_write_rsa_pubkey(p, start, key));
    } else
#endif
#if defined(MBEDTLS_PK_HAVE_ECC_KEYS)
    if (mbedtls_pk_get_type(key) == MBEDTLS_PK_ECKEY) {
        MBEDTLS_ASN1_CHK_ADD(len, pk_write_ec_pubkey(p, start, key));
    } else
#endif
#if defined(MBEDTLS_USE_PSA_CRYPTO)
    if (mbedtls_pk_get_type(key) == MBEDTLS_PK_OPAQUE) {
        MBEDTLS_ASN1_CHK_ADD(len, pk_write_opaque_pubkey(p, start, key));
    } else
#endif /* MBEDTLS_USE_PSA_CRYPTO */
    return MBEDTLS_ERR_PK_FEATURE_UNAVAILABLE;

    return (int) len;
}

int mbedtls_pk_write_pubkey_der(const mbedtls_pk_context *key, unsigned char *buf, size_t size)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    unsigned char *c;
    int has_par = 1;
    size_t len = 0, par_len = 0, oid_len = 0;
    mbedtls_pk_type_t pk_type;
#if defined(MBEDTLS_PK_HAVE_ECC_KEYS)
    mbedtls_ecp_group_id ec_grp_id = MBEDTLS_ECP_DP_NONE;
#endif
    const char *oid = NULL;

    if (size == 0) {
        return MBEDTLS_ERR_ASN1_BUF_TOO_SMALL;
    }

    c = buf + size;

    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_pk_write_pubkey(&c, buf, key));

    if (c - buf < 1) {
        return MBEDTLS_ERR_ASN1_BUF_TOO_SMALL;
    }

    /*
     *  SubjectPublicKeyInfo  ::=  SEQUENCE  {
     *       algorithm            AlgorithmIdentifier,
     *       subjectPublicKey     BIT STRING }
     */
    *--c = 0;
    len += 1;

    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_len(&c, buf, len));
    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_tag(&c, buf, MBEDTLS_ASN1_BIT_STRING));

    pk_type = mbedtls_pk_get_type(key);
#if defined(MBEDTLS_PK_HAVE_ECC_KEYS)
    if (pk_type == MBEDTLS_PK_ECKEY) {
        ec_grp_id = mbedtls_pk_get_group_id(key);
    }
#endif /* MBEDTLS_PK_HAVE_ECC_KEYS */
#if defined(MBEDTLS_USE_PSA_CRYPTO)
    if (pk_type == MBEDTLS_PK_OPAQUE) {
        psa_key_type_t opaque_key_type = pk_get_opaque_key_type(key);
#if defined(MBEDTLS_PK_HAVE_ECC_KEYS)
        if (PSA_KEY_TYPE_IS_ECC(opaque_key_type)) {
            pk_type = MBEDTLS_PK_ECKEY;
            ec_grp_id = mbedtls_pk_get_group_id(key);
        } else
#endif /* MBEDTLS_PK_HAVE_ECC_KEYS */
        if (PSA_KEY_TYPE_IS_RSA(opaque_key_type)) {
            /* The rest of the function works as for legacy RSA contexts. */
            pk_type = MBEDTLS_PK_RSA;
        }
    }
    /* `pk_type` will have been changed to non-opaque by here if this function can handle it */
    if (pk_type == MBEDTLS_PK_OPAQUE) {
        return MBEDTLS_ERR_PK_FEATURE_UNAVAILABLE;
    }
#endif /* MBEDTLS_USE_PSA_CRYPTO */

#if defined(MBEDTLS_PK_HAVE_ECC_KEYS)
    if (pk_type == MBEDTLS_PK_ECKEY) {
        /* Some groups have their own AlgorithmIdentifier OID, others are handled
         * by mbedtls_oid_get_oid_by_pk_alg() below */
        ret = mbedtls_oid_get_oid_by_ec_grp_algid(ec_grp_id, &oid, &oid_len);

        if (ret == 0) {
            /* Currently, none of the supported algorithms that have their own
             * AlgorithmIdentifier OID have any parameters */
            has_par = 0;
        } else if (ret == MBEDTLS_ERR_OID_NOT_FOUND) {
            MBEDTLS_ASN1_CHK_ADD(par_len, pk_write_ec_param(&c, buf, ec_grp_id));
        } else {
            return ret;
        }
    }
#endif /* MBEDTLS_PK_HAVE_ECC_KEYS */

    if (oid_len == 0) {
        if ((ret = mbedtls_oid_get_oid_by_pk_alg(pk_type, &oid,
                                                 &oid_len)) != 0) {
            return ret;
        }
    }

    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_algorithm_identifier_ext(&c, buf, oid, oid_len,
                                                                          par_len, has_par));

    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_len(&c, buf, len));
    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_tag(&c, buf, MBEDTLS_ASN1_CONSTRUCTED |
                                                     MBEDTLS_ASN1_SEQUENCE));

    return (int) len;
}

#if defined(MBEDTLS_PK_HAVE_ECC_KEYS)
#if defined(MBEDTLS_PK_HAVE_RFC8410_CURVES)
/*
 * RFC8410 section 7
 *
 * OneAsymmetricKey ::= SEQUENCE {
 *    version Version,
 *    privateKeyAlgorithm PrivateKeyAlgorithmIdentifier,
 *    privateKey PrivateKey,
 *    attributes [0] IMPLICIT Attributes OPTIONAL,
 *    ...,
 *    [[2: publicKey [1] IMPLICIT PublicKey OPTIONAL ]],
 *    ...
 * }
 * ...
 * CurvePrivateKey ::= OCTET STRING
 */
static int pk_write_ec_rfc8410_der(unsigned char **p, unsigned char *buf,
                                   const mbedtls_pk_context *pk)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    size_t len = 0;
    size_t oid_len = 0;
    const char *oid;
    mbedtls_ecp_group_id grp_id;

    /* privateKey */
    MBEDTLS_ASN1_CHK_ADD(len, pk_write_ec_private(p, buf, pk));
    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_len(p, buf, len));
    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_tag(p, buf, MBEDTLS_ASN1_OCTET_STRING));

    grp_id = mbedtls_pk_get_group_id(pk);
    /* privateKeyAlgorithm */
    if ((ret = mbedtls_oid_get_oid_by_ec_grp_algid(grp_id, &oid, &oid_len)) != 0) {
        return ret;
    }
    MBEDTLS_ASN1_CHK_ADD(len,
                         mbedtls_asn1_write_algorithm_identifier_ext(p, buf, oid, oid_len, 0, 0));

    /* version */
    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_int(p, buf, 0));

    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_len(p, buf, len));
    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_tag(p, buf, MBEDTLS_ASN1_CONSTRUCTED |
                                                     MBEDTLS_ASN1_SEQUENCE));

    return (int) len;
}
#endif /* MBEDTLS_PK_HAVE_RFC8410_CURVES */

/*
 * RFC 5915, or SEC1 Appendix C.4
 *
 * ECPrivateKey ::= SEQUENCE {
 *      version        INTEGER { ecPrivkeyVer1(1) } (ecPrivkeyVer1),
 *      privateKey     OCTET STRING,
 *      parameters [0] ECParameters {{ NamedCurve }} OPTIONAL,
 *      publicKey  [1] BIT STRING OPTIONAL
 *    }
 */
static int pk_write_ec_der(unsigned char **p, unsigned char *buf,
                           const mbedtls_pk_context *pk)
{
    size_t len = 0;
    int ret;
    size_t pub_len = 0, par_len = 0;
    mbedtls_ecp_group_id grp_id;

    /* publicKey */
    MBEDTLS_ASN1_CHK_ADD(pub_len, pk_write_ec_pubkey(p, buf, pk));

    if (*p - buf < 1) {
        return MBEDTLS_ERR_ASN1_BUF_TOO_SMALL;
    }
    (*p)--;
    **p = 0;
    pub_len += 1;

    MBEDTLS_ASN1_CHK_ADD(pub_len, mbedtls_asn1_write_len(p, buf, pub_len));
    MBEDTLS_ASN1_CHK_ADD(pub_len, mbedtls_asn1_write_tag(p, buf, MBEDTLS_ASN1_BIT_STRING));

    MBEDTLS_ASN1_CHK_ADD(pub_len, mbedtls_asn1_write_len(p, buf, pub_len));
    MBEDTLS_ASN1_CHK_ADD(pub_len, mbedtls_asn1_write_tag(p, buf,
                                                         MBEDTLS_ASN1_CONTEXT_SPECIFIC |
                                                         MBEDTLS_ASN1_CONSTRUCTED | 1));
    len += pub_len;

    /* parameters */
    grp_id = mbedtls_pk_get_group_id(pk);
    MBEDTLS_ASN1_CHK_ADD(par_len, pk_write_ec_param(p, buf, grp_id));
    MBEDTLS_ASN1_CHK_ADD(par_len, mbedtls_asn1_write_len(p, buf, par_len));
    MBEDTLS_ASN1_CHK_ADD(par_len, mbedtls_asn1_write_tag(p, buf,
                                                         MBEDTLS_ASN1_CONTEXT_SPECIFIC |
                                                         MBEDTLS_ASN1_CONSTRUCTED | 0));
    len += par_len;

    /* privateKey */
    MBEDTLS_ASN1_CHK_ADD(len, pk_write_ec_private(p, buf, pk));

    /* version */
    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_int(p, buf, 1));

    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_len(p, buf, len));
    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_tag(p, buf, MBEDTLS_ASN1_CONSTRUCTED |
                                                     MBEDTLS_ASN1_SEQUENCE));

    return (int) len;
}
#endif /* MBEDTLS_PK_HAVE_ECC_KEYS */

#if defined(MBEDTLS_RSA_C)
static int pk_write_rsa_der(unsigned char **p, unsigned char *buf,
                            const mbedtls_pk_context *pk)
{
    size_t len = 0;
    int ret;

#if defined(MBEDTLS_USE_PSA_CRYPTO)
    if (mbedtls_pk_get_type(pk) == MBEDTLS_PK_OPAQUE) {
        uint8_t tmp[PSA_EXPORT_KEY_PAIR_MAX_SIZE];
        size_t tmp_len = 0;

        if (psa_export_key(pk->priv_id, tmp, sizeof(tmp), &tmp_len) != PSA_SUCCESS) {
            return MBEDTLS_ERR_PK_BAD_INPUT_DATA;
        }
        *p -= tmp_len;
        memcpy(*p, tmp, tmp_len);
        len += tmp_len;
        mbedtls_platform_zeroize(tmp, sizeof(tmp));
    } else
#endif /* MBEDTLS_USE_PSA_CRYPTO */
    {
        mbedtls_mpi T; /* Temporary holding the exported parameters */
        mbedtls_rsa_context *rsa = mbedtls_pk_rsa(*pk);

        /*
         * Export the parameters one after another to avoid simultaneous copies.
         */

        mbedtls_mpi_init(&T);

        /* Export QP */
        if ((ret = mbedtls_rsa_export_crt(rsa, NULL, NULL, &T)) != 0 ||
            (ret = mbedtls_asn1_write_mpi(p, buf, &T)) < 0) {
            goto end_of_export;
        }
        len += ret;

        /* Export DQ */
        if ((ret = mbedtls_rsa_export_crt(rsa, NULL, &T, NULL)) != 0 ||
            (ret = mbedtls_asn1_write_mpi(p, buf, &T)) < 0) {
            goto end_of_export;
        }
        len += ret;

        /* Export DP */
        if ((ret = mbedtls_rsa_export_crt(rsa, &T, NULL, NULL)) != 0 ||
            (ret = mbedtls_asn1_write_mpi(p, buf, &T)) < 0) {
            goto end_of_export;
        }
        len += ret;

        /* Export Q */
        if ((ret = mbedtls_rsa_export(rsa, NULL, NULL,
                                      &T, NULL, NULL)) != 0 ||
            (ret = mbedtls_asn1_write_mpi(p, buf, &T)) < 0) {
            goto end_of_export;
        }
        len += ret;

        /* Export P */
        if ((ret = mbedtls_rsa_export(rsa, NULL, &T,
                                      NULL, NULL, NULL)) != 0 ||
            (ret = mbedtls_asn1_write_mpi(p, buf, &T)) < 0) {
            goto end_of_export;
        }
        len += ret;

        /* Export D */
        if ((ret = mbedtls_rsa_export(rsa, NULL, NULL,
                                      NULL, &T, NULL)) != 0 ||
            (ret = mbedtls_asn1_write_mpi(p, buf, &T)) < 0) {
            goto end_of_export;
        }
        len += ret;

        /* Export E */
        if ((ret = mbedtls_rsa_export(rsa, NULL, NULL,
                                      NULL, NULL, &T)) != 0 ||
            (ret = mbedtls_asn1_write_mpi(p, buf, &T)) < 0) {
            goto end_of_export;
        }
        len += ret;

        /* Export N */
        if ((ret = mbedtls_rsa_export(rsa, &T, NULL,
                                      NULL, NULL, NULL)) != 0 ||
            (ret = mbedtls_asn1_write_mpi(p, buf, &T)) < 0) {
            goto end_of_export;
        }
        len += ret;

end_of_export:

        mbedtls_mpi_free(&T);
        if (ret < 0) {
            return ret;
        }

        MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_int(p, buf, 0));
        MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_len(p, buf, len));
        MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_tag(p,
                                                         buf, MBEDTLS_ASN1_CONSTRUCTED |
                                                         MBEDTLS_ASN1_SEQUENCE));
    }

    return (int) len;
}
#endif /* MBEDTLS_RSA_C */

int mbedtls_pk_write_key_der(const mbedtls_pk_context *key, unsigned char *buf, size_t size)
{
    unsigned char *c;
#if defined(MBEDTLS_RSA_C)
    int is_rsa_opaque = 0;
#endif /* MBEDTLS_RSA_C */
#if defined(MBEDTLS_PK_HAVE_ECC_KEYS)
    int is_ec_opaque = 0;
#endif /* MBEDTLS_PK_HAVE_ECC_KEYS */
#if defined(MBEDTLS_USE_PSA_CRYPTO)
    psa_key_type_t opaque_key_type;
#endif /* MBEDTLS_USE_PSA_CRYPTO */

    if (size == 0) {
        return MBEDTLS_ERR_ASN1_BUF_TOO_SMALL;
    }

    c = buf + size;

#if defined(MBEDTLS_USE_PSA_CRYPTO)
    if (mbedtls_pk_get_type(key) == MBEDTLS_PK_OPAQUE) {
        opaque_key_type = pk_get_opaque_key_type(key);
#if defined(MBEDTLS_RSA_C)
        is_rsa_opaque = PSA_KEY_TYPE_IS_RSA(opaque_key_type);
#endif /* MBEDTLS_RSA_C */
#if defined(MBEDTLS_PK_HAVE_ECC_KEYS)
        is_ec_opaque = PSA_KEY_TYPE_IS_ECC(opaque_key_type);
#endif /* MBEDTLS_PK_HAVE_ECC_KEYS */
    }
#endif /* MBEDTLS_USE_PSA_CRYPTO */

#if defined(MBEDTLS_RSA_C)
    if ((mbedtls_pk_get_type(key) == MBEDTLS_PK_RSA) || is_rsa_opaque) {
        return pk_write_rsa_der(&c, buf, key);
    } else
#endif /* MBEDTLS_RSA_C */
#if defined(MBEDTLS_PK_HAVE_ECC_KEYS)
    if ((mbedtls_pk_get_type(key) == MBEDTLS_PK_ECKEY) || is_ec_opaque) {
#if defined(MBEDTLS_PK_HAVE_RFC8410_CURVES)
        if (mbedtls_pk_is_rfc8410(key)) {
            return pk_write_ec_rfc8410_der(&c, buf, key);
        }
#endif /* MBEDTLS_PK_HAVE_RFC8410_CURVES */
        return pk_write_ec_der(&c, buf, key);
    } else
#endif /* MBEDTLS_PK_HAVE_ECC_KEYS */
    return MBEDTLS_ERR_PK_FEATURE_UNAVAILABLE;
}

#if defined(MBEDTLS_PEM_WRITE_C)

#define PEM_BEGIN_PUBLIC_KEY    "-----BEGIN PUBLIC KEY-----\n"
#define PEM_END_PUBLIC_KEY      "-----END PUBLIC KEY-----\n"

#define PEM_BEGIN_PRIVATE_KEY_RSA   "-----BEGIN RSA PRIVATE KEY-----\n"
#define PEM_END_PRIVATE_KEY_RSA     "-----END RSA PRIVATE KEY-----\n"
#define PEM_BEGIN_PRIVATE_KEY_EC    "-----BEGIN EC PRIVATE KEY-----\n"
#define PEM_END_PRIVATE_KEY_EC      "-----END EC PRIVATE KEY-----\n"
#define PEM_BEGIN_PRIVATE_KEY_PKCS8 "-----BEGIN PRIVATE KEY-----\n"
#define PEM_END_PRIVATE_KEY_PKCS8   "-----END PRIVATE KEY-----\n"

#define PUB_DER_MAX_BYTES                                                   \
    (MBEDTLS_PK_RSA_PUB_DER_MAX_BYTES > MBEDTLS_PK_ECP_PUB_DER_MAX_BYTES ? \
     MBEDTLS_PK_RSA_PUB_DER_MAX_BYTES : MBEDTLS_PK_ECP_PUB_DER_MAX_BYTES)
#define PRV_DER_MAX_BYTES                                                   \
    (MBEDTLS_PK_RSA_PRV_DER_MAX_BYTES > MBEDTLS_PK_ECP_PRV_DER_MAX_BYTES ? \
     MBEDTLS_PK_RSA_PRV_DER_MAX_BYTES : MBEDTLS_PK_ECP_PRV_DER_MAX_BYTES)

int mbedtls_pk_write_pubkey_pem(const mbedtls_pk_context *key, unsigned char *buf, size_t size)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    unsigned char output_buf[PUB_DER_MAX_BYTES];
    size_t olen = 0;

    if ((ret = mbedtls_pk_write_pubkey_der(key, output_buf,
                                           sizeof(output_buf))) < 0) {
        return ret;
    }

    if ((ret = mbedtls_pem_write_buffer(PEM_BEGIN_PUBLIC_KEY, PEM_END_PUBLIC_KEY,
                                        output_buf + sizeof(output_buf) - ret,
                                        ret, buf, size, &olen)) != 0) {
        return ret;
    }

    return 0;
}

int mbedtls_pk_write_key_pem(const mbedtls_pk_context *key, unsigned char *buf, size_t size)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    unsigned char output_buf[PRV_DER_MAX_BYTES];
    const char *begin, *end;
    size_t olen = 0;
#if defined(MBEDTLS_PK_HAVE_ECC_KEYS)
    int is_ec_opaque = 0;
#if defined(MBEDTLS_PK_HAVE_RFC8410_CURVES)
    int is_montgomery_opaque = 0;
#endif /* MBEDTLS_PK_HAVE_RFC8410_CURVES */
#endif /* MBEDTLS_PK_HAVE_ECC_KEYS */
#if defined(MBEDTLS_RSA_C)
    int is_rsa_opaque = 0;
#endif

    if ((ret = mbedtls_pk_write_key_der(key, output_buf, sizeof(output_buf))) < 0) {
        return ret;
    }

#if defined(MBEDTLS_USE_PSA_CRYPTO)
    if (mbedtls_pk_get_type(key) == MBEDTLS_PK_OPAQUE) {
        psa_key_type_t opaque_key_type = pk_get_opaque_key_type(key);

#if defined(MBEDTLS_RSA_C)
        is_rsa_opaque = PSA_KEY_TYPE_IS_RSA(opaque_key_type);
#endif
#if defined(MBEDTLS_PK_HAVE_ECC_KEYS)
        is_ec_opaque = PSA_KEY_TYPE_IS_ECC(opaque_key_type);
#if defined(MBEDTLS_PK_HAVE_RFC8410_CURVES)
        if (pk_get_opaque_ec_family(key) == PSA_ECC_FAMILY_MONTGOMERY) {
            is_montgomery_opaque = 1;
        }
#endif /* MBEDTLS_PK_HAVE_RFC8410_CURVES */
#endif /* MBEDTLS_PK_HAVE_ECC_KEYS */
    }
#endif /* MBEDTLS_USE_PSA_CRYPTO */

#if defined(MBEDTLS_RSA_C)
    if ((mbedtls_pk_get_type(key) == MBEDTLS_PK_RSA) || is_rsa_opaque) {
        begin = PEM_BEGIN_PRIVATE_KEY_RSA;
        end = PEM_END_PRIVATE_KEY_RSA;
    } else
#endif
#if defined(MBEDTLS_PK_HAVE_ECC_KEYS)
    if ((mbedtls_pk_get_type(key) == MBEDTLS_PK_ECKEY) || is_ec_opaque) {
#if defined(MBEDTLS_PK_HAVE_RFC8410_CURVES)
        if (is_montgomery_opaque ||
            ((mbedtls_pk_get_type(key) == MBEDTLS_PK_ECKEY) &&
             (mbedtls_pk_is_rfc8410(key)))) {
            begin = PEM_BEGIN_PRIVATE_KEY_PKCS8;
            end = PEM_END_PRIVATE_KEY_PKCS8;
        } else
#endif /* MBEDTLS_PK_HAVE_RFC8410_CURVES */
        {
            begin = PEM_BEGIN_PRIVATE_KEY_EC;
            end = PEM_END_PRIVATE_KEY_EC;
        }
    } else
#endif /* MBEDTLS_PK_HAVE_ECC_KEYS */
    return MBEDTLS_ERR_PK_FEATURE_UNAVAILABLE;

    if ((ret = mbedtls_pem_write_buffer(begin, end,
                                        output_buf + sizeof(output_buf) - ret,
                                        ret, buf, size, &olen)) != 0) {
        return ret;
    }

    return 0;
}
#endif /* MBEDTLS_PEM_WRITE_C */

#endif /* MBEDTLS_PK_WRITE_C */
