/**
 * \file oid.c
 *
 * \brief Object Identifier (OID) database
 *
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

#include "common.h"

#if defined(MBEDTLS_OID_C)

#include "mbedtls/oid.h"
#include "mbedtls/rsa.h"
#include "mbedtls/error.h"

#include <stdio.h>
#include <string.h>

#include "mbedtls/platform.h"

/*
 * Macro to automatically add the size of #define'd OIDs
 */
#define ADD_LEN(s)      s, MBEDTLS_OID_SIZE(s)

/*
 * Macro to generate an internal function for oid_XXX_from_asn1() (used by
 * the other functions)
 */
#define FN_OID_TYPED_FROM_ASN1(TYPE_T, NAME, LIST)                    \
    static const TYPE_T *oid_ ## NAME ## _from_asn1(                   \
        const mbedtls_asn1_buf *oid)     \
    {                                                                   \
        const TYPE_T *p = (LIST);                                       \
        const mbedtls_oid_descriptor_t *cur =                           \
            (const mbedtls_oid_descriptor_t *) p;                       \
        if (p == NULL || oid == NULL) return NULL;                  \
        while (cur->asn1 != NULL) {                                    \
            if (cur->asn1_len == oid->len &&                            \
                memcmp(cur->asn1, oid->p, oid->len) == 0) {          \
                return p;                                            \
            }                                                           \
            p++;                                                        \
            cur = (const mbedtls_oid_descriptor_t *) p;                 \
        }                                                               \
        return NULL;                                                 \
    }

/*
 * Macro to generate a function for retrieving a single attribute from the
 * descriptor of an mbedtls_oid_descriptor_t wrapper.
 */
#define FN_OID_GET_DESCRIPTOR_ATTR1(FN_NAME, TYPE_T, TYPE_NAME, ATTR1_TYPE, ATTR1) \
    int FN_NAME(const mbedtls_asn1_buf *oid, ATTR1_TYPE * ATTR1)                  \
    {                                                                       \
        const TYPE_T *data = oid_ ## TYPE_NAME ## _from_asn1(oid);        \
        if (data == NULL) return MBEDTLS_ERR_OID_NOT_FOUND;            \
        *ATTR1 = data->descriptor.ATTR1;                                    \
        return 0;                                                        \
    }

/*
 * Macro to generate a function for retrieving a single attribute from an
 * mbedtls_oid_descriptor_t wrapper.
 */
#define FN_OID_GET_ATTR1(FN_NAME, TYPE_T, TYPE_NAME, ATTR1_TYPE, ATTR1) \
    int FN_NAME(const mbedtls_asn1_buf *oid, ATTR1_TYPE * ATTR1)                  \
    {                                                                       \
        const TYPE_T *data = oid_ ## TYPE_NAME ## _from_asn1(oid);        \
        if (data == NULL) return MBEDTLS_ERR_OID_NOT_FOUND;            \
        *ATTR1 = data->ATTR1;                                               \
        return 0;                                                        \
    }

/*
 * Macro to generate a function for retrieving two attributes from an
 * mbedtls_oid_descriptor_t wrapper.
 */
#define FN_OID_GET_ATTR2(FN_NAME, TYPE_T, TYPE_NAME, ATTR1_TYPE, ATTR1,     \
                         ATTR2_TYPE, ATTR2)                                 \
    int FN_NAME(const mbedtls_asn1_buf *oid, ATTR1_TYPE * ATTR1,               \
                ATTR2_TYPE * ATTR2)              \
    {                                                                           \
        const TYPE_T *data = oid_ ## TYPE_NAME ## _from_asn1(oid);            \
        if (data == NULL) return MBEDTLS_ERR_OID_NOT_FOUND;                 \
        *(ATTR1) = data->ATTR1;                                                 \
        *(ATTR2) = data->ATTR2;                                                 \
        return 0;                                                            \
    }

/*
 * Macro to generate a function for retrieving the OID based on a single
 * attribute from a mbedtls_oid_descriptor_t wrapper.
 */
#define FN_OID_GET_OID_BY_ATTR1(FN_NAME, TYPE_T, LIST, ATTR1_TYPE, ATTR1)   \
    int FN_NAME(ATTR1_TYPE ATTR1, const char **oid, size_t *olen)             \
    {                                                                           \
        const TYPE_T *cur = (LIST);                                             \
        while (cur->descriptor.asn1 != NULL) {                                 \
            if (cur->ATTR1 == (ATTR1)) {                                       \
                *oid = cur->descriptor.asn1;                                    \
                *olen = cur->descriptor.asn1_len;                               \
                return 0;                                                    \
            }                                                                   \
            cur++;                                                              \
        }                                                                       \
        return MBEDTLS_ERR_OID_NOT_FOUND;                                    \
    }

/*
 * Macro to generate a function for retrieving the OID based on two
 * attributes from a mbedtls_oid_descriptor_t wrapper.
 */
#define FN_OID_GET_OID_BY_ATTR2(FN_NAME, TYPE_T, LIST, ATTR1_TYPE, ATTR1,   \
                                ATTR2_TYPE, ATTR2)                          \
    int FN_NAME(ATTR1_TYPE ATTR1, ATTR2_TYPE ATTR2, const char **oid,         \
                size_t *olen)                                                 \
    {                                                                           \
        const TYPE_T *cur = (LIST);                                             \
        while (cur->descriptor.asn1 != NULL) {                                 \
            if (cur->ATTR1 == (ATTR1) && cur->ATTR2 == (ATTR2)) {              \
                *oid = cur->descriptor.asn1;                                    \
                *olen = cur->descriptor.asn1_len;                               \
                return 0;                                                    \
            }                                                                   \
            cur++;                                                              \
        }                                                                       \
        return MBEDTLS_ERR_OID_NOT_FOUND;                                   \
    }

/*
 * For X520 attribute types
 */
typedef struct {
    mbedtls_oid_descriptor_t    descriptor;
    const char          *short_name;
} oid_x520_attr_t;

static const oid_x520_attr_t oid_x520_attr_type[] =
{
    {
        { ADD_LEN(MBEDTLS_OID_AT_CN),          "id-at-commonName",               "Common Name" },
        "CN",
    },
    {
        { ADD_LEN(MBEDTLS_OID_AT_COUNTRY),     "id-at-countryName",              "Country" },
        "C",
    },
    {
        { ADD_LEN(MBEDTLS_OID_AT_LOCALITY),    "id-at-locality",                 "Locality" },
        "L",
    },
    {
        { ADD_LEN(MBEDTLS_OID_AT_STATE),       "id-at-state",                    "State" },
        "ST",
    },
    {
        { ADD_LEN(MBEDTLS_OID_AT_ORGANIZATION), "id-at-organizationName",         "Organization" },
        "O",
    },
    {
        { ADD_LEN(MBEDTLS_OID_AT_ORG_UNIT),    "id-at-organizationalUnitName",   "Org Unit" },
        "OU",
    },
    {
        { ADD_LEN(MBEDTLS_OID_PKCS9_EMAIL),    "emailAddress",                   "E-mail address" },
        "emailAddress",
    },
    {
        { ADD_LEN(MBEDTLS_OID_AT_SERIAL_NUMBER), "id-at-serialNumber",            "Serial number" },
        "serialNumber",
    },
    {
        { ADD_LEN(MBEDTLS_OID_AT_POSTAL_ADDRESS), "id-at-postalAddress",
          "Postal address" },
        "postalAddress",
    },
    {
        { ADD_LEN(MBEDTLS_OID_AT_POSTAL_CODE), "id-at-postalCode",               "Postal code" },
        "postalCode",
    },
    {
        { ADD_LEN(MBEDTLS_OID_AT_SUR_NAME),    "id-at-surName",                  "Surname" },
        "SN",
    },
    {
        { ADD_LEN(MBEDTLS_OID_AT_GIVEN_NAME),  "id-at-givenName",                "Given name" },
        "GN",
    },
    {
        { ADD_LEN(MBEDTLS_OID_AT_INITIALS),    "id-at-initials",                 "Initials" },
        "initials",
    },
    {
        { ADD_LEN(MBEDTLS_OID_AT_GENERATION_QUALIFIER), "id-at-generationQualifier",
          "Generation qualifier" },
        "generationQualifier",
    },
    {
        { ADD_LEN(MBEDTLS_OID_AT_TITLE),       "id-at-title",                    "Title" },
        "title",
    },
    {
        { ADD_LEN(MBEDTLS_OID_AT_DN_QUALIFIER), "id-at-dnQualifier",
          "Distinguished Name qualifier" },
        "dnQualifier",
    },
    {
        { ADD_LEN(MBEDTLS_OID_AT_PSEUDONYM),   "id-at-pseudonym",                "Pseudonym" },
        "pseudonym",
    },
    {
        { ADD_LEN(MBEDTLS_OID_DOMAIN_COMPONENT), "id-domainComponent",
          "Domain component" },
        "DC",
    },
    {
        { ADD_LEN(MBEDTLS_OID_AT_UNIQUE_IDENTIFIER), "id-at-uniqueIdentifier",
          "Unique Identifier" },
        "uniqueIdentifier",
    },
    {
        { NULL, 0, NULL, NULL },
        NULL,
    }
};

FN_OID_TYPED_FROM_ASN1(oid_x520_attr_t, x520_attr, oid_x520_attr_type)
FN_OID_GET_ATTR1(mbedtls_oid_get_attr_short_name,
                 oid_x520_attr_t,
                 x520_attr,
                 const char *,
                 short_name)

/*
 * For X509 extensions
 */
typedef struct {
    mbedtls_oid_descriptor_t    descriptor;
    int                 ext_type;
} oid_x509_ext_t;

static const oid_x509_ext_t oid_x509_ext[] =
{
    {
        { ADD_LEN(MBEDTLS_OID_BASIC_CONSTRAINTS),    "id-ce-basicConstraints",
          "Basic Constraints" },
        MBEDTLS_OID_X509_EXT_BASIC_CONSTRAINTS,
    },
    {
        { ADD_LEN(MBEDTLS_OID_KEY_USAGE),            "id-ce-keyUsage",            "Key Usage" },
        MBEDTLS_OID_X509_EXT_KEY_USAGE,
    },
    {
        { ADD_LEN(MBEDTLS_OID_EXTENDED_KEY_USAGE),   "id-ce-extKeyUsage",
          "Extended Key Usage" },
        MBEDTLS_OID_X509_EXT_EXTENDED_KEY_USAGE,
    },
    {
        { ADD_LEN(MBEDTLS_OID_SUBJECT_ALT_NAME),     "id-ce-subjectAltName",
          "Subject Alt Name" },
        MBEDTLS_OID_X509_EXT_SUBJECT_ALT_NAME,
    },
    {
        { ADD_LEN(MBEDTLS_OID_NS_CERT_TYPE),         "id-netscape-certtype",
          "Netscape Certificate Type" },
        MBEDTLS_OID_X509_EXT_NS_CERT_TYPE,
    },
    {
        { ADD_LEN(MBEDTLS_OID_CERTIFICATE_POLICIES), "id-ce-certificatePolicies",
          "Certificate Policies" },
        MBEDTLS_OID_X509_EXT_CERTIFICATE_POLICIES,
    },
    {
        { NULL, 0, NULL, NULL },
        0,
    },
};

FN_OID_TYPED_FROM_ASN1(oid_x509_ext_t, x509_ext, oid_x509_ext)
FN_OID_GET_ATTR1(mbedtls_oid_get_x509_ext_type, oid_x509_ext_t, x509_ext, int, ext_type)

static const mbedtls_oid_descriptor_t oid_ext_key_usage[] =
{
    { ADD_LEN(MBEDTLS_OID_SERVER_AUTH),      "id-kp-serverAuth",
      "TLS Web Server Authentication" },
    { ADD_LEN(MBEDTLS_OID_CLIENT_AUTH),      "id-kp-clientAuth",
      "TLS Web Client Authentication" },
    { ADD_LEN(MBEDTLS_OID_CODE_SIGNING),     "id-kp-codeSigning",      "Code Signing" },
    { ADD_LEN(MBEDTLS_OID_EMAIL_PROTECTION), "id-kp-emailProtection",  "E-mail Protection" },
    { ADD_LEN(MBEDTLS_OID_TIME_STAMPING),    "id-kp-timeStamping",     "Time Stamping" },
    { ADD_LEN(MBEDTLS_OID_OCSP_SIGNING),     "id-kp-OCSPSigning",      "OCSP Signing" },
    { ADD_LEN(MBEDTLS_OID_WISUN_FAN),        "id-kp-wisun-fan-device",
      "Wi-SUN Alliance Field Area Network (FAN)" },
    { NULL, 0, NULL, NULL },
};

FN_OID_TYPED_FROM_ASN1(mbedtls_oid_descriptor_t, ext_key_usage, oid_ext_key_usage)
FN_OID_GET_ATTR1(mbedtls_oid_get_extended_key_usage,
                 mbedtls_oid_descriptor_t,
                 ext_key_usage,
                 const char *,
                 description)

static const mbedtls_oid_descriptor_t oid_certificate_policies[] =
{
    { ADD_LEN(MBEDTLS_OID_ANY_POLICY),      "anyPolicy",       "Any Policy" },
    { NULL, 0, NULL, NULL },
};

FN_OID_TYPED_FROM_ASN1(mbedtls_oid_descriptor_t, certificate_policies, oid_certificate_policies)
FN_OID_GET_ATTR1(mbedtls_oid_get_certificate_policies,
                 mbedtls_oid_descriptor_t,
                 certificate_policies,
                 const char *,
                 description)

#if defined(MBEDTLS_MD_C)
/*
 * For SignatureAlgorithmIdentifier
 */
typedef struct {
    mbedtls_oid_descriptor_t    descriptor;
    mbedtls_md_type_t           md_alg;
    mbedtls_pk_type_t           pk_alg;
} oid_sig_alg_t;

static const oid_sig_alg_t oid_sig_alg[] =
{
#if defined(MBEDTLS_RSA_C)
#if defined(MBEDTLS_MD2_C)
    {
        { ADD_LEN(MBEDTLS_OID_PKCS1_MD2),        "md2WithRSAEncryption",     "RSA with MD2" },
        MBEDTLS_MD_MD2,      MBEDTLS_PK_RSA,
    },
#endif /* MBEDTLS_MD2_C */
#if defined(MBEDTLS_MD4_C)
    {
        { ADD_LEN(MBEDTLS_OID_PKCS1_MD4),        "md4WithRSAEncryption",     "RSA with MD4" },
        MBEDTLS_MD_MD4,      MBEDTLS_PK_RSA,
    },
#endif /* MBEDTLS_MD4_C */
#if defined(MBEDTLS_MD5_C)
    {
        { ADD_LEN(MBEDTLS_OID_PKCS1_MD5),        "md5WithRSAEncryption",     "RSA with MD5" },
        MBEDTLS_MD_MD5,      MBEDTLS_PK_RSA,
    },
#endif /* MBEDTLS_MD5_C */
#if defined(MBEDTLS_SHA1_C)
    {
        { ADD_LEN(MBEDTLS_OID_PKCS1_SHA1),       "sha-1WithRSAEncryption",   "RSA with SHA1" },
        MBEDTLS_MD_SHA1,     MBEDTLS_PK_RSA,
    },
#endif /* MBEDTLS_SHA1_C */
#if defined(MBEDTLS_SHA256_C)
    {
        { ADD_LEN(MBEDTLS_OID_PKCS1_SHA224),     "sha224WithRSAEncryption",  "RSA with SHA-224" },
        MBEDTLS_MD_SHA224,   MBEDTLS_PK_RSA,
    },
    {
        { ADD_LEN(MBEDTLS_OID_PKCS1_SHA256),     "sha256WithRSAEncryption",  "RSA with SHA-256" },
        MBEDTLS_MD_SHA256,   MBEDTLS_PK_RSA,
    },
#endif /* MBEDTLS_SHA256_C */
#if defined(MBEDTLS_SHA512_C)
    {
        { ADD_LEN(MBEDTLS_OID_PKCS1_SHA384),     "sha384WithRSAEncryption",  "RSA with SHA-384" },
        MBEDTLS_MD_SHA384,   MBEDTLS_PK_RSA,
    },
    {
        { ADD_LEN(MBEDTLS_OID_PKCS1_SHA512),     "sha512WithRSAEncryption",  "RSA with SHA-512" },
        MBEDTLS_MD_SHA512,   MBEDTLS_PK_RSA,
    },
#endif /* MBEDTLS_SHA512_C */
#if defined(MBEDTLS_SHA1_C)
    {
        { ADD_LEN(MBEDTLS_OID_RSA_SHA_OBS),      "sha-1WithRSAEncryption",   "RSA with SHA1" },
        MBEDTLS_MD_SHA1,     MBEDTLS_PK_RSA,
    },
#endif /* MBEDTLS_SHA1_C */
#endif /* MBEDTLS_RSA_C */
#if defined(MBEDTLS_ECDSA_C)
#if defined(MBEDTLS_SHA1_C)
    {
        { ADD_LEN(MBEDTLS_OID_ECDSA_SHA1),       "ecdsa-with-SHA1",      "ECDSA with SHA1" },
        MBEDTLS_MD_SHA1,     MBEDTLS_PK_ECDSA,
    },
#endif /* MBEDTLS_SHA1_C */
#if defined(MBEDTLS_SHA256_C)
    {
        { ADD_LEN(MBEDTLS_OID_ECDSA_SHA224),     "ecdsa-with-SHA224",    "ECDSA with SHA224" },
        MBEDTLS_MD_SHA224,   MBEDTLS_PK_ECDSA,
    },
    {
        { ADD_LEN(MBEDTLS_OID_ECDSA_SHA256),     "ecdsa-with-SHA256",    "ECDSA with SHA256" },
        MBEDTLS_MD_SHA256,   MBEDTLS_PK_ECDSA,
    },
#endif /* MBEDTLS_SHA256_C */
#if defined(MBEDTLS_SHA512_C)
    {
        { ADD_LEN(MBEDTLS_OID_ECDSA_SHA384),     "ecdsa-with-SHA384",    "ECDSA with SHA384" },
        MBEDTLS_MD_SHA384,   MBEDTLS_PK_ECDSA,
    },
    {
        { ADD_LEN(MBEDTLS_OID_ECDSA_SHA512),     "ecdsa-with-SHA512",    "ECDSA with SHA512" },
        MBEDTLS_MD_SHA512,   MBEDTLS_PK_ECDSA,
    },
#endif /* MBEDTLS_SHA512_C */
#endif /* MBEDTLS_ECDSA_C */
#if defined(MBEDTLS_RSA_C)
    {
        { ADD_LEN(MBEDTLS_OID_RSASSA_PSS),        "RSASSA-PSS",           "RSASSA-PSS" },
        MBEDTLS_MD_NONE,     MBEDTLS_PK_RSASSA_PSS,
    },
#endif /* MBEDTLS_RSA_C */
    {
        { NULL, 0, NULL, NULL },
        MBEDTLS_MD_NONE, MBEDTLS_PK_NONE,
    },
};

FN_OID_TYPED_FROM_ASN1(oid_sig_alg_t, sig_alg, oid_sig_alg)
FN_OID_GET_DESCRIPTOR_ATTR1(mbedtls_oid_get_sig_alg_desc,
                            oid_sig_alg_t,
                            sig_alg,
                            const char *,
                            description)
FN_OID_GET_ATTR2(mbedtls_oid_get_sig_alg,
                 oid_sig_alg_t,
                 sig_alg,
                 mbedtls_md_type_t,
                 md_alg,
                 mbedtls_pk_type_t,
                 pk_alg)
FN_OID_GET_OID_BY_ATTR2(mbedtls_oid_get_oid_by_sig_alg,
                        oid_sig_alg_t,
                        oid_sig_alg,
                        mbedtls_pk_type_t,
                        pk_alg,
                        mbedtls_md_type_t,
                        md_alg)
#endif /* MBEDTLS_MD_C */

/*
 * For PublicKeyInfo (PKCS1, RFC 5480)
 */
typedef struct {
    mbedtls_oid_descriptor_t    descriptor;
    mbedtls_pk_type_t           pk_alg;
} oid_pk_alg_t;

static const oid_pk_alg_t oid_pk_alg[] =
{
    {
        { ADD_LEN(MBEDTLS_OID_PKCS1_RSA),      "rsaEncryption",   "RSA" },
        MBEDTLS_PK_RSA,
    },
    {
        { ADD_LEN(MBEDTLS_OID_EC_ALG_UNRESTRICTED),  "id-ecPublicKey",   "Generic EC key" },
        MBEDTLS_PK_ECKEY,
    },
    {
        { ADD_LEN(MBEDTLS_OID_EC_ALG_ECDH),          "id-ecDH",          "EC key for ECDH" },
        MBEDTLS_PK_ECKEY_DH,
    },
    {
        { NULL, 0, NULL, NULL },
        MBEDTLS_PK_NONE,
    },
};

FN_OID_TYPED_FROM_ASN1(oid_pk_alg_t, pk_alg, oid_pk_alg)
FN_OID_GET_ATTR1(mbedtls_oid_get_pk_alg, oid_pk_alg_t, pk_alg, mbedtls_pk_type_t, pk_alg)
FN_OID_GET_OID_BY_ATTR1(mbedtls_oid_get_oid_by_pk_alg,
                        oid_pk_alg_t,
                        oid_pk_alg,
                        mbedtls_pk_type_t,
                        pk_alg)

#if defined(MBEDTLS_ECP_C)
/*
 * For namedCurve (RFC 5480)
 */
typedef struct {
    mbedtls_oid_descriptor_t    descriptor;
    mbedtls_ecp_group_id        grp_id;
} oid_ecp_grp_t;

static const oid_ecp_grp_t oid_ecp_grp[] =
{
#if defined(MBEDTLS_ECP_DP_SECP192R1_ENABLED)
    {
        { ADD_LEN(MBEDTLS_OID_EC_GRP_SECP192R1), "secp192r1",    "secp192r1" },
        MBEDTLS_ECP_DP_SECP192R1,
    },
#endif /* MBEDTLS_ECP_DP_SECP192R1_ENABLED */
#if defined(MBEDTLS_ECP_DP_SECP224R1_ENABLED)
    {
        { ADD_LEN(MBEDTLS_OID_EC_GRP_SECP224R1), "secp224r1",    "secp224r1" },
        MBEDTLS_ECP_DP_SECP224R1,
    },
#endif /* MBEDTLS_ECP_DP_SECP224R1_ENABLED */
#if defined(MBEDTLS_ECP_DP_SECP256R1_ENABLED)
    {
        { ADD_LEN(MBEDTLS_OID_EC_GRP_SECP256R1), "secp256r1",    "secp256r1" },
        MBEDTLS_ECP_DP_SECP256R1,
    },
#endif /* MBEDTLS_ECP_DP_SECP256R1_ENABLED */
#if defined(MBEDTLS_ECP_DP_SECP384R1_ENABLED)
    {
        { ADD_LEN(MBEDTLS_OID_EC_GRP_SECP384R1), "secp384r1",    "secp384r1" },
        MBEDTLS_ECP_DP_SECP384R1,
    },
#endif /* MBEDTLS_ECP_DP_SECP384R1_ENABLED */
#if defined(MBEDTLS_ECP_DP_SECP521R1_ENABLED)
    {
        { ADD_LEN(MBEDTLS_OID_EC_GRP_SECP521R1), "secp521r1",    "secp521r1" },
        MBEDTLS_ECP_DP_SECP521R1,
    },
#endif /* MBEDTLS_ECP_DP_SECP521R1_ENABLED */
#if defined(MBEDTLS_ECP_DP_SECP192K1_ENABLED)
    {
        { ADD_LEN(MBEDTLS_OID_EC_GRP_SECP192K1), "secp192k1",    "secp192k1" },
        MBEDTLS_ECP_DP_SECP192K1,
    },
#endif /* MBEDTLS_ECP_DP_SECP192K1_ENABLED */
#if defined(MBEDTLS_ECP_DP_SECP224K1_ENABLED)
    {
        { ADD_LEN(MBEDTLS_OID_EC_GRP_SECP224K1), "secp224k1",    "secp224k1" },
        MBEDTLS_ECP_DP_SECP224K1,
    },
#endif /* MBEDTLS_ECP_DP_SECP224K1_ENABLED */
#if defined(MBEDTLS_ECP_DP_SECP256K1_ENABLED)
    {
        { ADD_LEN(MBEDTLS_OID_EC_GRP_SECP256K1), "secp256k1",    "secp256k1" },
        MBEDTLS_ECP_DP_SECP256K1,
    },
#endif /* MBEDTLS_ECP_DP_SECP256K1_ENABLED */
#if defined(MBEDTLS_ECP_DP_BP256R1_ENABLED)
    {
        { ADD_LEN(MBEDTLS_OID_EC_GRP_BP256R1),   "brainpoolP256r1", "brainpool256r1" },
        MBEDTLS_ECP_DP_BP256R1,
    },
#endif /* MBEDTLS_ECP_DP_BP256R1_ENABLED */
#if defined(MBEDTLS_ECP_DP_BP384R1_ENABLED)
    {
        { ADD_LEN(MBEDTLS_OID_EC_GRP_BP384R1),   "brainpoolP384r1", "brainpool384r1" },
        MBEDTLS_ECP_DP_BP384R1,
    },
#endif /* MBEDTLS_ECP_DP_BP384R1_ENABLED */
#if defined(MBEDTLS_ECP_DP_BP512R1_ENABLED)
    {
        { ADD_LEN(MBEDTLS_OID_EC_GRP_BP512R1),   "brainpoolP512r1", "brainpool512r1" },
        MBEDTLS_ECP_DP_BP512R1,
    },
#endif /* MBEDTLS_ECP_DP_BP512R1_ENABLED */
    {
        { NULL, 0, NULL, NULL },
        MBEDTLS_ECP_DP_NONE,
    },
};

FN_OID_TYPED_FROM_ASN1(oid_ecp_grp_t, grp_id, oid_ecp_grp)
FN_OID_GET_ATTR1(mbedtls_oid_get_ec_grp, oid_ecp_grp_t, grp_id, mbedtls_ecp_group_id, grp_id)
FN_OID_GET_OID_BY_ATTR1(mbedtls_oid_get_oid_by_ec_grp,
                        oid_ecp_grp_t,
                        oid_ecp_grp,
                        mbedtls_ecp_group_id,
                        grp_id)
#endif /* MBEDTLS_ECP_C */

#if defined(MBEDTLS_CIPHER_C)
/*
 * For PKCS#5 PBES2 encryption algorithm
 */
typedef struct {
    mbedtls_oid_descriptor_t    descriptor;
    mbedtls_cipher_type_t       cipher_alg;
} oid_cipher_alg_t;

static const oid_cipher_alg_t oid_cipher_alg[] =
{
    {
        { ADD_LEN(MBEDTLS_OID_DES_CBC),              "desCBC",       "DES-CBC" },
        MBEDTLS_CIPHER_DES_CBC,
    },
    {
        { ADD_LEN(MBEDTLS_OID_DES_EDE3_CBC),         "des-ede3-cbc", "DES-EDE3-CBC" },
        MBEDTLS_CIPHER_DES_EDE3_CBC,
    },
    {
        { NULL, 0, NULL, NULL },
        MBEDTLS_CIPHER_NONE,
    },
};

FN_OID_TYPED_FROM_ASN1(oid_cipher_alg_t, cipher_alg, oid_cipher_alg)
FN_OID_GET_ATTR1(mbedtls_oid_get_cipher_alg,
                 oid_cipher_alg_t,
                 cipher_alg,
                 mbedtls_cipher_type_t,
                 cipher_alg)
#endif /* MBEDTLS_CIPHER_C */

#if defined(MBEDTLS_MD_C)
/*
 * For digestAlgorithm
 */
typedef struct {
    mbedtls_oid_descriptor_t    descriptor;
    mbedtls_md_type_t           md_alg;
} oid_md_alg_t;

static const oid_md_alg_t oid_md_alg[] =
{
#if defined(MBEDTLS_MD2_C)
    {
        { ADD_LEN(MBEDTLS_OID_DIGEST_ALG_MD2),       "id-md2",       "MD2" },
        MBEDTLS_MD_MD2,
    },
#endif /* MBEDTLS_MD2_C */
#if defined(MBEDTLS_MD4_C)
    {
        { ADD_LEN(MBEDTLS_OID_DIGEST_ALG_MD4),       "id-md4",       "MD4" },
        MBEDTLS_MD_MD4,
    },
#endif /* MBEDTLS_MD4_C */
#if defined(MBEDTLS_MD5_C)
    {
        { ADD_LEN(MBEDTLS_OID_DIGEST_ALG_MD5),       "id-md5",       "MD5" },
        MBEDTLS_MD_MD5,
    },
#endif /* MBEDTLS_MD5_C */
#if defined(MBEDTLS_SHA1_C)
    {
        { ADD_LEN(MBEDTLS_OID_DIGEST_ALG_SHA1),      "id-sha1",      "SHA-1" },
        MBEDTLS_MD_SHA1,
    },
#endif /* MBEDTLS_SHA1_C */
#if defined(MBEDTLS_SHA256_C)
    {
        { ADD_LEN(MBEDTLS_OID_DIGEST_ALG_SHA224),    "id-sha224",    "SHA-224" },
        MBEDTLS_MD_SHA224,
    },
    {
        { ADD_LEN(MBEDTLS_OID_DIGEST_ALG_SHA256),    "id-sha256",    "SHA-256" },
        MBEDTLS_MD_SHA256,
    },
#endif /* MBEDTLS_SHA256_C */
#if defined(MBEDTLS_SHA512_C)
    {
        { ADD_LEN(MBEDTLS_OID_DIGEST_ALG_SHA384),    "id-sha384",    "SHA-384" },
        MBEDTLS_MD_SHA384,
    },
    {
        { ADD_LEN(MBEDTLS_OID_DIGEST_ALG_SHA512),    "id-sha512",    "SHA-512" },
        MBEDTLS_MD_SHA512,
    },
#endif /* MBEDTLS_SHA512_C */
#if defined(MBEDTLS_RIPEMD160_C)
    {
        { ADD_LEN(MBEDTLS_OID_DIGEST_ALG_RIPEMD160),       "id-ripemd160",       "RIPEMD-160" },
        MBEDTLS_MD_RIPEMD160,
    },
#endif /* MBEDTLS_RIPEMD160_C */
    {
        { NULL, 0, NULL, NULL },
        MBEDTLS_MD_NONE,
    },
};

FN_OID_TYPED_FROM_ASN1(oid_md_alg_t, md_alg, oid_md_alg)
FN_OID_GET_ATTR1(mbedtls_oid_get_md_alg, oid_md_alg_t, md_alg, mbedtls_md_type_t, md_alg)
FN_OID_GET_OID_BY_ATTR1(mbedtls_oid_get_oid_by_md,
                        oid_md_alg_t,
                        oid_md_alg,
                        mbedtls_md_type_t,
                        md_alg)

/*
 * For HMAC digestAlgorithm
 */
typedef struct {
    mbedtls_oid_descriptor_t    descriptor;
    mbedtls_md_type_t           md_hmac;
} oid_md_hmac_t;

static const oid_md_hmac_t oid_md_hmac[] =
{
#if defined(MBEDTLS_SHA1_C)
    {
        { ADD_LEN(MBEDTLS_OID_HMAC_SHA1),      "hmacSHA1",      "HMAC-SHA-1" },
        MBEDTLS_MD_SHA1,
    },
#endif /* MBEDTLS_SHA1_C */
#if defined(MBEDTLS_SHA256_C)
    {
        { ADD_LEN(MBEDTLS_OID_HMAC_SHA224),    "hmacSHA224",    "HMAC-SHA-224" },
        MBEDTLS_MD_SHA224,
    },
    {
        { ADD_LEN(MBEDTLS_OID_HMAC_SHA256),    "hmacSHA256",    "HMAC-SHA-256" },
        MBEDTLS_MD_SHA256,
    },
#endif /* MBEDTLS_SHA256_C */
#if defined(MBEDTLS_SHA512_C)
    {
        { ADD_LEN(MBEDTLS_OID_HMAC_SHA384),    "hmacSHA384",    "HMAC-SHA-384" },
        MBEDTLS_MD_SHA384,
    },
    {
        { ADD_LEN(MBEDTLS_OID_HMAC_SHA512),    "hmacSHA512",    "HMAC-SHA-512" },
        MBEDTLS_MD_SHA512,
    },
#endif /* MBEDTLS_SHA512_C */
    {
        { NULL, 0, NULL, NULL },
        MBEDTLS_MD_NONE,
    },
};

FN_OID_TYPED_FROM_ASN1(oid_md_hmac_t, md_hmac, oid_md_hmac)
FN_OID_GET_ATTR1(mbedtls_oid_get_md_hmac, oid_md_hmac_t, md_hmac, mbedtls_md_type_t, md_hmac)
#endif /* MBEDTLS_MD_C */

#if defined(MBEDTLS_PKCS12_C)
/*
 * For PKCS#12 PBEs
 */
typedef struct {
    mbedtls_oid_descriptor_t    descriptor;
    mbedtls_md_type_t           md_alg;
    mbedtls_cipher_type_t       cipher_alg;
} oid_pkcs12_pbe_alg_t;

static const oid_pkcs12_pbe_alg_t oid_pkcs12_pbe_alg[] =
{
    {
        { ADD_LEN(MBEDTLS_OID_PKCS12_PBE_SHA1_DES3_EDE_CBC), "pbeWithSHAAnd3-KeyTripleDES-CBC",
          "PBE with SHA1 and 3-Key 3DES" },
        MBEDTLS_MD_SHA1,      MBEDTLS_CIPHER_DES_EDE3_CBC,
    },
    {
        { ADD_LEN(MBEDTLS_OID_PKCS12_PBE_SHA1_DES2_EDE_CBC), "pbeWithSHAAnd2-KeyTripleDES-CBC",
          "PBE with SHA1 and 2-Key 3DES" },
        MBEDTLS_MD_SHA1,      MBEDTLS_CIPHER_DES_EDE_CBC,
    },
    {
        { NULL, 0, NULL, NULL },
        MBEDTLS_MD_NONE, MBEDTLS_CIPHER_NONE,
    },
};

FN_OID_TYPED_FROM_ASN1(oid_pkcs12_pbe_alg_t, pkcs12_pbe_alg, oid_pkcs12_pbe_alg)
FN_OID_GET_ATTR2(mbedtls_oid_get_pkcs12_pbe_alg,
                 oid_pkcs12_pbe_alg_t,
                 pkcs12_pbe_alg,
                 mbedtls_md_type_t,
                 md_alg,
                 mbedtls_cipher_type_t,
                 cipher_alg)
#endif /* MBEDTLS_PKCS12_C */

/* Return the x.y.z.... style numeric string for the given OID */
int mbedtls_oid_get_numeric_string(char *buf, size_t size,
                                   const mbedtls_asn1_buf *oid)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    char *p = buf;
    size_t n = size;
    unsigned int value = 0;

    if (size > INT_MAX) {
        /* Avoid overflow computing return value */
        return MBEDTLS_ERR_ASN1_INVALID_LENGTH;
    }

    if (oid->len <= 0) {
        /* OID must not be empty */
        return MBEDTLS_ERR_ASN1_OUT_OF_DATA;
    }

    for (size_t i = 0; i < oid->len; i++) {
        /* Prevent overflow in value. */
        if (value > (UINT_MAX >> 7)) {
            return MBEDTLS_ERR_ASN1_INVALID_DATA;
        }
        if ((value == 0) && ((oid->p[i]) == 0x80)) {
            /* Overlong encoding is not allowed */
            return MBEDTLS_ERR_ASN1_INVALID_DATA;
        }

        value <<= 7;
        value |= oid->p[i] & 0x7F;

        if (!(oid->p[i] & 0x80)) {
            /* Last byte */
            if (n == size) {
                int component1;
                unsigned int component2;
                /* First subidentifier contains first two OID components */
                if (value >= 80) {
                    component1 = '2';
                    component2 = value - 80;
                } else if (value >= 40) {
                    component1 = '1';
                    component2 = value - 40;
                } else {
                    component1 = '0';
                    component2 = value;
                }
                ret = mbedtls_snprintf(p, n, "%c.%u", component1, component2);
            } else {
                ret = mbedtls_snprintf(p, n, ".%u", value);
            }
            if (ret < 2 || (size_t) ret >= n) {
                return MBEDTLS_ERR_OID_BUF_TOO_SMALL;
            }
            n -= (size_t) ret;
            p += ret;
            value = 0;
        }
    }

    if (value != 0) {
        /* Unterminated subidentifier */
        return MBEDTLS_ERR_ASN1_OUT_OF_DATA;
    }

    return (int) (size - n);
}

#endif /* MBEDTLS_OID_C */
