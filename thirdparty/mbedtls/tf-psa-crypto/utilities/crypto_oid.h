/**
 * \file crypto_oid.h
 *
 * \brief Object Identifier (OID) database
 */
/*
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */
#ifndef TF_PSA_CRYPTO_CRYPTO_OID_H
#define TF_PSA_CRYPTO_CRYPTO_OID_H
#include "mbedtls/private_access.h"

#include "tf-psa-crypto/build_info.h"

#include "mbedtls/asn1.h"
#include "mbedtls/pk.h"
#if defined(MBEDTLS_PK_HAVE_PRIVATE_HEADER)
#include <mbedtls/private/pk_private.h>
#endif /* MBEDTLS_PK_HAVE_PRIVATE_HEADER */

#include <stddef.h>

#if defined(MBEDTLS_CIPHER_C)
#include "mbedtls/private/cipher.h"
#endif

#include "mbedtls/md.h"

#define MBEDTLS_ERR_OID_NOT_FOUND                         PSA_ERROR_NOT_SUPPORTED

/*
 * Top level OID tuples
 */
#define MBEDTLS_OID_ISO_MEMBER_BODIES           "\x2a"          /* {iso(1) member-body(2)} */
#define MBEDTLS_OID_ISO_IDENTIFIED_ORG          "\x2b"          /* {iso(1) identified-organization(3)} */
#define MBEDTLS_OID_ISO_ITU_COUNTRY             "\x60"          /* {joint-iso-itu-t(2) country(16)} */

/*
 * ISO Member bodies OID parts
 */
#define MBEDTLS_OID_COUNTRY_US                  "\x86\x48"      /* {us(840)} */
#define MBEDTLS_OID_ORG_RSA_DATA_SECURITY       "\x86\xf7\x0d"  /* {rsadsi(113549)} */
#define MBEDTLS_OID_RSA_COMPANY                 MBEDTLS_OID_ISO_MEMBER_BODIES MBEDTLS_OID_COUNTRY_US \
    MBEDTLS_OID_ORG_RSA_DATA_SECURITY                           /* {iso(1) member-body(2) us(840) rsadsi(113549)} */
#define MBEDTLS_OID_ORG_ANSI_X9_62              "\xce\x3d" /* ansi-X9-62(10045) */
#define MBEDTLS_OID_ANSI_X9_62                  MBEDTLS_OID_ISO_MEMBER_BODIES MBEDTLS_OID_COUNTRY_US \
    MBEDTLS_OID_ORG_ANSI_X9_62

/*
 * ISO Identified organization OID parts
 */
#define MBEDTLS_OID_ORG_DOD                     "\x06"          /* {dod(6)} */
#define MBEDTLS_OID_ORG_OIW                     "\x0e"
#define MBEDTLS_OID_OIW_SECSIG                  MBEDTLS_OID_ORG_OIW "\x03"
#define MBEDTLS_OID_OIW_SECSIG_ALG              MBEDTLS_OID_OIW_SECSIG "\x02"
#define MBEDTLS_OID_OIW_SECSIG_SHA1             MBEDTLS_OID_OIW_SECSIG_ALG "\x1a"
#define MBEDTLS_OID_ORG_THAWTE                  "\x65"          /* thawte(101) */
#define MBEDTLS_OID_THAWTE                      MBEDTLS_OID_ISO_IDENTIFIED_ORG \
    MBEDTLS_OID_ORG_THAWTE
#define MBEDTLS_OID_ORG_CERTICOM                "\x81\x04"  /* certicom(132) */
#define MBEDTLS_OID_CERTICOM                    MBEDTLS_OID_ISO_IDENTIFIED_ORG \
    MBEDTLS_OID_ORG_CERTICOM
#define MBEDTLS_OID_ORG_TELETRUST               "\x24" /* teletrust(36) */
#define MBEDTLS_OID_TELETRUST                   MBEDTLS_OID_ISO_IDENTIFIED_ORG \
    MBEDTLS_OID_ORG_TELETRUST

/*
 * ISO ITU OID parts
 */
#define MBEDTLS_OID_ORGANIZATION                "\x01"          /* {organization(1)} */
#define MBEDTLS_OID_ISO_ITU_US_ORG              MBEDTLS_OID_ISO_ITU_COUNTRY MBEDTLS_OID_COUNTRY_US \
    MBEDTLS_OID_ORGANIZATION                                    /* {joint-iso-itu-t(2) country(16) us(840) organization(1)} */

#define MBEDTLS_OID_ORG_GOV                     "\x65"          /* {gov(101)} */
#define MBEDTLS_OID_GOV                         MBEDTLS_OID_ISO_ITU_US_ORG MBEDTLS_OID_ORG_GOV /* {joint-iso-itu-t(2) country(16) us(840) organization(1) gov(101)} */

#define MBEDTLS_OID_ORG_NETSCAPE                "\x86\xF8\x42"  /* {netscape(113730)} */
#define MBEDTLS_OID_NETSCAPE                    MBEDTLS_OID_ISO_ITU_US_ORG MBEDTLS_OID_ORG_NETSCAPE /* Netscape OID {joint-iso-itu-t(2) country(16) us(840) organization(1) netscape(113730)} */

#define MBEDTLS_OID_NIST_ALG                    MBEDTLS_OID_GOV "\x03\x04" /** { joint-iso-itu-t(2) country(16) us(840) organization(1) gov(101) csor(3) nistAlgorithm(4) */

/**
 * Private Internet Extensions
 * { iso(1) identified-organization(3) dod(6) internet(1)
 *                      security(5) mechanisms(5) pkix(7) }
 */
#define MBEDTLS_OID_INTERNET                    MBEDTLS_OID_ISO_IDENTIFIED_ORG MBEDTLS_OID_ORG_DOD \
    "\x01"

/*
 * PKCS definition OIDs
 */

#define MBEDTLS_OID_PKCS                MBEDTLS_OID_RSA_COMPANY "\x01" /**< pkcs OBJECT IDENTIFIER ::= { iso(1) member-body(2) us(840) rsadsi(113549) 1 } */
#define MBEDTLS_OID_PKCS1               MBEDTLS_OID_PKCS "\x01" /**< pkcs-1 OBJECT IDENTIFIER ::= { iso(1) member-body(2) us(840) rsadsi(113549) pkcs(1) 1 } */
#define MBEDTLS_OID_PKCS5               MBEDTLS_OID_PKCS "\x05" /**< pkcs-5 OBJECT IDENTIFIER ::= { iso(1) member-body(2) us(840) rsadsi(113549) pkcs(1) 5 } */

/*
 * PKCS#1 OIDs
 */
#define MBEDTLS_OID_PKCS1_RSA           MBEDTLS_OID_PKCS1 "\x01" /**< rsaEncryption OBJECT IDENTIFIER ::= { pkcs-1 1 } */

/*
 * Digest algorithms
 */
#define MBEDTLS_OID_DIGEST_ALG_MD5              MBEDTLS_OID_RSA_COMPANY "\x02\x05" /**< id-mbedtls_md5 OBJECT IDENTIFIER ::= { iso(1) member-body(2) us(840) rsadsi(113549) digestAlgorithm(2) 5 } */
#define MBEDTLS_OID_DIGEST_ALG_SHA1             MBEDTLS_OID_ISO_IDENTIFIED_ORG \
    MBEDTLS_OID_OIW_SECSIG_SHA1                                                 /**< id-mbedtls_sha1 OBJECT IDENTIFIER ::= { iso(1) identified-organization(3) oiw(14) secsig(3) algorithms(2) 26 } */
#define MBEDTLS_OID_DIGEST_ALG_SHA224           MBEDTLS_OID_NIST_ALG "\x02\x04" /**< id-sha224 OBJECT IDENTIFIER ::= { joint-iso-itu-t(2) country(16) us(840) organization(1) gov(101) csor(3) nistalgorithm(4) hashalgs(2) 4 } */
#define MBEDTLS_OID_DIGEST_ALG_SHA256           MBEDTLS_OID_NIST_ALG "\x02\x01" /**< id-mbedtls_sha256 OBJECT IDENTIFIER ::= { joint-iso-itu-t(2) country(16) us(840) organization(1) gov(101) csor(3) nistalgorithm(4) hashalgs(2) 1 } */

#define MBEDTLS_OID_DIGEST_ALG_SHA384           MBEDTLS_OID_NIST_ALG "\x02\x02" /**< id-sha384 OBJECT IDENTIFIER ::= { joint-iso-itu-t(2) country(16) us(840) organization(1) gov(101) csor(3) nistalgorithm(4) hashalgs(2) 2 } */

#define MBEDTLS_OID_DIGEST_ALG_SHA512           MBEDTLS_OID_NIST_ALG "\x02\x03" /**< id-mbedtls_sha512 OBJECT IDENTIFIER ::= { joint-iso-itu-t(2) country(16) us(840) organization(1) gov(101) csor(3) nistalgorithm(4) hashalgs(2) 3 } */

#define MBEDTLS_OID_DIGEST_ALG_RIPEMD160        MBEDTLS_OID_TELETRUST "\x03\x02\x01" /**< id-ripemd160 OBJECT IDENTIFIER :: { iso(1) identified-organization(3) teletrust(36) algorithm(3) hashAlgorithm(2) ripemd160(1) } */

#define MBEDTLS_OID_DIGEST_ALG_SHA3_224         MBEDTLS_OID_NIST_ALG "\x02\x07" /**< id-sha3-224 OBJECT IDENTIFIER ::= { joint-iso-itu-t(2) country(16) us(840) organization(1) gov(101) csor(3) nistAlgorithms(4) hashalgs(2) sha3-224(7) } */

#define MBEDTLS_OID_DIGEST_ALG_SHA3_256         MBEDTLS_OID_NIST_ALG "\x02\x08" /**< id-sha3-256 OBJECT IDENTIFIER ::= { joint-iso-itu-t(2) country(16) us(840) organization(1) gov(101) csor(3) nistAlgorithms(4) hashalgs(2) sha3-256(8) } */

#define MBEDTLS_OID_DIGEST_ALG_SHA3_384         MBEDTLS_OID_NIST_ALG "\x02\x09" /**< id-sha3-384 OBJECT IDENTIFIER ::= { joint-iso-itu-t(2) country(16) us(840) organization(1) gov(101) csor(3) nistAlgorithms(4) hashalgs(2) sha3-384(9) } */

#define MBEDTLS_OID_DIGEST_ALG_SHA3_512         MBEDTLS_OID_NIST_ALG "\x02\x0a" /**< id-sha3-512 OBJECT IDENTIFIER ::= { joint-iso-itu-t(2) country(16) us(840) organization(1) gov(101) csor(3) nistAlgorithms(4) hashalgs(2) sha3-512(10) } */

#define MBEDTLS_OID_HMAC_SHA1                   MBEDTLS_OID_RSA_COMPANY "\x02\x07" /**< id-hmacWithSHA1 OBJECT IDENTIFIER ::= { iso(1) member-body(2) us(840) rsadsi(113549) digestAlgorithm(2) 7 } */

#define MBEDTLS_OID_HMAC_SHA224                 MBEDTLS_OID_RSA_COMPANY "\x02\x08" /**< id-hmacWithSHA224 OBJECT IDENTIFIER ::= { iso(1) member-body(2) us(840) rsadsi(113549) digestAlgorithm(2) 8 } */

#define MBEDTLS_OID_HMAC_SHA256                 MBEDTLS_OID_RSA_COMPANY "\x02\x09" /**< id-hmacWithSHA256 OBJECT IDENTIFIER ::= { iso(1) member-body(2) us(840) rsadsi(113549) digestAlgorithm(2) 9 } */

#define MBEDTLS_OID_HMAC_SHA384                 MBEDTLS_OID_RSA_COMPANY "\x02\x0A" /**< id-hmacWithSHA384 OBJECT IDENTIFIER ::= { iso(1) member-body(2) us(840) rsadsi(113549) digestAlgorithm(2) 10 } */

#define MBEDTLS_OID_HMAC_SHA512                 MBEDTLS_OID_RSA_COMPANY "\x02\x0B" /**< id-hmacWithSHA512 OBJECT IDENTIFIER ::= { iso(1) member-body(2) us(840) rsadsi(113549) digestAlgorithm(2) 11 } */

#define MBEDTLS_OID_HMAC_SHA3_224               MBEDTLS_OID_NIST_ALG "\x02\x0d" /**< id-hmacWithSHA3-512 OBJECT IDENTIFIER ::= { joint-iso-itu-t(2) country(16) us(840) organization(1) gov(101) csor(3) nistAlgorithms(4) hashalgs(2) hmacWithSHA3-224(13) } */

#define MBEDTLS_OID_HMAC_SHA3_256               MBEDTLS_OID_NIST_ALG "\x02\x0e" /**< id-hmacWithSHA3-512 OBJECT IDENTIFIER ::= { joint-iso-itu-t(2) country(16) us(840) organization(1) gov(101) csor(3) nistAlgorithms(4) hashalgs(2) hmacWithSHA3-256(14) } */

#define MBEDTLS_OID_HMAC_SHA3_384               MBEDTLS_OID_NIST_ALG "\x02\x0f" /**< id-hmacWithSHA3-512 OBJECT IDENTIFIER ::= { joint-iso-itu-t(2) country(16) us(840) organization(1) gov(101) csor(3) nistAlgorithms(4) hashalgs(2) hmacWithSHA3-384(15) } */

#define MBEDTLS_OID_HMAC_SHA3_512               MBEDTLS_OID_NIST_ALG "\x02\x10" /**< id-hmacWithSHA3-512 OBJECT IDENTIFIER ::= { joint-iso-itu-t(2) country(16) us(840) organization(1) gov(101) csor(3) nistAlgorithms(4) hashalgs(2) hmacWithSHA3-512(16) } */

#define MBEDTLS_OID_HMAC_RIPEMD160              MBEDTLS_OID_INTERNET "\x05\x05\x08\x01\x04" /**< id-hmacWithSHA1 OBJECT IDENTIFIER ::= {iso(1) iso-identified-organization(3) dod(6) internet(1) security(5) mechanisms(5) ipsec(8) isakmpOakley(1) hmacRIPEMD160(4)} */

/*
 * Encryption algorithms,
 * the following standardized object identifiers are specified at
 * https://datatracker.ietf.org/doc/html/rfc8018#appendix-C.
 */
#define MBEDTLS_OID_AES                         MBEDTLS_OID_NIST_ALG "\x01" /** aes OBJECT IDENTIFIER ::= { joint-iso-itu-t(2) country(16) us(840) organization(1) gov(101) csor(3) nistAlgorithm(4) 1 } */
#define MBEDTLS_OID_AES_128_CBC                 MBEDTLS_OID_AES "\x02" /** aes128-cbc-pad OBJECT IDENTIFIER ::= { joint-iso-itu-t(2) country(16) us(840) organization(1) gov(101) csor(3) nistAlgorithms(4) aes(1) aes128-CBC-PAD(2) } */
#define MBEDTLS_OID_AES_192_CBC                 MBEDTLS_OID_AES "\x16" /** aes192-cbc-pad OBJECT IDENTIFIER ::= { joint-iso-itu-t(2) country(16) us(840) organization(1) gov(101) csor(3) nistAlgorithms(4) aes(1) aes192-CBC-PAD(22) } */
#define MBEDTLS_OID_AES_256_CBC                 MBEDTLS_OID_AES "\x2a" /** aes256-cbc-pad OBJECT IDENTIFIER ::= { joint-iso-itu-t(2) country(16) us(840) organization(1) gov(101) csor(3) nistAlgorithms(4) aes(1) aes256-CBC-PAD(42) } */

/*
 * PKCS#5 OIDs
 */
#define MBEDTLS_OID_PKCS5_PBKDF2                MBEDTLS_OID_PKCS5 "\x0c" /**< id-PBKDF2 OBJECT IDENTIFIER ::= {pkcs-5 12} */
#define MBEDTLS_OID_PKCS5_PBES2                 MBEDTLS_OID_PKCS5 "\x0d" /**< id-PBES2 OBJECT IDENTIFIER ::= {pkcs-5 13} */

/*
 * EC key algorithms from RFC 5480
 */

/* id-ecPublicKey OBJECT IDENTIFIER ::= {
 *       iso(1) member-body(2) us(840) ansi-X9-62(10045) keyType(2) 1 } */
#define MBEDTLS_OID_EC_ALG_UNRESTRICTED         MBEDTLS_OID_ANSI_X9_62 "\x02\01"

/*   id-ecDH OBJECT IDENTIFIER ::= {
 *     iso(1) identified-organization(3) certicom(132)
 *     schemes(1) ecdh(12) } */
#define MBEDTLS_OID_EC_ALG_ECDH                 MBEDTLS_OID_CERTICOM "\x01\x0c"

/*
 * ECParameters namedCurve identifiers, from RFC 5480, RFC 5639, and SEC2
 */

/* secp256r1 OBJECT IDENTIFIER ::= {
 *   iso(1) member-body(2) us(840) ansi-X9-62(10045) curves(3) prime(1) 7 } */
#define MBEDTLS_OID_EC_GRP_SECP256R1        MBEDTLS_OID_ANSI_X9_62 "\x03\x01\x07"

/* secp384r1 OBJECT IDENTIFIER ::= {
 *   iso(1) identified-organization(3) certicom(132) curve(0) 34 } */
#define MBEDTLS_OID_EC_GRP_SECP384R1        MBEDTLS_OID_CERTICOM "\x00\x22"

/* secp521r1 OBJECT IDENTIFIER ::= {
 *   iso(1) identified-organization(3) certicom(132) curve(0) 35 } */
#define MBEDTLS_OID_EC_GRP_SECP521R1        MBEDTLS_OID_CERTICOM "\x00\x23"

/* secp256k1 OBJECT IDENTIFIER ::= {
 *   iso(1) identified-organization(3) certicom(132) curve(0) 10 } */
#define MBEDTLS_OID_EC_GRP_SECP256K1        MBEDTLS_OID_CERTICOM "\x00\x0a"

/* RFC 5639 4.1
 * ecStdCurvesAndGeneration OBJECT IDENTIFIER::= {iso(1)
 * identified-organization(3) teletrust(36) algorithm(3) signature-
 * algorithm(3) ecSign(2) 8}
 * ellipticCurve OBJECT IDENTIFIER ::= {ecStdCurvesAndGeneration 1}
 * versionOne OBJECT IDENTIFIER ::= {ellipticCurve 1} */
#define MBEDTLS_OID_EC_BRAINPOOL_V1         MBEDTLS_OID_TELETRUST "\x03\x03\x02\x08\x01\x01"

/* brainpoolP256r1 OBJECT IDENTIFIER ::= {versionOne 7} */
#define MBEDTLS_OID_EC_GRP_BP256R1          MBEDTLS_OID_EC_BRAINPOOL_V1 "\x07"

/* brainpoolP384r1 OBJECT IDENTIFIER ::= {versionOne 11} */
#define MBEDTLS_OID_EC_GRP_BP384R1          MBEDTLS_OID_EC_BRAINPOOL_V1 "\x0B"

/* brainpoolP512r1 OBJECT IDENTIFIER ::= {versionOne 13} */
#define MBEDTLS_OID_EC_GRP_BP512R1          MBEDTLS_OID_EC_BRAINPOOL_V1 "\x0D"

/*
 * SEC1 C.1
 *
 * prime-field OBJECT IDENTIFIER ::= { id-fieldType 1 }
 * id-fieldType OBJECT IDENTIFIER ::= { ansi-X9-62 fieldType(1)}
 */
#define MBEDTLS_OID_ANSI_X9_62_FIELD_TYPE   MBEDTLS_OID_ANSI_X9_62 "\x01"
#define MBEDTLS_OID_ANSI_X9_62_PRIME_FIELD  MBEDTLS_OID_ANSI_X9_62_FIELD_TYPE "\x01"

/*
 * EC key algorithms from RFC 8410
 */

#define MBEDTLS_OID_X25519                  MBEDTLS_OID_THAWTE "\x6e" /**< id-X25519    OBJECT IDENTIFIER ::= { 1 3 101 110 } */
#define MBEDTLS_OID_X448                    MBEDTLS_OID_THAWTE "\x6f" /**< id-X448      OBJECT IDENTIFIER ::= { 1 3 101 111 } */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief Base OID descriptor structure
 */
typedef struct mbedtls_oid_descriptor_t {
    const char *MBEDTLS_PRIVATE(asn1);               /*!< OID ASN.1 representation       */
    size_t MBEDTLS_PRIVATE(asn1_len);                /*!< length of asn1                 */
} mbedtls_oid_descriptor_t;

#if defined(MBEDTLS_PK_PARSE_C) || defined(MBEDTLS_PK_WRITE_C)
/**
 * \brief          Translate PublicKeyAlgorithm OID into pk_type
 *
 * \param oid      OID to use
 * \param pk_alg   place to store public key algorithm
 *
 * \return         0 if successful, or MBEDTLS_ERR_OID_NOT_FOUND
 */
int mbedtls_oid_get_pk_alg(const mbedtls_asn1_buf *oid, mbedtls_pk_type_t *pk_alg);

/**
 * \brief          Translate pk_type into PublicKeyAlgorithm OID
 *
 * \param pk_alg   Public key type to look for
 * \param oid      place to store ASN.1 OID string pointer
 * \param olen     length of the OID
 *
 * \return         0 if successful, or MBEDTLS_ERR_OID_NOT_FOUND
 */
int mbedtls_oid_get_oid_by_pk_alg(mbedtls_pk_type_t pk_alg,
                                  const char **oid, size_t *olen);

#if defined(PSA_WANT_KEY_TYPE_ECC_PUBLIC_KEY)
/**
 * \brief          Translate NamedCurve OID into an EC group identifier
 *
 * \param oid      OID to use
 * \param grp_id   place to store group id
 *
 * \return         0 if successful, or MBEDTLS_ERR_OID_NOT_FOUND
 */
int mbedtls_oid_get_ec_grp(const mbedtls_asn1_buf *oid, mbedtls_ecp_group_id *grp_id);

/**
 * \brief          Translate EC group identifier into NamedCurve OID
 *
 * \param grp_id   EC group identifier
 * \param oid      place to store ASN.1 OID string pointer
 * \param olen     length of the OID
 *
 * \return         0 if successful, or MBEDTLS_ERR_OID_NOT_FOUND
 */
int mbedtls_oid_get_oid_by_ec_grp(mbedtls_ecp_group_id grp_id,
                                  const char **oid, size_t *olen);

/**
 * \brief          Translate AlgorithmIdentifier OID into an EC group identifier,
 *                 for curves that are directly encoded at this level
 *
 * \param oid      OID to use
 * \param grp_id   place to store group id
 *
 * \return         0 if successful, or MBEDTLS_ERR_OID_NOT_FOUND
 */
int mbedtls_oid_get_ec_grp_algid(const mbedtls_asn1_buf *oid, mbedtls_ecp_group_id *grp_id);

/**
 * \brief          Translate EC group identifier into AlgorithmIdentifier OID,
 *                 for curves that are directly encoded at this level
 *
 * \param grp_id   EC group identifier
 * \param oid      place to store ASN.1 OID string pointer
 * \param olen     length of the OID
 *
 * \return         0 if successful, or MBEDTLS_ERR_OID_NOT_FOUND
 */
int mbedtls_oid_get_oid_by_ec_grp_algid(mbedtls_ecp_group_id grp_id,
                                        const char **oid, size_t *olen);
#endif /* PSA_WANT_KEY_TYPE_ECC_PUBLIC_KEY */
#endif /* MBEDTLS_PK_PARSE_C || MBEDTLS_PK_WRITE_C */

#if defined(MBEDTLS_PKCS5_C) && defined(MBEDTLS_ASN1_PARSE_C)
/**
 * \brief          Translate hmac algorithm OID into md_type
 *
 * \param oid      OID to use
 * \param md_hmac  place to store message hmac algorithm
 *
 * \return         0 if successful, or MBEDTLS_ERR_OID_NOT_FOUND
 */
int mbedtls_oid_get_md_hmac(const mbedtls_asn1_buf *oid, mbedtls_md_type_t *md_hmac);

#if defined(MBEDTLS_CIPHER_C)
/**
 * \brief          Translate encryption algorithm OID into cipher_type
 *
 * \param oid           OID to use
 * \param cipher_alg    place to store cipher algorithm
 *
 * \return         0 if successful, or MBEDTLS_ERR_OID_NOT_FOUND
 */
int mbedtls_oid_get_cipher_alg(const mbedtls_asn1_buf *oid, mbedtls_cipher_type_t *cipher_alg);
#endif /* MBEDTLS_CIPHER_C */
#endif /* MBEDTLS_PKCS5_C && MBEDTLS_ASN1_PARSE_C */

#if defined(MBEDTLS_RSA_C) && defined(MBEDTLS_PKCS1_V15)
/**
 * \brief          Translate md_type into hash algorithm OID
 *
 * \param md_alg   message digest algorithm
 * \param oid      place to store ASN.1 OID string pointer
 * \param olen     length of the OID
 *
 * \return         0 if successful, or MBEDTLS_ERR_OID_NOT_FOUND
 */
int mbedtls_oid_get_oid_by_md(mbedtls_md_type_t md_alg, const char **oid, size_t *olen);
#endif /* MBEDTLS_RSA_C && MBEDTLS_PKCS1_V15 */

#ifdef __cplusplus
}
#endif

#endif /* TF_PSA_CRYPTO_CRYPTO_OID_H */
