/**
 * \file pkwrite.h
 *
 * \brief Internal defines shared by the PK write module
 */
/*
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */

#ifndef TF_PSA_CRYPTO_PKWRITE_H
#define TF_PSA_CRYPTO_PKWRITE_H

#include "tf-psa-crypto/build_info.h"

#include "mbedtls/pk.h"
#if defined(MBEDTLS_PK_HAVE_PRIVATE_HEADER)
#include <mbedtls/private/pk_private.h>
#endif /* MBEDTLS_PK_HAVE_PRIVATE_HEADER */

#include "psa/crypto.h"

/*
 * Max sizes of key per types. Shown as tag + len (+ content).
 */

#if defined(PSA_WANT_KEY_TYPE_RSA_PUBLIC_KEY)

/*
 * RSA public keys:
 *  SubjectPublicKeyInfo  ::=  SEQUENCE  {          1 + 3
 *       algorithm            AlgorithmIdentifier,  1 + 1 (sequence)
 *                                                + 1 + 1 + 9 (rsa oid)
 *                                                + 1 + 1 (params null)
 *       subjectPublicKey     BIT STRING            1 + 3 + [PSA format]
 *  }
 */
#define MBEDTLS_PK_RSA_PUB_DER_MAX_BYTES    \
    23 + PSA_KEY_EXPORT_RSA_PUBLIC_KEY_MAX_SIZE(PSA_VENDOR_RSA_MAX_KEY_BITS)

/*
 * RSA private keys: PSA export format
 */
#define MBEDTLS_PK_RSA_PRV_DER_MAX_BYTES    \
    PSA_KEY_EXPORT_RSA_KEY_PAIR_MAX_SIZE(PSA_VENDOR_RSA_MAX_KEY_BITS)

#else /* PSA_WANT_KEY_TYPE_RSA_PUBLIC_KEY */

#define MBEDTLS_PK_RSA_PUB_DER_MAX_BYTES   0
#define MBEDTLS_PK_RSA_PRV_DER_MAX_BYTES   0

#endif /* PSA_WANT_KEY_TYPE_RSA_PUBLIC_KEY */

#if defined(PSA_WANT_KEY_TYPE_ECC_PUBLIC_KEY)

/*
 * EC public keys:
 *  SubjectPublicKeyInfo  ::=  SEQUENCE  {      1 + 2
 *    algorithm         AlgorithmIdentifier,    1 + 1 (sequence)
 *                                            + 1 + 1 + 7 (ec oid)
 *                                            + 1 + 1 + 9 (namedCurve oid)
 *    subjectPublicKey  BIT STRING              1 + 2 + 1               [*]
 *                                            + [PSA export format]     [*]
 *  }
 */
#define MBEDTLS_PK_ECP_PUB_DER_MAX_BYTES    (29 + \
                                             PSA_KEY_EXPORT_ECC_PUBLIC_KEY_MAX_SIZE( \
                                                 PSA_VENDOR_ECC_MAX_CURVE_BITS))

/*
 * EC private keys:
 * ECPrivateKey ::= SEQUENCE {                  1 + 2
 *      version        INTEGER ,                1 + 1 + 1
 *      privateKey     OCTET STRING,            1 + 1 + [PSA export format]
 *      parameters [0] ECParameters OPTIONAL,   1 + 1 + (1 + 1 + 9)
 *      publicKey  [1] BIT STRING OPTIONAL      1 + 2 + [*] above
 *    }
 */
#define MBEDTLS_PK_ECP_PRV_DER_MAX_BYTES    (8 + \
                                             PSA_KEY_EXPORT_ECC_KEY_PAIR_MAX_SIZE( \
                                                 PSA_VENDOR_ECC_MAX_CURVE_BITS) + \
                                             16 + 4 + \
                                             PSA_KEY_EXPORT_ECC_PUBLIC_KEY_MAX_SIZE( \
                                                 PSA_VENDOR_ECC_MAX_CURVE_BITS))

#else /* PSA_WANT_KEY_TYPE_ECC_PUBLIC_KEY */

#define MBEDTLS_PK_ECP_PUB_DER_MAX_BYTES   0
#define MBEDTLS_PK_ECP_PRV_DER_MAX_BYTES   0

#endif /* PSA_WANT_KEY_TYPE_ECC_PUBLIC_KEY */

/* Define the maximum available public key DER length based on the supported
 * key types (EC and/or RSA). */
#if (MBEDTLS_PK_RSA_PUB_DER_MAX_BYTES > MBEDTLS_PK_ECP_PUB_DER_MAX_BYTES)
#define MBEDTLS_PK_WRITE_PUBKEY_MAX_SIZE    MBEDTLS_PK_RSA_PUB_DER_MAX_BYTES
#else
#define MBEDTLS_PK_WRITE_PUBKEY_MAX_SIZE    MBEDTLS_PK_ECP_PUB_DER_MAX_BYTES
#endif

#endif /* TF_PSA_CRYPTO_PKWRITE_H */
