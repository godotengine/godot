/**
 * \file tf-psa-crypto/private/crypto_adjust_config_synonyms.h
 * \brief Adjust PSA configuration: enable quasi-synonyms
 *
 * This is an internal header. Do not include it directly.
 *
 * When two features require almost the same code, we automatically enable
 * both when either one is requested, to reduce the combinatorics of
 * possible configurations.
 */
/*
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */

#ifndef TF_PSA_CRYPTO_PRIVATE_CRYPTO_ADJUST_CONFIG_SYNONYMS_H
#define TF_PSA_CRYPTO_PRIVATE_CRYPTO_ADJUST_CONFIG_SYNONYMS_H

/****************************************************************/
/* De facto synonyms */
/****************************************************************/

#if defined(PSA_WANT_ALG_ECDSA_ANY) && !defined(PSA_WANT_ALG_ECDSA)
#define PSA_WANT_ALG_ECDSA PSA_WANT_ALG_ECDSA_ANY
#elif !defined(PSA_WANT_ALG_ECDSA_ANY) && defined(PSA_WANT_ALG_ECDSA)
#define PSA_WANT_ALG_ECDSA_ANY PSA_WANT_ALG_ECDSA
#endif

#if defined(PSA_WANT_ALG_RSA_PKCS1V15_SIGN_RAW) && !defined(PSA_WANT_ALG_RSA_PKCS1V15_SIGN)
#define PSA_WANT_ALG_RSA_PKCS1V15_SIGN PSA_WANT_ALG_RSA_PKCS1V15_SIGN_RAW
#elif !defined(PSA_WANT_ALG_RSA_PKCS1V15_SIGN_RAW) && defined(PSA_WANT_ALG_RSA_PKCS1V15_SIGN)
#define PSA_WANT_ALG_RSA_PKCS1V15_SIGN_RAW PSA_WANT_ALG_RSA_PKCS1V15_SIGN
#endif

#if defined(PSA_WANT_ALG_RSA_PSS_ANY_SALT) && !defined(PSA_WANT_ALG_RSA_PSS)
#define PSA_WANT_ALG_RSA_PSS PSA_WANT_ALG_RSA_PSS_ANY_SALT
#elif !defined(PSA_WANT_ALG_RSA_PSS_ANY_SALT) && defined(PSA_WANT_ALG_RSA_PSS)
#define PSA_WANT_ALG_RSA_PSS_ANY_SALT PSA_WANT_ALG_RSA_PSS
#endif

#endif /* TF_PSA_CRYPTO_PRIVATE_CRYPTO_ADJUST_CONFIG_SYNONYMS_H */
