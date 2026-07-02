/**
 * \file tf-psa-crypto/private/crypto_adjust_config_key_pair_types.h
 * \brief Adjust PSA configuration for key pair types.
 *
 * This is an internal header. Do not include it directly.
 *
 * See docs/proposed/psa-conditional-inclusion-c.md.
 * - Support non-basic operations in a keypair type implicitly enables basic
 *   support for that keypair type.
 * - Support for a keypair type implicitly enables the corresponding public
 *   key type.
 * - Basic support for a keypair type implicilty enables import/export support
 *   for that keypair type. Warning: this is implementation-specific (mainly
 *   for the benefit of testing) and may change in the future!
 */
/*
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */

#ifndef TF_PSA_CRYPTO_PRIVATE_CRYPTO_ADJUST_CONFIG_KEY_PAIR_TYPES_H
#define TF_PSA_CRYPTO_PRIVATE_CRYPTO_ADJUST_CONFIG_KEY_PAIR_TYPES_H

/*****************************************************************
 * ANYTHING -> BASIC
 ****************************************************************/

#if defined(PSA_WANT_KEY_TYPE_ECC_KEY_PAIR_IMPORT) || \
    defined(PSA_WANT_KEY_TYPE_ECC_KEY_PAIR_EXPORT) || \
    defined(PSA_WANT_KEY_TYPE_ECC_KEY_PAIR_GENERATE) || \
    defined(PSA_WANT_KEY_TYPE_ECC_KEY_PAIR_DERIVE)
#define PSA_WANT_KEY_TYPE_ECC_KEY_PAIR_BASIC 1
#endif

#if defined(PSA_WANT_KEY_TYPE_RSA_KEY_PAIR_IMPORT) || \
    defined(PSA_WANT_KEY_TYPE_RSA_KEY_PAIR_EXPORT) || \
    defined(PSA_WANT_KEY_TYPE_RSA_KEY_PAIR_GENERATE) || \
    defined(PSA_WANT_KEY_TYPE_RSA_KEY_PAIR_DERIVE)
#define PSA_WANT_KEY_TYPE_RSA_KEY_PAIR_BASIC 1
#endif

#if defined(PSA_WANT_KEY_TYPE_DH_KEY_PAIR_IMPORT) || \
    defined(PSA_WANT_KEY_TYPE_DH_KEY_PAIR_EXPORT) || \
    defined(PSA_WANT_KEY_TYPE_DH_KEY_PAIR_GENERATE) || \
    defined(PSA_WANT_KEY_TYPE_DH_KEY_PAIR_DERIVE)
#define PSA_WANT_KEY_TYPE_DH_KEY_PAIR_BASIC 1
#endif

/*****************************************************************
 * BASIC -> corresponding PUBLIC
 ****************************************************************/

#if defined(PSA_WANT_KEY_TYPE_ECC_KEY_PAIR_BASIC)
#define PSA_WANT_KEY_TYPE_ECC_PUBLIC_KEY 1
#endif

#if defined(PSA_WANT_KEY_TYPE_RSA_KEY_PAIR_BASIC)
#define PSA_WANT_KEY_TYPE_RSA_PUBLIC_KEY 1
#endif

#if defined(PSA_WANT_KEY_TYPE_DH_KEY_PAIR_BASIC)
#define PSA_WANT_KEY_TYPE_DH_PUBLIC_KEY 1
#endif

/*****************************************************************
 * BASIC -> IMPORT+EXPORT
 *
 * (Implementation-specific, may change in the future.)
 ****************************************************************/

/* Even though KEY_PAIR symbols' feature several level of support (BASIC, IMPORT,
 * EXPORT, GENERATE, DERIVE) we're not planning to have support only for BASIC
 * without IMPORT/EXPORT since these last 2 features are strongly used in tests.
 * In general it is allowed to include more feature than what is strictly
 * requested.
 * As a consequence IMPORT and EXPORT features will be automatically enabled
 * as soon as the BASIC one is. */
#if defined(PSA_WANT_KEY_TYPE_ECC_KEY_PAIR_BASIC)
#define PSA_WANT_KEY_TYPE_ECC_KEY_PAIR_IMPORT 1
#define PSA_WANT_KEY_TYPE_ECC_KEY_PAIR_EXPORT 1
#endif

#if defined(PSA_WANT_KEY_TYPE_RSA_KEY_PAIR_BASIC)
#define PSA_WANT_KEY_TYPE_RSA_KEY_PAIR_IMPORT 1
#define PSA_WANT_KEY_TYPE_RSA_KEY_PAIR_EXPORT 1
#endif

#if defined(PSA_WANT_KEY_TYPE_DH_KEY_PAIR_BASIC)
#define PSA_WANT_KEY_TYPE_DH_KEY_PAIR_IMPORT 1
#define PSA_WANT_KEY_TYPE_DH_KEY_PAIR_EXPORT 1
#endif

#endif /* TF_PSA_CRYPTO_PRIVATE_CRYPTO_ADJUST_CONFIG_KEY_PAIR_TYPES_H */
