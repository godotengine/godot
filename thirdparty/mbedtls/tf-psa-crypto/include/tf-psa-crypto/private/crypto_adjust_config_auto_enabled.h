/**
 * \file tf-psa-crypto/private/crypto_adjust_config_auto_enabled.h
 * \brief Adjust PSA configuration: enable always-on features
 *
 * This is an internal header. Do not include it directly.
 *
 * Always enable certain features which require a negligible amount of code
 * to implement, to avoid some edge cases in the configuration combinatorics.
 */
/*
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */

#ifndef TF_PSA_CRYPTO_PRIVATE_CRYPTO_ADJUST_CONFIG_AUTO_ENABLED_H
#define TF_PSA_CRYPTO_PRIVATE_CRYPTO_ADJUST_CONFIG_AUTO_ENABLED_H

#define PSA_WANT_KEY_TYPE_DERIVE 1
#define PSA_WANT_KEY_TYPE_PASSWORD 1
#define PSA_WANT_KEY_TYPE_PASSWORD_HASH 1
#define PSA_WANT_KEY_TYPE_RAW_DATA 1

#endif /* TF_PSA_CRYPTO_PRIVATE_CRYPTO_ADJUST_CONFIG_AUTO_ENABLED_H */
