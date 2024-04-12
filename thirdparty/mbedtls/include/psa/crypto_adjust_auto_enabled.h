/**
 * \file psa/crypto_adjust_auto_enabled.h
 * \brief Adjust PSA configuration: enable always-on features
 *
 * Always enable certain features which require a negligible amount of code
 * to implement, to avoid some edge cases in the configuration combinatorics.
 */
/*
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */

#ifndef PSA_CRYPTO_ADJUST_AUTO_ENABLED_H
#define PSA_CRYPTO_ADJUST_AUTO_ENABLED_H

#define PSA_WANT_KEY_TYPE_DERIVE 1
#define PSA_WANT_KEY_TYPE_PASSWORD 1
#define PSA_WANT_KEY_TYPE_PASSWORD_HASH 1
#define PSA_WANT_KEY_TYPE_RAW_DATA 1

#endif /* PSA_CRYPTO_ADJUST_AUTO_ENABLED_H */
