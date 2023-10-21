/**
 * \file build_info.h
 *
 * \brief Build-time configuration info
 *
 *  Include this file if you need to depend on the
 *  configuration options defined in mbedtls_config.h or MBEDTLS_CONFIG_FILE
 */
/*
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

#ifndef MBEDTLS_BUILD_INFO_H
#define MBEDTLS_BUILD_INFO_H

/*
 * This set of compile-time defines can be used to determine the version number
 * of the Mbed TLS library used. Run-time variables for the same can be found in
 * version.h
 */

/**
 * The version number x.y.z is split into three parts.
 * Major, Minor, Patchlevel
 */
#define MBEDTLS_VERSION_MAJOR  3
#define MBEDTLS_VERSION_MINOR  4
#define MBEDTLS_VERSION_PATCH  1

/**
 * The single version number has the following structure:
 *    MMNNPP00
 *    Major version | Minor version | Patch version
 */
#define MBEDTLS_VERSION_NUMBER         0x03040100
#define MBEDTLS_VERSION_STRING         "3.4.1"
#define MBEDTLS_VERSION_STRING_FULL    "mbed TLS 3.4.1"

#if defined(_MSC_VER) && !defined(_CRT_SECURE_NO_DEPRECATE)
#define _CRT_SECURE_NO_DEPRECATE 1
#endif

/* Define `inline` on some non-C99-compliant compilers. */
#if (defined(__ARMCC_VERSION) || defined(_MSC_VER)) && \
    !defined(inline) && !defined(__cplusplus)
#define inline __inline
#endif

#if !defined(MBEDTLS_CONFIG_FILE)
#include "mbedtls/mbedtls_config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#if defined(MBEDTLS_CONFIG_VERSION) && ( \
    MBEDTLS_CONFIG_VERSION < 0x03000000 || \
                             MBEDTLS_CONFIG_VERSION > MBEDTLS_VERSION_NUMBER)
#error "Invalid config version, defined value of MBEDTLS_CONFIG_VERSION is unsupported"
#endif

/* Target and application specific configurations
 *
 * Allow user to override any previous default.
 *
 */
#if defined(MBEDTLS_USER_CONFIG_FILE)
#include MBEDTLS_USER_CONFIG_FILE
#endif

/* Auto-enable MBEDTLS_MD_LIGHT based on MBEDTLS_MD_C.
 * This allows checking for MD_LIGHT rather than MD_LIGHT || MD_C.
 */
#if defined(MBEDTLS_MD_C)
#define MBEDTLS_MD_LIGHT
#endif

/* Auto-enable MBEDTLS_MD_LIGHT if some module needs it.
 */
#if defined(MBEDTLS_PEM_PARSE_C) || \
    defined(MBEDTLS_RSA_C)
#define MBEDTLS_MD_LIGHT
#endif

/* If MBEDTLS_PSA_CRYPTO_C is defined, make sure MBEDTLS_PSA_CRYPTO_CLIENT
 * is defined as well to include all PSA code.
 */
#if defined(MBEDTLS_PSA_CRYPTO_C)
#define MBEDTLS_PSA_CRYPTO_CLIENT
#endif /* MBEDTLS_PSA_CRYPTO_C */

/* The PK wrappers need pk_write functions to format RSA key objects
 * when they are dispatching to the PSA API. This happens under USE_PSA_CRYPTO,
 * and also even without USE_PSA_CRYPTO for mbedtls_pk_sign_ext(). */
#if defined(MBEDTLS_PSA_CRYPTO_C) && defined(MBEDTLS_RSA_C)
#define MBEDTLS_PK_C
#define MBEDTLS_PK_WRITE_C
#define MBEDTLS_PK_PARSE_C
#endif

#if !defined(MBEDTLS_SSL_PROTO_TLS1_2)
#undef MBEDTLS_KEY_EXCHANGE_RSA_ENABLED
#undef MBEDTLS_KEY_EXCHANGE_DHE_RSA_ENABLED
#undef MBEDTLS_KEY_EXCHANGE_ECDHE_RSA_ENABLED
#undef MBEDTLS_KEY_EXCHANGE_ECDHE_ECDSA_ENABLED
#undef MBEDTLS_KEY_EXCHANGE_PSK_ENABLED
#undef MBEDTLS_KEY_EXCHANGE_DHE_PSK_ENABLED
#undef MBEDTLS_KEY_EXCHANGE_RSA_PSK_ENABLED
#undef MBEDTLS_KEY_EXCHANGE_ECDHE_PSK_ENABLED
#undef MBEDTLS_KEY_EXCHANGE_ECDH_RSA_ENABLED
#undef MBEDTLS_KEY_EXCHANGE_ECDH_ECDSA_ENABLED
#undef MBEDTLS_KEY_EXCHANGE_ECJPAKE_ENABLED
#endif

#if !defined(MBEDTLS_SSL_PROTO_TLS1_3)
#undef MBEDTLS_SSL_TLS1_3_KEY_EXCHANGE_MODE_PSK_ENABLED
#undef MBEDTLS_SSL_TLS1_3_KEY_EXCHANGE_MODE_EPHEMERAL_ENABLED
#undef MBEDTLS_SSL_TLS1_3_KEY_EXCHANGE_MODE_PSK_EPHEMERAL_ENABLED
#undef MBEDTLS_SSL_EARLY_DATA
#endif

#if defined(MBEDTLS_SSL_TLS1_3_KEY_EXCHANGE_MODE_PSK_ENABLED) || \
    defined(MBEDTLS_SSL_TLS1_3_KEY_EXCHANGE_MODE_PSK_EPHEMERAL_ENABLED)
#define MBEDTLS_SSL_TLS1_3_KEY_EXCHANGE_MODE_SOME_PSK_ENABLED
#endif

#if defined(MBEDTLS_SSL_TLS1_3_KEY_EXCHANGE_MODE_EPHEMERAL_ENABLED) || \
    defined(MBEDTLS_SSL_TLS1_3_KEY_EXCHANGE_MODE_PSK_EPHEMERAL_ENABLED)
#define MBEDTLS_SSL_TLS1_3_KEY_EXCHANGE_MODE_SOME_EPHEMERAL_ENABLED
#endif

/* Make sure all configuration symbols are set before including check_config.h,
 * even the ones that are calculated programmatically. */
#if defined(MBEDTLS_PSA_CRYPTO_CONFIG) /* PSA_WANT_xxx influences MBEDTLS_xxx */ || \
    defined(MBEDTLS_PSA_CRYPTO_C) /* MBEDTLS_xxx influences PSA_WANT_xxx */
#include "mbedtls/config_psa.h"
#endif

#include "mbedtls/check_config.h"

#endif /* MBEDTLS_BUILD_INFO_H */
