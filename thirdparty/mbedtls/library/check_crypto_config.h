/**
 * \file check_crypto_config.h
 *
 * \brief Consistency checks for PSA configuration options
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

/*
 * It is recommended to include this file from your crypto_config.h
 * in order to catch dependency issues early.
 */

#ifndef MBEDTLS_CHECK_CRYPTO_CONFIG_H
#define MBEDTLS_CHECK_CRYPTO_CONFIG_H

#if defined(PSA_WANT_ALG_CCM) && \
    !(defined(PSA_WANT_KEY_TYPE_AES) || \
    defined(PSA_WANT_KEY_TYPE_CAMELLIA))
#error "PSA_WANT_ALG_CCM defined, but not all prerequisites"
#endif

#if defined(PSA_WANT_ALG_CMAC) && \
    !(defined(PSA_WANT_KEY_TYPE_AES) || \
    defined(PSA_WANT_KEY_TYPE_CAMELLIA) || \
    defined(PSA_WANT_KEY_TYPE_DES))
#error "PSA_WANT_ALG_CMAC defined, but not all prerequisites"
#endif

#if defined(PSA_WANT_ALG_DETERMINISTIC_ECDSA) && \
    !(defined(PSA_WANT_KEY_TYPE_ECC_KEY_PAIR) || \
    defined(PSA_WANT_KEY_TYPE_ECC_PUBLIC_KEY))
#error "PSA_WANT_ALG_DETERMINISTIC_ECDSA defined, but not all prerequisites"
#endif

#if defined(PSA_WANT_ALG_ECDSA) && \
    !(defined(PSA_WANT_KEY_TYPE_ECC_KEY_PAIR) || \
    defined(PSA_WANT_KEY_TYPE_ECC_PUBLIC_KEY))
#error "PSA_WANT_ALG_ECDSA defined, but not all prerequisites"
#endif

#if defined(PSA_WANT_ALG_GCM) && \
    !(defined(PSA_WANT_KEY_TYPE_AES) || \
    defined(PSA_WANT_KEY_TYPE_CAMELLIA))
#error "PSA_WANT_ALG_GCM defined, but not all prerequisites"
#endif

#if defined(PSA_WANT_ALG_RSA_PKCS1V15_CRYPT) && \
    !(defined(PSA_WANT_KEY_TYPE_RSA_KEY_PAIR) || \
    defined(PSA_WANT_KEY_TYPE_RSA_PUBLIC_KEY))
#error "PSA_WANT_ALG_RSA_PKCS1V15_CRYPT defined, but not all prerequisites"
#endif

#if defined(PSA_WANT_ALG_RSA_PKCS1V15_SIGN) && \
    !(defined(PSA_WANT_KEY_TYPE_RSA_KEY_PAIR) || \
    defined(PSA_WANT_KEY_TYPE_RSA_PUBLIC_KEY))
#error "PSA_WANT_ALG_RSA_PKCS1V15_SIGN defined, but not all prerequisites"
#endif

#if defined(PSA_WANT_ALG_RSA_OAEP) && \
    !(defined(PSA_WANT_KEY_TYPE_RSA_KEY_PAIR) || \
    defined(PSA_WANT_KEY_TYPE_RSA_PUBLIC_KEY))
#error "PSA_WANT_ALG_RSA_OAEP defined, but not all prerequisites"
#endif

#if defined(PSA_WANT_ALG_RSA_PSS) && \
    !(defined(PSA_WANT_KEY_TYPE_RSA_KEY_PAIR) || \
    defined(PSA_WANT_KEY_TYPE_RSA_PUBLIC_KEY))
#error "PSA_WANT_ALG_RSA_PSS defined, but not all prerequisites"
#endif

#if defined(PSA_WANT_KEY_TYPE_ECC_KEY_PAIR) && \
    !defined(PSA_WANT_KEY_TYPE_ECC_PUBLIC_KEY)
#error "PSA_WANT_KEY_TYPE_ECC_KEY_PAIR defined, but not all prerequisites"
#endif

#if defined(MBEDTLS_SSL_PROTO_TLS1_2) && defined(MBEDTLS_USE_PSA_CRYPTO) && \
    !(defined(PSA_WANT_ALG_SHA_1) || defined(PSA_WANT_ALG_SHA_256) || defined(PSA_WANT_ALG_SHA_512))
#error "MBEDTLS_SSL_PROTO_TLS1_2 defined, but not all prerequisites"
#endif

#if defined(PSA_WANT_ALG_TLS12_ECJPAKE_TO_PMS) && \
    !defined(PSA_WANT_ALG_SHA_256)
#error "PSA_WANT_ALG_TLS12_ECJPAKE_TO_PMS defined, but not all prerequisites"
#endif

#endif /* MBEDTLS_CHECK_CRYPTO_CONFIG_H */
