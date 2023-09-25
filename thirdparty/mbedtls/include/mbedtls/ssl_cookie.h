/**
 * \file ssl_cookie.h
 *
 * \brief DTLS cookie callbacks implementation
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
#ifndef MBEDTLS_SSL_COOKIE_H
#define MBEDTLS_SSL_COOKIE_H
#include "mbedtls/private_access.h"

#include "mbedtls/build_info.h"

#include "mbedtls/ssl.h"

#if !defined(MBEDTLS_USE_PSA_CRYPTO)
#if defined(MBEDTLS_THREADING_C)
#include "mbedtls/threading.h"
#endif
#endif /* !MBEDTLS_USE_PSA_CRYPTO */

/**
 * \name SECTION: Module settings
 *
 * The configuration options you can set for this module are in this section.
 * Either change them in mbedtls_config.h or define them on the compiler command line.
 * \{
 */
#ifndef MBEDTLS_SSL_COOKIE_TIMEOUT
#define MBEDTLS_SSL_COOKIE_TIMEOUT     60 /**< Default expiration delay of DTLS cookies, in seconds if HAVE_TIME, or in number of cookies issued */
#endif

/** \} name SECTION: Module settings */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief          Context for the default cookie functions.
 */
typedef struct mbedtls_ssl_cookie_ctx {
#if defined(MBEDTLS_USE_PSA_CRYPTO)
    mbedtls_svc_key_id_t    MBEDTLS_PRIVATE(psa_hmac_key);  /*!< key id for the HMAC portion   */
    psa_algorithm_t         MBEDTLS_PRIVATE(psa_hmac_alg);  /*!< key algorithm for the HMAC portion   */
#else
    mbedtls_md_context_t    MBEDTLS_PRIVATE(hmac_ctx);   /*!< context for the HMAC portion   */
#endif /* MBEDTLS_USE_PSA_CRYPTO */
#if !defined(MBEDTLS_HAVE_TIME)
    unsigned long   MBEDTLS_PRIVATE(serial);     /*!< serial number for expiration   */
#endif
    unsigned long   MBEDTLS_PRIVATE(timeout);    /*!< timeout delay, in seconds if HAVE_TIME,
                                                    or in number of tickets issued */

#if !defined(MBEDTLS_USE_PSA_CRYPTO)
#if defined(MBEDTLS_THREADING_C)
    mbedtls_threading_mutex_t MBEDTLS_PRIVATE(mutex);
#endif
#endif /* !MBEDTLS_USE_PSA_CRYPTO */
} mbedtls_ssl_cookie_ctx;

/**
 * \brief          Initialize cookie context
 */
void mbedtls_ssl_cookie_init(mbedtls_ssl_cookie_ctx *ctx);

/**
 * \brief          Setup cookie context (generate keys)
 */
int mbedtls_ssl_cookie_setup(mbedtls_ssl_cookie_ctx *ctx,
                             int (*f_rng)(void *, unsigned char *, size_t),
                             void *p_rng);

/**
 * \brief          Set expiration delay for cookies
 *                 (Default MBEDTLS_SSL_COOKIE_TIMEOUT)
 *
 * \param ctx      Cookie context
 * \param delay    Delay, in seconds if HAVE_TIME, or in number of cookies
 *                 issued in the meantime.
 *                 0 to disable expiration (NOT recommended)
 */
void mbedtls_ssl_cookie_set_timeout(mbedtls_ssl_cookie_ctx *ctx, unsigned long delay);

/**
 * \brief          Free cookie context
 */
void mbedtls_ssl_cookie_free(mbedtls_ssl_cookie_ctx *ctx);

/**
 * \brief          Generate cookie, see \c mbedtls_ssl_cookie_write_t
 */
mbedtls_ssl_cookie_write_t mbedtls_ssl_cookie_write;

/**
 * \brief          Verify cookie, see \c mbedtls_ssl_cookie_write_t
 */
mbedtls_ssl_cookie_check_t mbedtls_ssl_cookie_check;

#ifdef __cplusplus
}
#endif

#endif /* ssl_cookie.h */
