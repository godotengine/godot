/*
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */

#ifndef MBEDTLS_SSL_TLS13_INVASIVE_H
#define MBEDTLS_SSL_TLS13_INVASIVE_H

#include "ssl_misc.h"

#if defined(MBEDTLS_SSL_PROTO_TLS1_3)

#include "psa/crypto.h"

#if defined(MBEDTLS_TEST_HOOKS)
int mbedtls_ssl_tls13_parse_certificate(mbedtls_ssl_context *ssl,
                                        const unsigned char *buf,
                                        const unsigned char *end);
#endif /* MBEDTLS_TEST_HOOKS */

#if defined(MBEDTLS_TEST_HOOKS) && defined(MBEDTLS_SSL_SRV_C)
int mbedtls_ssl_tls13_reset_handshake_state_for_hrr(mbedtls_ssl_context *ssl);
#if defined(MBEDTLS_SSL_TLS1_3_KEY_EXCHANGE_MODE_SOME_EPHEMERAL_ENABLED)
int mbedtls_ssl_tls13_parse_key_shares_ext(mbedtls_ssl_context *ssl,
                                           const unsigned char *buf,
                                           const unsigned char *end);
#endif
#endif /* MBEDTLS_TEST_HOOKS && MBEDTLS_SSL_SRV_C */

#endif /* MBEDTLS_SSL_PROTO_TLS1_3 */

#endif /* MBEDTLS_SSL_TLS13_INVASIVE_H */
