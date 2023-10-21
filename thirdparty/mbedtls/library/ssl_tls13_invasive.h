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

#ifndef MBEDTLS_SSL_TLS13_INVASIVE_H
#define MBEDTLS_SSL_TLS13_INVASIVE_H

#include "common.h"

#if defined(MBEDTLS_SSL_PROTO_TLS1_3)

#include "psa/crypto.h"

#if defined(MBEDTLS_TEST_HOOKS)
int mbedtls_ssl_tls13_parse_certificate(mbedtls_ssl_context *ssl,
                                        const unsigned char *buf,
                                        const unsigned char *end);
#endif /* MBEDTLS_TEST_HOOKS */

#endif /* MBEDTLS_SSL_PROTO_TLS1_3 */

#endif /* MBEDTLS_SSL_TLS13_INVASIVE_H */
