// Copyright 2015-2016 Espressif Systems (Shanghai) PTE LTD
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef _SSL_X509_H_
#define _SSL_X509_H_

#ifdef __cplusplus
 extern "C" {
#endif

#include "ssl_types.h"
#include "ssl_stack.h"

DEFINE_STACK_OF(X509_NAME)

/**
 * @brief create a X509 certification object according to input X509 certification
 *
 * @param ix - input X509 certification point
 *
 * @return new X509 certification object point
 */
X509* __X509_new(X509 *ix);

/**
 * @brief create a X509 certification object
 *
 * @param none
 *
 * @return X509 certification object point
 */
X509* X509_new(void);

/**
 * @brief load a character certification context into system context. If '*cert' is pointed to the
 *        certification, then load certification into it. Or create a new X509 certification object
 *
 * @param cert   - a point pointed to X509 certification
 * @param buffer - a point pointed to the certification context memory point
 * @param length - certification bytes
 *
 * @return X509 certification object point
 */
X509* d2i_X509(X509 **cert, const unsigned char *buffer, long len);

/**
 * @brief free a X509 certification object
 *
 * @param x - X509 certification object point
 *
 * @return none
 */
void X509_free(X509 *x);

/**
 * @brief set SSL context client CA certification
 *
 * @param ctx - SSL context point
 * @param x   - X509 certification point
 *
 * @return result
 *     0 : failed
 *     1 : OK
 */
int SSL_CTX_add_client_CA(SSL_CTX *ctx, X509 *x);

/**
 * @brief add CA client certification into the SSL
 *
 * @param ssl - SSL point
 * @param x   - X509 certification point
 *
 * @return result
 *     0 : failed
 *     1 : OK
 */
int SSL_add_client_CA(SSL *ssl, X509 *x);

/**
 * @brief load certification into the SSL
 *
 * @param ssl - SSL point
 * @param len - data bytes
 * @param d   - data point
 *
 * @return result
 *     0 : failed
 *     1 : OK
 *
 */
int SSL_use_certificate_ASN1(SSL *ssl, int len, const unsigned char *d);

const char *X509_verify_cert_error_string(long n);

#ifdef __cplusplus
}
#endif

#endif
