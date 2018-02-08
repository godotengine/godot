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

#ifndef _SSL_PKEY_H_
#define _SSL_PKEY_H_

#ifdef __cplusplus
 extern "C" {
#endif

#include "ssl_types.h"

/**
 * @brief create a private key object according to input private key
 *
 * @param ipk - input private key point
 *
 * @return new private key object point
 */
EVP_PKEY* __EVP_PKEY_new(EVP_PKEY *ipk);

/**
 * @brief create a private key object
 *
 * @param none
 *
 * @return private key object point
 */
EVP_PKEY* EVP_PKEY_new(void);

/**
 * @brief load a character key context into system context. If '*a' is pointed to the
 *        private key, then load key into it. Or create a new private key object
 *
 * @param type   - private key type
 * @param a      - a point pointed to a private key point
 * @param pp     - a point pointed to the key context memory point
 * @param length - key bytes
 *
 * @return private key object point
 */
EVP_PKEY* d2i_PrivateKey(int type,
                         EVP_PKEY **a,
                         const unsigned char **pp,
                         long length);

/**
 * @brief free a private key object
 *
 * @param pkey - private key object point
 *
 * @return none
 */
void EVP_PKEY_free(EVP_PKEY *x);

/**
 * @brief load private key into the SSL
 *
 * @param type - private key type
 * @param ssl  - SSL point
 * @param len  - data bytes
 * @param d    - data point
 *
 * @return result
 *     0 : failed
 *     1 : OK 
 */
 int SSL_use_PrivateKey_ASN1(int type, SSL *ssl, const unsigned char *d, long len);


#ifdef __cplusplus
}
#endif

#endif
