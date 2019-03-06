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

#ifndef _SSL_CERT_H_
#define _SSL_CERT_H_

#ifdef __cplusplus
 extern "C" {
#endif

#include "ssl_types.h"

/**
 * @brief create a certification object include private key object according to input certification
 *
 * @param ic - input certification point
 *
 * @return certification object point
 */
CERT *__ssl_cert_new(CERT *ic);

/**
 * @brief create a certification object include private key object
 *
 * @param none
 *
 * @return certification object point
 */
CERT* ssl_cert_new(void);

/**
 * @brief free a certification object
 *
 * @param cert - certification object point
 *
 * @return none
 */
void ssl_cert_free(CERT *cert);

#ifdef __cplusplus
}
#endif

#endif
