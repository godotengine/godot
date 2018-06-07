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

#include "ssl_methods.h"
#include "ssl_pm.h"

/**
 * TLS method function collection
 */
IMPLEMENT_TLS_METHOD_FUNC(TLS_method_func,
        ssl_pm_new, ssl_pm_free,
        ssl_pm_handshake, ssl_pm_shutdown, ssl_pm_clear,
        ssl_pm_read, ssl_pm_send, ssl_pm_pending,
        ssl_pm_set_fd, ssl_pm_get_fd,
        ssl_pm_set_bufflen,
        ssl_pm_get_verify_result,
        ssl_pm_get_state);

/**
 * TLS or SSL client method collection
 */
IMPLEMENT_TLS_METHOD(TLS_ANY_VERSION, 0, TLS_method_func, TLS_client_method);

IMPLEMENT_TLS_METHOD(TLS1_2_VERSION, 0, TLS_method_func, TLSv1_2_client_method);

IMPLEMENT_TLS_METHOD(TLS1_1_VERSION, 0, TLS_method_func, TLSv1_1_client_method);

IMPLEMENT_TLS_METHOD(TLS1_VERSION, 0, TLS_method_func, TLSv1_client_method);

IMPLEMENT_SSL_METHOD(SSL3_VERSION, 0, TLS_method_func, SSLv3_client_method);

/**
 * TLS or SSL server method collection
 */
IMPLEMENT_TLS_METHOD(TLS_ANY_VERSION, 1, TLS_method_func, TLS_server_method);

IMPLEMENT_TLS_METHOD(TLS1_1_VERSION, 1, TLS_method_func, TLSv1_1_server_method);

IMPLEMENT_TLS_METHOD(TLS1_2_VERSION, 1, TLS_method_func, TLSv1_2_server_method);

IMPLEMENT_TLS_METHOD(TLS1_VERSION, 0, TLS_method_func, TLSv1_server_method);

IMPLEMENT_SSL_METHOD(SSL3_VERSION, 1, TLS_method_func, SSLv3_server_method);

/**
 * TLS or SSL method collection
 */
IMPLEMENT_TLS_METHOD(TLS_ANY_VERSION, -1, TLS_method_func, TLS_method);

IMPLEMENT_SSL_METHOD(TLS1_2_VERSION, -1, TLS_method_func, TLSv1_2_method);

IMPLEMENT_SSL_METHOD(TLS1_1_VERSION, -1, TLS_method_func, TLSv1_1_method);

IMPLEMENT_SSL_METHOD(TLS1_VERSION, -1, TLS_method_func, TLSv1_method);

IMPLEMENT_SSL_METHOD(SSL3_VERSION, -1, TLS_method_func, SSLv3_method);

/**
 * @brief get X509 object method
 */
IMPLEMENT_X509_METHOD(X509_method,
            x509_pm_new, x509_pm_free,
            x509_pm_load, x509_pm_show_info);

/**
 * @brief get private key object method
 */
IMPLEMENT_PKEY_METHOD(EVP_PKEY_method,
            pkey_pm_new, pkey_pm_free,
            pkey_pm_load);
