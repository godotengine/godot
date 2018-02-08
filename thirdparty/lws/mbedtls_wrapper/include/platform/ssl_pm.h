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

#ifndef _SSL_PM_H_
#define _SSL_PM_H_

#ifdef __cplusplus
 extern "C" {
#endif

#include <string.h>
#include "ssl_types.h"
#include "ssl_port.h"

#define LOCAL_ATRR

int ssl_pm_new(SSL *ssl);
void ssl_pm_free(SSL *ssl);

int ssl_pm_handshake(SSL *ssl);
int ssl_pm_shutdown(SSL *ssl);
int ssl_pm_clear(SSL *ssl);

int ssl_pm_read(SSL *ssl, void *buffer, int len);
int ssl_pm_send(SSL *ssl, const void *buffer, int len);
int ssl_pm_pending(const SSL *ssl);

void ssl_pm_set_fd(SSL *ssl, int fd, int mode);
int ssl_pm_get_fd(const SSL *ssl, int mode);

OSSL_HANDSHAKE_STATE ssl_pm_get_state(const SSL *ssl);

void ssl_pm_set_bufflen(SSL *ssl, int len);

int x509_pm_show_info(X509 *x);
int x509_pm_new(X509 *x, X509 *m_x);
void x509_pm_free(X509 *x);
int x509_pm_load(X509 *x, const unsigned char *buffer, int len);

int pkey_pm_new(EVP_PKEY *pk, EVP_PKEY *m_pk);
void pkey_pm_free(EVP_PKEY *pk);
int pkey_pm_load(EVP_PKEY *pk, const unsigned char *buffer, int len);

long ssl_pm_get_verify_result(const SSL *ssl);

#ifdef __cplusplus
 }
#endif

#endif
