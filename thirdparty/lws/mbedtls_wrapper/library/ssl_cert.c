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

#include "ssl_cert.h"
#include "ssl_pkey.h"
#include "ssl_x509.h"
#include "ssl_dbg.h"
#include "ssl_port.h"

/**
 * @brief create a certification object according to input certification
 */
CERT *__ssl_cert_new(CERT *ic)
{
    CERT *cert;

    X509 *ix;
    EVP_PKEY *ipk;

    cert = ssl_mem_zalloc(sizeof(CERT));
    if (!cert) {
        SSL_DEBUG(SSL_CERT_ERROR_LEVEL, "no enough memory > (cert)");
        goto no_mem;
    }

    if (ic) {
        ipk = ic->pkey;
        ix = ic->x509;
    } else {
        ipk = NULL;
        ix = NULL;
    }

    cert->pkey = __EVP_PKEY_new(ipk);
    if (!cert->pkey) {
        SSL_DEBUG(SSL_CERT_ERROR_LEVEL, "__EVP_PKEY_new() return NULL");
        goto pkey_err;
    }

    cert->x509 = __X509_new(ix);
    if (!cert->x509) {
        SSL_DEBUG(SSL_CERT_ERROR_LEVEL, "__X509_new() return NULL");
        goto x509_err;
    }

    return cert;

x509_err:
    EVP_PKEY_free(cert->pkey);
pkey_err:
    ssl_mem_free(cert);
no_mem:
    return NULL;
}

/**
 * @brief create a certification object include private key object
 */
CERT *ssl_cert_new(void)
{
    return __ssl_cert_new(NULL);
}

/**
 * @brief free a certification object
 */
void ssl_cert_free(CERT *cert)
{
    SSL_ASSERT3(cert);

    X509_free(cert->x509);

    EVP_PKEY_free(cert->pkey);

    ssl_mem_free(cert);
}
