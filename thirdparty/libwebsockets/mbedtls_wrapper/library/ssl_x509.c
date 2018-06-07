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

#include "ssl_x509.h"
#include "ssl_methods.h"
#include "ssl_dbg.h"
#include "ssl_port.h"

/**
 * @brief show X509 certification information
 */
int __X509_show_info(X509 *x)
{
    return X509_METHOD_CALL(show_info, x);
}

/**
 * @brief create a X509 certification object according to input X509 certification
 */
X509* __X509_new(X509 *ix)
{
    int ret;
    X509 *x;

    x = ssl_mem_zalloc(sizeof(X509));
    if (!x) {
        SSL_DEBUG(SSL_X509_ERROR_LEVEL, "no enough memory > (x)");
        goto no_mem;
    }

    if (ix)
        x->method = ix->method;
    else
        x->method = X509_method();

    ret = X509_METHOD_CALL(new, x, ix);
    if (ret) {
        SSL_DEBUG(SSL_PKEY_ERROR_LEVEL, "X509_METHOD_CALL(new) return %d", ret);
        goto failed;
    }

    return x;

failed:
    ssl_mem_free(x);
no_mem:
    return NULL;
}

/**
 * @brief create a X509 certification object
 */
X509* X509_new(void)
{
    return __X509_new(NULL);
}

/**
 * @brief free a X509 certification object
 */
void X509_free(X509 *x)
{
    SSL_ASSERT3(x);

    X509_METHOD_CALL(free, x);

    ssl_mem_free(x);
};

/**
 * @brief load a character certification context into system context. If '*cert' is pointed to the
 *        certification, then load certification into it. Or create a new X509 certification object
 */
X509* d2i_X509(X509 **cert, const unsigned char *buffer, long len)
{
    int m = 0;
    int ret;
    X509 *x;

    SSL_ASSERT2(buffer);
    SSL_ASSERT2(len);

    if (cert && *cert) {
        x = *cert;
    } else {
        x = X509_new();
        if (!x) {
            SSL_DEBUG(SSL_PKEY_ERROR_LEVEL, "X509_new() return NULL");
            goto failed1;
        }
        m = 1;
    }

    ret = X509_METHOD_CALL(load, x, buffer, len);
    if (ret) {
        SSL_DEBUG(SSL_PKEY_ERROR_LEVEL, "X509_METHOD_CALL(load) return %d", ret);
        goto failed2;
    }

    return x;

failed2:
    if (m)
        X509_free(x);
failed1:
    return NULL;
}

/**
 * @brief return SSL X509 verify parameters
 */

X509_VERIFY_PARAM *SSL_get0_param(SSL *ssl)
{
	return &ssl->param;
}

/**
 * @brief set X509 host verification flags
 */

int X509_VERIFY_PARAM_set_hostflags(X509_VERIFY_PARAM *param,
				    unsigned long flags)
{
	/* flags not supported yet */
	return 0;
}

/**
 * @brief clear X509 host verification flags
 */

int X509_VERIFY_PARAM_clear_hostflags(X509_VERIFY_PARAM *param,
				      unsigned long flags)
{
	/* flags not supported yet */
	return 0;
}

/**
 * @brief set SSL context client CA certification
 */
int SSL_CTX_add_client_CA(SSL_CTX *ctx, X509 *x)
{
    SSL_ASSERT1(ctx);
    SSL_ASSERT1(x);

    if (ctx->client_CA == x)
        return 1;

    X509_free(ctx->client_CA);

    ctx->client_CA = x;

    return 1;
}

/**
 * @brief add CA client certification into the SSL
 */
int SSL_add_client_CA(SSL *ssl, X509 *x)
{
    SSL_ASSERT1(ssl);
    SSL_ASSERT1(x);

    if (ssl->client_CA == x)
        return 1;

    X509_free(ssl->client_CA);

    ssl->client_CA = x;

    return 1;
}

/**
 * @brief set the SSL context certification
 */
int SSL_CTX_use_certificate(SSL_CTX *ctx, X509 *x)
{
    SSL_ASSERT1(ctx);
    SSL_ASSERT1(x);

    if (ctx->cert->x509 == x)
        return 1;

    X509_free(ctx->cert->x509);

    ctx->cert->x509 = x;

    return 1;
}

/**
 * @brief set the SSL certification
 */
int SSL_use_certificate(SSL *ssl, X509 *x)
{
    SSL_ASSERT1(ssl);
    SSL_ASSERT1(x);

    if (ssl->cert->x509 == x)
        return 1;

    X509_free(ssl->cert->x509);

    ssl->cert->x509 = x;

    return 1;
}

/**
 * @brief get the SSL certification point
 */
X509 *SSL_get_certificate(const SSL *ssl)
{
    SSL_ASSERT2(ssl);

    return ssl->cert->x509;
}

/**
 * @brief load certification into the SSL context
 */
int SSL_CTX_use_certificate_ASN1(SSL_CTX *ctx, int len,
                                 const unsigned char *d)
{
    int ret;
    X509 *x;

    x = d2i_X509(NULL, d, len);
    if (!x) {
        SSL_DEBUG(SSL_PKEY_ERROR_LEVEL, "d2i_X509() return NULL");
        goto failed1;
    }

    ret = SSL_CTX_use_certificate(ctx, x);
    if (!ret) {
        SSL_DEBUG(SSL_PKEY_ERROR_LEVEL, "SSL_CTX_use_certificate() return %d", ret);
        goto failed2;
    }

    return 1;

failed2:
    X509_free(x);
failed1:
    return 0;
}

/**
 * @brief load certification into the SSL
 */
int SSL_use_certificate_ASN1(SSL *ssl, int len,
                             const unsigned char *d)
{
    int ret;
    X509 *x;

    x = d2i_X509(NULL, d, len);
    if (!x) {
        SSL_DEBUG(SSL_PKEY_ERROR_LEVEL, "d2i_X509() return NULL");
        goto failed1;
    }

    ret = SSL_use_certificate(ssl, x);
    if (!ret) {
        SSL_DEBUG(SSL_PKEY_ERROR_LEVEL, "SSL_use_certificate() return %d", ret);
        goto failed2;
    }

    return 1;

failed2:
    X509_free(x);
failed1:
    return 0;
}

/**
 * @brief load the certification file into SSL context
 */
int SSL_CTX_use_certificate_file(SSL_CTX *ctx, const char *file, int type)
{
    return 0;
}

/**
 * @brief load the certification file into SSL
 */
int SSL_use_certificate_file(SSL *ssl, const char *file, int type)
{
    return 0;
}

/**
 * @brief get peer certification
 */
X509 *SSL_get_peer_certificate(const SSL *ssl)
{
    SSL_ASSERT2(ssl);

    return ssl->session->peer;
}

int X509_STORE_CTX_get_error(X509_STORE_CTX *ctx)
{
	return X509_V_ERR_UNSPECIFIED;
}

int X509_STORE_CTX_get_error_depth(X509_STORE_CTX *ctx)
{
	return 0;
}

const char *X509_verify_cert_error_string(long n)
{
	return "unknown";
}
