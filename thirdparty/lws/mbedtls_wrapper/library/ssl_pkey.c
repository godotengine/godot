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

#include "ssl_pkey.h"
#include "ssl_methods.h"
#include "ssl_dbg.h"
#include "ssl_port.h"

/**
 * @brief create a private key object according to input private key
 */
EVP_PKEY* __EVP_PKEY_new(EVP_PKEY *ipk)
{
    int ret;
    EVP_PKEY *pkey;

    pkey = ssl_mem_zalloc(sizeof(EVP_PKEY));
    if (!pkey) {
        SSL_DEBUG(SSL_PKEY_ERROR_LEVEL, "no enough memory > (pkey)");
        goto no_mem;
    }

    if (ipk) {
        pkey->method = ipk->method;
    } else {
        pkey->method = EVP_PKEY_method();
    }

    ret = EVP_PKEY_METHOD_CALL(new, pkey, ipk);
    if (ret) {
        SSL_DEBUG(SSL_PKEY_ERROR_LEVEL, "EVP_PKEY_METHOD_CALL(new) return %d", ret);
        goto failed;
    }

    return pkey;

failed:
    ssl_mem_free(pkey);
no_mem:
    return NULL;
}

/**
 * @brief create a private key object
 */
EVP_PKEY* EVP_PKEY_new(void)
{
    return __EVP_PKEY_new(NULL);
}

/**
 * @brief free a private key object
 */
void EVP_PKEY_free(EVP_PKEY *pkey)
{
    SSL_ASSERT3(pkey);

    EVP_PKEY_METHOD_CALL(free, pkey);

    ssl_mem_free(pkey);
}

/**
 * @brief load a character key context into system context. If '*a' is pointed to the
 *        private key, then load key into it. Or create a new private key object
 */
EVP_PKEY *d2i_PrivateKey(int type,
                         EVP_PKEY **a,
                         const unsigned char **pp,
                         long length)
{
    int m = 0;
    int ret;
    EVP_PKEY *pkey;

    SSL_ASSERT2(pp);
    SSL_ASSERT2(*pp);
    SSL_ASSERT2(length);

    if (a && *a) {
        pkey = *a;
    } else {
        pkey = EVP_PKEY_new();;
        if (!pkey) {
            SSL_DEBUG(SSL_PKEY_ERROR_LEVEL, "EVP_PKEY_new() return NULL");
            goto failed1;
        }

        m = 1;
    }

    ret = EVP_PKEY_METHOD_CALL(load, pkey, *pp, length);
    if (ret) {
        SSL_DEBUG(SSL_PKEY_ERROR_LEVEL, "EVP_PKEY_METHOD_CALL(load) return %d", ret);
        goto failed2;
    }

    if (a)
        *a = pkey;

    return pkey;

failed2:
    if (m)
        EVP_PKEY_free(pkey);
failed1:
    return NULL;
}

/**
 * @brief set the SSL context private key
 */
int SSL_CTX_use_PrivateKey(SSL_CTX *ctx, EVP_PKEY *pkey)
{
    SSL_ASSERT1(ctx);
    SSL_ASSERT1(pkey);

    if (ctx->cert->pkey == pkey)
        return 1;

    if (ctx->cert->pkey)
        EVP_PKEY_free(ctx->cert->pkey);

    ctx->cert->pkey = pkey;

    return 1;
}

/**
 * @brief set the SSL private key
 */
int SSL_use_PrivateKey(SSL *ssl, EVP_PKEY *pkey)
{
    SSL_ASSERT1(ssl);
    SSL_ASSERT1(pkey);

    if (ssl->cert->pkey == pkey)
        return 1;

    if (ssl->cert->pkey)
        EVP_PKEY_free(ssl->cert->pkey);

    ssl->cert->pkey = pkey;

    return 1;
}

/**
 * @brief load private key into the SSL context
 */
int SSL_CTX_use_PrivateKey_ASN1(int type, SSL_CTX *ctx,
                                const unsigned char *d, long len)
{
    int ret;
    EVP_PKEY *pk;

    pk = d2i_PrivateKey(0, NULL, &d, len);
    if (!pk) {
        SSL_DEBUG(SSL_PKEY_ERROR_LEVEL, "d2i_PrivateKey() return NULL");
        goto failed1;
    }

    ret = SSL_CTX_use_PrivateKey(ctx, pk);
    if (!ret) {
        SSL_DEBUG(SSL_PKEY_ERROR_LEVEL, "SSL_CTX_use_PrivateKey() return %d", ret);
        goto failed2;
    }

    return 1;

failed2:
    EVP_PKEY_free(pk);
failed1:
    return 0;
}

/**
 * @brief load private key into the SSL
 */
int SSL_use_PrivateKey_ASN1(int type, SSL *ssl,
                                const unsigned char *d, long len)
{
    int ret;
    EVP_PKEY *pk;

    pk = d2i_PrivateKey(0, NULL, &d, len);
    if (!pk) {
        SSL_DEBUG(SSL_PKEY_ERROR_LEVEL, "d2i_PrivateKey() return NULL");
        goto failed1;
    }

    ret = SSL_use_PrivateKey(ssl, pk);
    if (!ret) {
        SSL_DEBUG(SSL_PKEY_ERROR_LEVEL, "SSL_use_PrivateKey() return %d", ret);
        goto failed2;
    }

    return 1;

failed2:
    EVP_PKEY_free(pk);
failed1:
    return 0;
}

/**
 * @brief load the private key file into SSL context
 */
int SSL_CTX_use_PrivateKey_file(SSL_CTX *ctx, const char *file, int type)
{
    return 0;
}

/**
 * @brief load the private key file into SSL
 */
int SSL_use_PrivateKey_file(SSL_CTX *ctx, const char *file, int type)
{
    return 0;
}

/**
 * @brief load the RSA ASN1 private key into SSL context
 */
int SSL_CTX_use_RSAPrivateKey_ASN1(SSL_CTX *ctx, const unsigned char *d, long len)
{
    return SSL_CTX_use_PrivateKey_ASN1(0, ctx, d, len);
}
