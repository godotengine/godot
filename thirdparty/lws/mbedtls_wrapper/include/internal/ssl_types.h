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

#ifndef _SSL_TYPES_H_
#define _SSL_TYPES_H_

#ifdef __cplusplus
 extern "C" {
#endif

#include <lws_config.h>
#if defined(LWS_WITH_ESP32)
#undef MBEDTLS_CONFIG_FILE
#define MBEDTLS_CONFIG_FILE <mbedtls/esp_config.h>
#endif

#include "ssl_code.h"

typedef void SSL_CIPHER;

typedef void X509_STORE_CTX;
typedef void X509_STORE;

typedef void RSA;

typedef void STACK;
typedef void BIO;

#define ossl_inline inline

#define SSL_METHOD_CALL(f, s, ...)        s->method->func->ssl_##f(s, ##__VA_ARGS__)
#define X509_METHOD_CALL(f, x, ...)       x->method->x509_##f(x, ##__VA_ARGS__)
#define EVP_PKEY_METHOD_CALL(f, k, ...)   k->method->pkey_##f(k, ##__VA_ARGS__)

typedef int (*OPENSSL_sk_compfunc)(const void *, const void *);

struct stack_st;
typedef struct stack_st OPENSSL_STACK;

struct ssl_method_st;
typedef struct ssl_method_st SSL_METHOD;

struct ssl_method_func_st;
typedef struct ssl_method_func_st SSL_METHOD_FUNC;

struct record_layer_st;
typedef struct record_layer_st RECORD_LAYER;

struct ossl_statem_st;
typedef struct ossl_statem_st OSSL_STATEM;

struct ssl_session_st;
typedef struct ssl_session_st SSL_SESSION;

struct ssl_ctx_st;
typedef struct ssl_ctx_st SSL_CTX;

struct ssl_st;
typedef struct ssl_st SSL;

struct cert_st;
typedef struct cert_st CERT;

struct x509_st;
typedef struct x509_st X509;

struct X509_VERIFY_PARAM_st;
typedef struct X509_VERIFY_PARAM_st X509_VERIFY_PARAM;

struct evp_pkey_st;
typedef struct evp_pkey_st EVP_PKEY;

struct x509_method_st;
typedef struct x509_method_st X509_METHOD;

struct pkey_method_st;
typedef struct pkey_method_st PKEY_METHOD;

struct stack_st {

    char **data;

    int num_alloc;

    OPENSSL_sk_compfunc c;
};

struct evp_pkey_st {

    void *pkey_pm;

    const PKEY_METHOD *method;
};

struct x509_st {

    /* X509 certification platform private point */
    void *x509_pm;

    const X509_METHOD *method;
};

struct cert_st {

    int sec_level;

    X509 *x509;

    EVP_PKEY *pkey;

};

struct ossl_statem_st {

    MSG_FLOW_STATE state;

    int hand_state;
};

struct record_layer_st {

    int rstate;

    int read_ahead;
};

struct ssl_session_st {

    long timeout;

    long time;

    X509 *peer;
};

struct X509_VERIFY_PARAM_st {

    int depth;

};

typedef int (*next_proto_cb)(SSL *ssl, unsigned char **out,
                             unsigned char *outlen, const unsigned char *in,
                             unsigned int inlen, void *arg);

struct ssl_ctx_st
{
    int version;

    int references;

    unsigned long options;

    const SSL_METHOD *method;

    CERT *cert;

    X509 *client_CA;

    const char **alpn_protos;

    next_proto_cb alpn_cb;

    int verify_mode;

    int (*default_verify_callback) (int ok, X509_STORE_CTX *ctx);

    long session_timeout;

    int read_ahead;

    int read_buffer_len;

    X509_VERIFY_PARAM param;
};

struct ssl_st
{
    /* protocol version(one of SSL3.0, TLS1.0, etc.) */
    int version;

    unsigned long options;

    /* shut things down(0x01 : sent, 0x02 : received) */
    int shutdown;

    CERT *cert;

    X509 *client_CA;

    SSL_CTX  *ctx;

    const SSL_METHOD *method;

    RECORD_LAYER rlayer;

    /* where we are */
    OSSL_STATEM statem;

    SSL_SESSION *session;

    int verify_mode;

    int (*verify_callback) (int ok, X509_STORE_CTX *ctx);

    int rwstate;
    int interrupted_remaining_write;

    long verify_result;

    X509_VERIFY_PARAM param;

    int err;

    void (*info_callback) (const SSL *ssl, int type, int val);

    /* SSL low-level system arch point */
    void *ssl_pm;
};

struct ssl_method_st {
    /* protocol version(one of SSL3.0, TLS1.0, etc.) */
    int version;

    /* SSL mode(client(0) , server(1), not known(-1)) */
    int endpoint;

    const SSL_METHOD_FUNC *func;
};

struct ssl_method_func_st {

    int (*ssl_new)(SSL *ssl);

    void (*ssl_free)(SSL *ssl);

    int (*ssl_handshake)(SSL *ssl);

    int (*ssl_shutdown)(SSL *ssl);

    int (*ssl_clear)(SSL *ssl);

    int (*ssl_read)(SSL *ssl, void *buffer, int len);

    int (*ssl_send)(SSL *ssl, const void *buffer, int len);

    int (*ssl_pending)(const SSL *ssl);

    void (*ssl_set_fd)(SSL *ssl, int fd, int mode);

    int (*ssl_get_fd)(const SSL *ssl, int mode);

    void (*ssl_set_bufflen)(SSL *ssl, int len);

    long (*ssl_get_verify_result)(const SSL *ssl);

    OSSL_HANDSHAKE_STATE (*ssl_get_state)(const SSL *ssl);
};

struct x509_method_st {

    int (*x509_new)(X509 *x, X509 *m_x);

    void (*x509_free)(X509 *x);

    int (*x509_load)(X509 *x, const unsigned char *buf, int len);

    int (*x509_show_info)(X509 *x);
};

struct pkey_method_st {

    int (*pkey_new)(EVP_PKEY *pkey, EVP_PKEY *m_pkey);

    void (*pkey_free)(EVP_PKEY *pkey);

    int (*pkey_load)(EVP_PKEY *pkey, const unsigned char *buf, int len);
};

#define OPENSSL_NPN_NEGOTIATED 1

#ifdef __cplusplus
}
#endif

#endif
