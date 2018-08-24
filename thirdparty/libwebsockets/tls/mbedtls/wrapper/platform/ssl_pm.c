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

#include "ssl_pm.h"
#include "ssl_port.h"
#include "ssl_dbg.h"

/* mbedtls include */
#include "mbedtls/platform.h"
#include "mbedtls/net_sockets.h"
#include "mbedtls/debug.h"
#include "mbedtls/entropy.h"
#include "mbedtls/ctr_drbg.h"
#include "mbedtls/error.h"
#include "mbedtls/certs.h"

#include <libwebsockets.h>

#define X509_INFO_STRING_LENGTH 8192

struct ssl_pm
{
    /* local socket file description */
    mbedtls_net_context fd;
    /* remote client socket file description */
    mbedtls_net_context cl_fd;

    mbedtls_ssl_config conf;

    mbedtls_ctr_drbg_context ctr_drbg;

    mbedtls_ssl_context ssl;

    mbedtls_entropy_context entropy;

    SSL *owner;
};

struct x509_pm
{
    mbedtls_x509_crt *x509_crt;

    mbedtls_x509_crt *ex_crt;
};

struct pkey_pm
{
    mbedtls_pk_context *pkey;

    mbedtls_pk_context *ex_pkey;
};

unsigned int max_content_len;

/*********************************************************************************************/
/************************************ SSL arch interface *************************************/

//#ifdef CONFIG_OPENSSL_LOWLEVEL_DEBUG

/* mbedtls debug level */
#define MBEDTLS_DEBUG_LEVEL 4

/**
 * @brief mbedtls debug function
 */
static void ssl_platform_debug(void *ctx, int level,
                     const char *file, int line,
                     const char *str)
{
    /* Shorten 'file' from the whole file path to just the filename

       This is a bit wasteful because the macros are compiled in with
       the full _FILE_ path in each case.
    */
//    char *file_sep = rindex(file, '/');
  //  if(file_sep)
    //    file = file_sep + 1;

    printf("%s:%d %s", file, line, str);
}
//#endif

/**
 * @brief create SSL low-level object
 */
int ssl_pm_new(SSL *ssl)
{
    struct ssl_pm *ssl_pm;
    int ret;

    const unsigned char pers[] = "OpenSSL PM";
    size_t pers_len = sizeof(pers);

    int endpoint;
    int version;

    const SSL_METHOD *method = ssl->method;

    ssl_pm = ssl_mem_zalloc(sizeof(struct ssl_pm));
    if (!ssl_pm) {
        SSL_DEBUG(SSL_PLATFORM_ERROR_LEVEL, "no enough memory > (ssl_pm)");
        goto no_mem;
    }

    ssl_pm->owner = ssl;

    if (!ssl->ctx->read_buffer_len)
	    ssl->ctx->read_buffer_len = 2048;

    max_content_len = ssl->ctx->read_buffer_len;
    // printf("ssl->ctx->read_buffer_len = %d ++++++++++++++++++++\n", ssl->ctx->read_buffer_len);

    mbedtls_net_init(&ssl_pm->fd);
    mbedtls_net_init(&ssl_pm->cl_fd);

    mbedtls_ssl_config_init(&ssl_pm->conf);
    mbedtls_ctr_drbg_init(&ssl_pm->ctr_drbg);
    mbedtls_entropy_init(&ssl_pm->entropy);
    mbedtls_ssl_init(&ssl_pm->ssl);

    ret = mbedtls_ctr_drbg_seed(&ssl_pm->ctr_drbg, mbedtls_entropy_func, &ssl_pm->entropy, pers, pers_len);
    if (ret) {
        SSL_DEBUG(SSL_PLATFORM_ERROR_LEVEL, "mbedtls_ctr_drbg_seed() return -0x%x", -ret);
        goto mbedtls_err1;
    }

    if (method->endpoint) {
        endpoint = MBEDTLS_SSL_IS_SERVER;
    } else {
        endpoint = MBEDTLS_SSL_IS_CLIENT;
    }
    ret = mbedtls_ssl_config_defaults(&ssl_pm->conf, endpoint, MBEDTLS_SSL_TRANSPORT_STREAM, MBEDTLS_SSL_PRESET_DEFAULT);
    if (ret) {
        SSL_DEBUG(SSL_PLATFORM_ERROR_LEVEL, "mbedtls_ssl_config_defaults() return -0x%x", -ret);
        goto mbedtls_err2;
    }

    if (TLS_ANY_VERSION != ssl->version) {
        if (TLS1_2_VERSION == ssl->version)
            version = MBEDTLS_SSL_MINOR_VERSION_3;
        else if (TLS1_1_VERSION == ssl->version)
            version = MBEDTLS_SSL_MINOR_VERSION_2;
        else if (TLS1_VERSION == ssl->version)
            version = MBEDTLS_SSL_MINOR_VERSION_1;
        else
            version = MBEDTLS_SSL_MINOR_VERSION_0;

        mbedtls_ssl_conf_max_version(&ssl_pm->conf, MBEDTLS_SSL_MAJOR_VERSION_3, version);
        mbedtls_ssl_conf_min_version(&ssl_pm->conf, MBEDTLS_SSL_MAJOR_VERSION_3, version);
    } else {
        mbedtls_ssl_conf_max_version(&ssl_pm->conf, MBEDTLS_SSL_MAJOR_VERSION_3, MBEDTLS_SSL_MINOR_VERSION_3);
        mbedtls_ssl_conf_min_version(&ssl_pm->conf, MBEDTLS_SSL_MAJOR_VERSION_3, MBEDTLS_SSL_MINOR_VERSION_0);
    }

    mbedtls_ssl_conf_rng(&ssl_pm->conf, mbedtls_ctr_drbg_random, &ssl_pm->ctr_drbg);

//#ifdef CONFIG_OPENSSL_LOWLEVEL_DEBUG
 //   mbedtls_debug_set_threshold(MBEDTLS_DEBUG_LEVEL);
//    mbedtls_ssl_conf_dbg(&ssl_pm->conf, ssl_platform_debug, NULL);
//#else
    mbedtls_ssl_conf_dbg(&ssl_pm->conf, ssl_platform_debug, NULL);
//#endif

    ret = mbedtls_ssl_setup(&ssl_pm->ssl, &ssl_pm->conf);
    if (ret) {
        SSL_DEBUG(SSL_PLATFORM_ERROR_LEVEL, "mbedtls_ssl_setup() return -0x%x", -ret);
        goto mbedtls_err2;
    }

    mbedtls_ssl_set_bio(&ssl_pm->ssl, &ssl_pm->fd, mbedtls_net_send, mbedtls_net_recv, NULL);

    ssl->ssl_pm = ssl_pm;

    return 0;

mbedtls_err2:
    mbedtls_ssl_config_free(&ssl_pm->conf);
    mbedtls_ctr_drbg_free(&ssl_pm->ctr_drbg);
mbedtls_err1:
    mbedtls_entropy_free(&ssl_pm->entropy);
    ssl_mem_free(ssl_pm);
no_mem:
    return -1;
}

/**
 * @brief free SSL low-level object
 */
void ssl_pm_free(SSL *ssl)
{
    struct ssl_pm *ssl_pm = (struct ssl_pm *)ssl->ssl_pm;

    mbedtls_ctr_drbg_free(&ssl_pm->ctr_drbg);
    mbedtls_entropy_free(&ssl_pm->entropy);
    mbedtls_ssl_config_free(&ssl_pm->conf);
    mbedtls_ssl_free(&ssl_pm->ssl);

    ssl_mem_free(ssl_pm);
    ssl->ssl_pm = NULL;
}

/**
 * @brief reload SSL low-level certification object
 */
static int ssl_pm_reload_crt(SSL *ssl)
{
    int ret;
    int mode;
    struct ssl_pm *ssl_pm = ssl->ssl_pm;
    struct x509_pm *ca_pm = (struct x509_pm *)ssl->client_CA->x509_pm;

    struct pkey_pm *pkey_pm = (struct pkey_pm *)ssl->cert->pkey->pkey_pm;
    struct x509_pm *crt_pm = (struct x509_pm *)ssl->cert->x509->x509_pm;

    if (ssl->verify_mode == SSL_VERIFY_PEER)
        mode = MBEDTLS_SSL_VERIFY_OPTIONAL;
    else if (ssl->verify_mode == SSL_VERIFY_FAIL_IF_NO_PEER_CERT)
        mode = MBEDTLS_SSL_VERIFY_OPTIONAL;
    else if (ssl->verify_mode == SSL_VERIFY_CLIENT_ONCE)
        mode = MBEDTLS_SSL_VERIFY_UNSET;
    else
        mode = MBEDTLS_SSL_VERIFY_NONE;

    mbedtls_ssl_conf_authmode(&ssl_pm->conf, mode);

    if (ca_pm->x509_crt) {
        mbedtls_ssl_conf_ca_chain(&ssl_pm->conf, ca_pm->x509_crt, NULL);
    } else if (ca_pm->ex_crt) {
        mbedtls_ssl_conf_ca_chain(&ssl_pm->conf, ca_pm->ex_crt, NULL);
    }

    if (crt_pm->x509_crt && pkey_pm->pkey) {
        ret = mbedtls_ssl_conf_own_cert(&ssl_pm->conf, crt_pm->x509_crt, pkey_pm->pkey);
    } else if (crt_pm->ex_crt && pkey_pm->ex_pkey) {
        ret = mbedtls_ssl_conf_own_cert(&ssl_pm->conf, crt_pm->ex_crt, pkey_pm->ex_pkey);
    } else {
        ret = 0;
    }

    if (ret) {
        SSL_DEBUG(SSL_PLATFORM_ERROR_LEVEL, "mbedtls_ssl_conf_own_cert() return -0x%x", -ret);
        ret = -1;
    }

    return ret;
}

/*
 * Perform the mbedtls SSL handshake instead of mbedtls_ssl_handshake.
 * We can add debug here.
 */
static int mbedtls_handshake( mbedtls_ssl_context *ssl )
{
    int ret = 0;

    while (ssl->state != MBEDTLS_SSL_HANDSHAKE_OVER) {
        ret = mbedtls_ssl_handshake_step(ssl);

        lwsl_info("%s: ssl ret -%x state %d\n", __func__, -ret, ssl->state);

        if (ret != 0)
            break;
    }

    return ret;
}

#include <errno.h>

int ssl_pm_handshake(SSL *ssl)
{
    int ret;
    struct ssl_pm *ssl_pm = (struct ssl_pm *)ssl->ssl_pm;

    ssl->err = 0;
    errno = 0;

    ret = ssl_pm_reload_crt(ssl);
    if (ret) {
	    printf("%s: cert reload failed\n", __func__);
        return 0;
    }

    if (ssl_pm->ssl.state != MBEDTLS_SSL_HANDSHAKE_OVER) {
	    ssl_speed_up_enter();

	   /* mbedtls return codes
	    * 0 = successful, or MBEDTLS_ERR_SSL_WANT_READ/WRITE
	    * anything else = death
	    */
	    ret = mbedtls_handshake(&ssl_pm->ssl);
	    ssl_speed_up_exit();
    } else
	    ret = 0;

    /*
     * OpenSSL return codes:
     *   0 = did not complete, but may be retried
     *   1 = successfully completed
     *   <0 = death
     */
    if (ret == MBEDTLS_ERR_SSL_WANT_READ || ret == MBEDTLS_ERR_SSL_WANT_WRITE) {
	    ssl->err = ret;
        SSL_DEBUG(SSL_PLATFORM_ERROR_LEVEL, "mbedtls_ssl_handshake() return -0x%x", -ret);
        return 0; /* OpenSSL: did not complete but may be retried */
    }

    if (ret == 0) { /* successful */
        struct x509_pm *x509_pm = (struct x509_pm *)ssl->session->peer->x509_pm;

        x509_pm->ex_crt = (mbedtls_x509_crt *)mbedtls_ssl_get_peer_cert(&ssl_pm->ssl);
        return 1; /* openssl successful */
    }

    if (errno == 11) {
	    ssl->err = ret == MBEDTLS_ERR_SSL_WANT_READ;

	    return 0;
    }

    printf("%s: mbedtls_ssl_handshake() returned -0x%x\n", __func__, -ret);

    /* it's had it */

    ssl->err = SSL_ERROR_SYSCALL;

    return -1; /* openssl death */
}

mbedtls_x509_crt *
ssl_ctx_get_mbedtls_x509_crt(SSL_CTX *ssl_ctx)
{
	struct x509_pm *x509_pm = (struct x509_pm *)ssl_ctx->cert->x509->x509_pm;

	if (!x509_pm)
		return NULL;

	return x509_pm->x509_crt;
}

mbedtls_x509_crt *
ssl_get_peer_mbedtls_x509_crt(SSL *ssl)
{
	struct x509_pm *x509_pm = (struct x509_pm *)ssl->session->peer->x509_pm;

	if (!x509_pm)
		return NULL;

	return x509_pm->ex_crt;
}

int ssl_pm_shutdown(SSL *ssl)
{
    int ret;
    struct ssl_pm *ssl_pm = (struct ssl_pm *)ssl->ssl_pm;

    ret = mbedtls_ssl_close_notify(&ssl_pm->ssl);
    if (ret) {
        SSL_DEBUG(SSL_PLATFORM_ERROR_LEVEL, "mbedtls_ssl_close_notify() return -0x%x", -ret);
        if (ret == MBEDTLS_ERR_NET_CONN_RESET)
		ssl->err = SSL_ERROR_SYSCALL;
	 ret = -1; /* OpenSSL: "Call SSL_get_error with the return value to find the reason */
    } else {
        struct x509_pm *x509_pm = (struct x509_pm *)ssl->session->peer->x509_pm;

        x509_pm->ex_crt = NULL;
        ret = 1; /* OpenSSL: "The shutdown was successfully completed"
		     ...0 means retry */
    }

    return ret;
}

int ssl_pm_clear(SSL *ssl)
{
    return ssl_pm_shutdown(ssl);
}


int ssl_pm_read(SSL *ssl, void *buffer, int len)
{
    int ret;
    struct ssl_pm *ssl_pm = (struct ssl_pm *)ssl->ssl_pm;

    ret = mbedtls_ssl_read(&ssl_pm->ssl, buffer, len);
    if (ret < 0) {
	 //   lwsl_notice("%s: mbedtls_ssl_read says -0x%x\n", __func__, -ret);
        SSL_DEBUG(SSL_PLATFORM_ERROR_LEVEL, "mbedtls_ssl_read() return -0x%x", -ret);
        if (ret == MBEDTLS_ERR_NET_CONN_RESET ||
            ret <= MBEDTLS_ERR_SSL_NO_USABLE_CIPHERSUITE) /* fatal errors */
		ssl->err = SSL_ERROR_SYSCALL;
        ret = -1;
    }

    return ret;
}

/*
 * This returns -1, or the length sent.
 * If -1, then you need to find out if the error was
 * fatal or recoverable using SSL_get_error()
 */
int ssl_pm_send(SSL *ssl, const void *buffer, int len)
{
    int ret;
    struct ssl_pm *ssl_pm = (struct ssl_pm *)ssl->ssl_pm;

    ret = mbedtls_ssl_write(&ssl_pm->ssl, buffer, len);
    /*
     * We can get a positive number, which may be less than len... that
     * much was sent successfully and you can call again to send more.
     *
     * We can get a negative mbedtls error code... if WANT_WRITE or WANT_READ,
     * it's nonfatal and means it should be retried as-is.  If something else,
     * it's fatal actually.
     *
     * If this function returns something other than a positive value or
     * MBEDTLS_ERR_SSL_WANT_READ/WRITE, the ssl context becomes unusable, and
     * you should either free it or call mbedtls_ssl_session_reset() on it
     * before re-using it for a new connection; the current connection must
     * be closed.
     *
     * When this function returns MBEDTLS_ERR_SSL_WANT_WRITE/READ, it must be
     * called later with the same arguments, until it returns a positive value.
     */

    if (ret < 0) {
	    SSL_DEBUG(SSL_PLATFORM_ERROR_LEVEL, "mbedtls_ssl_write() return -0x%x", -ret);
	switch (ret) {
	case MBEDTLS_ERR_NET_SEND_FAILED:
	case MBEDTLS_ERR_NET_CONN_RESET:
		ssl->err = SSL_ERROR_SYSCALL;
		break;
	case MBEDTLS_ERR_SSL_WANT_WRITE:
		ssl->err = SSL_ERROR_WANT_WRITE;
		break;
	case MBEDTLS_ERR_SSL_WANT_READ:
		ssl->err = SSL_ERROR_WANT_READ;
		break;
	default:
		break;
	}

	ret = -1;
    }

    return ret;
}

int ssl_pm_pending(const SSL *ssl)
{
    struct ssl_pm *ssl_pm = (struct ssl_pm *)ssl->ssl_pm;

    return mbedtls_ssl_get_bytes_avail(&ssl_pm->ssl);
}

void ssl_pm_set_fd(SSL *ssl, int fd, int mode)
{
    struct ssl_pm *ssl_pm = (struct ssl_pm *)ssl->ssl_pm;

    ssl_pm->fd.fd = fd;
}

int ssl_pm_get_fd(const SSL *ssl, int mode)
{
    struct ssl_pm *ssl_pm = (struct ssl_pm *)ssl->ssl_pm;

    return ssl_pm->fd.fd;
}

OSSL_HANDSHAKE_STATE ssl_pm_get_state(const SSL *ssl)
{
    OSSL_HANDSHAKE_STATE state;

    struct ssl_pm *ssl_pm = (struct ssl_pm *)ssl->ssl_pm;

    switch (ssl_pm->ssl.state)
    {
        case MBEDTLS_SSL_CLIENT_HELLO:
            state = TLS_ST_CW_CLNT_HELLO;
            break;
        case MBEDTLS_SSL_SERVER_HELLO:
            state = TLS_ST_SW_SRVR_HELLO;
            break;
        case MBEDTLS_SSL_SERVER_CERTIFICATE:
            state = TLS_ST_SW_CERT;
            break;
        case MBEDTLS_SSL_SERVER_HELLO_DONE:
            state = TLS_ST_SW_SRVR_DONE;
            break;
        case MBEDTLS_SSL_CLIENT_KEY_EXCHANGE:
            state = TLS_ST_CW_KEY_EXCH;
            break;
        case MBEDTLS_SSL_CLIENT_CHANGE_CIPHER_SPEC:
            state = TLS_ST_CW_CHANGE;
            break;
        case MBEDTLS_SSL_CLIENT_FINISHED:
            state = TLS_ST_CW_FINISHED;
            break;
        case MBEDTLS_SSL_SERVER_CHANGE_CIPHER_SPEC:
            state = TLS_ST_SW_CHANGE;
            break;
        case MBEDTLS_SSL_SERVER_FINISHED:
            state = TLS_ST_SW_FINISHED;
            break;
        case MBEDTLS_SSL_CLIENT_CERTIFICATE:
            state = TLS_ST_CW_CERT;
            break;
        case MBEDTLS_SSL_SERVER_KEY_EXCHANGE:
            state = TLS_ST_SR_KEY_EXCH;
            break;
        case MBEDTLS_SSL_SERVER_NEW_SESSION_TICKET:
            state = TLS_ST_SW_SESSION_TICKET;
            break;
        case MBEDTLS_SSL_SERVER_HELLO_VERIFY_REQUEST_SENT:
            state = TLS_ST_SW_CERT_REQ;
            break;
        case MBEDTLS_SSL_HANDSHAKE_OVER:
            state = TLS_ST_OK;
            break;
        default :
            state = TLS_ST_BEFORE;
            break;
    }

    return state;
}

int x509_pm_show_info(X509 *x)
{
    int ret;
    char *buf;
    mbedtls_x509_crt *x509_crt;
    struct x509_pm *x509_pm = x->x509_pm;

    if (x509_pm->x509_crt)
        x509_crt = x509_pm->x509_crt;
    else if (x509_pm->ex_crt)
        x509_crt = x509_pm->ex_crt;
    else
        x509_crt = NULL;

    if (!x509_crt)
        return -1;

    buf = ssl_mem_malloc(X509_INFO_STRING_LENGTH);
    if (!buf) {
        SSL_DEBUG(SSL_PLATFORM_ERROR_LEVEL, "no enough memory > (buf)");
        goto no_mem;
    }

    ret = mbedtls_x509_crt_info(buf, X509_INFO_STRING_LENGTH - 1, "", x509_crt);
    if (ret <= 0) {
        SSL_DEBUG(SSL_PLATFORM_ERROR_LEVEL, "mbedtls_x509_crt_info() return -0x%x", -ret);
        goto mbedtls_err1;
    }

    buf[ret] = 0;

    ssl_mem_free(buf);

    SSL_DEBUG(SSL_DEBUG_ON, "%s", buf);

    return 0;

mbedtls_err1:
    ssl_mem_free(buf);
no_mem:
    return -1;
}

int x509_pm_new(X509 *x, X509 *m_x)
{
    struct x509_pm *x509_pm;

    x509_pm = ssl_mem_zalloc(sizeof(struct x509_pm));
    if (!x509_pm) {
        SSL_DEBUG(SSL_PLATFORM_ERROR_LEVEL, "no enough memory > (x509_pm)");
        goto failed1;
    }

    x->x509_pm = x509_pm;

    if (m_x) {
        struct x509_pm *m_x509_pm = (struct x509_pm *)m_x->x509_pm;

        x509_pm->ex_crt = m_x509_pm->x509_crt;
    }

    return 0;

failed1:
    return -1;
}

void x509_pm_free(X509 *x)
{
    struct x509_pm *x509_pm = (struct x509_pm *)x->x509_pm;

    if (x509_pm->x509_crt) {
        mbedtls_x509_crt_free(x509_pm->x509_crt);

        ssl_mem_free(x509_pm->x509_crt);
        x509_pm->x509_crt = NULL;
    }

    ssl_mem_free(x->x509_pm);
    x->x509_pm = NULL;
}

int x509_pm_load(X509 *x, const unsigned char *buffer, int len)
{
    int ret;
    unsigned char *load_buf;
    struct x509_pm *x509_pm = (struct x509_pm *)x->x509_pm;

	if (x509_pm->x509_crt)
        mbedtls_x509_crt_free(x509_pm->x509_crt);

    if (!x509_pm->x509_crt) {
        x509_pm->x509_crt = ssl_mem_malloc(sizeof(mbedtls_x509_crt));
        if (!x509_pm->x509_crt) {
            SSL_DEBUG(SSL_PLATFORM_ERROR_LEVEL, "no enough memory > (x509_pm->x509_crt)");
            goto no_mem;
        }
    }

    mbedtls_x509_crt_init(x509_pm->x509_crt);
    if (buffer[0] != 0x30) {
	    load_buf = ssl_mem_malloc(len + 1);
	    if (!load_buf) {
		SSL_DEBUG(SSL_PLATFORM_ERROR_LEVEL, "no enough memory > (load_buf)");
		goto failed;
	    }

	    ssl_memcpy(load_buf, buffer, len);
	    load_buf[len] = '\0';

	    ret = mbedtls_x509_crt_parse(x509_pm->x509_crt, load_buf, len + 1);
	    ssl_mem_free(load_buf);
    } else {
	    printf("parsing as der\n");

	    ret = mbedtls_x509_crt_parse_der(x509_pm->x509_crt, buffer, len);
    }

    if (ret) {
        printf("mbedtls_x509_crt_parse return -0x%x", -ret);
        goto failed;
    }

    return 0;

failed:
    mbedtls_x509_crt_free(x509_pm->x509_crt);
    ssl_mem_free(x509_pm->x509_crt);
    x509_pm->x509_crt = NULL;
no_mem:
    return -1;
}

int pkey_pm_new(EVP_PKEY *pk, EVP_PKEY *m_pkey)
{
    struct pkey_pm *pkey_pm;

    pkey_pm = ssl_mem_zalloc(sizeof(struct pkey_pm));
    if (!pkey_pm)
        return -1;

    pk->pkey_pm = pkey_pm;

    if (m_pkey) {
        struct pkey_pm *m_pkey_pm = (struct pkey_pm *)m_pkey->pkey_pm;

        pkey_pm->ex_pkey = m_pkey_pm->pkey;
    }

    return 0;
}

void pkey_pm_free(EVP_PKEY *pk)
{
    struct pkey_pm *pkey_pm = (struct pkey_pm *)pk->pkey_pm;

    if (pkey_pm->pkey) {
        mbedtls_pk_free(pkey_pm->pkey);

        ssl_mem_free(pkey_pm->pkey);
        pkey_pm->pkey = NULL;
    }

    ssl_mem_free(pk->pkey_pm);
    pk->pkey_pm = NULL;
}

int pkey_pm_load(EVP_PKEY *pk, const unsigned char *buffer, int len)
{
    int ret;
    unsigned char *load_buf;
    struct pkey_pm *pkey_pm = (struct pkey_pm *)pk->pkey_pm;

    if (pkey_pm->pkey)
        mbedtls_pk_free(pkey_pm->pkey);

    if (!pkey_pm->pkey) {
        pkey_pm->pkey = ssl_mem_malloc(sizeof(mbedtls_pk_context));
        if (!pkey_pm->pkey) {
            SSL_DEBUG(SSL_PLATFORM_ERROR_LEVEL, "no enough memory > (pkey_pm->pkey)");
            goto no_mem;
        }
    }

    load_buf = ssl_mem_malloc(len + 1);
    if (!load_buf) {
        SSL_DEBUG(SSL_PLATFORM_ERROR_LEVEL, "no enough memory > (load_buf)");
        goto failed;
    }

    ssl_memcpy(load_buf, buffer, len);
    load_buf[len] = '\0';

    mbedtls_pk_init(pkey_pm->pkey);

    ret = mbedtls_pk_parse_key(pkey_pm->pkey, load_buf, len + 1, NULL, 0);
    ssl_mem_free(load_buf);

    if (ret) {
        SSL_DEBUG(SSL_PLATFORM_ERROR_LEVEL, "mbedtls_pk_parse_key return -0x%x", -ret);
        goto failed;
    }

    return 0;

failed:
    mbedtls_pk_free(pkey_pm->pkey);
    ssl_mem_free(pkey_pm->pkey);
    pkey_pm->pkey = NULL;
no_mem:
    return -1;
}



void ssl_pm_set_bufflen(SSL *ssl, int len)
{
    max_content_len = len;
}

long ssl_pm_get_verify_result(const SSL *ssl)
{
	uint32_t ret;
	long verify_result;
	struct ssl_pm *ssl_pm = (struct ssl_pm *)ssl->ssl_pm;

	ret = mbedtls_ssl_get_verify_result(&ssl_pm->ssl);
	if (!ret)
		return X509_V_OK;

	if (ret & MBEDTLS_X509_BADCERT_NOT_TRUSTED ||
		(ret & MBEDTLS_X509_BADCRL_NOT_TRUSTED))
		verify_result = X509_V_ERR_INVALID_CA;

	else if (ret & MBEDTLS_X509_BADCERT_CN_MISMATCH)
		verify_result = X509_V_ERR_HOSTNAME_MISMATCH;

	else if ((ret & MBEDTLS_X509_BADCERT_BAD_KEY) ||
		(ret & MBEDTLS_X509_BADCRL_BAD_KEY))
		verify_result = X509_V_ERR_CA_KEY_TOO_SMALL;

	else if ((ret & MBEDTLS_X509_BADCERT_BAD_MD) ||
		(ret & MBEDTLS_X509_BADCRL_BAD_MD))
		verify_result = X509_V_ERR_CA_MD_TOO_WEAK;

	else if ((ret & MBEDTLS_X509_BADCERT_FUTURE) ||
		(ret & MBEDTLS_X509_BADCRL_FUTURE))
		verify_result = X509_V_ERR_CERT_NOT_YET_VALID;

	else if ((ret & MBEDTLS_X509_BADCERT_EXPIRED) ||
		(ret & MBEDTLS_X509_BADCRL_EXPIRED))
		verify_result = X509_V_ERR_CERT_HAS_EXPIRED;

	else
		verify_result = X509_V_ERR_UNSPECIFIED;

	SSL_DEBUG(SSL_PLATFORM_ERROR_LEVEL,
		  "mbedtls_ssl_get_verify_result() return 0x%x", ret);

	return verify_result;
}

/**
 * @brief set expected hostname on peer cert CN
 */

int X509_VERIFY_PARAM_set1_host(X509_VERIFY_PARAM *param,
                                const char *name, size_t namelen)
{
	SSL *ssl = (SSL *)((char *)param - offsetof(SSL, param));
	struct ssl_pm *ssl_pm = (struct ssl_pm *)ssl->ssl_pm;
	char *name_cstr = NULL;

	if (namelen) {
		name_cstr = malloc(namelen + 1);
		if (!name_cstr)
			return 0;
		memcpy(name_cstr, name, namelen);
		name_cstr[namelen] = '\0';
		name = name_cstr;
	}

	mbedtls_ssl_set_hostname(&ssl_pm->ssl, name);

	if (namelen)
		free(name_cstr);

	return 1;
}

void _ssl_set_alpn_list(const SSL *ssl)
{
	if (ssl->alpn_protos) {
		if (mbedtls_ssl_conf_alpn_protocols(&((struct ssl_pm *)(ssl->ssl_pm))->conf, ssl->alpn_protos))
			fprintf(stderr, "mbedtls_ssl_conf_alpn_protocols failed\n");

		return;
	}
	if (!ssl->ctx->alpn_protos)
		return;
	if (mbedtls_ssl_conf_alpn_protocols(&((struct ssl_pm *)(ssl->ssl_pm))->conf, ssl->ctx->alpn_protos))
		fprintf(stderr, "mbedtls_ssl_conf_alpn_protocols failed\n");
}

void SSL_get0_alpn_selected(const SSL *ssl, const unsigned char **data,
                            unsigned int *len)
{
	const char *alp = mbedtls_ssl_get_alpn_protocol(&((struct ssl_pm *)(ssl->ssl_pm))->ssl);

	*data = (const unsigned char *)alp;
	if (alp)
		*len = strlen(alp);
	else
		*len = 0;
}

int SSL_set_sni_callback(SSL *ssl, int(*cb)(void *, mbedtls_ssl_context *,
			 const unsigned char *, size_t), void *param)
{
	struct ssl_pm *ssl_pm = (struct ssl_pm *)ssl->ssl_pm;

	mbedtls_ssl_conf_sni(&ssl_pm->conf, cb, param);

	return 0;
}

SSL *SSL_SSL_from_mbedtls_ssl_context(mbedtls_ssl_context *msc)
{
	struct ssl_pm *ssl_pm = (struct ssl_pm *)((char *)msc - offsetof(struct ssl_pm, ssl));

	return ssl_pm->owner;
}

#include "ssl_cert.h"

void SSL_set_SSL_CTX(SSL *ssl, SSL_CTX *ctx)
{
	struct ssl_pm *ssl_pm = ssl->ssl_pm;
	struct x509_pm *x509_pm = (struct x509_pm *)ctx->cert->x509->x509_pm;
	struct x509_pm *x509_pm_ca = (struct x509_pm *)ctx->client_CA->x509_pm;

	struct pkey_pm *pkey_pm = (struct pkey_pm *)ctx->cert->pkey->pkey_pm;
	int mode;

	if (ssl->cert)
		ssl_cert_free(ssl->cert);
	ssl->ctx = ctx;
	ssl->cert = __ssl_cert_new(ctx->cert);

	if (ctx->verify_mode == SSL_VERIFY_PEER)
		mode = MBEDTLS_SSL_VERIFY_OPTIONAL;
	else if (ctx->verify_mode == SSL_VERIFY_FAIL_IF_NO_PEER_CERT)
		mode = MBEDTLS_SSL_VERIFY_OPTIONAL;
	else if (ctx->verify_mode == SSL_VERIFY_CLIENT_ONCE)
		mode = MBEDTLS_SSL_VERIFY_UNSET;
	else
	        mode = MBEDTLS_SSL_VERIFY_NONE;

	    // printf("ssl: %p, client ca x509_crt %p, mbedtls mode %d\n", ssl, x509_pm_ca->x509_crt, mode);

	/* apply new ctx cert to ssl */

	ssl->verify_mode = ctx->verify_mode;

	mbedtls_ssl_set_hs_ca_chain(&ssl_pm->ssl, x509_pm_ca->x509_crt, NULL);
	mbedtls_ssl_set_hs_own_cert(&ssl_pm->ssl, x509_pm->x509_crt, pkey_pm->pkey);
	mbedtls_ssl_set_hs_authmode(&ssl_pm->ssl, mode);
}
