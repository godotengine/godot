/*************************************************************************/
/*  ssl_context_mbedtls.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "ssl_context_mbedtls.h"

static void my_debug(void *ctx, int level,
		const char *file, int line,
		const char *str) {
	printf("%s:%04d: %s", file, line, str);
	fflush(stdout);
}

void SSLContextMbedTLS::print_mbedtls_error(int p_ret) {
	printf("mbedtls error: returned -0x%x\n\n", -p_ret);
	fflush(stdout);
}

/// CookieContextMbedTLS

Error CookieContextMbedTLS::setup() {
	ERR_FAIL_COND_V_MSG(inited, ERR_ALREADY_IN_USE, "This cookie context is already in use");

	mbedtls_ctr_drbg_init(&ctr_drbg);
	mbedtls_entropy_init(&entropy);
	mbedtls_ssl_cookie_init(&cookie_ctx);
	inited = true;

	int ret = mbedtls_ctr_drbg_seed(&ctr_drbg, mbedtls_entropy_func, &entropy, nullptr, 0);
	if (ret != 0) {
		clear(); // Never leave unusable resources around.
		ERR_FAIL_V_MSG(FAILED, "mbedtls_ctr_drbg_seed returned an error " + itos(ret));
	}

	ret = mbedtls_ssl_cookie_setup(&cookie_ctx, mbedtls_ctr_drbg_random, &ctr_drbg);
	if (ret != 0) {
		clear();
		ERR_FAIL_V_MSG(FAILED, "mbedtls_ssl_cookie_setup returned an error " + itos(ret));
	}
	return OK;
}

void CookieContextMbedTLS::clear() {
	if (!inited) {
		return;
	}
	mbedtls_ctr_drbg_free(&ctr_drbg);
	mbedtls_entropy_free(&entropy);
	mbedtls_ssl_cookie_free(&cookie_ctx);
}

CookieContextMbedTLS::CookieContextMbedTLS() {
	inited = false;
}

CookieContextMbedTLS::~CookieContextMbedTLS() {
	clear();
}

/// SSLContextMbedTLS

Error SSLContextMbedTLS::_setup(int p_endpoint, int p_transport, int p_authmode) {
	ERR_FAIL_COND_V_MSG(inited, ERR_ALREADY_IN_USE, "This SSL context is already active");

	mbedtls_ssl_init(&ssl);
	mbedtls_ssl_config_init(&conf);
	mbedtls_ctr_drbg_init(&ctr_drbg);
	mbedtls_entropy_init(&entropy);
	inited = true;

	int ret = mbedtls_ctr_drbg_seed(&ctr_drbg, mbedtls_entropy_func, &entropy, nullptr, 0);
	if (ret != 0) {
		clear(); // Never leave unusable resources around.
		ERR_FAIL_V_MSG(FAILED, "mbedtls_ctr_drbg_seed returned an error " + itos(ret));
	}

	ret = mbedtls_ssl_config_defaults(&conf, p_endpoint, p_transport, MBEDTLS_SSL_PRESET_DEFAULT);
	if (ret != 0) {
		clear();
		ERR_FAIL_V_MSG(FAILED, "mbedtls_ssl_config_defaults returned an error" + itos(ret));
	}
	mbedtls_ssl_conf_authmode(&conf, p_authmode);
	mbedtls_ssl_conf_rng(&conf, mbedtls_ctr_drbg_random, &ctr_drbg);
	mbedtls_ssl_conf_dbg(&conf, my_debug, stdout);
	return OK;
}

Error SSLContextMbedTLS::init_server(int p_transport, int p_authmode, Ref<CryptoKeyMbedTLS> p_pkey, Ref<X509CertificateMbedTLS> p_cert, Ref<CookieContextMbedTLS> p_cookies) {
	ERR_FAIL_COND_V(!p_pkey.is_valid(), ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(!p_cert.is_valid(), ERR_INVALID_PARAMETER);

	Error err = _setup(MBEDTLS_SSL_IS_SERVER, p_transport, p_authmode);
	ERR_FAIL_COND_V(err != OK, err);

	// Locking key and certificate(s)
	pkey = p_pkey;
	certs = p_cert;
	if (pkey.is_valid()) {
		pkey->lock();
	}
	if (certs.is_valid()) {
		certs->lock();
	}

	// Adding key and certificate
	int ret = mbedtls_ssl_conf_own_cert(&conf, &(certs->cert), &(pkey->pkey));
	if (ret != 0) {
		clear();
		ERR_FAIL_V_MSG(ERR_INVALID_PARAMETER, "Invalid cert/key combination " + itos(ret));
	}
	// Adding CA chain if available.
	if (certs->cert.next) {
		mbedtls_ssl_conf_ca_chain(&conf, certs->cert.next, nullptr);
	}
	// DTLS Cookies
	if (p_transport == MBEDTLS_SSL_TRANSPORT_DATAGRAM) {
		if (p_cookies.is_null() || !p_cookies->inited) {
			clear();
			ERR_FAIL_V(ERR_BUG);
		}
		cookies = p_cookies;
		mbedtls_ssl_conf_dtls_cookies(&conf, mbedtls_ssl_cookie_write, mbedtls_ssl_cookie_check, &(cookies->cookie_ctx));
	}
	mbedtls_ssl_setup(&ssl, &conf);
	return OK;
}

Error SSLContextMbedTLS::init_client(int p_transport, int p_authmode, Ref<X509CertificateMbedTLS> p_valid_cas) {
	Error err = _setup(MBEDTLS_SSL_IS_CLIENT, p_transport, p_authmode);
	ERR_FAIL_COND_V(err != OK, err);

	X509CertificateMbedTLS *cas = nullptr;

	if (p_valid_cas.is_valid()) {
		// Locking CA certificates
		certs = p_valid_cas;
		certs->lock();
		cas = certs.ptr();
	} else {
		// Fall back to default certificates (no need to lock those).
		cas = CryptoMbedTLS::get_default_certificates();
		if (cas == nullptr) {
			clear();
			ERR_FAIL_V_MSG(ERR_UNCONFIGURED, "SSL module failed to initialize!");
		}
	}

	// Set valid CAs
	mbedtls_ssl_conf_ca_chain(&conf, &(cas->cert), nullptr);
	mbedtls_ssl_setup(&ssl, &conf);
	return OK;
}

void SSLContextMbedTLS::clear() {
	if (!inited) {
		return;
	}
	mbedtls_ssl_free(&ssl);
	mbedtls_ssl_config_free(&conf);
	mbedtls_ctr_drbg_free(&ctr_drbg);
	mbedtls_entropy_free(&entropy);

	// Unlock and key and certificates
	if (certs.is_valid()) {
		certs->unlock();
	}
	certs = Ref<X509Certificate>();
	if (pkey.is_valid()) {
		pkey->unlock();
	}
	pkey = Ref<CryptoKeyMbedTLS>();
	cookies = Ref<CookieContextMbedTLS>();
	inited = false;
}

mbedtls_ssl_context *SSLContextMbedTLS::get_context() {
	ERR_FAIL_COND_V(!inited, nullptr);
	return &ssl;
}

SSLContextMbedTLS::SSLContextMbedTLS() {
	inited = false;
}

SSLContextMbedTLS::~SSLContextMbedTLS() {
	clear();
}
