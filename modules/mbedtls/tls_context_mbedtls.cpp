/**************************************************************************/
/*  tls_context_mbedtls.cpp                                               */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "tls_context_mbedtls.h"

#include "core/config/project_settings.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_settings.h"
#endif // TOOLS_ENABLED

static void my_debug(void *ctx, int level,
		const char *file, int line,
		const char *str) {
	printf("%s:%04d: %s", file, line, str);
	fflush(stdout);
}

void TLSContextMbedTLS::print_mbedtls_error(int p_ret) {
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
}

CookieContextMbedTLS::~CookieContextMbedTLS() {
	clear();
}

/// TLSContextMbedTLS

Error TLSContextMbedTLS::_setup(int p_endpoint, int p_transport, int p_authmode) {
	ERR_FAIL_COND_V_MSG(inited, ERR_ALREADY_IN_USE, "This SSL context is already active");

	mbedtls_ssl_init(&tls);
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

Error TLSContextMbedTLS::init_server(int p_transport, Ref<TLSOptions> p_options, Ref<CookieContextMbedTLS> p_cookies) {
	ERR_FAIL_COND_V(p_options.is_null() || !p_options->is_server(), ERR_INVALID_PARAMETER);

	// Check key and certificate(s)
	pkey = p_options->get_private_key();
	certs = p_options->get_own_certificate();
	ERR_FAIL_COND_V(pkey.is_null() || certs.is_null(), ERR_INVALID_PARAMETER);

	Error err = _setup(MBEDTLS_SSL_IS_SERVER, p_transport, MBEDTLS_SSL_VERIFY_NONE); // TODO client auth.
	ERR_FAIL_COND_V(err != OK, err);

	// Locking key and certificate(s)
	pkey->lock();
	certs->lock();

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

#if MBEDTLS_VERSION_MAJOR >= 3
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		if (!EditorSettings::get_singleton()->get_setting("network/tls/enable_tls_v1.3").operator bool()) {
			mbedtls_ssl_conf_max_tls_version(&conf, MBEDTLS_SSL_VERSION_TLS1_2);
		}
	} else
#endif
	{
		if (!GLOBAL_GET("network/tls/enable_tls_v1.3").operator bool()) {
			mbedtls_ssl_conf_max_tls_version(&conf, MBEDTLS_SSL_VERSION_TLS1_2);
		}
	}
#endif

	mbedtls_ssl_setup(&tls, &conf);
	return OK;
}

Error TLSContextMbedTLS::init_client(int p_transport, const String &p_hostname, Ref<TLSOptions> p_options) {
	ERR_FAIL_COND_V(p_options.is_null() || p_options->is_server(), ERR_INVALID_PARAMETER);

	int authmode = MBEDTLS_SSL_VERIFY_REQUIRED;
	bool unsafe = p_options->is_unsafe_client();
	if (unsafe && p_options->get_trusted_ca_chain().is_null()) {
		authmode = MBEDTLS_SSL_VERIFY_NONE;
	}

	Error err = _setup(MBEDTLS_SSL_IS_CLIENT, p_transport, authmode);
	ERR_FAIL_COND_V(err != OK, err);

	if (unsafe) {
		// No hostname verification for unsafe clients.
		mbedtls_ssl_set_hostname(&tls, nullptr);
	} else {
		String cn = p_options->get_common_name_override();
		if (cn.is_empty()) {
			cn = p_hostname;
		}
		mbedtls_ssl_set_hostname(&tls, cn.utf8().get_data());
	}

	X509CertificateMbedTLS *cas = nullptr;

	if (p_options->get_trusted_ca_chain().is_valid()) {
		// Locking CA certificates
		certs = p_options->get_trusted_ca_chain();
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

#if MBEDTLS_VERSION_MAJOR >= 3
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		if (!EditorSettings::get_singleton()->get_setting("network/tls/enable_tls_v1.3").operator bool()) {
			mbedtls_ssl_conf_max_tls_version(&conf, MBEDTLS_SSL_VERSION_TLS1_2);
		}
	} else
#endif
	{
		if (!GLOBAL_GET("network/tls/enable_tls_v1.3").operator bool()) {
			mbedtls_ssl_conf_max_tls_version(&conf, MBEDTLS_SSL_VERSION_TLS1_2);
		}
	}
#endif

	// Set valid CAs
	mbedtls_ssl_conf_ca_chain(&conf, &(cas->cert), nullptr);
	mbedtls_ssl_setup(&tls, &conf);
	return OK;
}

void TLSContextMbedTLS::clear() {
	if (!inited) {
		return;
	}
	mbedtls_ssl_free(&tls);
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

mbedtls_ssl_context *TLSContextMbedTLS::get_context() {
	ERR_FAIL_COND_V(!inited, nullptr);
	return &tls;
}

TLSContextMbedTLS::TLSContextMbedTLS() {
}

TLSContextMbedTLS::~TLSContextMbedTLS() {
	clear();
}
