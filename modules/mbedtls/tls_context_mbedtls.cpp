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
#include "core/os/os.h"

#include <mbedtls/error.h>

#ifdef TOOLS_ENABLED
#include "editor/settings/editor_settings.h"
#endif // TOOLS_ENABLED

static void my_debug(void *ctx, int level,
		const char *file, int line,
		const char *str) {
	printf("%s:%04d: %s", file, line, str);
	fflush(stdout);
}

void TLSContextMbedTLS::print_mbedtls_error(int p_ret) {
	const int error_buflen = 245;
	char error_buf[error_buflen];
	mbedtls_strerror(p_ret, error_buf, error_buflen);
	printf("mbedtls error: returned -0x%x: %s\n\n", -p_ret, error_buf);
	fflush(stdout);
}

/// CookieContextMbedTLS

Error CookieContextMbedTLS::setup() {
	ERR_FAIL_COND_V_MSG(inited, ERR_ALREADY_IN_USE, "This cookie context is already in use");

	mbedtls_ssl_cookie_init(&cookie_ctx);
	inited = true;

	int ret = mbedtls_ssl_cookie_setup(GODOT_MBEDTLS_COMPAT_ARGS(&cookie_ctx));
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
	mbedtls_ssl_cookie_free(&cookie_ctx);
	inited = false;
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
	mbedtls_pk_init(&pk);
	inited = true;

	int ret = mbedtls_ssl_config_defaults(&conf, p_endpoint, p_transport, MBEDTLS_SSL_PRESET_DEFAULT);
	if (ret != 0) {
		clear();
		ERR_FAIL_V_MSG(FAILED, "mbedtls_ssl_config_defaults returned an error" + itos(ret));
	}
	mbedtls_ssl_conf_authmode(&conf, p_authmode);
	GODOT_MBEDTLS_COMPAT_SSL_CONF;
	mbedtls_ssl_conf_dbg(&conf, my_debug, stdout);
	return OK;
}

void TLSContextMbedTLS::encode_mbedtls_x509_to_vector(mbedtls_x509_crt *p_crt, Vector<PackedByteArray> &r_crts) {
	int num_crt = 0;
	mbedtls_x509_crt *wlkr = p_crt;

	// Count the number of certificates to size crts correctly.
	while (wlkr) {
		num_crt++;
		wlkr = wlkr->next;
	}

	r_crts.resize(num_crt);

	// Copy the DER data in the mbedtls_x509_crt into the PackedByteArray
	wlkr = p_crt;
	num_crt = 0;

	while (wlkr) {
		PackedByteArray *der_data = r_crts.ptrw() + num_crt;
		der_data->resize_uninitialized(wlkr->raw.len);

		memcpy(der_data->ptrw(), wlkr->raw.p, wlkr->raw.len);

		wlkr = wlkr->next;
		num_crt++;
	}
}

// In a single pass, collect all of the certificates provided by the server during the SSL handshake
// and pass them to the OS implementation for verification.
//
// Any non-zero return value is treated as validation failure and the connection aborted.
//
// Only perform this validation once per handshake on the peer certificate (depth is 0).
int TLSContextMbedTLS::do_cert_validation(mbedtls_x509_crt *p_crt, int p_depth, uint32_t *p_flags) {
	if (p_depth > 0) {
		// Only validate on the leaf certificate once.
		return 0;
	}

	Vector<PackedByteArray> ssl_crts;
	Vector<PackedByteArray> ca_crts;

	encode_mbedtls_x509_to_vector(p_crt, ssl_crts);

	// Use any CA certificate overrides configured via get_trusted_ca_chain() in the options.
	if (certs.is_valid()) {
		encode_mbedtls_x509_to_vector(certs->get_context(), ca_crts);
	}

	// Validate entire chain in a single pass.
	Error result = OS::get_singleton()->verify_tls_certificate_chain(mbedtls_ssl_get_hostname(&tls), ssl_crts, ca_crts);

	if (result != OK) {
		// While the OS could supply additional options, x509_crt_verify_chain would have already failed
		// if the certificate chain did not validate correctly before this method is invoked, leaving
		// explicit revocation as the most likely cause.
		*p_flags = MBEDTLS_X509_BADCERT_REVOKED;
	}

	return 0;
}

Error TLSContextMbedTLS::_apply_ca_bundle(Ref<TLSOptions> p_options) {
	Ref<X509CertificateMbedTLS> cas;

	if (p_options->get_trusted_ca_chain().is_valid()) {
		// Locking CA certificates.
		certs = p_options->get_trusted_ca_chain();
		certs->lock();
		cas = certs;
	} else {
		// Fall back to default certificates (no need to lock those).
		cas = CryptoMbedTLS::get_default_certificates();
		if (cas.is_null()) {
			clear();
			ERR_FAIL_V_MSG(ERR_UNCONFIGURED, "SSL module failed to initialize!");
		}
	}

	// Set valid CAs.
	mbedtls_ssl_conf_ca_chain(&conf, cas->get_context(), nullptr);
	return OK;
}

Error TLSContextMbedTLS::_use_os_validation(Ref<TLSOptions> p_options) {
	// Applies the correct CA bundle to mbedTLS which does CA chain validation
	// regardless of any custom verification steps.  This also locks `certs`
	// for use during the OS validation.
	Error err = _apply_ca_bundle(p_options);
	ERR_FAIL_COND_V(err != OK, err);

	// Use the OS supplied validation code for this connection.
	mbedtls_ssl_set_verify(&tls, TLSContextMbedTLS::_os_cert_validation, this);

	return OK;
}

Error TLSContextMbedTLS::_use_ca_only_validation(Ref<TLSOptions> p_options) {
	return _apply_ca_bundle(p_options);
}

Error TLSContextMbedTLS::init_server(int p_transport, Ref<TLSOptions> p_options, Ref<CookieContextMbedTLS> p_cookies) {
	ERR_FAIL_COND_V(p_options.is_null() || !p_options->is_server(), ERR_INVALID_PARAMETER);

	// Check key and certificate(s)
	Ref<CryptoKeyMbedTLS> pkey = p_options->get_private_key();
	certs = p_options->get_own_certificate();
	ERR_FAIL_COND_V(pkey.is_null() || certs.is_null(), ERR_INVALID_PARAMETER);

	Error err = _setup(MBEDTLS_SSL_IS_SERVER, p_transport, MBEDTLS_SSL_VERIFY_NONE); // TODO client auth.
	ERR_FAIL_COND_V(err != OK, err);

	// Locking key and certificate(s)
	certs->lock();

	// Adding key and certificate
	int ret = mbedtls_pk_copy_from_psa(pkey->get_key_id(), &pk);
	if (ret != 0) {
		clear();
		ERR_FAIL_V_MSG(ERR_INVALID_PARAMETER, "Failed to import key " + itos(ret));
	}
	ret = mbedtls_ssl_conf_own_cert(&conf, certs->get_context(), &pk);
	if (ret != 0) {
		clear();
		ERR_FAIL_V_MSG(ERR_INVALID_PARAMETER, "Invalid cert/key combination " + itos(ret));
	}
	// Adding CA chain if available.
	if (certs->get_context()->next) {
		mbedtls_ssl_conf_ca_chain(&conf, certs->get_context()->next, nullptr);
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

#ifdef TOOLS_ENABLED
	if (EditorSettings::get_singleton()) {
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

#ifdef TOOLS_ENABLED
	if (EditorSettings::get_singleton()) {
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

	if (OS::get_singleton()->has_feature("tls_cert_validation")) {
		err = _use_os_validation(p_options);
	} else {
		err = _use_ca_only_validation(p_options);
	}
	ERR_FAIL_COND_V(err != OK, err);

	mbedtls_ssl_setup(&tls, &conf);
	return OK;
}

void TLSContextMbedTLS::clear() {
	if (!inited) {
		return;
	}
	mbedtls_ssl_free(&tls);
	mbedtls_ssl_config_free(&conf);
	mbedtls_pk_free(&pk);
	// Unlock certificates
	if (certs.is_valid()) {
		certs->unlock();
	}
	certs = Ref<X509Certificate>();
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
