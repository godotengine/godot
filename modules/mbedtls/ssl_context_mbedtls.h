/*************************************************************************/
/*  ssl_context_mbedtls.h                                                */
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

#ifndef SSL_CONTEXT_MBEDTLS_H
#define SSL_CONTEXT_MBEDTLS_H

#include "crypto_mbedtls.h"

#include "core/os/file_access.h"
#include "core/pool_vector.h"
#include "core/reference.h"

#include <mbedtls/config.h>
#include <mbedtls/ctr_drbg.h>
#include <mbedtls/debug.h>
#include <mbedtls/entropy.h>
#include <mbedtls/ssl.h>
#include <mbedtls/ssl_cookie.h>

class SSLContextMbedTLS;

class CookieContextMbedTLS : public Reference {
	friend class SSLContextMbedTLS;

protected:
	bool inited;
	mbedtls_entropy_context entropy;
	mbedtls_ctr_drbg_context ctr_drbg;
	mbedtls_ssl_cookie_ctx cookie_ctx;

public:
	Error setup();
	void clear();

	CookieContextMbedTLS();
	~CookieContextMbedTLS();
};

class SSLContextMbedTLS : public Reference {
protected:
	bool inited;

public:
	static void print_mbedtls_error(int p_ret);

	Ref<X509CertificateMbedTLS> certs;
	mbedtls_entropy_context entropy;
	mbedtls_ctr_drbg_context ctr_drbg;
	mbedtls_ssl_context ssl;
	mbedtls_ssl_config conf;

	Ref<CookieContextMbedTLS> cookies;
	Ref<CryptoKeyMbedTLS> pkey;

	Error _setup(int p_endpoint, int p_transport, int p_authmode);
	Error init_server(int p_transport, int p_authmode, Ref<CryptoKeyMbedTLS> p_pkey, Ref<X509CertificateMbedTLS> p_cert, Ref<CookieContextMbedTLS> p_cookies = Ref<CookieContextMbedTLS>());
	Error init_client(int p_transport, int p_authmode, Ref<X509CertificateMbedTLS> p_valid_cas);
	void clear();

	mbedtls_ssl_context *get_context();

	SSLContextMbedTLS();
	~SSLContextMbedTLS();
};

#endif // SSL_CONTEXT_MBEDTLS_H
