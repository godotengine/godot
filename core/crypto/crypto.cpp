/*************************************************************************/
/*  crypto.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "crypto.h"

#include "core/engine.h"
#include "core/io/certs_compressed.gen.h"
#include "core/io/compression.h"

/// Resources

CryptoKey *(*CryptoKey::_create)() = nullptr;
CryptoKey *CryptoKey::create() {
	if (_create)
		return _create();
	return nullptr;
}

void CryptoKey::_bind_methods() {
	ClassDB::bind_method(D_METHOD("save", "path"), &CryptoKey::save);
	ClassDB::bind_method(D_METHOD("load", "path"), &CryptoKey::load);
}

X509Certificate *(*X509Certificate::_create)() = nullptr;
X509Certificate *X509Certificate::create() {
	if (_create)
		return _create();
	return nullptr;
}

void X509Certificate::_bind_methods() {
	ClassDB::bind_method(D_METHOD("save", "path"), &X509Certificate::save);
	ClassDB::bind_method(D_METHOD("load", "path"), &X509Certificate::load);
}

/// Crypto

void (*Crypto::_load_default_certificates)(String p_path) = nullptr;
Crypto *(*Crypto::_create)() = nullptr;
Crypto *Crypto::create() {
	if (_create)
		return _create();
	return memnew(Crypto);
}

void Crypto::load_default_certificates(String p_path) {

	if (_load_default_certificates)
		_load_default_certificates(p_path);
}

void Crypto::_bind_methods() {
	ClassDB::bind_method(D_METHOD("generate_random_bytes", "size"), &Crypto::generate_random_bytes);
	ClassDB::bind_method(D_METHOD("generate_rsa", "size"), &Crypto::generate_rsa);
	ClassDB::bind_method(D_METHOD("generate_self_signed_certificate", "key", "issuer_name", "not_before", "not_after"), &Crypto::generate_self_signed_certificate, DEFVAL("CN=myserver,O=myorganisation,C=IT"), DEFVAL("20140101000000"), DEFVAL("20340101000000"));
}

PackedByteArray Crypto::generate_random_bytes(int p_bytes) {
	ERR_FAIL_V_MSG(PackedByteArray(), "generate_random_bytes is not available when mbedtls module is disabled.");
}

Ref<CryptoKey> Crypto::generate_rsa(int p_bytes) {
	ERR_FAIL_V_MSG(nullptr, "generate_rsa is not available when mbedtls module is disabled.");
}

Ref<X509Certificate> Crypto::generate_self_signed_certificate(Ref<CryptoKey> p_key, String p_issuer_name, String p_not_before, String p_not_after) {
	ERR_FAIL_V_MSG(nullptr, "generate_self_signed_certificate is not available when mbedtls module is disabled.");
}

Crypto::Crypto() {
}

/// Resource loader/saver

RES ResourceFormatLoaderCrypto::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress) {

	String el = p_path.get_extension().to_lower();
	if (el == "crt") {
		X509Certificate *cert = X509Certificate::create();
		if (cert)
			cert->load(p_path);
		return cert;
	} else if (el == "key") {
		CryptoKey *key = CryptoKey::create();
		if (key)
			key->load(p_path);
		return key;
	}
	return nullptr;
}

void ResourceFormatLoaderCrypto::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("crt");
	p_extensions->push_back("key");
}

bool ResourceFormatLoaderCrypto::handles_type(const String &p_type) const {

	return p_type == "X509Certificate" || p_type == "CryptoKey";
}

String ResourceFormatLoaderCrypto::get_resource_type(const String &p_path) const {

	String el = p_path.get_extension().to_lower();
	if (el == "crt")
		return "X509Certificate";
	else if (el == "key")
		return "CryptoKey";
	return "";
}

Error ResourceFormatSaverCrypto::save(const String &p_path, const RES &p_resource, uint32_t p_flags) {

	Error err;
	Ref<X509Certificate> cert = p_resource;
	Ref<CryptoKey> key = p_resource;
	if (cert.is_valid()) {
		err = cert->save(p_path);
	} else if (key.is_valid()) {
		err = key->save(p_path);
	} else {
		ERR_FAIL_V(ERR_INVALID_PARAMETER);
	}
	ERR_FAIL_COND_V_MSG(err != OK, err, "Cannot save Crypto resource to file '" + p_path + "'.");
	return OK;
}

void ResourceFormatSaverCrypto::get_recognized_extensions(const RES &p_resource, List<String> *p_extensions) const {

	const X509Certificate *cert = Object::cast_to<X509Certificate>(*p_resource);
	const CryptoKey *key = Object::cast_to<CryptoKey>(*p_resource);
	if (cert) {
		p_extensions->push_back("crt");
	}
	if (key) {
		p_extensions->push_back("key");
	}
}
bool ResourceFormatSaverCrypto::recognize(const RES &p_resource) const {

	return Object::cast_to<X509Certificate>(*p_resource) || Object::cast_to<CryptoKey>(*p_resource);
}
