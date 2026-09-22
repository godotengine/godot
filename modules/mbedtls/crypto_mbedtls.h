/**************************************************************************/
/*  crypto_mbedtls.h                                                      */
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

#pragma once

#include "core/crypto/crypto.h"

#include <mbedtls/ssl.h>

#if MBEDTLS_VERSION_MAJOR < 4
int godot_mbedtls_random_compat(void *p_rng, unsigned char *r_output, size_t p_output_len);
#define GODOT_MBEDTLS_COMPAT_ARGS(x) x, godot_mbedtls_random_compat, nullptr
#define GODOT_MBEDTLS_COMPAT_SSL_CONF mbedtls_ssl_conf_rng(&conf, godot_mbedtls_random_compat, nullptr);
#else
#define GODOT_MBEDTLS_COMPAT_ARGS(x) x
#define GODOT_MBEDTLS_COMPAT_SSL_CONF
#endif

class CryptoMbedTLS;
class TLSContextMbedTLS;
class CryptoKeyMbedTLS : public CryptoKey {
	GDSOFTCLASS(CryptoKeyMbedTLS, CryptoKey);

private:
	psa_key_id_t pk_slot = PSA_KEY_ID_NULL;
	psa_key_type_t pk_type = PSA_KEY_TYPE_NONE;
	size_t pk_size = 0;

	bool public_only = true;

	void _clear();
	int _parse_key(const uint8_t *p_buf, int p_size, bool p_public_only);

public:
	static CryptoKey *create(bool p_notify_postinitialize = true);
	static void make_default() { CryptoKey::_create = create; }
	static void finalize() { CryptoKey::_create = nullptr; }

	static Ref<CryptoKeyMbedTLS> generate(psa_key_type_t p_type, size_t p_size);

	psa_key_id_t get_key_id() const { return pk_slot; }
	psa_key_type_t get_key_type() const { return pk_type; }
	size_t get_key_size() const { return pk_size; }

	psa_key_id_t reimport(psa_key_usage_t p_usage, psa_algorithm_t p_alg); // You *MUST* call psa_destroy_key when you are done with the key.

	Error load(const String &p_path, bool p_public_only) override;
	Error save(const String &p_path, bool p_public_only) override;
	String save_to_string(bool p_public_only) override;
	Error load_from_string(const String &p_string_key, bool p_public_only) override;
	bool is_public_only() const override { return public_only; }

	CryptoKeyMbedTLS() {}
	~CryptoKeyMbedTLS() override {
		_clear();
	}
};

class X509CertificateMbedTLS : public X509Certificate {
	GDSOFTCLASS(X509CertificateMbedTLS, X509Certificate);

private:
	mbedtls_x509_crt cert;
	int locks;

public:
	static X509Certificate *create(bool p_notify_postinitialize = true);
	static void make_default() { X509Certificate::_create = create; }
	static void finalize() { X509Certificate::_create = nullptr; }

	mbedtls_x509_crt *get_context() { return &cert; }

	Error load(const String &p_path) override;
	Error load_from_memory(const uint8_t *p_buffer, int p_len) override;
	Error save(const String &p_path) override;
	String save_to_string() override;
	Error load_from_string(const String &p_string_key) override;

	X509CertificateMbedTLS() {
		mbedtls_x509_crt_init(&cert);
		locks = 0;
	}
	~X509CertificateMbedTLS() override {
		mbedtls_x509_crt_free(&cert);
		ERR_FAIL_COND(locks);
	}

	_FORCE_INLINE_ void lock() { locks++; }
	_FORCE_INLINE_ void unlock() { locks--; }
};

class HMACContextMbedTLS : public HMACContext {
private:
	psa_mac_operation_t mac_op = PSA_MAC_OPERATION_INIT;
	psa_key_id_t key_slot = PSA_KEY_ID_NULL;
	int hash_len = 0;

	void _clear();

public:
	static HMACContext *create(bool p_notify_postinitialize = true);
	static void make_default() { HMACContext::_create = create; }
	static void finalize() { HMACContext::_create = nullptr; }

	static bool is_hash_type_allowed(HashingContext::HashType p_hash_type);

	Error start(HashingContext::HashType p_hash_type, const PackedByteArray &p_key) override;
	Error update(const PackedByteArray &p_data) override;
	PackedByteArray finish() override;

	HMACContextMbedTLS() {}
	~HMACContextMbedTLS() override;
};

class CryptoMbedTLS : public Crypto {
private:
	static Ref<X509CertificateMbedTLS> default_certs;

public:
	static Crypto *create(bool p_notify_postinitialize = true);
	static void initialize_crypto();
	static void finalize_crypto();
	static Ref<X509CertificateMbedTLS> get_default_certificates();
	static void load_default_certificates(const String &p_path);
	static psa_algorithm_t psa_alg_from_hashtype(HashingContext::HashType p_hash_type, int &r_size);
	static mbedtls_md_type_t md_type_from_hashtype(HashingContext::HashType p_hash_type, int &r_size);

	PackedByteArray generate_random_bytes(int p_bytes) override;
	Ref<CryptoKey> generate_rsa(int p_bits) override;
	Ref<X509Certificate> generate_self_signed_certificate(Ref<CryptoKey> p_key, const String &p_issuer_name, const String &p_not_before, const String &p_not_after) override;
	Vector<uint8_t> sign(HashingContext::HashType p_hash_type, const Vector<uint8_t> &p_hash, Ref<CryptoKey> p_key) override;
	bool verify(HashingContext::HashType p_hash_type, const Vector<uint8_t> &p_hash, const Vector<uint8_t> &p_signature, Ref<CryptoKey> p_key) override;
	Vector<uint8_t> encrypt(Ref<CryptoKey> p_key, const Vector<uint8_t> &p_plaintext) override;
	Vector<uint8_t> decrypt(Ref<CryptoKey> p_key, const Vector<uint8_t> &p_ciphertext) override;

	CryptoMbedTLS();
	~CryptoMbedTLS() override;
};
