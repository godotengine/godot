/**************************************************************************/
/*  crypto_mbedtls.cpp                                                    */
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

#include "crypto_mbedtls.h"

#include "core/io/certs_compressed.gen.h"
#include "core/io/compression.h"
#include "core/io/file_access.h"
#include "core/object/class_db.h"
#include "core/os/os.h"

#include <mbedtls/debug.h>
#include <mbedtls/md.h>
#include <mbedtls/pem.h>

#define PEM_BEGIN_CRT "-----BEGIN CERTIFICATE-----\n"
#define PEM_END_CRT "-----END CERTIFICATE-----\n"
#define PEM_MIN_SIZE 54

#if MBEDTLS_VERSION_MAJOR < 4
int godot_mbedtls_random_compat(void *p_rng, unsigned char *r_output, size_t p_output_len) {
	return psa_generate_random(r_output, p_output_len);
}
#endif

CryptoKey *CryptoKeyMbedTLS::create(bool p_notify_postinitialize) {
	return static_cast<CryptoKey *>(ClassDB::creator<CryptoKeyMbedTLS>(p_notify_postinitialize));
}

psa_key_type_t CryptoKeyMbedTLS::get_key_type() const {
	psa_key_attributes_t attr = PSA_KEY_ATTRIBUTES_INIT;
	int ret = psa_get_key_attributes(pk_slot, &attr);
	ERR_FAIL_COND_V_MSG(ret, PSA_KEY_TYPE_NONE, "Failed to read key attributes: " + itos(ret));
	return psa_get_key_type(&attr);
}

Error CryptoKeyMbedTLS::load(const String &p_path, bool p_public_only) {
	PackedByteArray out;
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ);
	ERR_FAIL_COND_V_MSG(f.is_null(), ERR_INVALID_PARAMETER, "Cannot open CryptoKeyMbedTLS file '" + p_path + "'.");

	uint64_t flen = f->get_length();
	out.resize(flen + 1);
	f->get_buffer(out.ptrw(), flen);
	out.write[flen] = 0; // string terminator

	int ret = _parse_key(out.ptr(), out.size(), p_public_only);
	// We MUST zeroize the memory for safety!
	mbedtls_platform_zeroize(out.ptrw(), out.size());
	ERR_FAIL_COND_V_MSG(ret, FAILED, "Error parsing key '" + itos(ret) + "'.");

	public_only = p_public_only;
	return OK;
}

Error CryptoKeyMbedTLS::save(const String &p_path, bool p_public_only) {
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::WRITE);
	ERR_FAIL_COND_V_MSG(f.is_null(), ERR_INVALID_PARAMETER, "Cannot save CryptoKeyMbedTLS file '" + p_path + "'.");

	mbedtls_pk_context ctx;
	mbedtls_pk_init(&ctx);
	int ret = 0;
	bool pub_only = p_public_only || public_only;
	if (pub_only) {
		ret = mbedtls_pk_copy_public_from_psa(pk_slot, &ctx);
	} else {
		ret = mbedtls_pk_copy_from_psa(pk_slot, &ctx);
	}
	if (ret) {
		mbedtls_pk_free(&ctx);
		ERR_FAIL_V_MSG(FAILED, "Failed to import key: " + itos(ret));
	}

	unsigned char w[16000];
	memset(w, 0, sizeof(w));
	if (pub_only) {
		ret = mbedtls_pk_write_pubkey_pem(&ctx, w, sizeof(w));
	} else {
		ret = mbedtls_pk_write_key_pem(&ctx, w, sizeof(w));
	}
	mbedtls_pk_free(&ctx);

	if (ret != 0) {
		mbedtls_platform_zeroize(w, sizeof(w)); // Zeroize anything we might have written.
		ERR_FAIL_V_MSG(FAILED, "Error writing key '" + itos(ret) + "'.");
	}

	size_t len = strlen((char *)w);
	f->store_buffer(w, len);
	mbedtls_platform_zeroize(w, sizeof(w)); // Zeroize temporary buffer.
	return OK;
}

Error CryptoKeyMbedTLS::load_from_string(const String &p_string_key, bool p_public_only) {
	const CharString string_key_utf8 = p_string_key.utf8();
	int ret = _parse_key((const unsigned char *)string_key_utf8.get_data(), string_key_utf8.size(), p_public_only);
	ERR_FAIL_COND_V_MSG(ret, FAILED, "Error parsing key '" + itos(ret) + "'.");

	public_only = p_public_only;
	return OK;
}

String CryptoKeyMbedTLS::save_to_string(bool p_public_only) {
	mbedtls_pk_context ctx;
	mbedtls_pk_init(&ctx);
	int ret = 0;
	bool pub_only = p_public_only || public_only;
	if (pub_only) {
		ret = mbedtls_pk_copy_public_from_psa(pk_slot, &ctx);
	} else {
		ret = mbedtls_pk_copy_from_psa(pk_slot, &ctx);
	}
	if (ret) {
		mbedtls_pk_free(&ctx);
		ERR_FAIL_V_MSG("", "Failed to import key: " + itos(ret));
	}

	unsigned char w[16000];
	memset(w, 0, sizeof(w));
	if (pub_only) {
		ret = mbedtls_pk_write_pubkey_pem(&ctx, w, sizeof(w));
	} else {
		ret = mbedtls_pk_write_key_pem(&ctx, w, sizeof(w));
	}
	mbedtls_pk_free(&ctx);
	if (ret != 0) {
		mbedtls_platform_zeroize(w, sizeof(w));
		ERR_FAIL_V_MSG("", "Error saving key '" + itos(ret) + "'.");
	}
	String s = String::utf8((char *)w);
	return s;
}

psa_key_id_t CryptoKeyMbedTLS::reimport(psa_key_usage_t p_usage, psa_algorithm_t p_alg) {
	ERR_FAIL_COND_V(pk_slot == PSA_KEY_ID_NULL, PSA_KEY_ID_NULL);
	psa_key_id_t slot = PSA_KEY_ID_NULL;
	// We can't copy the key, as we are not allowed to change the usage/algorithm.
	// We will need to export and reimport the key.
	psa_key_attributes_t attr = PSA_KEY_ATTRIBUTES_INIT;
	int ret = psa_get_key_attributes(pk_slot, &attr);
	ERR_FAIL_COND_V_MSG(ret, slot, "Failed to read key attributes: " + itos(ret));

	// Export key.
	psa_key_type_t type = psa_get_key_type(&attr);
	int bits = psa_get_key_bits(&attr);
	int size = PSA_EXPORT_KEY_OUTPUT_SIZE(type, bits);
	uint8_t *buf = (uint8_t *)alloca(size);
	size_t out_size = 0;
	ret = psa_export_key(pk_slot, buf, size, &out_size);
	ERR_FAIL_COND_V_MSG(ret, slot, "Failed to export the key: " + itos(ret));

	// Re-import key.
	psa_reset_key_attributes(&attr);
	psa_set_key_type(&attr, type);
	psa_set_key_bits(&attr, bits);
	psa_set_key_usage_flags(&attr, p_usage);
	psa_set_key_algorithm(&attr, p_alg);
	ret = psa_import_key(&attr, buf, out_size, &slot);
	mbedtls_platform_zeroize(buf, size);

	ERR_FAIL_COND_V_MSG(ret, slot, "Failed to re-import key: " + itos(ret));
	return slot;
}

int CryptoKeyMbedTLS::_parse_key(const uint8_t *p_buf, int p_size, bool p_public_only) {
	// Not yet fully supported via PSA
	// See: https://github.com/ARM-software/psa-api/issues/50
	psa_key_attributes_t import_attr = PSA_KEY_ATTRIBUTES_INIT;
	psa_key_usage_t flags = PSA_KEY_USAGE_VERIFY_HASH;
	mbedtls_pk_context ctx;
	mbedtls_pk_init(&ctx);
	int ret = -1;
	if (p_public_only) {
		ret = mbedtls_pk_parse_public_key(&ctx, p_buf, p_size);
	} else {
		flags = PSA_KEY_USAGE_SIGN_HASH;
		ret = mbedtls_pk_parse_key(&ctx, p_buf, p_size, nullptr, GODOT_MBEDTLS_COMPAT_ARGS(0));
	}
	if (ret) {
		mbedtls_pk_free(&ctx);
		ERR_FAIL_V_MSG(PSA_ERROR_GENERIC_ERROR, "Failed to parse key " + itos(ret));
	}
	ret = mbedtls_pk_get_psa_attributes(&ctx, flags, &import_attr);
	if (ret) {
		mbedtls_pk_free(&ctx);
		ERR_FAIL_V_MSG(PSA_ERROR_GENERIC_ERROR, "Failed to get key attributes " + itos(ret));
	}
	psa_key_attributes_t attr = PSA_KEY_ATTRIBUTES_INIT;
	psa_set_key_type(&attr, psa_get_key_type(&import_attr));
	psa_set_key_usage_flags(&attr, PSA_KEY_USAGE_EXPORT | PSA_KEY_USAGE_COPY);
	psa_set_key_bits(&attr, psa_get_key_bits(&import_attr));
	ret = mbedtls_pk_import_into_psa(&ctx, &attr, &pk_slot);
	mbedtls_pk_free(&ctx);
	return ret;
}

X509Certificate *X509CertificateMbedTLS::create(bool p_notify_postinitialize) {
	return static_cast<X509Certificate *>(ClassDB::creator<X509CertificateMbedTLS>(p_notify_postinitialize));
}

Error X509CertificateMbedTLS::load(const String &p_path) {
	ERR_FAIL_COND_V_MSG(locks, ERR_ALREADY_IN_USE, "Certificate is already in use.");

	PackedByteArray out;
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ);
	ERR_FAIL_COND_V_MSG(f.is_null(), ERR_INVALID_PARAMETER, vformat("Cannot open X509CertificateMbedTLS file '%s'.", p_path));

	uint64_t flen = f->get_length();
	out.resize(flen + 1);
	f->get_buffer(out.ptrw(), flen);
	out.write[flen] = 0; // string terminator

	int ret = mbedtls_x509_crt_parse(&cert, out.ptr(), out.size());
	ERR_FAIL_COND_V_MSG(ret < 0, FAILED, vformat("Error parsing X509 certificates from file '%s': %d.", p_path, ret));
	if (ret > 0) { // Some certs parsed fine, don't error.
		print_verbose(vformat("MbedTLS: Some X509 certificates could not be parsed from file '%s' (%d certificates skipped).", p_path, ret));
	}

	return OK;
}

Error X509CertificateMbedTLS::load_from_memory(const uint8_t *p_buffer, int p_len) {
	ERR_FAIL_COND_V_MSG(locks, ERR_ALREADY_IN_USE, "Certificate is already in use.");

	int ret = mbedtls_x509_crt_parse(&cert, p_buffer, p_len);
	ERR_FAIL_COND_V_MSG(ret < 0, FAILED, vformat("Error parsing X509 certificates: %d.", ret));
	if (ret > 0) { // Some certs parsed fine, don't error.
		print_verbose(vformat("MbedTLS: Some X509 certificates could not be parsed (%d certificates skipped).", ret));
	}
	return OK;
}

Error X509CertificateMbedTLS::save(const String &p_path) {
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::WRITE);
	ERR_FAIL_COND_V_MSG(f.is_null(), ERR_INVALID_PARAMETER, vformat("Cannot save X509CertificateMbedTLS file '%s'.", p_path));

	mbedtls_x509_crt *crt = &cert;
	while (crt) {
		unsigned char w[4096];
		size_t wrote = 0;
		int ret = mbedtls_pem_write_buffer(PEM_BEGIN_CRT, PEM_END_CRT, cert.raw.p, cert.raw.len, w, sizeof(w), &wrote);
		if (ret != 0 || wrote == 0) {
			ERR_FAIL_V_MSG(FAILED, "Error writing certificate '" + itos(ret) + "'.");
		}

		f->store_buffer(w, wrote - 1); // don't write the string terminator
		crt = crt->next;
	}
	return OK;
}

String X509CertificateMbedTLS::save_to_string() {
	String buffer;
	mbedtls_x509_crt *crt = &cert;
	while (crt) {
		unsigned char w[4096];
		size_t wrote = 0;
		int ret = mbedtls_pem_write_buffer(PEM_BEGIN_CRT, PEM_END_CRT, cert.raw.p, cert.raw.len, w, sizeof(w), &wrote);
		ERR_FAIL_COND_V_MSG(ret != 0 || wrote == 0, String(), "Error saving the certificate.");

		// PEM is base64, aka ascii
		buffer += String::ascii(Span((char *)w, wrote));
		crt = crt->next;
	}
	if (buffer.length() <= PEM_MIN_SIZE) {
		// When the returned value of variable 'buffer' would consist of no Base-64 data, return an empty String instead.
		return String();
	}
	return buffer;
}

Error X509CertificateMbedTLS::load_from_string(const String &p_string_key) {
	ERR_FAIL_COND_V_MSG(locks, ERR_ALREADY_IN_USE, "Certificate is already in use.");
	CharString cs = p_string_key.utf8();

	int ret = mbedtls_x509_crt_parse(&cert, (const unsigned char *)cs.get_data(), cs.size());
	ERR_FAIL_COND_V_MSG(ret < 0, FAILED, vformat("Error parsing X509 certificates: %d.", ret));
	if (ret > 0) { // Some certs parsed fine, don't error.
		print_verbose(vformat("MbedTLS: Some X509 certificates could not be parsed (%d certificates skipped).", ret));
	}

	return OK;
}

bool HMACContextMbedTLS::is_hash_type_allowed(HashingContext::HashType p_hash_type) {
	switch (p_hash_type) {
		case HashingContext::HASH_SHA1:
		case HashingContext::HASH_SHA256:
			return true;
		default:
			return false;
	}
}

HMACContext *HMACContextMbedTLS::create(bool p_notify_postinitialize) {
	return static_cast<HMACContext *>(ClassDB::creator<HMACContextMbedTLS>(p_notify_postinitialize));
}

Error HMACContextMbedTLS::start(HashingContext::HashType p_hash_type, const PackedByteArray &p_key) {
	ERR_FAIL_COND_V_MSG(key_slot != PSA_KEY_ID_NULL, ERR_FILE_ALREADY_IN_USE, "HMACContext already started.");

	// HMAC keys can be any size.
	ERR_FAIL_COND_V_MSG(p_key.is_empty(), ERR_INVALID_PARAMETER, "Key must not be empty.");

	bool allowed = HMACContextMbedTLS::is_hash_type_allowed(p_hash_type);
	ERR_FAIL_COND_V_MSG(!allowed, ERR_INVALID_PARAMETER, "Unsupported hash type.");

	psa_algorithm_t ht = PSA_ALG_HMAC(CryptoMbedTLS::psa_alg_from_hashtype(p_hash_type, hash_len));
	psa_key_attributes_t attr = PSA_KEY_ATTRIBUTES_INIT;
	psa_set_key_type(&attr, PSA_KEY_TYPE_HMAC);
	psa_set_key_algorithm(&attr, ht);
	psa_set_key_usage_flags(&attr, PSA_KEY_USAGE_SIGN_HASH);
	int ret = psa_import_key(&attr, p_key.ptr(), p_key.size(), &key_slot);
	ERR_FAIL_COND_V_MSG(ret != PSA_SUCCESS, FAILED, "Error import key: " + itos(ret));

	ret = psa_mac_sign_setup(&mac_op, key_slot, ht);
	if (ret != PSA_SUCCESS) {
		_clear();
		ERR_FAIL_V_MSG(FAILED, "Failed to setup HMAC signing: " + itos(ret));
	}
	return OK;
}

Error HMACContextMbedTLS::update(const PackedByteArray &p_data) {
	ERR_FAIL_COND_V_MSG(key_slot == PSA_KEY_ID_NULL, ERR_INVALID_DATA, "Start must be called before update.");

	ERR_FAIL_COND_V_MSG(p_data.is_empty(), ERR_INVALID_PARAMETER, "Src must not be empty.");

	int ret = psa_mac_update(&mac_op, (const uint8_t *)p_data.ptr(), (size_t)p_data.size());
	ERR_FAIL_COND_V_MSG(ret != PSA_SUCCESS, FAILED, "Failed to update HMAC: " + itos(ret));
	return OK;
}

PackedByteArray HMACContextMbedTLS::finish() {
	ERR_FAIL_COND_V_MSG(key_slot == PSA_KEY_ID_NULL, PackedByteArray(), "Start must be called before finish.");
	ERR_FAIL_COND_V_MSG(hash_len == 0, PackedByteArray(), "Unsupported hash type.");

	PackedByteArray out;
	out.resize(hash_len);
	size_t mac_length = 0;
	int ret = psa_mac_sign_finish(&mac_op, out.ptrw(), (size_t)out.size(), &mac_length);
	_clear();
	ERR_FAIL_COND_V_MSG(ret, PackedByteArray(), "Error received while finishing HMAC");
	ERR_FAIL_COND_V(mac_length != (size_t)out.size(), PackedByteArray()); // Bug?
	return out;
}

void HMACContextMbedTLS::_clear() {
	if (key_slot == PSA_KEY_ID_NULL) {
		return;
	}
	hash_len = 0;
	psa_mac_abort(&mac_op);
	mac_op = PSA_MAC_OPERATION_INIT;
	psa_destroy_key(key_slot);
	key_slot = PSA_KEY_ID_NULL;
}

HMACContextMbedTLS::~HMACContextMbedTLS() {
	_clear();
}

Crypto *CryptoMbedTLS::create(bool p_notify_postinitialize) {
	return static_cast<Crypto *>(ClassDB::creator<CryptoMbedTLS>(p_notify_postinitialize));
}

void CryptoMbedTLS::initialize_crypto() {
	Crypto::_create = create;
	Crypto::_load_default_certificates = load_default_certificates;
	X509CertificateMbedTLS::make_default();
	CryptoKeyMbedTLS::make_default();
	HMACContextMbedTLS::make_default();
}

void CryptoMbedTLS::finalize_crypto() {
	Crypto::_create = nullptr;
	Crypto::_load_default_certificates = nullptr;
	default_certs = nullptr;
	X509CertificateMbedTLS::finalize();
	CryptoKeyMbedTLS::finalize();
	HMACContextMbedTLS::finalize();
}

CryptoMbedTLS::CryptoMbedTLS() {
}

CryptoMbedTLS::~CryptoMbedTLS() {
}

Ref<X509CertificateMbedTLS> CryptoMbedTLS::default_certs;

Ref<X509CertificateMbedTLS> CryptoMbedTLS::get_default_certificates() {
	return default_certs;
}

void CryptoMbedTLS::load_default_certificates(const String &p_path) {
	ERR_FAIL_COND(default_certs.is_valid());

	default_certs = memnew(X509CertificateMbedTLS);
	ERR_FAIL_COND(default_certs.is_null());

	if (!p_path.is_empty()) {
		// Use certs defined in project settings.
		default_certs->load(p_path);
	} else {
		// Try to use system certs otherwise.
		String system_certs = OS::get_singleton()->get_system_ca_certificates();
		if (!system_certs.is_empty()) {
			CharString cs = system_certs.utf8();
			default_certs->load_from_memory((const uint8_t *)cs.get_data(), cs.size());
			print_verbose("Loaded system CA certificates");
		}
#ifdef BUILTIN_CERTS_ENABLED
		else {
			// Use builtin certs if there are no system certs.
			PackedByteArray certs;
			certs.resize(_certs_uncompressed_size + 1);
			const int64_t decompressed_size = Compression::decompress(certs.ptrw(), _certs_uncompressed_size, _certs_compressed, _certs_compressed_size, Compression::MODE_DEFLATE);
			ERR_FAIL_COND_MSG(decompressed_size != _certs_uncompressed_size, "Error decompressing builtin CA certificates. Decompressed size did not match expected size.");
			certs.write[_certs_uncompressed_size] = 0; // Make sure it ends with string terminator
			default_certs->load_from_memory(certs.ptr(), certs.size());
			print_verbose("Loaded builtin CA certificates");
		}
#endif
	}
}

Ref<CryptoKey> CryptoMbedTLS::generate_rsa(int p_bits) {
	Ref<CryptoKeyMbedTLS> out;
	out.instantiate();
	psa_key_attributes_t attr = PSA_KEY_ATTRIBUTES_INIT;
	psa_set_key_type(&attr, PSA_KEY_TYPE_RSA_KEY_PAIR);
	psa_set_key_usage_flags(&attr, PSA_KEY_USAGE_EXPORT | PSA_KEY_USAGE_COPY);
	psa_set_key_bits(&attr, p_bits);
	int ret = psa_generate_key(&attr, &(out->pk_slot));
	ERR_FAIL_COND_V(ret != 0, nullptr);
	out->public_only = false;
	return out;
}

Ref<X509Certificate> CryptoMbedTLS::generate_self_signed_certificate(Ref<CryptoKey> p_key, const String &p_issuer_name, const String &p_not_before, const String &p_not_after) {
	Ref<CryptoKeyMbedTLS> key = static_cast<Ref<CryptoKeyMbedTLS>>(p_key);
	ERR_FAIL_COND_V_MSG(key.is_null(), nullptr, "Invalid private key argument.");

	uint8_t rand_serial[20] = {};
	int ret = psa_generate_random(rand_serial, sizeof(rand_serial));
	ERR_FAIL_COND_V(ret, nullptr);

	mbedtls_pk_context ctx;
	mbedtls_pk_init(&ctx);
	ret = mbedtls_pk_copy_from_psa(key->pk_slot, &ctx);
	if (ret) {
		mbedtls_pk_free(&ctx);
		ERR_FAIL_V_MSG(nullptr, "Failed to import private key: " + itos(ret));
	}

	mbedtls_x509write_cert crt;
	mbedtls_x509write_crt_init(&crt);
	mbedtls_x509write_crt_set_subject_key(&crt, &ctx);
	mbedtls_x509write_crt_set_issuer_key(&crt, &ctx);
	mbedtls_x509write_crt_set_subject_name(&crt, p_issuer_name.utf8().get_data());
	mbedtls_x509write_crt_set_issuer_name(&crt, p_issuer_name.utf8().get_data());
	mbedtls_x509write_crt_set_version(&crt, MBEDTLS_X509_CRT_VERSION_3);
	mbedtls_x509write_crt_set_md_alg(&crt, MBEDTLS_MD_SHA256);
	mbedtls_x509write_crt_set_serial_raw(&crt, rand_serial, sizeof(rand_serial));
	mbedtls_x509write_crt_set_validity(&crt, p_not_before.utf8().get_data(), p_not_after.utf8().get_data());
	mbedtls_x509write_crt_set_basic_constraints(&crt, 1, 0);

	unsigned char buf[4096];
	memset(buf, 0, 4096);

	ret = mbedtls_x509write_crt_pem(&crt, buf, GODOT_MBEDTLS_COMPAT_ARGS(4096));
	mbedtls_x509write_crt_free(&crt);
	mbedtls_pk_free(&ctx);
	ERR_FAIL_COND_V_MSG(ret != 0, nullptr, "Failed to generate certificate: " + itos(ret));
	buf[4095] = '\0'; // Make sure strlen can't fail.

	Ref<X509CertificateMbedTLS> out;
	out.instantiate();
	out->load_from_memory(buf, strlen((char *)buf) + 1); // Use strlen to find correct output size.
	return out;
}

PackedByteArray CryptoMbedTLS::generate_random_bytes(int p_bytes) {
	ERR_FAIL_COND_V(p_bytes < 0, PackedByteArray());
	PackedByteArray out;
	out.resize(p_bytes);
	int ret = psa_generate_random(out.ptrw(), out.size());
	ERR_FAIL_COND_V(ret != PSA_SUCCESS, PackedByteArray());
	return out;
}

psa_algorithm_t CryptoMbedTLS::psa_alg_from_hashtype(HashingContext::HashType p_hash_type, int &r_size) {
	switch (p_hash_type) {
		case HashingContext::HASH_MD5:
			r_size = 16;
			return PSA_ALG_MD5;
		case HashingContext::HASH_SHA1:
			r_size = 20;
			return PSA_ALG_SHA_1;
		case HashingContext::HASH_SHA256:
			r_size = 32;
			return PSA_ALG_SHA_256;
		default:
			r_size = 0;
			ERR_FAIL_V_MSG(PSA_ALG_NONE, "Invalid hash type.");
	}
}

mbedtls_md_type_t CryptoMbedTLS::md_type_from_hashtype(HashingContext::HashType p_hash_type, int &r_size) {
	switch (p_hash_type) {
		case HashingContext::HASH_MD5:
			r_size = 16;
			return MBEDTLS_MD_MD5;
		case HashingContext::HASH_SHA1:
			r_size = 20;
			return MBEDTLS_MD_SHA1;
		case HashingContext::HASH_SHA256:
			r_size = 32;
			return MBEDTLS_MD_SHA256;
		default:
			r_size = 0;
			ERR_FAIL_V_MSG(MBEDTLS_MD_NONE, "Invalid hash type.");
	}
}

Vector<uint8_t> CryptoMbedTLS::sign(HashingContext::HashType p_hash_type, const Vector<uint8_t> &p_hash, Ref<CryptoKey> p_key) {
	int size;
	psa_algorithm_t hash_type = CryptoMbedTLS::psa_alg_from_hashtype(p_hash_type, size);
	ERR_FAIL_COND_V_MSG(hash_type == PSA_ALG_NONE, Vector<uint8_t>(), "Invalid hash type.");
	ERR_FAIL_COND_V_MSG(p_hash.size() != size, Vector<uint8_t>(), "Invalid hash provided. Size must be " + itos(size));
	Ref<CryptoKeyMbedTLS> key = static_cast<Ref<CryptoKeyMbedTLS>>(p_key);
	ERR_FAIL_COND_V_MSG(key.is_null(), Vector<uint8_t>(), "Invalid key provided.");
	ERR_FAIL_COND_V_MSG(key->is_public_only(), Vector<uint8_t>(), "Invalid key provided. Cannot sign with public_only keys.");
	size_t sig_size = 0;
	psa_key_type_t key_type = key->get_key_type();
	psa_algorithm_t alg = PSA_ALG_NONE;
	if (PSA_KEY_TYPE_IS_RSA(key_type)) {
		alg = PSA_ALG_RSA_PKCS1V15_SIGN(hash_type);
	} else if (PSA_KEY_TYPE_IS_ECC(key_type)) {
		alg = PSA_ALG_ECDSA(hash_type);
	}
	ERR_FAIL_COND_V_MSG(alg == PSA_ALG_NONE, Vector<uint8_t>(), "Unknown key type: " + itos(key_type));
	unsigned char buf[PSA_SIGNATURE_MAX_SIZE];
	Vector<uint8_t> out;
	psa_key_id_t pk = key->reimport(PSA_KEY_USAGE_SIGN_HASH, alg);
	int ret = psa_sign_hash(pk, alg, p_hash.ptr(), p_hash.size(), buf, sizeof(buf), &sig_size);
	psa_destroy_key(pk);
	ERR_FAIL_COND_V_MSG(ret, out, "Error while signing: " + itos(ret));
	out.resize(sig_size);
	memcpy(out.ptrw(), buf, sig_size);
	return out;
}

bool CryptoMbedTLS::verify(HashingContext::HashType p_hash_type, const Vector<uint8_t> &p_hash, const Vector<uint8_t> &p_signature, Ref<CryptoKey> p_key) {
	int size;
	psa_algorithm_t hash_type = CryptoMbedTLS::psa_alg_from_hashtype(p_hash_type, size);
	ERR_FAIL_COND_V_MSG(hash_type == PSA_ALG_NONE, false, "Invalid hash type.");
	ERR_FAIL_COND_V_MSG(p_hash.size() != size, false, "Invalid hash provided. Size must be " + itos(size));
	Ref<CryptoKeyMbedTLS> key = static_cast<Ref<CryptoKeyMbedTLS>>(p_key);
	ERR_FAIL_COND_V_MSG(key.is_null(), false, "Invalid key provided.");
	psa_key_type_t key_type = key->get_key_type();
	psa_algorithm_t alg = PSA_ALG_NONE;
	if (PSA_KEY_TYPE_IS_RSA(key_type)) {
		alg = PSA_ALG_RSA_PKCS1V15_SIGN(hash_type);
	} else if (PSA_KEY_TYPE_IS_ECC(key_type)) {
		alg = PSA_ALG_ECDSA(hash_type);
	}
	ERR_FAIL_COND_V_MSG(alg == PSA_ALG_NONE, false, "Unknown key type: " + itos(key_type));
	psa_key_id_t pk = key->reimport(PSA_KEY_USAGE_VERIFY_HASH, alg);
	int ret = psa_verify_hash(pk, alg, p_hash.ptr(), p_hash.size(), p_signature.ptr(), p_signature.size());
	psa_destroy_key(pk);
	return ret == PSA_SUCCESS;
}

Vector<uint8_t> CryptoMbedTLS::encrypt(Ref<CryptoKey> p_key, const Vector<uint8_t> &p_plaintext) {
	Ref<CryptoKeyMbedTLS> key = static_cast<Ref<CryptoKeyMbedTLS>>(p_key);
	ERR_FAIL_COND_V_MSG(key.is_null(), Vector<uint8_t>(), "Invalid key provided.");
	uint8_t buf[PSA_ASYMMETRIC_ENCRYPT_OUTPUT_MAX_SIZE];
	size_t size;
	psa_key_id_t pk = key->reimport(PSA_KEY_USAGE_ENCRYPT, PSA_ALG_RSA_PKCS1V15_CRYPT);
	int ret = psa_asymmetric_encrypt(pk, PSA_ALG_RSA_PKCS1V15_CRYPT, p_plaintext.ptr(), p_plaintext.size(), nullptr, 0, buf, sizeof(buf), &size);
	psa_destroy_key(pk);
	ERR_FAIL_COND_V_MSG(ret, Vector<uint8_t>(), "Error while encrypting: " + itos(ret));
	Vector<uint8_t> out;
	out.resize(size);
	memcpy(out.ptrw(), buf, size);
	return out;
}

Vector<uint8_t> CryptoMbedTLS::decrypt(Ref<CryptoKey> p_key, const Vector<uint8_t> &p_ciphertext) {
	Ref<CryptoKeyMbedTLS> key = static_cast<Ref<CryptoKeyMbedTLS>>(p_key);
	ERR_FAIL_COND_V_MSG(key.is_null(), Vector<uint8_t>(), "Invalid key provided.");
	ERR_FAIL_COND_V_MSG(key->is_public_only(), Vector<uint8_t>(), "Invalid key provided. Cannot decrypt using a public_only key.");
	psa_key_id_t pk = key->reimport(PSA_KEY_USAGE_DECRYPT, PSA_ALG_RSA_PKCS1V15_CRYPT);
	uint8_t buf[PSA_ASYMMETRIC_DECRYPT_OUTPUT_MAX_SIZE];
	size_t size;
	int ret = psa_asymmetric_decrypt(pk, PSA_ALG_RSA_PKCS1V15_CRYPT, p_ciphertext.ptr(), p_ciphertext.size(), nullptr, 0, buf, sizeof(buf), &size);
	psa_destroy_key(pk);
	ERR_FAIL_COND_V_MSG(ret, Vector<uint8_t>(), "Error while decrypting: " + itos(ret));
	Vector<uint8_t> out;
	out.resize(size);
	memcpy(out.ptrw(), buf, size);
	return out;
}
