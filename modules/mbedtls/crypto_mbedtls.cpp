/*************************************************************************/
/*  crypto_mbedtls.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "crypto_mbedtls.h"

#include "core/os/file_access.h"

#include "core/engine.h"
#include "core/io/certs_compressed.gen.h"
#include "core/io/compression.h"
#include "core/project_settings.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_settings.h"
#endif
#define PEM_BEGIN_CRT "-----BEGIN CERTIFICATE-----\n"
#define PEM_END_CRT "-----END CERTIFICATE-----\n"

#include "mbedtls/pem.h"
#include <mbedtls/debug.h>

CryptoKey *CryptoKeyMbedTLS::create() {
	return memnew(CryptoKeyMbedTLS);
}

Error CryptoKeyMbedTLS::load(String p_path) {
	ERR_FAIL_COND_V_MSG(locks, ERR_ALREADY_IN_USE, "Key is in use");

	PoolByteArray out;
	FileAccess *f = FileAccess::open(p_path, FileAccess::READ);
	ERR_FAIL_COND_V_MSG(!f, ERR_INVALID_PARAMETER, "Cannot open CryptoKeyMbedTLS file '" + p_path + "'.");

	int flen = f->get_len();
	out.resize(flen + 1);
	{
		PoolByteArray::Write w = out.write();
		f->get_buffer(w.ptr(), flen);
		w[flen] = 0; //end f string
	}
	memdelete(f);

	int ret = mbedtls_pk_parse_key(&pkey, out.read().ptr(), out.size(), NULL, 0);
	// We MUST zeroize the memory for safety!
	mbedtls_platform_zeroize(out.write().ptr(), out.size());
	ERR_FAIL_COND_V_MSG(ret, FAILED, "Error parsing private key '" + itos(ret) + "'.");

	return OK;
}

Error CryptoKeyMbedTLS::save(String p_path) {
	FileAccess *f = FileAccess::open(p_path, FileAccess::WRITE);
	ERR_FAIL_COND_V_MSG(!f, ERR_INVALID_PARAMETER, "Cannot save CryptoKeyMbedTLS file '" + p_path + "'.");

	unsigned char w[16000];
	memset(w, 0, sizeof(w));

	int ret = mbedtls_pk_write_key_pem(&pkey, w, sizeof(w));
	if (ret != 0) {
		memdelete(f);
		memset(w, 0, sizeof(w)); // Zeroize anything we might have written.
		ERR_FAIL_V_MSG(FAILED, "Error writing key '" + itos(ret) + "'.");
	}

	size_t len = strlen((char *)w);
	f->store_buffer(w, len);
	memdelete(f);
	memset(w, 0, sizeof(w)); // Zeroize temporary buffer.
	return OK;
}

X509Certificate *X509CertificateMbedTLS::create() {
	return memnew(X509CertificateMbedTLS);
}

Error X509CertificateMbedTLS::load(String p_path) {
	ERR_FAIL_COND_V_MSG(locks, ERR_ALREADY_IN_USE, "Certificate is in use");

	PoolByteArray out;
	FileAccess *f = FileAccess::open(p_path, FileAccess::READ);
	ERR_FAIL_COND_V_MSG(!f, ERR_INVALID_PARAMETER, "Cannot open X509CertificateMbedTLS file '" + p_path + "'.");

	int flen = f->get_len();
	out.resize(flen + 1);
	{
		PoolByteArray::Write w = out.write();
		f->get_buffer(w.ptr(), flen);
		w[flen] = 0; //end f string
	}
	memdelete(f);

	int ret = mbedtls_x509_crt_parse(&cert, out.read().ptr(), out.size());
	ERR_FAIL_COND_V_MSG(ret, FAILED, "Error parsing some certificates: " + itos(ret));

	return OK;
}

Error X509CertificateMbedTLS::load_from_memory(const uint8_t *p_buffer, int p_len) {
	ERR_FAIL_COND_V_MSG(locks, ERR_ALREADY_IN_USE, "Certificate is in use");

	int ret = mbedtls_x509_crt_parse(&cert, p_buffer, p_len);
	ERR_FAIL_COND_V_MSG(ret, FAILED, "Error parsing certificates: " + itos(ret));
	return OK;
}

Error X509CertificateMbedTLS::save(String p_path) {
	FileAccess *f = FileAccess::open(p_path, FileAccess::WRITE);
	ERR_FAIL_COND_V_MSG(!f, ERR_INVALID_PARAMETER, "Cannot save X509CertificateMbedTLS file '" + p_path + "'.");

	mbedtls_x509_crt *crt = &cert;
	while (crt) {
		unsigned char w[4096];
		size_t wrote = 0;
		int ret = mbedtls_pem_write_buffer(PEM_BEGIN_CRT, PEM_END_CRT, cert.raw.p, cert.raw.len, w, sizeof(w), &wrote);
		if (ret != 0 || wrote == 0) {
			memdelete(f);
			ERR_FAIL_V_MSG(FAILED, "Error writing certificate '" + itos(ret) + "'.");
		}

		f->store_buffer(w, wrote - 1); // don't write the string terminator
		crt = crt->next;
	}
	memdelete(f);
	return OK;
}

Crypto *CryptoMbedTLS::create() {
	return memnew(CryptoMbedTLS);
}

void CryptoMbedTLS::initialize_crypto() {

#ifdef DEBUG_ENABLED
	mbedtls_debug_set_threshold(1);
#endif

	Crypto::_create = create;
	Crypto::_load_default_certificates = load_default_certificates;
	X509CertificateMbedTLS::make_default();
	CryptoKeyMbedTLS::make_default();
}

void CryptoMbedTLS::finalize_crypto() {
	Crypto::_create = NULL;
	Crypto::_load_default_certificates = NULL;
	if (default_certs) {
		memdelete(default_certs);
		default_certs = NULL;
	}
	X509CertificateMbedTLS::finalize();
	CryptoKeyMbedTLS::finalize();
}

CryptoMbedTLS::CryptoMbedTLS() {
	mbedtls_ctr_drbg_init(&ctr_drbg);
	mbedtls_entropy_init(&entropy);
	int ret = mbedtls_ctr_drbg_seed(&ctr_drbg, mbedtls_entropy_func, &entropy, NULL, 0);
	if (ret != 0) {
		ERR_PRINTS(" failed\n  ! mbedtls_ctr_drbg_seed returned an error" + itos(ret));
	}
}

CryptoMbedTLS::~CryptoMbedTLS() {
	mbedtls_ctr_drbg_free(&ctr_drbg);
	mbedtls_entropy_free(&entropy);
}

X509CertificateMbedTLS *CryptoMbedTLS::default_certs = NULL;

X509CertificateMbedTLS *CryptoMbedTLS::get_default_certificates() {
	return default_certs;
}

void CryptoMbedTLS::load_default_certificates(String p_path) {
	ERR_FAIL_COND(default_certs != NULL);

	default_certs = memnew(X509CertificateMbedTLS);
	ERR_FAIL_COND(default_certs == NULL);

	String certs_path = GLOBAL_DEF("network/ssl/certificates", "");

	if (p_path != "") {
		// Use certs defined in project settings.
		default_certs->load(p_path);
	}
#ifdef BUILTIN_CERTS_ENABLED
	else {
		// Use builtin certs only if user did not override it in project settings.
		PoolByteArray out;
		out.resize(_certs_uncompressed_size + 1);
		PoolByteArray::Write w = out.write();
		Compression::decompress(w.ptr(), _certs_uncompressed_size, _certs_compressed, _certs_compressed_size, Compression::MODE_DEFLATE);
		w[_certs_uncompressed_size] = 0; // Make sure it ends with string terminator
#ifdef DEBUG_ENABLED
		print_verbose("Loaded builtin certs");
#endif
		default_certs->load_from_memory(out.read().ptr(), out.size());
	}
#endif
}

Ref<CryptoKey> CryptoMbedTLS::generate_rsa(int p_bytes) {
	Ref<CryptoKeyMbedTLS> out;
	out.instance();
	int ret = mbedtls_pk_setup(&(out->pkey), mbedtls_pk_info_from_type(MBEDTLS_PK_RSA));
	ERR_FAIL_COND_V(ret != 0, NULL);
	ret = mbedtls_rsa_gen_key(mbedtls_pk_rsa(out->pkey), mbedtls_ctr_drbg_random, &ctr_drbg, p_bytes, 65537);
	ERR_FAIL_COND_V(ret != 0, NULL);
	return out;
}

Ref<X509Certificate> CryptoMbedTLS::generate_self_signed_certificate(Ref<CryptoKey> p_key, String p_issuer_name, String p_not_before, String p_not_after) {
	Ref<CryptoKeyMbedTLS> key = static_cast<Ref<CryptoKeyMbedTLS> >(p_key);
	mbedtls_x509write_cert crt;
	mbedtls_x509write_crt_init(&crt);

	mbedtls_x509write_crt_set_subject_key(&crt, &(key->pkey));
	mbedtls_x509write_crt_set_issuer_key(&crt, &(key->pkey));
	mbedtls_x509write_crt_set_subject_name(&crt, p_issuer_name.utf8().get_data());
	mbedtls_x509write_crt_set_issuer_name(&crt, p_issuer_name.utf8().get_data());
	mbedtls_x509write_crt_set_version(&crt, MBEDTLS_X509_CRT_VERSION_3);
	mbedtls_x509write_crt_set_md_alg(&crt, MBEDTLS_MD_SHA256);

	mbedtls_mpi serial;
	mbedtls_mpi_init(&serial);
	uint8_t rand_serial[20];
	mbedtls_ctr_drbg_random(&ctr_drbg, rand_serial, 20);
	ERR_FAIL_COND_V(mbedtls_mpi_read_binary(&serial, rand_serial, 20), NULL);
	mbedtls_x509write_crt_set_serial(&crt, &serial);

	mbedtls_x509write_crt_set_validity(&crt, p_not_before.utf8().get_data(), p_not_after.utf8().get_data());
	mbedtls_x509write_crt_set_basic_constraints(&crt, 1, -1);
	mbedtls_x509write_crt_set_basic_constraints(&crt, 1, 0);

	unsigned char buf[4096];
	memset(buf, 0, 4096);
	Ref<X509CertificateMbedTLS> out;
	out.instance();
	mbedtls_x509write_crt_pem(&crt, buf, 4096, mbedtls_ctr_drbg_random, &ctr_drbg);

	int err = mbedtls_x509_crt_parse(&(out->cert), buf, 4096);
	if (err != 0) {
		mbedtls_mpi_free(&serial);
		mbedtls_x509write_crt_free(&crt);
		ERR_PRINTS("Generated invalid certificate: " + itos(err));
		return NULL;
	}

	mbedtls_mpi_free(&serial);
	mbedtls_x509write_crt_free(&crt);
	return out;
}

PoolByteArray CryptoMbedTLS::generate_random_bytes(int p_bytes) {
	PoolByteArray out;
	out.resize(p_bytes);
	mbedtls_ctr_drbg_random(&ctr_drbg, out.write().ptr(), p_bytes);
	return out;
}
