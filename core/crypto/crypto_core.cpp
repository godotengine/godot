/**************************************************************************/
/*  crypto_core.cpp                                                       */
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

#include "crypto_core.h"

#include "core/os/os.h"
#include "core/string/ustring.h"

#include <mbedtls/base64.h>
#include <psa/crypto.h>

// MD5
CryptoCore::MD5Context::MD5Context() {
	ctx = memalloc_zeroed(sizeof(psa_hash_operation_t));
}

CryptoCore::MD5Context::~MD5Context() {
	psa_hash_abort((psa_hash_operation_t *)ctx);
	memfree((psa_hash_operation_t *)ctx);
}

Error CryptoCore::MD5Context::start() {
	int ret = psa_hash_setup((psa_hash_operation_t *)ctx, PSA_ALG_MD5);
	return ret ? FAILED : OK;
}

Error CryptoCore::MD5Context::update(const uint8_t *p_src, size_t p_len) {
	int ret = psa_hash_update((psa_hash_operation_t *)ctx, p_src, p_len);
	return ret ? FAILED : OK;
}

Error CryptoCore::MD5Context::finish(unsigned char r_hash[16]) {
	size_t size = 0;
	int ret = psa_hash_finish((psa_hash_operation_t *)ctx, r_hash, 16, &size);
	if (ret) {
		psa_hash_abort((psa_hash_operation_t *)ctx);
		return FAILED;
	}
	ERR_FAIL_COND_V(size != 16, ERR_BUG);
	return OK;
}

// SHA1
CryptoCore::SHA1Context::SHA1Context() {
	ctx = memalloc_zeroed(sizeof(psa_hash_operation_t));
}

CryptoCore::SHA1Context::~SHA1Context() {
	psa_hash_abort((psa_hash_operation_t *)ctx);
	memfree((psa_hash_operation_t *)ctx);
}

Error CryptoCore::SHA1Context::start() {
	int ret = psa_hash_setup((psa_hash_operation_t *)ctx, PSA_ALG_SHA_1);
	return ret ? FAILED : OK;
}

Error CryptoCore::SHA1Context::update(const uint8_t *p_src, size_t p_len) {
	int ret = psa_hash_update((psa_hash_operation_t *)ctx, p_src, p_len);
	return ret ? FAILED : OK;
}

Error CryptoCore::SHA1Context::finish(unsigned char r_hash[20]) {
	size_t size = 0;
	int ret = psa_hash_finish((psa_hash_operation_t *)ctx, r_hash, 20, &size);
	if (ret) {
		psa_hash_abort((psa_hash_operation_t *)ctx);
		return FAILED;
	}
	ERR_FAIL_COND_V(size != 20, ERR_BUG);
	return OK;
}

// SHA256
CryptoCore::SHA256Context::SHA256Context() {
	ctx = memalloc_zeroed(sizeof(psa_hash_operation_t));
}

CryptoCore::SHA256Context::~SHA256Context() {
	psa_hash_abort((psa_hash_operation_t *)ctx);
	memfree((psa_hash_operation_t *)ctx);
}

Error CryptoCore::SHA256Context::start() {
	int ret = psa_hash_setup((psa_hash_operation_t *)ctx, PSA_ALG_SHA_256);
	return ret ? FAILED : OK;
}

Error CryptoCore::SHA256Context::update(const uint8_t *p_src, size_t p_len) {
	int ret = psa_hash_update((psa_hash_operation_t *)ctx, p_src, p_len);
	return ret ? FAILED : OK;
}

Error CryptoCore::SHA256Context::finish(unsigned char r_hash[32]) {
	size_t size = 0;
	int ret = psa_hash_finish((psa_hash_operation_t *)ctx, r_hash, 32, &size);
	if (ret) {
		psa_hash_abort((psa_hash_operation_t *)ctx);
		return FAILED;
	}
	ERR_FAIL_COND_V(size != 32, ERR_BUG);
	return OK;
}

// AES256
CryptoCore::AESContext::AESContext() {
	ctx = memalloc_zeroed(sizeof(psa_cipher_operation_t));
}

CryptoCore::AESContext::~AESContext() {
	psa_cipher_abort((psa_cipher_operation_t *)ctx);
	psa_destroy_key(key_id);
	memfree((psa_cipher_operation_t *)ctx);
}

void CryptoCore::AESContext::reset() {
	psa_cipher_abort((psa_cipher_operation_t *)ctx);
	psa_destroy_key(key_id);
	key_id = PSA_KEY_ID_NULL;
	alg = PSA_ALG_NONE;
}

Error CryptoCore::AESContext::setup(Mode p_mode, Cipher p_cipher, const uint8_t *p_key, size_t p_key_length, const uint8_t *p_iv, size_t p_iv_size) {
	reset();
	psa_key_attributes_t attr = PSA_KEY_ATTRIBUTES_INIT;
	switch (p_mode) {
		case Mode::NONE:
			return OK;
		case Mode::ENCRYPT:
		case Mode::DECRYPT:
			break;
		default:
			return ERR_INVALID_PARAMETER;
	}
	switch (p_cipher) {
		case Cipher::CBC:
			alg = PSA_ALG_CBC_NO_PADDING;
			break;
		case Cipher::ECB:
			alg = PSA_ALG_ECB_NO_PADDING;
			break;
		case Cipher::CFB:
			alg = PSA_ALG_CFB;
			break;
		default:
			return ERR_INVALID_PARAMETER;
	}
	psa_set_key_usage_flags(&attr, PSA_KEY_USAGE_ENCRYPT | PSA_KEY_USAGE_DECRYPT);
	psa_set_key_algorithm(&attr, alg);
	psa_set_key_type(&attr, PSA_KEY_TYPE_AES);
	psa_set_key_bits(&attr, p_key_length << 3);
	int ret = psa_import_key(&attr, p_key, p_key_length, &key_id);
	if (ret) {
		return FAILED;
	}
	if (p_mode == Mode::ENCRYPT) {
		ret = psa_cipher_encrypt_setup((psa_cipher_operation_t *)ctx, key_id, alg);
	} else {
		ret = psa_cipher_decrypt_setup((psa_cipher_operation_t *)ctx, key_id, alg);
	}
	if (ret == PSA_SUCCESS && p_iv_size) {
		ret = psa_cipher_set_iv((psa_cipher_operation_t *)ctx, p_iv, p_iv_size);
	}
	if (ret) {
		reset();
		return FAILED;
	}
	return OK;
}

Error CryptoCore::AESContext::update(const uint8_t *p_src, size_t p_src_size, uint8_t *r_dst, size_t p_dst_size) {
	size_t size = 0;
	int ret = psa_cipher_update((psa_cipher_operation_t *)ctx, p_src, p_src_size, r_dst, p_dst_size, &size);
	if (ret) {
		return FAILED;
	}
	ERR_FAIL_COND_V(p_dst_size != size, ERR_BUG);
	return OK;
}

Error CryptoCore::AESContext::finish(uint8_t *r_dst, size_t p_dst_size) {
	size_t size = 0;
	int ret = psa_cipher_finish((psa_cipher_operation_t *)ctx, r_dst, p_dst_size, &size);
	reset();
	if (ret) {
		return FAILED;
	}
	ERR_FAIL_COND_V(p_dst_size != size, ERR_BUG);
	return OK;
}

// CryptoCore
Error CryptoCore::generate_random(uint8_t *r_buffer, size_t p_buffer_len) {
	if (unlikely(!initialized)) {
		// Fall back to raw get_entropy (might affect performance).
		return OS::get_singleton()->get_entropy(r_buffer, p_buffer_len);
	}
	int ret = psa_generate_random(r_buffer, p_buffer_len);
	ERR_FAIL_COND_V(ret, FAILED);
	return OK;
}

String CryptoCore::b64_encode_str(const uint8_t *p_src, size_t p_src_len) {
	size_t b64len = p_src_len / 3 * 4 + 4 + 1;
	Vector<uint8_t> b64buff;
	b64buff.resize(b64len);
	uint8_t *w64 = b64buff.ptrw();
	size_t strlen = 0;
	int ret = b64_encode(&w64[0], b64len, &strlen, p_src, p_src_len);
	w64[strlen] = 0;
	return ret ? String() : (const char *)&w64[0];
}

Error CryptoCore::b64_encode(uint8_t *r_dst, size_t p_dst_len, size_t *r_len, const uint8_t *p_src, size_t p_src_len) {
	int ret = mbedtls_base64_encode(r_dst, p_dst_len, r_len, p_src, p_src_len);
	return ret ? FAILED : OK;
}

Error CryptoCore::b64_decode(uint8_t *r_dst, size_t p_dst_len, size_t *r_len, const uint8_t *p_src, size_t p_src_len) {
	int ret = mbedtls_base64_decode(r_dst, p_dst_len, r_len, p_src, p_src_len);
	return ret ? FAILED : OK;
}

Error CryptoCore::md5(const uint8_t *p_src, size_t p_src_len, unsigned char r_hash[16]) {
	ERR_FAIL_COND_V(PSA_HASH_LENGTH(PSA_ALG_MD5) != 16, ERR_BUG);
	size_t size = 0;
	int ret = psa_hash_compute(PSA_ALG_MD5, p_src, p_src_len, r_hash, 16, &size);
	return ret ? FAILED : OK;
}

Error CryptoCore::sha1(const uint8_t *p_src, size_t p_src_len, unsigned char r_hash[20]) {
	ERR_FAIL_COND_V(PSA_HASH_LENGTH(PSA_ALG_SHA_1) != 20, ERR_BUG);
	size_t size = 0;
	int ret = psa_hash_compute(PSA_ALG_SHA_1, p_src, p_src_len, r_hash, 20, &size);
	return ret ? FAILED : OK;
}

Error CryptoCore::sha256(const uint8_t *p_src, size_t p_src_len, unsigned char r_hash[32]) {
	ERR_FAIL_COND_V(PSA_HASH_LENGTH(PSA_ALG_SHA_256) != 32, ERR_BUG);
	size_t size = 0;
	int ret = psa_hash_compute(PSA_ALG_SHA_256, p_src, p_src_len, r_hash, 32, &size);
	return ret ? FAILED : OK;
}

Error CryptoCore::encrypt_cfb(const uint8_t *p_src, uint8_t *r_dst, size_t p_length, const uint8_t *p_key, size_t p_key_length, const uint8_t p_iv[16]) {
	AESContext ctx;
	Error err = ctx.setup(AESContext::Mode::ENCRYPT, AESContext::Cipher::CFB, p_key, p_key_length, p_iv, 16);
	ERR_FAIL_COND_V(err, err);
	err = ctx.update(p_src, p_length, r_dst, p_length);
	ERR_FAIL_COND_V(err, err);
	err = ctx.finish(nullptr, 0);
	ERR_FAIL_COND_V(err, err);
	return OK;
}

Error CryptoCore::decrypt_cfb(const uint8_t *p_src, uint8_t *r_dst, size_t p_length, const uint8_t *p_key, size_t p_key_length, const uint8_t p_iv[16]) {
	AESContext ctx;
	Error err = ctx.setup(AESContext::Mode::DECRYPT, AESContext::Cipher::CFB, p_key, p_key_length, p_iv, 16);
	ERR_FAIL_COND_V(err, err);
	err = ctx.update(p_src, p_length, r_dst, p_length);
	ERR_FAIL_COND_V(err, err);
	err = ctx.finish(nullptr, 0);
	ERR_FAIL_COND_V(err, err);
	return OK;
}

bool CryptoCore::initialized = false;
void CryptoCore::initialize() {
	ERR_FAIL_COND(initialized);
#ifdef GODOT_MBEDTLS_PLATFORM
	godot_mbedtls_platform_init();
#endif

	int status = psa_crypto_init();
	ERR_FAIL_COND_MSG(status != PSA_SUCCESS, "Failed to initialize psa crypto. Cryptographic functions will not work.");
	initialized = true;
}
void CryptoCore::finalize() {
	if (!initialized) {
		return;
	}
	initialized = false;
	mbedtls_psa_crypto_free(); // Not part of PSA for reasons.
#ifdef GODOT_MBEDTLS_PLATFORM
	godot_mbedtls_platform_free();
#endif
}
