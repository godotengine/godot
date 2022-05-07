/*************************************************************************/
/*  crypto_core.cpp                                                      */
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

#include "crypto_core.h"

#include <mbedtls/aes.h>
#include <mbedtls/base64.h>
#include <mbedtls/md5.h>
#include <mbedtls/sha1.h>
#include <mbedtls/sha256.h>

// MD5
CryptoCore::MD5Context::MD5Context() {
	ctx = memalloc(sizeof(mbedtls_md5_context));
	mbedtls_md5_init((mbedtls_md5_context *)ctx);
}

CryptoCore::MD5Context::~MD5Context() {
	mbedtls_md5_free((mbedtls_md5_context *)ctx);
	memfree((mbedtls_md5_context *)ctx);
}

Error CryptoCore::MD5Context::start() {
	int ret = mbedtls_md5_starts_ret((mbedtls_md5_context *)ctx);
	return ret ? FAILED : OK;
}

Error CryptoCore::MD5Context::update(const uint8_t *p_src, size_t p_len) {
	int ret = mbedtls_md5_update_ret((mbedtls_md5_context *)ctx, p_src, p_len);
	return ret ? FAILED : OK;
}

Error CryptoCore::MD5Context::finish(unsigned char r_hash[16]) {
	int ret = mbedtls_md5_finish_ret((mbedtls_md5_context *)ctx, r_hash);
	return ret ? FAILED : OK;
}

// SHA1
CryptoCore::SHA1Context::SHA1Context() {
	ctx = memalloc(sizeof(mbedtls_sha1_context));
	mbedtls_sha1_init((mbedtls_sha1_context *)ctx);
}

CryptoCore::SHA1Context::~SHA1Context() {
	mbedtls_sha1_free((mbedtls_sha1_context *)ctx);
	memfree((mbedtls_sha1_context *)ctx);
}

Error CryptoCore::SHA1Context::start() {
	int ret = mbedtls_sha1_starts_ret((mbedtls_sha1_context *)ctx);
	return ret ? FAILED : OK;
}

Error CryptoCore::SHA1Context::update(const uint8_t *p_src, size_t p_len) {
	int ret = mbedtls_sha1_update_ret((mbedtls_sha1_context *)ctx, p_src, p_len);
	return ret ? FAILED : OK;
}

Error CryptoCore::SHA1Context::finish(unsigned char r_hash[20]) {
	int ret = mbedtls_sha1_finish_ret((mbedtls_sha1_context *)ctx, r_hash);
	return ret ? FAILED : OK;
}

// SHA256
CryptoCore::SHA256Context::SHA256Context() {
	ctx = memalloc(sizeof(mbedtls_sha256_context));
	mbedtls_sha256_init((mbedtls_sha256_context *)ctx);
}

CryptoCore::SHA256Context::~SHA256Context() {
	mbedtls_sha256_free((mbedtls_sha256_context *)ctx);
	memfree((mbedtls_sha256_context *)ctx);
}

Error CryptoCore::SHA256Context::start() {
	int ret = mbedtls_sha256_starts_ret((mbedtls_sha256_context *)ctx, 0);
	return ret ? FAILED : OK;
}

Error CryptoCore::SHA256Context::update(const uint8_t *p_src, size_t p_len) {
	int ret = mbedtls_sha256_update_ret((mbedtls_sha256_context *)ctx, p_src, p_len);
	return ret ? FAILED : OK;
}

Error CryptoCore::SHA256Context::finish(unsigned char r_hash[32]) {
	int ret = mbedtls_sha256_finish_ret((mbedtls_sha256_context *)ctx, r_hash);
	return ret ? FAILED : OK;
}

// AES256
CryptoCore::AESContext::AESContext() {
	ctx = memalloc(sizeof(mbedtls_aes_context));
	mbedtls_aes_init((mbedtls_aes_context *)ctx);
}

CryptoCore::AESContext::~AESContext() {
	mbedtls_aes_free((mbedtls_aes_context *)ctx);
	memfree((mbedtls_aes_context *)ctx);
}

Error CryptoCore::AESContext::set_encode_key(const uint8_t *p_key, size_t p_bits) {
	int ret = mbedtls_aes_setkey_enc((mbedtls_aes_context *)ctx, p_key, p_bits);
	return ret ? FAILED : OK;
}

Error CryptoCore::AESContext::set_decode_key(const uint8_t *p_key, size_t p_bits) {
	int ret = mbedtls_aes_setkey_dec((mbedtls_aes_context *)ctx, p_key, p_bits);
	return ret ? FAILED : OK;
}

Error CryptoCore::AESContext::encrypt_ecb(const uint8_t p_src[16], uint8_t r_dst[16]) {
	int ret = mbedtls_aes_crypt_ecb((mbedtls_aes_context *)ctx, MBEDTLS_AES_ENCRYPT, p_src, r_dst);
	return ret ? FAILED : OK;
}

Error CryptoCore::AESContext::decrypt_ecb(const uint8_t p_src[16], uint8_t r_dst[16]) {
	int ret = mbedtls_aes_crypt_ecb((mbedtls_aes_context *)ctx, MBEDTLS_AES_DECRYPT, p_src, r_dst);
	return ret ? FAILED : OK;
}

Error CryptoCore::AESContext::encrypt_cbc(size_t p_length, uint8_t r_iv[16], const uint8_t *p_src, uint8_t *r_dst) {
	int ret = mbedtls_aes_crypt_cbc((mbedtls_aes_context *)ctx, MBEDTLS_AES_ENCRYPT, p_length, r_iv, p_src, r_dst);
	return ret ? FAILED : OK;
}

Error CryptoCore::AESContext::decrypt_cbc(size_t p_length, uint8_t r_iv[16], const uint8_t *p_src, uint8_t *r_dst) {
	int ret = mbedtls_aes_crypt_cbc((mbedtls_aes_context *)ctx, MBEDTLS_AES_DECRYPT, p_length, r_iv, p_src, r_dst);
	return ret ? FAILED : OK;
}

// CryptoCore
String CryptoCore::b64_encode_str(const uint8_t *p_src, int p_src_len) {
	int b64len = p_src_len / 3 * 4 + 4 + 1;
	PoolVector<uint8_t> b64buff;
	b64buff.resize(b64len);
	PoolVector<uint8_t>::Write w64 = b64buff.write();
	size_t strlen = 0;
	int ret = b64_encode(&w64[0], b64len, &strlen, p_src, p_src_len);
	w64[strlen] = 0;
	return ret ? String() : (const char *)&w64[0];
}

Error CryptoCore::b64_encode(uint8_t *r_dst, int p_dst_len, size_t *r_len, const uint8_t *p_src, int p_src_len) {
	int ret = mbedtls_base64_encode(r_dst, p_dst_len, r_len, p_src, p_src_len);
	return ret ? FAILED : OK;
}

Error CryptoCore::b64_decode(uint8_t *r_dst, int p_dst_len, size_t *r_len, const uint8_t *p_src, int p_src_len) {
	int ret = mbedtls_base64_decode(r_dst, p_dst_len, r_len, p_src, p_src_len);
	return ret ? FAILED : OK;
}

Error CryptoCore::md5(const uint8_t *p_src, int p_src_len, unsigned char r_hash[16]) {
	int ret = mbedtls_md5_ret(p_src, p_src_len, r_hash);
	return ret ? FAILED : OK;
}

Error CryptoCore::sha1(const uint8_t *p_src, int p_src_len, unsigned char r_hash[20]) {
	int ret = mbedtls_sha1_ret(p_src, p_src_len, r_hash);
	return ret ? FAILED : OK;
}

Error CryptoCore::sha256(const uint8_t *p_src, int p_src_len, unsigned char r_hash[32]) {
	int ret = mbedtls_sha256_ret(p_src, p_src_len, r_hash, 0);
	return ret ? FAILED : OK;
}
