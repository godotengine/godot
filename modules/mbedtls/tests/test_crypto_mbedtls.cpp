/**************************************************************************/
/*  test_crypto_mbedtls.cpp                                               */
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

#include "test_crypto_mbedtls.h"

#include "../crypto_mbedtls.h"

#include "core/io/file_access.h"
#include "tests/test_macros.h"
#include "tests/test_utils.h"

namespace TestCryptoMbedTLS {

void hmac_digest_test(HashingContext::HashType ht, String expected_hex) {
	CryptoMbedTLS crypto;
	PackedByteArray key = String("supersecretkey").to_utf8_buffer();
	PackedByteArray msg = String("Return of the MAC!").to_utf8_buffer();
	PackedByteArray digest = crypto.hmac_digest(ht, key, msg);
	String hex = String::hex_encode_buffer(digest.ptr(), digest.size());
	CHECK(hex == expected_hex);
}

void hmac_context_digest_test(HashingContext::HashType ht, String expected_hex) {
	HMACContextMbedTLS ctx;
	PackedByteArray key = String("supersecretkey").to_utf8_buffer();
	PackedByteArray msg1 = String("Return of ").to_utf8_buffer();
	PackedByteArray msg2 = String("the MAC!").to_utf8_buffer();
	Error err = ctx.start(ht, key);
	CHECK(err == OK);
	err = ctx.update(msg1);
	CHECK(err == OK);
	err = ctx.update(msg2);
	CHECK(err == OK);
	PackedByteArray digest = ctx.finish();
	String hex = String::hex_encode_buffer(digest.ptr(), digest.size());
	CHECK(hex == expected_hex);
}

Ref<CryptoKey> load_key(const String &p_key_path, bool p_public_only) {
	Ref<CryptoKey> key(memnew(CryptoKeyMbedTLS));
	REQUIRE(key.is_valid());
	Error err = key->load(p_key_path, p_public_only);
	REQUIRE(err == OK);
	return key;
}

String read_file_s(const String &p_file_path) {
	Ref<FileAccess> file_access = FileAccess::open(p_file_path, FileAccess::READ);
	REQUIRE(file_access.is_valid());
	return file_access->get_as_utf8_string();
}

bool files_equal(const String &p_in_path, const String &p_out_path) {
	const String s_in = read_file_s(p_in_path);
	const String s_out = read_file_s(p_out_path);
	return s_in == s_out;
}

void crypto_key_public_only_test(const String &p_key_path, bool p_public_only) {
	Ref<CryptoKey> crypto_key = load_key(p_key_path, p_public_only);
	bool is_equal = crypto_key->is_public_only() == p_public_only;
	CHECK(is_equal);
}

void crypto_key_save_test(const String &p_in_path, const String &p_out_path, bool p_public_only) {
	Ref<CryptoKey> crypto_key = load_key(p_in_path, p_public_only);
	crypto_key->save(p_out_path, p_public_only);
	bool is_equal = files_equal(p_in_path, p_out_path);
	CHECK(is_equal);
}

void crypto_key_save_public_only_test(const String &p_in_priv_path, const String &p_in_pub_path, const String &p_out_path) {
	Ref<CryptoKey> crypto_key = load_key(p_in_priv_path, false);
	crypto_key->save(p_out_path, true);
	bool is_equal = files_equal(p_in_pub_path, p_out_path);
	CHECK(is_equal);
}

void crypto_key_generate_save_load(const String &p_out_priv_path, const String &p_out_pub_path) {
	Ref<Crypto> c(memnew(CryptoMbedTLS));
	REQUIRE(c.is_valid());
	Ref<CryptoKey> key = c->generate_rsa(1024);
	CHECK(key->save_to_string() != key->save_to_string(true));
	Error err = key->save(p_out_priv_path);
	REQUIRE(err == OK);
	err = key->save(p_out_pub_path, true);
	REQUIRE(err == OK);
	Ref<CryptoKey> priv = load_key(p_out_priv_path, false);
	Ref<CryptoKey> pub = load_key(p_out_pub_path, true);
	CHECK(pub->save_to_string(true) != priv->save_to_string());
	CHECK(pub->save_to_string(true) == priv->save_to_string(true));
	CHECK(pub->save_to_string(true) == key->save_to_string(true));
	CHECK(key->save_to_string() == priv->save_to_string());
}

void crypto_generate_sign_verify_test(HashingContext::HashType p_hash_type, const Vector<uint8_t> &p_hash) {
	Ref<Crypto> c(memnew(CryptoMbedTLS));
	REQUIRE(c.is_valid());
	Ref<CryptoKey> key = c->generate_rsa(1024);
	Vector<uint8_t> signature = c->sign(p_hash_type, p_hash, key);
	CHECK(signature.size() > 0);
	bool valid = c->verify(p_hash_type, p_hash, signature, key);
	CHECK(valid);
	Ref<CryptoKey> pub(memnew(CryptoKeyMbedTLS));
	pub->load_from_string(key->save_to_string(true), true);
	valid = c->verify(p_hash_type, p_hash, signature, pub);
	CHECK(valid);
}

void crypto_sign_verify_test(const String &p_in_priv_path, const String &p_in_pub_path, HashingContext::HashType p_hash_type, const Vector<uint8_t> &p_hash) {
	Ref<CryptoKey> priv = load_key(p_in_priv_path, false);
	Ref<CryptoKey> pub = load_key(p_in_pub_path, true);
	Ref<Crypto> c(memnew(CryptoMbedTLS));
	REQUIRE(c.is_valid());
	Vector<uint8_t> signature = c->sign(p_hash_type, p_hash, priv);
	CHECK(signature.size() > 0);
	bool valid = c->verify(p_hash_type, p_hash, signature, pub);
	CHECK(valid);
}

void crypto_verify_test(const String &p_in_pub_path, HashingContext::HashType p_hash_type, const Vector<uint8_t> &p_hash, const Vector<uint8_t> &p_signature, bool p_expected) {
	Ref<CryptoKey> pub = load_key(p_in_pub_path, true);
	Ref<Crypto> c(memnew(CryptoMbedTLS));
	REQUIRE(c.is_valid());
	bool valid = c->verify(p_hash_type, p_hash, p_signature, pub);
	CHECK_EQ(valid, p_expected);
}

bool equals(const PackedByteArray &p_a, const PackedByteArray &p_b) {
	if (p_a.size() != p_b.size()) {
		return false;
	}
	if (p_a.is_empty()) {
		return true;
	}
	return memcmp(p_a.ptr(), p_b.ptr(), p_a.size()) == 0;
}

void crypto_encrypt_decrypt_test(const String &p_in_priv_path, const String &p_in_pub_path, const Vector<uint8_t> &p_message) {
	Ref<CryptoKey> priv = load_key(p_in_priv_path, false);
	Ref<CryptoKey> pub = load_key(p_in_pub_path, true);
	Ref<Crypto> c(memnew(CryptoMbedTLS));
	REQUIRE(c.is_valid());
	Vector<uint8_t> ciphertext = c->encrypt(pub, p_message);
	Vector<uint8_t> plaintext = c->decrypt(priv, ciphertext);
	CHECK(equals(plaintext, p_message));
}

void crypto_decrypt_test(const String &p_in_priv_path, const Vector<uint8_t> &p_ciphertext, const Vector<uint8_t> &p_message, bool p_valid, bool p_try_pub_key) {
	Ref<CryptoKey> priv = load_key(p_in_priv_path, p_try_pub_key);
	Ref<Crypto> c(memnew(CryptoMbedTLS));
	REQUIRE(c.is_valid());
	Vector<uint8_t> plaintext = c->decrypt(priv, p_ciphertext);
	CHECK_EQ(equals(plaintext, p_message), p_valid);
}

} // namespace TestCryptoMbedTLS
