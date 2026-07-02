/**************************************************************************/
/*  test_aes_context.cpp                                                  */
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

#include "tests/test_macros.h"

TEST_FORCE_LINK(test_aes_context)

#include "core/crypto/aes_context.h"

namespace TestAes_context {

bool equals(const PackedByteArray &p_a, const PackedByteArray &p_b) {
	if (p_a.size() != p_b.size()) {
		return false;
	}
	if (p_a.is_empty()) {
		return true;
	}
	return memcmp(p_a.ptr(), p_b.ptr(), p_a.size()) == 0;
}

Vector<uint8_t> to_vec(String s) {
	return s.to_utf8_buffer();
}

TEST_CASE("[AESContext] ECB start") {
	AESContext::Mode modes[2] = { AESContext::MODE_ECB_ENCRYPT, AESContext::MODE_ECB_DECRYPT };
	Ref<AESContext> ctx;
	ctx.instantiate();
	for (int i = 0; i < 2; i++) {
		AESContext::Mode mode = modes[i];
		// ECB has no IV
		PackedByteArray iv;
		// Key must be either 16 or 32 bytes.
		Vector<uint8_t> key;

		// Valid key (16)
		key.resize(16);
		Error err = ctx->start(mode, key, iv);
		CHECK_EQ(err, OK);
		ctx->finish();

		// Valid key (32)
		key.resize(32);
		err = ctx->start(mode, key, iv);
		CHECK_EQ(err, OK);

		ERR_PRINT_OFF;

		// Already in use
		err = ctx->start(mode, key, iv);
		CHECK_EQ(err, ERR_ALREADY_IN_USE);
		ctx->finish();

		// Should ignore IV
		iv.resize(1);
		err = ctx->start(mode, key, iv);
		CHECK_EQ(err, OK);
		ctx->finish();
		iv.resize(16);
		err = ctx->start(mode, key, iv);
		CHECK_EQ(err, OK);
		ctx->finish();
		iv.clear();

		// Invalid keys
		key.resize(0);
		err = ctx->start(mode, key, iv);
		CHECK_EQ(err, ERR_INVALID_PARAMETER);
		key.resize(15);
		err = ctx->start(mode, key, iv);
		CHECK_EQ(err, ERR_INVALID_PARAMETER);
		key.resize(17);
		err = ctx->start(mode, key, iv);
		CHECK_EQ(err, ERR_INVALID_PARAMETER);
		key.resize(33);
		err = ctx->start(mode, key, iv);
		CHECK_EQ(err, ERR_INVALID_PARAMETER);

		ERR_PRINT_ON;

		// Valid again
		key.resize(16);
		err = ctx->start(mode, key, iv);
		CHECK_EQ(err, OK);
		ctx->finish();
	}
}

TEST_CASE("[AESContext] ECB update") {
	// ECB has no IV
	PackedByteArray iv;
	// Key must be either 16 or 32 bytes.
	Vector<uint8_t> key;
	// Data size must be multiple of 16 bytes, apply padding if needed.
	String data = "My secret text!!";
	Ref<AESContext> ctx;
	ctx.instantiate();
	Vector<uint8_t> enc;
	for (int i = 0; i < 2; i++) {
		// Test with 16 and 32 bytes key
		key.resize(i ? 32 : 16);
		Error err = ctx->start(AESContext::MODE_ECB_ENCRYPT, key, iv);
		CHECK_EQ(err, OK);

		enc = ctx->update(to_vec(data));
		CHECK_EQ(enc.size(), to_vec(data).size());

		enc = ctx->update(to_vec(data + data));
		CHECK_EQ(enc.size(), to_vec(data + data).size());

		enc = ctx->update(to_vec(data + data + data));
		CHECK_EQ(enc.size(), to_vec(data + data + data).size());

		// Update size must be a multiple of 16
		ERR_PRINT_OFF;
		enc = ctx->update(to_vec(data + "1"));
		CHECK_EQ(enc.size(), 0);
		enc = ctx->update(to_vec(data.substr(1)));
		CHECK_EQ(enc.size(), 0);
		enc = ctx->update(to_vec(data + data + "1"));
		CHECK_EQ(enc.size(), 0);
		ERR_PRINT_ON;
		ctx->finish();
	}

	// The context must be started
	ERR_PRINT_OFF;
	enc = ctx->update(to_vec(data));
	CHECK_EQ(enc.size(), 0);
	ERR_PRINT_ON;
}

TEST_CASE("[AESContext] ECB encrypt/decrypt") {
	// ECB has no IV
	Vector<uint8_t> iv;
	// Key must be either 16 or 32 bytes.
	Vector<uint8_t> key;
	// Data size must be multiple of 16 bytes, apply padding if needed.
	Vector<uint8_t> data;
	data.resize(64);
	uint8_t *d = data.ptrw();
	for (int i = 0; i < 64; i++) {
		d[i] = i;
	}
	Ref<AESContext> ctx;
	ctx.instantiate();
	for (int i = 0; i < 2; i++) {
		key.resize(i ? 32 : 16);
		// Encrypt
		Error err = ctx->start(AESContext::MODE_ECB_ENCRYPT, key, iv);
		CHECK_EQ(err, OK);
		Vector<uint8_t> enc = ctx->update(data.slice(0, 16));
		CHECK_EQ(enc.size(), 16);
		enc.append_array(ctx->update(data.slice(16)));
		CHECK(!equals(data, enc));
		ctx->finish();
		// Decrypt
		err = ctx->start(AESContext::MODE_ECB_DECRYPT, key, iv);
		CHECK_EQ(err, OK);
		Vector<uint8_t> dec = ctx->update(enc.slice(0, 16));
		CHECK_EQ(dec.size(), 16);
		dec.append_array(ctx->update(enc.slice(16)));
		CHECK(equals(dec, data));
		ctx->finish();
	}

	// Test known values
	String wk_key = "My secret key!!!";
	String wk_data = "My secret text!!My secret text!!";
	Vector<uint8_t> wk_enc = { 103, 81, 126, 234, 18, 0, 206, 76, 28, 20, 158, 138, 208, 20, 117, 53, 103, 81, 126, 234, 18, 0, 206, 76, 28, 20, 158, 138, 208, 20, 117, 53 };
	Error err = ctx->start(AESContext::MODE_ECB_ENCRYPT, to_vec(wk_key));
	Vector<uint8_t> enc = ctx->update(to_vec(wk_data));
	CHECK_EQ(err, OK);
	CHECK(equals(wk_enc, enc));
	ctx->finish();

	wk_key = "My secret key!!!My secret key!!!";
	wk_enc = { 154, 26, 109, 2, 146, 25, 9, 217, 179, 234, 4, 129, 2, 148, 181, 80, 154, 26, 109, 2, 146, 25, 9, 217, 179, 234, 4, 129, 2, 148, 181, 80 };
	err = ctx->start(AESContext::MODE_ECB_ENCRYPT, to_vec(wk_key));
	enc = ctx->update(to_vec(wk_data));
	CHECK_EQ(err, OK);
	CHECK(equals(wk_enc, enc));
	ctx->finish();
}

TEST_CASE("[AESContext] CBC start") {
	AESContext::Mode modes[2] = { AESContext::MODE_CBC_ENCRYPT, AESContext::MODE_CBC_DECRYPT };
	Ref<AESContext> ctx;
	ctx.instantiate();
	for (int i = 0; i < 2; i++) {
		AESContext::Mode mode = modes[i];
		// CBC requires 16 bytes of IV
		PackedByteArray iv;
		iv.resize(16);
		// Key must be either 16 or 32 bytes.
		Vector<uint8_t> key;

		// Valid key (16)
		key.resize(16);
		Error err = ctx->start(mode, key, iv);
		CHECK_EQ(err, OK);
		ctx->finish();

		// Valid key (32)
		err = ctx->start(mode, key, iv);
		CHECK_EQ(err, OK);

		ERR_PRINT_OFF;

		// Already in use
		err = ctx->start(mode, key, iv);
		CHECK_EQ(err, ERR_ALREADY_IN_USE);
		ctx->finish();

		// Should require exactly 16 bytes of IV
		iv.resize(0);
		err = ctx->start(mode, key, iv);
		CHECK_EQ(err, ERR_INVALID_PARAMETER);
		iv.resize(15);
		err = ctx->start(mode, key, iv);
		CHECK_EQ(err, ERR_INVALID_PARAMETER);
		iv.resize(17);
		err = ctx->start(mode, key, iv);
		CHECK_EQ(err, ERR_INVALID_PARAMETER);
		iv.resize(16);

		// Invalid keys
		key.resize(0);
		err = ctx->start(mode, key, iv);
		CHECK_EQ(err, ERR_INVALID_PARAMETER);
		key.resize(15);
		err = ctx->start(mode, key, iv);
		CHECK_EQ(err, ERR_INVALID_PARAMETER);
		key.resize(17);
		err = ctx->start(mode, key, iv);
		CHECK_EQ(err, ERR_INVALID_PARAMETER);
		key.resize(33);
		err = ctx->start(mode, key, iv);
		CHECK_EQ(err, ERR_INVALID_PARAMETER);

		ERR_PRINT_ON;

		// Valid again
		key.resize(16);
		err = ctx->start(mode, key, iv);
		CHECK_EQ(err, OK);
		ctx->finish();
	}
}

TEST_CASE("[AESContext] CBC encrypt/decrypt") {
	// IV must be exactly 16 bytes.
	Vector<uint8_t> iv;
	iv.resize(16);
	// Key must be either 16 or 32 bytes.
	Vector<uint8_t> key;
	// Data size must be multiple of 16 bytes, apply padding if needed.
	Vector<uint8_t> data;
	data.resize(64);
	uint8_t *d = data.ptrw();
	for (int i = 0; i < 64; i++) {
		d[i] = i;
	}
	Ref<AESContext> ctx;
	ctx.instantiate();
	for (int i = 0; i < 2; i++) {
		key.resize(i ? 32 : 16);
		// Encrypt
		Error err = ctx->start(AESContext::MODE_CBC_ENCRYPT, key, iv);
		CHECK_EQ(err, OK);
		Vector<uint8_t> enc = ctx->update(data.slice(0, 16));
		CHECK_EQ(enc.size(), 16);
		enc.append_array(ctx->update(data.slice(16)));
		CHECK(!equals(data, enc));
		ctx->finish();
		// Decrypt
		err = ctx->start(AESContext::MODE_CBC_DECRYPT, key, iv);
		CHECK_EQ(err, OK);
		Vector<uint8_t> dec = ctx->update(enc.slice(0, 16));
		CHECK_EQ(dec.size(), 16);
		dec.append_array(ctx->update(enc.slice(16)));
		CHECK(equals(dec, data));
		ctx->finish();
	}

	// Test known values
	String wk_key = "My secret key!!!";
	String wk_data = "My secret text!!My secret text!!";
	String wk_iv = "My secret iv!!!!";
	Vector<uint8_t> wk_enc = { 5, 69, 219, 146, 167, 176, 184, 99, 29, 246, 79, 191, 234, 6, 46, 39, 157, 47, 126, 102, 152, 34, 50, 4, 137, 167, 31, 110, 63, 141, 84, 145 };
	Error err = ctx->start(AESContext::MODE_CBC_ENCRYPT, to_vec(wk_key), to_vec(wk_iv));
	Vector<uint8_t> enc = ctx->update(to_vec(wk_data));
	CHECK_EQ(err, OK);
	CHECK(equals(wk_enc, enc));
	ctx->finish();

	wk_key = "My secret key!!!My secret key!!!";
	wk_enc = { 166, 176, 138, 196, 78, 18, 148, 120, 139, 21, 172, 33, 92, 181, 5, 239, 175, 111, 229, 214, 22, 186, 129, 95, 223, 46, 192, 103, 141, 105, 183, 91 };
	err = ctx->start(AESContext::MODE_CBC_ENCRYPT, to_vec(wk_key), to_vec(wk_iv));
	enc = ctx->update(to_vec(wk_data));
	CHECK_EQ(err, OK);
	CHECK(equals(wk_enc, enc));
	ctx->finish();
}

} // namespace TestAes_context
