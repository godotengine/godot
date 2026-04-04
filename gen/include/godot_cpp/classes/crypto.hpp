/**************************************************************************/
/*  crypto.hpp                                                            */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/hashing_context.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/string.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class CryptoKey;
class X509Certificate;

class Crypto : public RefCounted {
	GDEXTENSION_CLASS(Crypto, RefCounted)

public:
	PackedByteArray generate_random_bytes(int32_t p_size);
	Ref<CryptoKey> generate_rsa(int32_t p_size);
	Ref<X509Certificate> generate_self_signed_certificate(const Ref<CryptoKey> &p_key, const String &p_issuer_name = "CN=myserver,O=myorganisation,C=IT", const String &p_not_before = "20140101000000", const String &p_not_after = "20340101000000");
	PackedByteArray sign(HashingContext::HashType p_hash_type, const PackedByteArray &p_hash, const Ref<CryptoKey> &p_key);
	bool verify(HashingContext::HashType p_hash_type, const PackedByteArray &p_hash, const PackedByteArray &p_signature, const Ref<CryptoKey> &p_key);
	PackedByteArray encrypt(const Ref<CryptoKey> &p_key, const PackedByteArray &p_plaintext);
	PackedByteArray decrypt(const Ref<CryptoKey> &p_key, const PackedByteArray &p_ciphertext);
	PackedByteArray hmac_digest(HashingContext::HashType p_hash_type, const PackedByteArray &p_key, const PackedByteArray &p_msg);
	bool constant_time_compare(const PackedByteArray &p_trusted, const PackedByteArray &p_received);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

