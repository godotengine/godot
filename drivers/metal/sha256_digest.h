/**************************************************************************/
/*  sha256_digest.h                                                       */
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

#import <CommonCrypto/CommonDigest.h>
#import <simd/simd.h>
#import <zlib.h>

#include "core/templates/local_vector.h"

struct SHA256Digest {
	unsigned char data[CC_SHA256_DIGEST_LENGTH];

	static constexpr size_t serialized_size() { return CC_SHA256_DIGEST_LENGTH; }

	uint32_t hash() const {
		uint32_t c = crc32(0, data, CC_SHA256_DIGEST_LENGTH);
		return c;
	}

	SHA256Digest() {
		bzero(data, CC_SHA256_DIGEST_LENGTH);
	}

	SHA256Digest(const char *p_hash) {
		memcpy(data, p_hash, CC_SHA256_DIGEST_LENGTH);
	}

	SHA256Digest(const char *p_data, size_t p_length) {
		CC_SHA256(p_data, (CC_LONG)p_length, data);
	}

	_FORCE_INLINE_ uint32_t short_sha() const {
		return __builtin_bswap32(*(uint32_t *)&data[0]);
	}

	LocalVector<uint8_t> serialize() const {
		LocalVector<uint8_t> result;
		result.resize(CC_SHA256_DIGEST_LENGTH);
		memcpy(result.ptr(), data, CC_SHA256_DIGEST_LENGTH);
		return result;
	}

	static SHA256Digest deserialize(LocalVector<uint8_t> p_ser) {
		return SHA256Digest((const char *)p_ser.ptr());
	}
};
