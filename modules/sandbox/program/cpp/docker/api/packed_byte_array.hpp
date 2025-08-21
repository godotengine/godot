/**************************************************************************/
/*  packed_byte_array.hpp                                                 */
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
#include "packed_array.hpp"
#include "string.hpp"

struct PackedByteArray final : public PackedArray<uint8_t> {
	using PackedArray::PackedArray;
	constexpr PackedByteArray(const PackedArray<uint8_t> &other) :
			PackedArray<uint8_t>(other) {}

	operator Variant() const {
		return *static_cast<const PackedArray<uint8_t> *>(this);
	}

	PackedByteArray decompress(int64_t buffer_size, int64_t compression_mode) const {
		return static_cast<const PackedArray<uint8_t> *>(this)->operator()("decompress", buffer_size, compression_mode).as_byte_array();
	}

	String get_string_from_ascii() const {
		return static_cast<const PackedArray<uint8_t> *>(this)->operator()("get_string_from_ascii");
	}
	String get_string_from_utf8() const {
		return static_cast<const PackedArray<uint8_t> *>(this)->operator()("get_string_from_utf8");
	}
	String get_string_from_utf16() const {
		return static_cast<const PackedArray<uint8_t> *>(this)->operator()("get_string_from_utf16");
	}
	String get_string_from_utf32() const {
		return static_cast<const PackedArray<uint8_t> *>(this)->operator()("get_string_from_utf32");
	}

	PackedFloat32Array to_float32_array() const {
		return static_cast<const PackedArray<uint8_t> *>(this)->operator()("to_float32");
	}
	PackedFloat64Array to_float64_array() const {
		return static_cast<const PackedArray<uint8_t> *>(this)->operator()("to_float64");
	}
	PackedInt32Array to_int32_array() const {
		return static_cast<const PackedArray<uint8_t> *>(this)->operator()("to_int32");
	}
	PackedInt64Array to_int64_array() const {
		return static_cast<const PackedArray<uint8_t> *>(this)->operator()("to_int64");
	}
};
