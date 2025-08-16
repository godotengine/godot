#pragma once
#include "packed_array.hpp"
#include "string.hpp"

struct PackedByteArray final : public PackedArray<uint8_t> {
	using PackedArray::PackedArray;
	constexpr PackedByteArray(const PackedArray<uint8_t> &other) : PackedArray<uint8_t>(other) {}

	operator Variant() const {
		return *static_cast<const PackedArray<uint8_t>*>(this);
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
		return static_cast<const PackedArray<uint8_t>*>(this)->operator ()("to_float32");
	}
	PackedFloat64Array to_float64_array() const {
		return static_cast<const PackedArray<uint8_t>*>(this)->operator ()("to_float64");
	}
	PackedInt32Array to_int32_array() const {
		return static_cast<const PackedArray<uint8_t>*>(this)->operator ()("to_int32");
	}
	PackedInt64Array to_int64_array() const {
		return static_cast<const PackedArray<uint8_t>*>(this)->operator ()("to_int64");
	}
};
