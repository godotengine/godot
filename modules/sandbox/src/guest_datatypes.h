/**************************************************************************/
/*  guest_datatypes.h                                                     */
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

#include "core/math/math_defs.h"
#include "core/object/object.h"
#include "core/string/ustring.h"
#include "core/typedefs.h"
#include "core/variant/array.h"
#include "core/variant/variant.h"
#include <cstdint>
#include <libriscv/machine.hpp>
#include <string>

// Forward declarations
class Sandbox;

#define RISCV_ARCH riscv::RISCV64
using gaddr_t = riscv::address_type<RISCV_ARCH>;
using machine_t = riscv::Machine<RISCV_ARCH>;

// -= Fast-path Variant Arguments =-

struct GDNativeVariant {
	uint8_t type;
	uint8_t padding[7];
	union {
		struct {
			double flt;
			uint64_t flt_padding1;
		};
		struct {
			uint64_t value;
			uint64_t i64_padding;
		};
		struct {
			real_t vec2_flt[2];
		};
		struct {
			int32_t ivec2_int[2];
		};
		struct {
			real_t vec3_flt[3];
		};
		struct {
			int32_t ivec3_int[3];
		};
		struct {
			real_t vec4_flt[4];
		};
		struct {
			int32_t ivec4_int[4];
		};
		struct {
			float color_flt[4];
		};
		struct {
			uint64_t object_id;
			Object *object_ptr;
		};
	};

	Object *to_object() const {
		if (object_ptr == nullptr)
			return nullptr;
		return object_ptr;
	}

} __attribute__((packed));

// -= Guest Data Types =-
struct GuestStdU32String {
	gaddr_t ptr;
	gaddr_t size;
	gaddr_t capacity;

	char32_t *to_array(const machine_t &machine) const {
		return machine.memory.memarray<char32_t>(ptr, size);
	}
	std::u32string to_u32string(const machine_t &machine, std::size_t max_len = 4UL << 20) const {
		if (size > max_len)
			throw std::runtime_error("Guest std::u32string too large (size > 4MB)");
		// Copy the string from guest memory
		const std::u32string_view view{ to_array(machine), size_t(size) };
		return std::u32string(view.begin(), view.end());
	}

	String to_godot_string(const machine_t &machine, std::size_t max_len = 1UL << 20) const {
		if (size > max_len)
			throw std::runtime_error("Guest std::u32string too large (size > 4MB)");
		// Get a view of the string from guest memory, including the null terminator
		const std::u32string_view view{ to_array(machine), size_t(size + 1) };
		if (view.back() == U'\0') {
			// Convert the string to a godot String directly
			return String(view.data());
		} else {
			// Use a temporary std::u32string to convert the string to a godot String
			std::u32string str(view.begin(), view.end() - 1);
			return String(str.c_str());
		}
	}

	void set_string(machine_t &machine, gaddr_t self, const char32_t *str, std::size_t len) {
		// Allocate memory for the string
		this->ptr = machine.arena().malloc(len * sizeof(char32_t));
		this->size = len;
		this->capacity = len;
		// Copy the string to guest memory
		char32_t *guest_ptr = machine.memory.memarray<char32_t>(this->ptr, len);
		std::memcpy(guest_ptr, str, len * sizeof(char32_t));
	}

	void free(machine_t &machine) {
		if (ptr != 0x0)
			machine.arena().free(ptr);
	}
};

struct GuestVariant {
	/**
	 * @brief Creates a new godot Variant from a GuestVariant that comes from a sandbox.
	 *
	 * @param emu The sandbox that the GuestVariant comes from.
	 * @return Variant The new godot Variant.
	 **/
	Variant toVariant(const Sandbox &emu) const;

	/**
	 * @brief Returns a pointer to a Variant that comes from a sandbox. This Variant must be
	 * already stored in the sandbox's state. This usually means it's a complex Variant that is
	 * never copied to the guest memory. Most complex types are stored in the call state and are
	 * referenced by their index.
	 *
	 * @param emu The sandbox that the GuestVariant comes from.
	 * @return const Variant* The pointer to the Variant.
	 **/
	const Variant *toVariantPtr(const Sandbox &emu) const;

	/**
	 * @brief Sets the value of the GuestVariant from a godot Variant.
	 *
	 * @param emu The sandbox that the GuestVariant comes from.
	 * @param value The godot Variant.
	 * @param implicit_trust If true, creating a complex type will not throw an error.
	 * @throw std::runtime_error If the type is not supported.
	 * @throw std::runtime_error If the type is complex and implicit_trust is false.
	 **/
	void set(Sandbox &emu, const Variant &value, bool implicit_trust = false);

	/**
	 * @brief Sets the value of the GuestVariant to a godot Object. The object is added to the
	 * scoped objects list and the GuestVariant stores the untranslated address to the object.
	 * @warning The object is implicitly trusted, treated as allowed.
	 *
	 * @param emu The sandbox that the GuestVariant comes from.
	 * @param obj The godot Object.
	 **/
	void set_object(Sandbox &emu, Object *obj);

	/**
	 * @brief Creates a new GuestVariant from a godot Variant. Trust is implicit for complex types.
	 * The variant is constructed directly into the call state, and the GuestVariant is an index
	 * to it. For objects, the object is added to the scoped objects list and the GuestVariant
	 * stores the untranslated address to the object.
	 *
	 * @param emu The sandbox that the GuestVariant comes from.
	 * @param value The godot Variant.
	 * @throw std::runtime_error If the type is not supported.
	 **/
	void create(Sandbox &emu, Variant &&value);

	/**
	 * @brief Check if the GuestVariant is implemented using an index to a scoped Variant.
	 *
	 * @return true If the type of the GuestVariant is implemented using an index to a scoped Variant.
	 */
	bool is_scoped_variant() const noexcept;

	Variant::Type type = Variant::NIL;
	union alignas(8) {
		int64_t i = 0;
		bool b;
		double f;
		std::array<real_t, 2> v2f;
		std::array<real_t, 3> v3f;
		std::array<real_t, 4> v4f;
		std::array<int32_t, 2> v2i;
		std::array<int32_t, 3> v3i;
		std::array<int32_t, 4> v4i;
	} v;

	void free(Sandbox &emu);

	static const char *type_name(int type);
	static const char *type_name(Variant::Type type) { return type_name(int(type)); }
};

inline bool GuestVariant::is_scoped_variant() const noexcept {
	switch (type) {
		case Variant::STRING:
		case Variant::TRANSFORM2D:
		case Variant::QUATERNION:
		case Variant::AABB:
		case Variant::BASIS:
		case Variant::TRANSFORM3D:
		case Variant::PROJECTION:
		case Variant::DICTIONARY:
		case Variant::ARRAY:
		case Variant::CALLABLE:
		case Variant::STRING_NAME:
		case Variant::NODE_PATH:
		case Variant::RID:
		case Variant::PACKED_BYTE_ARRAY:
		case Variant::PACKED_FLOAT32_ARRAY:
		case Variant::PACKED_FLOAT64_ARRAY:
		case Variant::PACKED_INT32_ARRAY:
		case Variant::PACKED_INT64_ARRAY:
		case Variant::PACKED_VECTOR2_ARRAY:
		case Variant::PACKED_VECTOR3_ARRAY:
		case Variant::PACKED_VECTOR4_ARRAY:
		case Variant::PACKED_COLOR_ARRAY:
		case Variant::PACKED_STRING_ARRAY: {
			return true;
		}
		case Variant::OBJECT: // Objects are raw pointers.
		default:
			return false;
	}
}

static inline void hash_combine(gaddr_t &seed, gaddr_t hash) {
	hash += 0x9e3779b9 + (seed << 6) + (seed >> 2);
	seed ^= hash;
}

#include "syscalls_helpers.hpp"

inline String to_godot_string(const riscv::CppString *string, machine_t &machine, std::size_t max_len = 4UL << 20) {
	std::string_view view = string->to_view(machine, max_len);
	return String::utf8(view.data(), view.size());
}
inline PackedByteArray to_godot_packed_byte_array(const riscv::CppString *string, machine_t &machine, std::size_t max_len = 4UL << 20) {
	std::string_view view = string->to_view(machine, max_len);
	PackedByteArray arr;
	arr.resize(view.size());
	std::memcpy(arr.ptrw(), view.data(), view.size());
	return arr;
}
