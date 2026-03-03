/**************************************************************************/
/*  variant_caster.h                                                      */
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

#include "core/object/object.h"
#include "core/variant/type_info.h"

enum class HatDir;
enum class HatMask;
enum class JoyAxis;
enum class JoyButton;

enum class MIDIMessage;
enum class MouseButton;
enum class MouseButtonMask;

enum class Key;
enum class KeyModifierMask;
enum class KeyLocation;

// Variant cannot define an implicit cast operator for every Object subclass, so the
// casting is done here, to allow binding methods with parameters more specific than Object *

template <typename T>
struct VariantCaster {
	static _FORCE_INLINE_ T cast(const Variant &p_variant) {
		using TStripped = std::remove_pointer_t<T>;
		if constexpr (std::is_base_of_v<Object, TStripped>) {
			return Object::cast_to<TStripped>(p_variant);
		} else {
			return p_variant;
		}
	}
};

template <typename T>
struct VariantCaster<T &> {
	static _FORCE_INLINE_ T cast(const Variant &p_variant) {
		using TStripped = std::remove_pointer_t<T>;
		if constexpr (std::is_base_of_v<Object, TStripped>) {
			return Object::cast_to<TStripped>(p_variant);
		} else {
			return p_variant;
		}
	}
};

template <typename T>
struct VariantCaster<const T &> {
	static _FORCE_INLINE_ T cast(const Variant &p_variant) {
		using TStripped = std::remove_pointer_t<T>;
		if constexpr (std::is_base_of_v<Object, TStripped>) {
			return Object::cast_to<TStripped>(p_variant);
		} else {
			return p_variant;
		}
	}
};

// Object enum casts must go here
VARIANT_ENUM_CAST(Object::ConnectFlags);

VARIANT_ENUM_CAST(Vector2::Axis);
VARIANT_ENUM_CAST(Vector2i::Axis);
VARIANT_ENUM_CAST(Vector3::Axis);
VARIANT_ENUM_CAST(Vector3i::Axis);
VARIANT_ENUM_CAST(Vector4::Axis);
VARIANT_ENUM_CAST(Vector4i::Axis);
VARIANT_ENUM_CAST(EulerOrder);
VARIANT_ENUM_CAST(Projection::Planes);

VARIANT_ENUM_CAST(Error);
VARIANT_ENUM_CAST(Side);
VARIANT_ENUM_CAST(ClockDirection);
VARIANT_ENUM_CAST(Corner);
VARIANT_ENUM_CAST(HatDir);
VARIANT_BITFIELD_CAST(HatMask);
VARIANT_ENUM_CAST(JoyAxis);
VARIANT_ENUM_CAST(JoyButton);

VARIANT_ENUM_CAST(MIDIMessage);
VARIANT_ENUM_CAST(MouseButton);
VARIANT_BITFIELD_CAST(MouseButtonMask);
VARIANT_ENUM_CAST(Orientation);
VARIANT_ENUM_CAST(HorizontalAlignment);
VARIANT_ENUM_CAST(VerticalAlignment);
VARIANT_ENUM_CAST(InlineAlignment);
VARIANT_ENUM_CAST(PropertyHint);
VARIANT_BITFIELD_CAST(PropertyUsageFlags);
VARIANT_ENUM_CAST(Variant::Type);
VARIANT_ENUM_CAST(Variant::Operator);

// Key

VARIANT_ENUM_CAST(Key);
VARIANT_BITFIELD_CAST(KeyModifierMask);
VARIANT_ENUM_CAST(KeyLocation);

template <>
struct VariantCaster<char32_t> {
	static _FORCE_INLINE_ char32_t cast(const Variant &p_variant) {
		return (char32_t)p_variant.operator int();
	}
};

template <typename T>
struct VariantObjectClassChecker {
	static _FORCE_INLINE_ bool check(const Variant &p_variant) {
		using TStripped = std::remove_pointer_t<T>;
		if constexpr (std::is_base_of_v<Object, TStripped>) {
			Object *obj = p_variant;
			return Object::cast_to<TStripped>(p_variant) || !obj;
		} else {
			return true;
		}
	}
};

template <typename T>
class Ref;

template <typename T>
struct VariantObjectClassChecker<const Ref<T> &> {
	static _FORCE_INLINE_ bool check(const Variant &p_variant) {
		Object *obj = p_variant;
		const Ref<T> node = p_variant;
		return node.ptr() || !obj;
	}
};

#ifdef DEBUG_ENABLED

template <typename T>
struct VariantCasterAndValidate {
	static _FORCE_INLINE_ T cast(const Variant **p_args, uint32_t p_arg_idx, Callable::CallError &r_error) {
		Variant::Type argtype = GetTypeInfo<T>::VARIANT_TYPE;
		if (!Variant::can_convert_strict(p_args[p_arg_idx]->get_type(), argtype) ||
				!VariantObjectClassChecker<T>::check(*p_args[p_arg_idx])) {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = p_arg_idx;
			r_error.expected = argtype;
		}

		return VariantCaster<T>::cast(*p_args[p_arg_idx]);
	}
};

template <typename T>
struct VariantCasterAndValidate<T &> {
	static _FORCE_INLINE_ T cast(const Variant **p_args, uint32_t p_arg_idx, Callable::CallError &r_error) {
		Variant::Type argtype = GetTypeInfo<T>::VARIANT_TYPE;
		if (!Variant::can_convert_strict(p_args[p_arg_idx]->get_type(), argtype) ||
				!VariantObjectClassChecker<T>::check(*p_args[p_arg_idx])) {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = p_arg_idx;
			r_error.expected = argtype;
		}

		return VariantCaster<T>::cast(*p_args[p_arg_idx]);
	}
};

template <typename T>
struct VariantCasterAndValidate<const T &> {
	static _FORCE_INLINE_ T cast(const Variant **p_args, uint32_t p_arg_idx, Callable::CallError &r_error) {
		Variant::Type argtype = GetTypeInfo<T>::VARIANT_TYPE;
		if (!Variant::can_convert_strict(p_args[p_arg_idx]->get_type(), argtype) ||
				!VariantObjectClassChecker<T>::check(*p_args[p_arg_idx])) {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_error.argument = p_arg_idx;
			r_error.expected = argtype;
		}

		return VariantCaster<T>::cast(*p_args[p_arg_idx]);
	}
};

#endif // DEBUG_ENABLED
