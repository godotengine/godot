/**************************************************************************/
/*  variant_struct_native.h                                               */
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

#include "core/variant/variant_struct.h"

#include "core/variant/binder_common.h"
// #include "core/variant/type_info.h"
// #include "core/variant/typed_array.h"
// #include "core/variant/typed_dictionary.h"
/*constepxr auto ini = std::is_trivially_constructible_v<T> ? &StructDefinition::generic_constructor : &init_struct; \
constepxr auto cop = std::is_trivially_copy_constructible_v<T> ? &StructDefinition::generic_copy_constructor : &copy_struct; \
constepxr auto des = std::is_trivially_destructible_v<T> ? &StructDefinition::trivial_destructor : &deinit_struct; \*/

#define STRINGIFY_MACRO(s) #s

#define VARIANT_STRUCT_DEFINITION(m_parent_class, m_struct_class, ...)                                                     \
	template <>                                                                                                            \
	StructDefinition *NativeStructDefinition<m_parent_class::m_struct_class>::build_definition() {                         \
		using T = m_parent_class::m_struct_class;                                                                          \
		StructDefinition *sd = StructDefinition::create(                                                                   \
				{##__VA_ARGS__##},                                                                                         \
				STRINGIFY_MACRO(m_parent_class.m_struct_class),                                                            \
				sizeof(m_parent_class::m_struct_class),                                                                    \
				(!std::is_trivially_constructible_v<T> ? &init_struct : &StructDefinition::generic_constructor),           \
				(!std::is_trivially_copy_constructible_v<T> ? &copy_struct : &StructDefinition::generic_copy_constructor), \
				(!std::is_trivially_destructible_v<T> ? &deinit_struct : &StructDefinition::trivial_destructor));          \
		return sd;                                                                                                         \
	}

#define VARIANT_STRUCT_PROPERTY(m_property_name) \
	StructDefinition::build_native_property(#m_property_name, &T::##m_property_name)

#define REGISTER_INBUILT_STRUCT(m_parent_class, m_struct_class)                                                                          \
	{                                                                                                                                    \
		ClassDB::bind_struct(#m_parent_class, #m_struct_class, &NativeStructDefinition<m_parent_class::m_struct_class>::get_definition); \
	}

/////////////////////////////////////

template <class T>
size_t Internal::NativeTypeInfo<T>::size() const {
// size_t StructDefinition::NativeTypeInfo<T>::size() const {
	return sizeof(T);
}

template <class T>
size_t Internal::NativeTypeInfo<T>::align() const {
// size_t StructDefinition::NativeTypeInfo<T>::align() const {
	return alignof(T);
}

template <class T>
Variant::Type Internal::NativeTypeInfo<T>::get_variant_type() const {
// Variant::Type StructDefinition::NativeTypeInfo<T>::get_variant_type() const {
	if constexpr (std::is_same_v<T, nullptr_t>) {
		return Variant::NIL;
	} else {
		return GetTypeInfo<T>::VARIANT_TYPE;
	}
}

template <class T>
const StringName Internal::NativeTypeInfo<T>::get_class_name() const {
// const StringName StructDefinition::NativeTypeInfo<T>::get_class_name() const {
	if constexpr (std::is_base_of_v<T, Object *>) {
		return std::remove_pointer_t<T>::get_class_static();
	} else {
		// Only pointers to Object have class names
		// (dev-note: not sure if we want to fail)
		// ERR_FAIL_V_MSG(StringName(),"Cannot get class name for struct properties that are not Object pointers");
		return StringName();
	}
}

template <class T>
void Internal::NativeTypeInfo<T>::construct(void *p_target) const {
// void StructDefinition::NativeTypeInfo<T>::construct(void *p_target) const {
	if constexpr (std::is_same_v<T, nullptr_t>) {
		// Do Nothing
	} else if constexpr (!std::is_trivially_constructible_v<T>) {
		new (reinterpret_cast<T *>(p_target)) T;
	} else {
		// TODO: assign default values for various types
		// (should use same logic, like how Variant ints are always 0)
	}
}

template <class T>
void Internal::NativeTypeInfo<T>::copy_construct(void *p_target, const void *p_value) const {
// void StructDefinition::NativeTypeInfo<T>::copy_construct(void *p_target, const void *p_value) const {
	if constexpr (std::is_same_v<T, nullptr_t>) {
		// Do Nothing
	} else if constexpr (!std::is_trivially_copy_constructible_v<T>) {
		new (reinterpret_cast<T *>(p_target)) T(*reinterpret_cast<const T *>(p_value));
	} else {
		*reinterpret_cast<T *>(p_target) = *reinterpret_cast<const T *>(p_value);
	}
}

template <class T>
void Internal::NativeTypeInfo<T>::destruct(void *p_target) const {
// void StructDefinition::NativeTypeInfo<T>::destruct(void *p_target) const {
	if constexpr (std::is_same_v<T, nullptr_t>) {
		// Do Nothing
	} else if constexpr (!std::is_trivially_destructible_v<T>) {
		reinterpret_cast<T *>(p_target)->~T();
	}
}

template <class T>
Variant Internal::NativeTypeInfo<T>::read(const void *p_target) const {
// Variant StructDefinition::NativeTypeInfo<T>::read(const void *p_target) const {
	if constexpr (std::is_same_v<T, nullptr_t>) {
		return Variant(); // Return NIL
	} else {
		return Variant(*reinterpret_cast<const T *>(p_target));
	}
}

template <class T>
void Internal::NativeTypeInfo<T>::ptr_get(const void *p_target, void *p_into) const {
// void StructDefinition::NativeTypeInfo<T>::ptr_get(const void *p_target, void *p_into) const {
	if constexpr (std::is_same_v<T, nullptr_t>) {
		// Do Nothing
	} else {
		*reinterpret_cast<T *>(p_into) = *reinterpret_cast<const T *>(p_target);
	}
}

template <class T>
void Internal::NativeTypeInfo<T>::write(void *p_target, const Variant &p_value) const {
// void StructDefinition::NativeTypeInfo<T>::write(void *p_target, const Variant &p_value) const {
	if constexpr (std::is_same_v<T, nullptr_t>) {
		// Do Nothing
	} else if constexpr (std::is_same_v<T, ObjectID>) {
		// TODO: analyze why this is necessary
		// Something about ambiguous casting?
		// Should check with the maintainers on if this is intentional before making any changes to the engine
		*reinterpret_cast<ObjectID *>(p_target) = static_cast<ObjectID>(static_cast<uint64_t>(p_value));
	} else {
		*reinterpret_cast<T *>(p_target) = static_cast<T>(p_value);
	}
}

template <class T>
void Internal::NativeTypeInfo<T>::ptr_set(void *p_target, const void *p_value) const {
// void StructDefinition::NativeTypeInfo<T>::ptr_set(void *p_target, const void *p_value) const {
	if constexpr (std::is_same_v<T, nullptr_t>) {
		// Do Nothing
	} else {
		*reinterpret_cast<T *>(p_target) = *reinterpret_cast<const T *>(p_value);
	}
}
