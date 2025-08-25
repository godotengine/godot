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

#include "variant_struct.h"

#include "core/variant/binder_common.h"

/////////////////////////////////////

template <class T>
size_t Internal::NativeTypeInfo<T>::size() const {
	return sizeof(T);
}

template <class T>
size_t Internal::NativeTypeInfo<T>::align() const {
	return alignof(T);
}

template <class T>
Variant::Type Internal::NativeTypeInfo<T>::get_variant_type() const {
	if constexpr (std::is_same_v<T, nullptr_t>) {
		return Variant::NIL;
	} else {
		return GetTypeInfo<T>::VARIANT_TYPE;
	}
}

template <class T>
void Internal::NativeTypeInfo<T>::construct(void *p_target) const {
	if constexpr (std::is_same_v<T, nullptr_t>) {
		// Do Nothing
	} else if constexpr (!std::is_trivially_constructible_v<T>) {
		memnew_placement(p_target, T);
	} else {
		*(T *)p_target = 0;
	}
}

template <class T>
void Internal::NativeTypeInfo<T>::copy_construct(void *p_target, const void *p_value) const {
	if constexpr (std::is_same_v<T, nullptr_t>) {
		// Do Nothing
	} else {
		memnew_placement(p_target, T(*(const T *)p_value));
	}
}

template <class T>
void Internal::NativeTypeInfo<T>::destruct(void *p_target) const {
	if constexpr (std::is_same_v<T, nullptr_t>) {
		// Do Nothing
	} else if constexpr (!std::is_trivially_destructible_v<T>) {
		((T *)p_target)->~T();
	}
}

template <class T>
Variant Internal::NativeTypeInfo<T>::read(const void *p_target) const {
	if constexpr (std::is_same_v<T, nullptr_t>) {
		return Variant(); // Return NIL
	} else {
		return Variant(*(const T *)p_target);
	}
}

template <class T>
void Internal::NativeTypeInfo<T>::ptr_get(const void *p_target, void *p_into) const {
	if constexpr (std::is_same_v<T, nullptr_t>) {
		// Do Nothing
	} else {
		*(T *)p_into = *(const T *)p_target;
	}
}

template <class T>
void Internal::NativeTypeInfo<T>::write(void *p_target, const Variant &p_value) const {
	if constexpr (std::is_same_v<T, nullptr_t>) {
		// Do Nothing
	} else if constexpr (std::is_same_v<T, ObjectID>) {
		// TODO: analyze why this is necessary
		// Something about ambiguous casting?
		// Should check with the maintainers on if this is intentional before making any changes to the engine
		*(ObjectID *)p_target = static_cast<ObjectID>(static_cast<uint64_t>(p_value));
	} else {
		*(T *)p_target = static_cast<const T>(p_value);
	}
}

template <class T>
void Internal::NativeTypeInfo<T>::ptr_set(void *p_target, const void *p_value) const {
	if constexpr (std::is_same_v<T, nullptr_t>) {
		// Do Nothing
	} else {
		*(T *)p_target = *(const T *)p_value;
	}
}

/////////////////////////////////////

template <class T>
struct NativeStructDefinition {
	using StructPtrType = Internal::StructPtrType;
	friend class VariantStruct;

private:
	static StructDefinition *sdef;
	static StructDefinition *build_definition();

	// If there are any non-trivial properties, C++ should be able to handle these operations faster than the generic variants above
	static void init_struct(const StructDefinition *p_definition, StructPtrType p_target) {
		if constexpr (!std::is_trivially_constructible_v<T>) {
			memnew_placement(p_target.get_heap_(), T());
		}
	}
	static void copy_struct(const StructDefinition *p_definition, StructPtrType p_target, const StructPtrType p_other) {
		memnew_placement(p_target.get_heap_(), T(*p_other.get_heap<T>()));
	}
	static void deinit_struct(const StructDefinition *p_definition, StructPtrType p_target) {
		if constexpr (!std::is_trivially_destructible_v<T>) {
			p_target.get_heap<T>()->~T();
		}
	}

public:
	static const StructDefinition *get_definition() {
		if (sdef == nullptr) {
			sdef = build_definition();
			StructDefinition::_register_native_definition(&sdef);
		}
		return sdef;
	}
};

template <class T>
StructDefinition *NativeStructDefinition<T>::sdef = nullptr;

/////////////////////////////////////

template <class T>
class NativeVariantStruct : public VariantStruct {
public:
	_ALWAYS_INLINE_ T *operator->() {
		return (T *)(instance.get_heap_());
	}
	_ALWAYS_INLINE_ T &get_struct() {
		return *(T *)(instance.get_heap_());
	}

	// Copy an existing struct instance into the heap
	_ALWAYS_INLINE_ NativeVariantStruct(const T &p_struct) {
		_init(NativeStructDefinition<T>::get_definition());
		memnew_placement(instance.get_heap_(), T(p_struct));
	}
	_ALWAYS_INLINE_ NativeVariantStruct(T &&p_struct) {
		_init(NativeStructDefinition<T>::get_definition());
		memnew_placement(instance.get_heap_(), T(std::move(p_struct)));
	}

	// or *must* be constructed in one of the following ways
	_ALWAYS_INLINE_ explicit NativeVariantStruct(void) {
		_init(NativeStructDefinition<T>::get_definition());
		if constexpr (!std::is_trivially_constructible_v<T>) {
			memnew_placement(instance.get_heap_(), T());
		}
	}
	template <typename... VarArgs>
	_ALWAYS_INLINE_ explicit NativeVariantStruct(VarArgs... p_args) {
		_init(NativeStructDefinition<T>::get_definition());
		memnew_placement(instance.get_heap_(), T(p_args...));
	}

	_ALWAYS_INLINE_ NativeVariantStruct(const VariantStruct &p_struct) :
			VariantStruct(p_struct) {
		DEV_ASSERT(is_empty() || definition == NativeStructDefinition<T>::get_definition());
	}
	_ALWAYS_INLINE_ NativeVariantStruct(VariantStruct &&p_struct) :
			VariantStruct(std::move(p_struct)) {
		DEV_ASSERT(is_empty() || definition == NativeStructDefinition<T>::get_definition());
	}

	_ALWAYS_INLINE_ NativeVariantStruct<T> &operator=(const NativeStructDefinition<T> &p_other) {
		_ref(p_other.instance, p_other.definition);
		return *this;
	}
	_ALWAYS_INLINE_ NativeVariantStruct<T> &operator=(NativeStructDefinition<T> &&p_other) {
		std::swap(instance, p_other.instance);
		std::swap(definition, p_other.definition);
		return *this;
	}
};

/////////////////////////////////////

namespace Internal {

template <class _Ty, typename = void>
struct remove_ref {
	using type = _Ty;
	using _Const_thru_ref_type = _Ty;
};
template <class _Ty>
struct remove_ref<Ref<_Ty>, std::enable_if_t<std::is_base_of_v<_Ty, RefCounted>>> {
	using type = _Ty;
	using _Const_thru_ref_type = Ref<_Ty>;
};
template <class _Ty>
struct is_ref {
	constexpr static bool v = !std::is_same_v<remove_ref<_Ty>::type, _Ty>;
};

template <class ST, typename MT>
_ALWAYS_INLINE_ static StructPropertyInfo build_native_property(StringName const &p_name, const MT ST::*p_member_pointer) {
	if constexpr (std::is_pointer_v<MT> && std::is_base_of_v<Object, std::remove_pointer_t<MT>>) {
		return StructPropertyInfo(StructPropertyInfo::NATIVE_OBJECT_WEAK, p_name, p_member_pointer, std::remove_pointer_t<MT>::get_class_static());

	} else if constexpr (is_ref<MT>::v) {
		return StructPropertyInfo(StructPropertyInfo::NATIVE_OBJECT_REF, p_name, p_member_pointer, remove_ref<MT>::type::get_class_static());

	} else {
		return StructPropertyInfo(p_name, p_member_pointer, NativeTypeInfo<MT>());
	}
}

} //namespace Internal

/////////////////////////////////////

#define STRINGIFY_MACRO(s) #s

#define VARIANT_STRUCT_DEFINITION(m_parent_class, m_struct_class, ...)                                            \
	template <>                                                                                                   \
	StructDefinition *NativeStructDefinition<m_parent_class::m_struct_class>::build_definition() {                \
		using T = m_parent_class::m_struct_class;                                                                 \
		StructDefinition *sd = StructDefinition::create(                                                          \
				{##__VA_ARGS__##},                                                                                \
				STRINGIFY_MACRO(m_parent_class.m_struct_class),                                                   \
				sizeof(m_parent_class::m_struct_class),                                                           \
				(!std::is_trivially_constructible_v<T> ? &init_struct : &StructDefinition::generic_constructor),  \
				(&copy_struct),                                                                                   \
				(!std::is_trivially_destructible_v<T> ? &deinit_struct : &StructDefinition::trivial_destructor)); \
		return sd;                                                                                                \
	}

#define VARIANT_STRUCT_PROPERTY(m_property_name) \
	Internal::build_native_property(#m_property_name, &T::##m_property_name)

#define REGISTER_INBUILT_STRUCT(m_parent_class, m_struct_class)                                                                          \
	{                                                                                                                                    \
		ClassDB::bind_struct(#m_parent_class, #m_struct_class, &NativeStructDefinition<m_parent_class::m_struct_class>::get_definition); \
	}

/////////////////////////////////////

template <class T>
T &VariantStruct::is_struct() {
	if (is_empty()) {
		return false;
	}
	return NativeStructDefinition<T>::get_definition() == definition;
}

template <class T>
T &VariantStruct::get_struct() {
	CRASH_COND_MSG(!is_struct<T>(), "Type Mismatch. Should have called is_struct<T>().");

	return *(T *)(instance.get_heap_());
}
