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
const StringName Internal::NativeTypeInfo<T>::get_class_name() const {
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
	if constexpr (std::is_same_v<T, nullptr_t>) {
		// Do Nothing
	} else if constexpr (!std::is_trivially_constructible_v<T>) {
		memnew_placement(p_target, T);
	} else {
		// TODO: assign default values for various types
		// (should use same logic, like how Variant ints are always 0)
	}
}

template <class T>
void Internal::NativeTypeInfo<T>::copy_construct(void *p_target, const void *p_value) const {
	if constexpr (std::is_same_v<T, nullptr_t>) {
		// Do Nothing
	} else if constexpr (!std::is_trivially_copy_constructible_v<T>) {
		memnew_placement(p_target, T(*(T *)p_value));
	} else {
		*reinterpret_cast<T *>(p_target) = *reinterpret_cast<const T *>(p_value);
	}
}

template <class T>
void Internal::NativeTypeInfo<T>::destruct(void *p_target) const {
	if constexpr (std::is_same_v<T, nullptr_t>) {
		// Do Nothing
	} else if constexpr (!std::is_trivially_destructible_v<T>) {
		reinterpret_cast<T *>(p_target)->~T();
	}
}

template <class T>
Variant Internal::NativeTypeInfo<T>::read(const void *p_target) const {
	if constexpr (std::is_same_v<T, nullptr_t>) {
		return Variant(); // Return NIL
	} else {
		return Variant(*reinterpret_cast<const T *>(p_target));
	}
}

template <class T>
void Internal::NativeTypeInfo<T>::ptr_get(const void *p_target, void *p_into) const {
	if constexpr (std::is_same_v<T, nullptr_t>) {
		// Do Nothing
	} else {
		*reinterpret_cast<T *>(p_into) = *reinterpret_cast<const T *>(p_target);
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
		*reinterpret_cast<ObjectID *>(p_target) = static_cast<ObjectID>(static_cast<uint64_t>(p_value));
	} else {
		*reinterpret_cast<T *>(p_target) = static_cast<T>(p_value);
	}
}

template <class T>
void Internal::NativeTypeInfo<T>::ptr_set(void *p_target, const void *p_value) const {
	if constexpr (std::is_same_v<T, nullptr_t>) {
		// Do Nothing
	} else {
		*reinterpret_cast<T *>(p_target) = *reinterpret_cast<const T *>(p_value);
	}
}

/////////////////////////////////////

template <class T>
struct NativeStructDefinition {
	friend class VariantStruct;
	using VarStructRef = Internal::VarStructRef;

private:
	static StructDefinition *sdef;
	static StructDefinition *build_definition();

	// If there are any non-trivial properties, C++ should be able to handle these operations faster than the generic variants above
	static void init_struct(VarStructRef p_struct) {
		if constexpr (!std::is_trivially_constructible_v<T>) {
			memnew_placement(p_struct.instance.get_heap_(), T());
		}
	}
	static void copy_struct(VarStructRef p_struct, const VarStructRef p_other) {
		memnew_placement(p_struct.instance.get_heap_(), T(*p_other.instance.get_heap<T>()));
	}
	static void assign_struct(VarStructRef p_struct, const VarStructRef p_other) {
		*p_struct.instance.get_heap<T>() = *p_other.instance.get_heap<T>();
	}
	static void deinit_struct(VarStructRef p_struct) {
		if constexpr (!std::is_trivially_destructible_v<T>) {
			p_struct.instance.get_heap<T>()->~T();
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

#ifdef VSTRUCT_IS_REFERENCE_TYPE
template <class T>
class NativeVariantStruct : public VariantStruct {
	using StructPtrType = Internal::StructPtrType;
	using VarStructRef = Internal::VarStructRef;

public:
	_ALWAYS_INLINE_ T *operator->() {
		return (T *)(instance.get_heap_());
	}
	_ALWAYS_INLINE_ T &get_struct() {
		return *(T *)(instance.get_heap_());
	}

	_ALWAYS_INLINE_ explicit NativeVariantStruct(bool p_should_construct) {
		VarStructRef ref = NativeStructDefinition<T>::get_definition()->allocate_instance();
		instance = ref.instance;
		definition = ref.definition;
		instance->refcount.init();
		instance->state = Internal::InstanceMetaData::STATE_VALID;
		if (p_should_construct) {
			memnew_placement(instance.get_heap_(), T);
		}
	}
	_ALWAYS_INLINE_ NativeVariantStruct(const T &p_struct) {
		VarStructRef ref = NativeStructDefinition<T>::get_definition()->allocate_instance();
		instance = ref.instance;
		definition = ref.definition;
		instance->refcount.init();
		instance->state = Internal::InstanceMetaData::STATE_VALID;
		memnew_placement(instance.get_heap_(), T(p_struct));
	}
	_ALWAYS_INLINE_ NativeVariantStruct(T &&p_struct) {
		VarStructRef ref = NativeStructDefinition<T>::get_definition()->allocate_instance();
		instance = ref.instance;
		definition = ref.definition;
		instance->refcount.init();
		instance->state = Internal::InstanceMetaData::STATE_VALID;
		memnew_placement(instance.get_heap_(), T(std::move(p_struct)));
	}

	_ALWAYS_INLINE_ NativeVariantStruct(const VariantStruct &p_struct) :
			VariantStruct(p_struct) {
		DEV_ASSERT(definition == NativeStructDefinition<T>::get_definition());
	}
	_ALWAYS_INLINE_ NativeVariantStruct(VariantStruct &&p_struct) :
			VariantStruct(std::move(p_struct)) {
		DEV_ASSERT(definition == NativeStructDefinition<T>::get_definition());
	}

	_ALWAYS_INLINE_ explicit NativeVariantStruct(VarStructRef p_ref) :
			VariantStruct(p_ref) {
		DEV_ASSERT(definition == NativeStructDefinition<T>::get_definition());
	}

	_ALWAYS_INLINE_ NativeVariantStruct<T> &operator=(const NativeStructDefinition<T> &p_other) {
		_ref({ p_other.instance, p_other.definition });
		return *this;
	}
	_ALWAYS_INLINE_ NativeVariantStruct<T> &operator=(NativeStructDefinition<T> &&p_other) {
		std::swap(instance, p_other.instance);
		std::swap(definition, p_other.definition);
		return *this;
	}
};
#endif

// /////////////////////////////////////

// Inplace is a helpful template for creating an instance of a struct that can be easily converted to VariantStruct
// It will instance the struct on the stack as normal, and then copy the data to the heap when cast to VariantStruct
template <class T>
class Inplace : private Internal::InstanceMetaData, public T {
public:
	_ALWAYS_INLINE_ Inplace() {
		state = Internal::InstanceMetaData::STATE_VALID;
		store = Internal::InstanceMetaData::IN_PLACE;
	}
	template <typename... VarArgs>
	_ALWAYS_INLINE_ Inplace(VarArgs... p_args) :
			T(p_args...) {
		state = Internal::InstanceMetaData::STATE_VALID;
		store = Internal::InstanceMetaData::IN_PLACE;
	}
	// Can be directly cast to VariantStruct by copying to the heap through NativeVariantStruct
	_ALWAYS_INLINE_ operator VariantStruct() {
		return NativeVariantStruct(static_cast<T &>(*this));
	}
};

/////////////////////////////////////

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
	Internal::build_native_property(#m_property_name, &T::##m_property_name)

#define REGISTER_INBUILT_STRUCT(m_parent_class, m_struct_class)                                                                          \
	{                                                                                                                                    \
		ClassDB::bind_struct(#m_parent_class, #m_struct_class, &NativeStructDefinition<m_parent_class::m_struct_class>::get_definition); \
	}

/////////////////////////////////////

template <class T>
_ALWAYS_INLINE_ T &VariantStruct::get_struct() {
	if (NativeStructDefinition<T>::get_definition() != definition) {
		ERR_FAIL_MSG("Type Mis-match");
	}

	return *(T *)(instance.get_heap_());
}
