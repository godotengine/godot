/**************************************************************************/
/*  variant_struct.h                                                      */
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

#include "core/config/variant_struct_dev_settings.h" // (dev-note: should remove when squashed)

#include "core/templates/vector.h"
#include "core/variant/variant.h"

class VariantStruct;

class StructDefinition {
	struct AbstractTypeInfo {
		// IMPORTANT: *NEVER* add any properties to this!

		// info
		virtual size_t size() const = 0;
		virtual size_t align() const = 0;
		virtual Variant::Type get_variant_type() const = 0;
		virtual const StringName get_class_name() const = 0;

		// operations
		virtual void construct(void *p_target) const = 0;
		virtual void copy_construct(void *p_target, const void *p_value) const = 0;
		virtual void destruct(void *p_target) const = 0;

		virtual Variant read(const void *p_target) const = 0;
		virtual void ptr_get(const void *p_target, void *p_into) const = 0;

		virtual void write(void *p_target, const Variant &p_value) const = 0;
		virtual void ptr_set(void *p_target, const void *p_value) const = 0;
	};

	template <typename T>
	struct NativeTypeInfo : public AbstractTypeInfo {
		// IMPORTANT: *NEVER* add any properties to this!

		size_t size() const override {
			return sizeof(T);
		}
		size_t align() const override {
			return alignof(T);
		}
		// Variant::Type get_variant_type() const override {
		// 	if constexpr (std::is_same_v<T,nullptr_t>) {
		// 		return Variant::NIL;
		// 	} else {
		// 		return GetTypeInfo<T>::VARIANT_TYPE;
		// 	}
		// }
		const StringName get_class_name() const override {
			if constexpr (std::is_base_of_v<T, Object *>) {
				return std::remove_pointer_t<T>::get_class_static();
			} else {
				// Only pointers to Object have class names
				// (dev-note: not sure if we want to fail)
				// ERR_FAIL_V_MSG(StringName(),"Cannot get class name for struct properties that are not Object pointers");
				return StringName();
			}
		}

		void construct(void *p_target) const override {
			if constexpr (std::is_same_v<T, nullptr_t>) {
				// Do Nothing
			} else if constexpr (!std::is_trivially_constructible_v<T>) {
				new (reinterpret_cast<T *>(p_target)) T;
			} else {
// TODO: assign default values for various types
				// (should use same logic, like how Variant ints are always 0)
			}
		}
		void copy_construct(void *p_target, const void *p_value) const override {
			if constexpr (std::is_same_v<T, nullptr_t>) {
				// Do Nothing
			} else if constexpr (!std::is_trivially_copy_constructible_v<T>) {
				new (reinterpret_cast<T *>(p_target)) T(*reinterpret_cast<const T *>(p_value));
			} else {
				*reinterpret_cast<T *>(p_target) = *reinterpret_cast<const T *>(p_value);
			}
		}
		void destruct(void *p_target) const override {
			if constexpr (std::is_same_v<T, nullptr_t>) {
				// Do Nothing
			} else if constexpr (!std::is_trivially_destructible_v<T>) {
				reinterpret_cast<T *>(p_target)->~T();
			}
		}

		Variant read(const void *p_target) const override {
			if constexpr (std::is_same_v<T, nullptr_t>) {
				return Variant(); // Return NIL
			} else {
				return Variant(*reinterpret_cast<const T *>(p_target));
			}
		}
		void ptr_get(const void *p_target, void *p_into) const override {
			if constexpr (std::is_same_v<T, nullptr_t>) {
				// Do Nothing
			} else {
				*reinterpret_cast<T *>(p_into) = *reinterpret_cast<const T *>(p_target);
			}
		}

		void write(void *p_target, const Variant &p_value) const override {
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
		void ptr_set(void *p_target, const void *p_value) const override {
			if constexpr (std::is_same_v<T, nullptr_t>) {
				// Do Nothing
			} else {
				*reinterpret_cast<T *>(p_target) = *reinterpret_cast<const T *>(p_value);
			}
		}

		Variant::Type get_variant_type() const override {
			if constexpr (std::is_same_v<T, nullptr_t>) {
				return Variant::Type::NIL;
			}

			// atomic types
			if constexpr (std::is_same_v<T, bool>) {
				return Variant::Type::BOOL;
			}
			if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, int32_t> || std::is_same_v<T, int16_t> || std::is_same_v<T, int8_t> || std::is_same_v<T, uint64_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, uint16_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, ObjectID>) {
				return Variant::Type::INT;
			}
			if constexpr (std::is_same_v<T, real_t> || std::is_same_v<T, float> || std::is_same_v<T, double>) {
				return Variant::Type::FLOAT;
			}
			if constexpr (std::is_same_v<T, String> || std::is_same_v<T, char *> || std::is_same_v<T, char32_t *> || std::is_same_v<T, IPAddress>) {
				return Variant::Type::STRING;
			}

			// math types
			if constexpr (std::is_same_v<T, Vector2>) {
				return Variant::Type::VECTOR2;
			}
			if constexpr (std::is_same_v<T, Vector2i>) {
				return Variant::Type::VECTOR2I;
			}
			if constexpr (std::is_same_v<T, Rect2>) {
				return Variant::Type::RECT2;
			}
			if constexpr (std::is_same_v<T, Rect2i>) {
				return Variant::Type::RECT2I;
			}
			if constexpr (std::is_same_v<T, Vector3>) {
				return Variant::Type::VECTOR3;
			}
			if constexpr (std::is_same_v<T, Vector3i>) {
				return Variant::Type::VECTOR3I;
			}
			if constexpr (std::is_same_v<T, Transform2D>) {
				return Variant::Type::TRANSFORM2D;
			}
			if constexpr (std::is_same_v<T, Vector4>) {
				return Variant::Type::VECTOR4;
			}
			if constexpr (std::is_same_v<T, Vector4i>) {
				return Variant::Type::VECTOR4I;
			}
			if constexpr (std::is_same_v<T, Plane>) {
				return Variant::Type::PLANE;
			}
			if constexpr (std::is_same_v<T, Quaternion>) {
				return Variant::Type::QUATERNION;
			}
			if constexpr (std::is_same_v<T, ::AABB>) {
				return Variant::Type::AABB;
			}
			if constexpr (std::is_same_v<T, Basis>) {
				return Variant::Type::BASIS;
			}
			if constexpr (std::is_same_v<T, Transform3D>) {
				return Variant::Type::TRANSFORM3D;
			}
			if constexpr (std::is_same_v<T, Projection>) {
				return Variant::Type::PROJECTION;
			}

			// misc types
			if constexpr (std::is_same_v<T, Color>) {
				return Variant::Type::COLOR;
			}
			if constexpr (std::is_same_v<T, StringName>) {
				return Variant::Type::STRING_NAME;
			}
			if constexpr (std::is_same_v<T, NodePath>) {
				return Variant::Type::NODE_PATH;
			}
			if constexpr (std::is_same_v<T, ::RID>) {
				return Variant::Type::RID;
			}
			if constexpr (std::is_base_of_v<Object *, T>) {
				return Variant::Type::OBJECT;
			}
			if constexpr (std::is_same_v<T, Callable>) {
				return Variant::Type::CALLABLE;
			}
			if constexpr (std::is_same_v<T, Signal>) {
				return Variant::Type::SIGNAL;
			}
			if constexpr (std::is_same_v<T, Dictionary>) {
				return Variant::Type::DICTIONARY;
			}
			if constexpr (std::is_same_v<T, VariantStruct>) {
				return Variant::Type::STRUCT;
			}
			if constexpr (std::is_same_v<T, Array>) {
				return Variant::Type::ARRAY;
			}

			// typed arrays
			if constexpr (std::is_same_v<T, T>) {
				return Variant::Type::PACKED_BYTE_ARRAY;
			}
			if constexpr (std::is_same_v<T, T>) {
				return Variant::Type::PACKED_INT32_ARRAY;
			}
			if constexpr (std::is_same_v<T, T>) {
				return Variant::Type::PACKED_INT64_ARRAY;
			}
			if constexpr (std::is_same_v<T, T>) {
				return Variant::Type::PACKED_FLOAT32_ARRAY;
			}
			if constexpr (std::is_same_v<T, T>) {
				return Variant::Type::PACKED_FLOAT64_ARRAY;
			}
			if constexpr (std::is_same_v<T, T>) {
				return Variant::Type::PACKED_STRING_ARRAY;
			}
			if constexpr (std::is_same_v<T, T>) {
				return Variant::Type::PACKED_VECTOR2_ARRAY;
			}
			if constexpr (std::is_same_v<T, T>) {
				return Variant::Type::PACKED_VECTOR3_ARRAY;
			}
			if constexpr (std::is_same_v<T, T>) {
				return Variant::Type::PACKED_COLOR_ARRAY;
			}
			if constexpr (std::is_same_v<T, T>) {
				return Variant::Type::PACKED_VECTOR4_ARRAY;
			}
		}
	};

public:
	// TypeInfo wraps NativeTypeInfo, allowing us to easily access the virtual table pointer while being compatible with arrays
	struct TypeInfo {
		uint8_t x[sizeof(NativeTypeInfo<int>)];
		_FORCE_INLINE_ const AbstractTypeInfo *const operator->() const {
			return reinterpret_cast<const AbstractTypeInfo *const>(this);
		}

	private:
		TypeInfo() {} // should only be "constructed" through get_native_type_info
	};

	template <typename T>
	_ALWAYS_INLINE_ static TypeInfo get_native_type_info() {
		NativeTypeInfo<T> ret;
		return *reinterpret_cast<TypeInfo *>(&ret);
	}

private:
	template <size_t ByteWidth>
	struct select_uint;
	template <>
	struct select_uint<1> {
		using type = uint8_t;
	};
	template <>
	struct select_uint<2> {
		using type = uint16_t;
	};
	template <>
	struct select_uint<4> {
		using type = uint32_t;
	};
	template <>
	struct select_uint<8> {
		using type = uint64_t;
	};

public:
	// MemberAddress is defined as the equivalent integer of any pointer-to-member
	using MemberAddress = select_uint<sizeof(int StructDefinition::*)>::type;

	struct StructPropertyInfo {
		StringName name;
		MemberAddress address;
		TypeInfo type;

		StructPropertyInfo() :
				address(0), type{ get_native_type_info<nullptr_t>() } {}
#ifdef SHOULD_PRE_CALC_OFFSET_ADDRESS_BY_REFCOUNTSIZE
		_ALWAYS_INLINE_ StructPropertyInfo(StringName p_name, MemberAddress p_address, TypeInfo p_type);
#else
		_ALWAYS_INLINE_ StructPropertyInfo(StringName p_name, MemberAddress p_address, TypeInfo p_type) :
				name{ p_name }, address{ p_address }, type{ p_type } {}
#endif
	};

	template <typename ST, typename MT>
	_ALWAYS_INLINE_ static StructPropertyInfo build_native_property(StringName const &p_name, const MT ST::*const &p_member_pointer) {
		return StructPropertyInfo(p_name, *reinterpret_cast<const MemberAddress *>(&p_member_pointer), get_native_type_info<MT>());
	}

	typedef void (*StructConstructor)(void *, const StructDefinition *);
	typedef void (*StructCopyConstructor)(void *, const StructDefinition *, const void *);
	typedef void (*StructDestructor)(void *, const StructDefinition *);

	// -- StructDefinition definition starts here
public:
	// StructDefinition Properties
	Vector<StructPropertyInfo> properties;
	StringName qualified_name;
	StructConstructor constructor;
	StructCopyConstructor copy_constructor;
	StructDestructor destructor;

	const StructPropertyInfo *get_property_info(const int &p_property_index) const;
	// const StructPropertyInfo *get_property_info(const int &p_property_index) const {
	// 	ERR_FAIL_INDEX_V(p_property_index, properties.size(), nullptr);
	// 	return &properties[p_property_index];
	// }
	const StructPropertyInfo *get_property_info(const StringName &p_property) const;
	const StructPropertyInfo *get_property_info(const String &p_property) const;
	const StructPropertyInfo *get_property_info(const Variant &p_property) const;

	static void generic_constructor(void *p_struct, const StructDefinition *p_definition);
	static void generic_copy_constructor(void *p_struct, const StructDefinition *p_definition, const void *p_other);
	static void generic_destructor(void *p_struct, const StructDefinition *p_definition);
	static void trivial_destructor(void *p_struct, const StructDefinition *p_definition) {}

	const size_t get_size() const;
	static const StructDefinition *get_native(const StringName &p_name);
	static void _register_native_definition(StructDefinition **p_definition);
	static void unregister_native_types();
	static void _register_struct_definition(StructDefinition *p_definition, bool to_clear = true);
	static void clean_struct_definitions();

	StructDefinition(Vector<StructPropertyInfo> p_properties, StringName p_qualified_name, StructConstructor p_constructor, StructCopyConstructor p_copy_constructor, StructDestructor p_destructor) :
			properties{ p_properties }, qualified_name{ p_qualified_name }, constructor{ p_constructor }, copy_constructor{ p_copy_constructor }, destructor{ p_destructor } {}
};

///////////////////////////////

template <class T>
struct NativeStructDefinition {
	friend class VariantStruct;

private:
	static StructDefinition *sdef;
	static StructDefinition *build_definition();

	// These shouldn't ever be needed, as any native struct that should be used for this should not generally define any of these
	// static void move_construct (void *p_struct, StructDefinition *p_definition, void *p_other) {
	// 	if constexpr (!std::is_trivially_constructible_v<T>) {
	// 		new (reinterpret_cast<T *>(p_struct)) T (std::move(*reinterpret_cast<T *>(p_other)));
	// 	}
	// }
	// static void copy_assign (void *p_struct, StructDefinition *p_definition, void *p_other) {
	// 	if constexpr (!std::is_trivially_constructible_v<T>) {
	// 		*reinterpret_cast<T *>(p_struct) = *reinterpret_cast<T *>(p_other);
	// 	}
	// }

	// If there are any non-trivial properties, C++ should be able to handle these operations faster than the generic variants above
	static void init_struct(void *p_struct, const StructDefinition *p_definition) {
		if constexpr (!std::is_trivially_constructible_v<T>) {
			new (reinterpret_cast<T *>(p_struct)) T();
		}
	}
	static void copy_construct(void *p_struct, const StructDefinition *p_definition, const void *p_other) {
		if constexpr (!std::is_trivially_copy_constructible_v<T>) {
			new (reinterpret_cast<T *>(p_struct)) T(*reinterpret_cast<T *>(p_other));
		}
	}
	static void deinit_struct(void *p_struct, const StructDefinition *p_definition) {
		if constexpr (!std::is_trivially_destructible_v<T>) {
			reinterpret_cast<T *>(p_struct)->~T();
		}
	}

public:
	static void clear_definition() {
		if (sdef != nullptr) {
			memdelete(sdef);
			sdef = nullptr;
		}
	}
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

///////////////////////////////

class VariantStruct {
	friend class StructDefinition;
	static constexpr size_t STRUCT_OFFSET = sizeof(uintptr_t);

#define ptr_math(m_ptr, m_operant, m_operand) reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(m_ptr) m_operant m_operand)

protected:
	const StructDefinition *definition;
	void *instance;

#ifdef SHOULD_PRE_CALC_OFFSET_ADDRESS_BY_REFCOUNTSIZE
	_ALWAYS_INLINE_ void *struct_ptr() {
		return ptr_math(instance, +, STRUCT_OFFSET);
	}
	_ALWAYS_INLINE_ SafeRefCount &getrefcount() {
		return *reinterpret_cast<SafeRefCount *>(instance);
	}
#else
	_ALWAYS_INLINE_ void *struct_ptr() {
		return instance;
	}
	_ALWAYS_INLINE_ SafeRefCount &getrefcount() {
		return *reinterpret_cast<SafeRefCount *>(ptr_math(instance, -, STRUCT_OFFSET));
	}
#endif

	// Assigns memory for the given struct definition
	// NOTE: does not construct it, but does initialise the reference counter
	// void allocate(const StructDefinition *p_definition);
	void allocate(const StructDefinition *p_definition) {
		if (instance) {
			_unref();
		}
		definition = p_definition;
#ifdef SHOULD_PRE_CALC_OFFSET_ADDRESS_BY_REFCOUNTSIZE
		instance = Memory::alloc_static(definition->get_size() + STRUCT_OFFSET);
#else
		instance = ptr_math(Memory::alloc_static(definition->get_size() + STRUCT_OFFSET), +, STRUCT_OFFSET);
#endif
		getrefcount().init();
	}

#ifdef VSTRUCT_IS_REFERENCE_TYPE
	// NOTE: member_ptr DOES NOT check if instance != nullptr
	// It is presumed that anywhere these are called has already done that
	_ALWAYS_INLINE_ void *member_ptr(const StructDefinition::MemberAddress &address) {
		return ptr_math(instance, +, address);
	}
	_ALWAYS_INLINE_ const void *member_ptr(const StructDefinition::MemberAddress &address) const {
		return ptr_math(instance, +, address);
	}
	_ALWAYS_INLINE_ void *member_ptr(const StructDefinition::StructPropertyInfo *prop) {
		return member_ptr(prop->address);
	}
	_ALWAYS_INLINE_ const void *member_ptr(const StructDefinition::StructPropertyInfo *prop) const {
		return member_ptr(prop->address);
	}
#else
	void _copy_on_write() {
		if (getrefcount().get() > 1) {
			void *copy_from = struct_ptr();
			allocate(definition);
			definition->copy_constructor(struct_ptr(), definition, copy_from);
		}
	}

	// NOTE: member_ptr and member_ptrw DO NOT check if instance != nullptr
	// It is presumed that anywhere these are called has already done that
	_ALWAYS_INLINE_ void *member_ptrw(const StructDefinition::MemberAddress &address) {
		_copy_on_write();
		return ptr_math(instance, +, address);
	}
	_ALWAYS_INLINE_ void *member_ptrw(const StructDefinition::StructPropertyInfo *prop) {
		return member_ptrw(prop->address);
	}
	_ALWAYS_INLINE_ const void *member_ptr(const StructDefinition::MemberAddress &address) const {
		return ptr_math(instance, +, address);
	}
	_ALWAYS_INLINE_ const void *member_ptr(const StructDefinition::StructPropertyInfo *prop) const {
		return member_ptr(prop->address);
	}
#endif

	void _unref() {
		if (getrefcount().unref()) {
			// Call the destructor for the type, then free the memory
			if (definition->destructor != StructDefinition::trivial_destructor) {
				definition->destructor(struct_ptr(), definition);
			}
#ifdef SHOULD_PRE_CALC_OFFSET_ADDRESS_BY_REFCOUNTSIZE
			Memory::free_static(instance, false);
#else
			Memory::free_static(ptr_math(instance, -, STRUCT_OFFSET), false);
#endif
		}
		instance = nullptr;
	}

public:
	void set(const StringName &p_name, const Variant &p_value, bool &r_valid);
	Variant get(const StringName &p_name, bool &r_valid) const;

	bool is_empty() const {
		return instance == nullptr;
	}

	~VariantStruct() {
		if (instance) {
			_unref();
		}
	}

	VariantStruct() :
			definition(nullptr), instance(nullptr) {}

	VariantStruct(const VariantStruct &p_struct) :
			definition(p_struct.definition), instance(p_struct.instance) {
		if (instance) {
			getrefcount().ref();
		}
	}

	VariantStruct(VariantStruct &&p_struct) :
			definition(p_struct.definition), instance(p_struct.instance) {
		if (p_struct.instance) {
			p_struct.instance = nullptr;
		}
	}

	_FORCE_INLINE_ void operator=(const VariantStruct &p_struct) {
		if (instance) {
			_unref();
		}

		definition = p_struct.definition;
		if (p_struct.instance) {
			instance = p_struct.instance;
			getrefcount().ref();
		}
	}

	_FORCE_INLINE_ void operator=(VariantStruct &&p_struct) {
		if (instance) {
			_unref();
		}

		definition = p_struct.definition;
		if (p_struct.instance) {
			instance = p_struct.instance;
			p_struct.instance = nullptr;
		}
	}
};

// (I couldn't figure out how to get random things to stop trying to use the template constructor for VariantStruct)
template <class T>
class NativeVariantStruct : public VariantStruct {
public:
	NativeVariantStruct(const VariantStruct &p_struct) :
			VariantStruct(p_struct) {}
	NativeVariantStruct(VariantStruct &&p_struct) :
			VariantStruct(p_struct) {}

	NativeVariantStruct(const T &p_struct) {
		allocate(NativeStructDefinition<T>::get_definition());
		definition->copy_constructor(struct_ptr(), definition, &p_struct);
	}
	// NativeVariantStruct(T &&p_struct) {
	// 	allocate(NativeStructDefinition<T>::get_definition());
	// 	definition->move_constructor(struct_ptr(), definition, &p_struct);
	// }
};

#ifdef SHOULD_PRE_CALC_OFFSET_ADDRESS_BY_REFCOUNTSIZE
StructDefinition::StructPropertyInfo::StructPropertyInfo(StringName p_name, MemberAddress p_address, TypeInfo p_type) :
		name{ p_name }, address{ *reinterpret_cast<const MemberAddress *>(&p_address) + VariantStruct::STRUCT_OFFSET }, type{ p_type } {}
#endif
