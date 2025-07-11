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
		virtual bool is_trivial() const = 0;

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
		bool is_trivial() const override {
			if constexpr (std::is_trivially_constructible_v<T> && std::is_trivially_destructible_v<T>) {
				return true;
			} else {
				return false;
			}
		}

		void copy_construct(void *p_target, const void *p_value) const override {
			if constexpr (std::is_same_v<T, nullptr_t>) {
				// Do Nothing
			} else if constexpr (!std::is_trivially_constructible_v<T>) {
				new (reinterpret_cast<T *>(p_target)) T(*reinterpret_cast<const T *>(p_value));
			}
		}
		void construct(void *p_target) const override {
			if constexpr (std::is_same_v<T, nullptr_t>) {
				// Do Nothing
			} else if constexpr (!std::is_trivially_constructible_v<T>) {
				new (reinterpret_cast<T *>(p_target)) T;
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
		// Properties
		StringName name;
		MemberAddress address;
		TypeInfo type;

		StructPropertyInfo() :
				address(0), type{ get_native_type_info<nullptr_t>() } {}
		StructPropertyInfo(StringName p_name, MemberAddress p_address, TypeInfo p_type) :
				name{ p_name }, address{ p_address }, type{ p_type } {}
	};

	template <typename ST, typename MT>
	_ALWAYS_INLINE_ static StructPropertyInfo build_native_property(StringName const &p_name, const MT ST::*const &p_member_pointer) {
		return StructPropertyInfo(p_name, *reinterpret_cast<const MemberAddress *>(&p_member_pointer), get_native_type_info<MT>());
	}

	typedef void (*StructConstructor)(void *, StructDefinition *);
	typedef void (*StructDestructor)(void *, StructDefinition *);

	// -- StructDefinition definition starts here
public:
	// Properties
	Vector<StructPropertyInfo> properties;
	StringName qualified_name;
	StructConstructor constructor;
	StructDestructor destructor;

	const StructPropertyInfo *get_property_info(const int &p_property_index) const;
	// const StructPropertyInfo *get_property_info(const int &p_property_index) const {
	// 	ERR_FAIL_INDEX_V(p_property_index, properties.size(), nullptr);
	// 	return &properties[p_property_index];
	// }
	const StructPropertyInfo *get_property_info(const StringName &p_property) const;
	const StructPropertyInfo *get_property_info(const String &p_property) const;
	const StructPropertyInfo *get_property_info(const Variant &p_property) const;

	static void generic_constructor(void *p_struct, StructDefinition *p_definition);
	static void generic_destructor(void *p_struct, StructDefinition *p_definition);

	const size_t get_size() const;
	static const StructDefinition *get_native(const StringName &p_name);
	static void _register_native_definition(StructDefinition **p_definition);
	static void unregister_native_types();
	static void _register_struct_definition(StructDefinition *p_definition, bool to_clear = true);
	static void clean_struct_definitions();

	StructDefinition(Vector<StructPropertyInfo> p_properties, StringName p_qualified_name, StructConstructor p_constructor, StructDestructor p_destructor) :
			properties{ p_properties }, qualified_name{ p_qualified_name }, constructor{ p_constructor }, destructor{ p_destructor } {}
};

///////////////////////////////

template <class T>
struct NativeStructDefinition {
	friend class VariantStruct;

private:
	static StructDefinition *sdef;
	static StructDefinition *build_definition();

	// These shouldn't ever be needed, as any native struct that should be used for this should not generally define any of these
	// static void copy_construct (void *p_struct, StructDefinition *p_definition, void *p_other) {
	// 	if constexpr (!std::is_trivially_constructible_v<T>) {
	// 		new (reinterpret_cast<T *>(p_struct)) T (*reinterpret_cast<T *>(p_other));
	// 	}
	// }
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

	static void init_struct(void *p_struct, StructDefinition *p_definition) {
		if constexpr (!std::is_trivially_constructible_v<T>) {
			new (reinterpret_cast<T *>(p_struct)) T();
		}
	}
	static void deinit_struct(void *p_struct, StructDefinition *p_definition) {
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
	static StructDefinition *get_definition() {
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
protected:
	// IMPORTANT: Do not add additional properties
	// This needs to fit squarely inside Variant::_data
	const StructDefinition *definition = nullptr;
#ifdef VSTRUCT_IS_REFERENCE_TYPE
	void *instance = nullptr;
#else
	CowData<uint8_t> _cowdata;
#endif

#ifdef VSTRUCT_IS_REFERENCE_TYPE
	// (dev-note: didn't have time to re-write this)
	// _ALWAYS_INLINE_ void *member_ptrw(const StructDefinition::MemberAddress &address) {
	// 	uintptr_t ptr = reinterpret_cast<uintptr_t>(_cowdata.ptrw()) + address;
	// 	return reinterpret_cast<void *>(ptr);
	// }
	// _ALWAYS_INLINE_ const void *member_ptr(const StructDefinition::MemberAddress &address) const {
	// 	uintptr_t ptr = reinterpret_cast<uintptr_t>(_cowdata.ptr()) + address;
	// 	return reinterpret_cast<void *>(ptr);
	// }
	// _ALWAYS_INLINE_ void *member_ptrw(const StructDefinition::StructPropertyInfo *prop) {
	// 	return member_ptrw(prop->address);
	// }
	// _ALWAYS_INLINE_ const void *member_ptr(const StructDefinition::StructPropertyInfo *prop) const {
	// 	return member_ptr(prop->address);
	// }
#else
	_ALWAYS_INLINE_ void *member_ptrw(const StructDefinition::MemberAddress &address) {
		// If this causes a CoW, we need to ensure that any appropriate copy-constructors are called for non-trivial types

		// (dev-note: optimally in the finaly implementation, this would all be handled internally)
		// (this is because CowData is already copying the bytes over; it just isn't aware of the actual types of the data)
		// (so it would be better to not perform those needless writes; in the meantime, this ensures that any appropriate copy constructor is called)
		// (which is important for anything non-trivial)
		const void *r = _cowdata.ptr();
		void *w = _cowdata.ptrw();
		if (r != w) {
			for (const StructDefinition::StructPropertyInfo &E : definition->properties) {
				if (!E.type->is_trivial()) {
					void *here = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(w) + E.address);
					const void *there = reinterpret_cast<void *>(reinterpret_cast<const uintptr_t>(r) + E.address);
					E.type->copy_construct(here, there);
				}
			}
		}

		uintptr_t ptr = reinterpret_cast<uintptr_t>(w) + address;
		return reinterpret_cast<void *>(ptr);
	}
	_ALWAYS_INLINE_ const void *member_ptr(const StructDefinition::MemberAddress &address) const {
		uintptr_t ptr = reinterpret_cast<uintptr_t>(_cowdata.ptr()) + address;
		return reinterpret_cast<void *>(ptr);
	}
	_ALWAYS_INLINE_ void *member_ptrw(const StructDefinition::StructPropertyInfo *prop) {
		return member_ptrw(prop->address);
	}
	_ALWAYS_INLINE_ const void *member_ptr(const StructDefinition::StructPropertyInfo *prop) const {
		return member_ptr(prop->address);
	}
#endif

public:
	void set(const StringName &p_name, const Variant &p_value, bool &r_valid);
	Variant get(const StringName &p_name, bool &r_valid) const;


	bool is_empty() const {
#ifdef VSTRUCT_IS_REFERENCE_TYPE
		return instance == nullptr;
#else
		return _cowdata.is_empty();
#endif
	}

#ifdef VSTRUCT_IS_REFERENCE_TYPE
private:
	_ALWAYS_INLINE_ SafeRefCount &getrefcount() {
		uintptr_t ptr = reinterpret_cast<uintptr_t>(instance) - sizeof(SafeRefCount);
		return *reinterpret_cast<SafeRefCount *>(reinterpret_cast<void *>(ptr));
	}
	void _ref(void *p_instance) {
		if (instance) {
			_unref();
		}
		instance = p_instance;
		if (instance) {
			getrefcount().ref();
		}
	}
	void _unref() {
		if (getrefcount().unref()) {
			memdelete(&getrefcount());
		}
		instance = nullptr;
	}
	void allocate(const StructDefinition *p_definition) {
		if (instance) {
			_unref();
		}
		definition = p_definition;
		uintptr_t ptr = reinterpret_cast<uintptr_t>(memalloc(definition->get_size() + sizeof(SafeRefCount)));
		instance = reinterpret_cast<void *>(ptr + sizeof(SafeRefCount));
		getrefcount().init();
	}

public:
	~VariantStruct() {
		if (instance) {
			_unref();
		}
	}
	VariantStruct() {}
	VariantStruct(const VariantStruct &p_struct) :
			definition(p_struct.definition) {
		if (p_struct.instance) {
			_ref(p_struct.instance);
		}
	}
	VariantStruct(VariantStruct &&p_struct) :
			definition(p_struct.definition) {
		if (p_struct.instance) {
			instance = p_struct.instance;
			p_struct.instance = nullptr;
		}
	}
#else
private:
	// note: part of super hacky solution below
	typedef uint64_t USize;
	constexpr static size_t REF_COUNT_OFFSET = 0;
	constexpr static size_t SIZE_OFFSET = ((REF_COUNT_OFFSET + sizeof(SafeNumeric<USize>)) % alignof(USize) == 0) ? (REF_COUNT_OFFSET + sizeof(SafeNumeric<USize>)) : ((REF_COUNT_OFFSET + sizeof(SafeNumeric<USize>)) + alignof(USize) - ((REF_COUNT_OFFSET + sizeof(SafeNumeric<USize>)) % alignof(USize)));
	constexpr static size_t DATA_OFFSET = ((SIZE_OFFSET + sizeof(USize)) % alignof(max_align_t) == 0) ? (SIZE_OFFSET + sizeof(USize)) : ((SIZE_OFFSET + sizeof(USize)) + alignof(max_align_t) - ((SIZE_OFFSET + sizeof(USize)) % alignof(max_align_t)));

public:
	VariantStruct() {}
	VariantStruct(const VariantStruct &p_struct) :
			definition(p_struct.definition), _cowdata(p_struct._cowdata) {}
	VariantStruct(VariantStruct &&p_struct) :
			definition(p_struct.definition), _cowdata(std::move(p_struct._cowdata)) {}
	~VariantStruct() {
		// If this causes a CoW, we need to ensure that any appropriate de-constructors are called for non-trivial types

		// (dev-note: optimally in the finaly implementation, this would all be handled internally)
		// (however, because we currently rely on CowData, and it isn't aware of the actual types of the data)
		// (we need to deconstruct any non-trivial type)

		// (so, for now, we have this super hacky solution!)

		if (!_cowdata.is_empty()) {
			const void *r = _cowdata.ptr();
			SafeNumeric<USize> *sn = (SafeNumeric<USize> *)((uint8_t *)r - DATA_OFFSET + REF_COUNT_OFFSET);

			// check if it is the last instance, then deconstruct any non-trivial properties
			if (sn->get() <= 1) {
				for (const StructDefinition::StructPropertyInfo &E : definition->properties) {
					if (!E.type->is_trivial()) {
						void *ptr = reinterpret_cast<void *>(reinterpret_cast<const uintptr_t>(r) + E.address);
						E.type->destruct(ptr);
					}
				}
			}
		}
	}
#endif

	_FORCE_INLINE_ void operator=(const VariantStruct &p_struct) {
		definition = p_struct.definition;
#ifdef VSTRUCT_IS_REFERENCE_TYPE
		if (instance) {
			_unref();
		}
		definition = p_struct.definition;
		_ref(p_struct.instance);
#else
		definition = p_struct.definition;
		_cowdata = p_struct._cowdata;
#endif
	}
	_FORCE_INLINE_ void operator=(VariantStruct &&p_struct) {
#ifdef VSTRUCT_IS_REFERENCE_TYPE
		if (instance) {
			_unref();
		}
		definition = p_struct.definition;
		if (p_struct.instance) {
			instance = p_struct.instance;
			p_struct.instance = nullptr;
		}
#else
		definition = p_struct.definition;
		_cowdata = std::move(p_struct._cowdata);
#endif
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
#ifdef VSTRUCT_IS_REFERENCE_TYPE
		allocate(NativeStructDefinition<T>::get_definition());
		new (reinterpret_cast<T *>(instance)) T(p_struct);
#else
		definition = NativeStructDefinition<T>::get_definition();
		_cowdata.resize(definition->get_size());
		new (reinterpret_cast<T *>(_cowdata.ptrw())) T(p_struct);
#endif
	}
	NativeVariantStruct(T &&p_struct) {
#ifdef VSTRUCT_IS_REFERENCE_TYPE
		allocate(NativeStructDefinition<T>::get_definition());
		new (reinterpret_cast<T *>(instance)) T(p_struct);
#else
		definition = NativeStructDefinition<T>::get_definition();
		_cowdata.resize(definition->get_size());
		new (reinterpret_cast<T *>(_cowdata.ptrw())) T(p_struct);
#endif
	}
};