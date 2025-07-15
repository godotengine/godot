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

#include "core/templates/heap_object.h"
#include "core/templates/vtable_pointer.h"
#include "core/variant/variant.h"

namespace Internal {
// class VarStructRef;

// It is important that the meta data is sized to a multiple of 8 bytes
// This ensures that we know the struct data will start with a 8-byte alignment
struct InstanceMetaData {
	SafeRefCount refcount; // 4 bytes
	enum StructType : uint8_t {
		PASS_BY_REFERENCE,
		PASS_BY_VALUE_COW,
	};
	StructType type; // 1 byte
	enum StoreType : uint8_t {
		IN_PLACE,
		IN_HEAP,
		IN_PLACE_IN_HEAP, // <-- should only be possible for PASS_BY_VALUE_COW
	};
	StoreType store; // 1 byte
	uint16_t __padding__; // 2 bytes
};
using MemberPtrType = VarHeapPointer<InstanceMetaData>::MemberDataPointer;

} //namespace Internal

//

namespace Internal {

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

template <class T>
struct NativeTypeInfo : public AbstractTypeInfo {
	// IMPORTANT: *NEVER* add any properties to this!

	// info
	size_t size() const override;
	size_t align() const override;
	Variant::Type get_variant_type() const override;
	const StringName get_class_name() const override;

	// operations
	void construct(void *p_target) const override;
	void copy_construct(void *p_target, const void *p_value) const override;
	void destruct(void *p_target) const override;

	Variant read(const void *p_target) const override;
	void ptr_get(const void *p_target, void *p_into) const override;

	void write(void *p_target, const Variant &p_value) const override;
	void ptr_set(void *p_target, const void *p_value) const override;
};

} //namespace Internal

namespace Internal {

struct StructPropertyInfo : public MemberPtrType {
	using MemberPtrType = Internal::MemberPtrType;
	using AbstractTypeInfo = Internal::AbstractTypeInfo;

	enum Kind : uint32_t {
		NATIVE_BUILTIN,
		NATIVE_OBJECT,
		KIND_MAX,
	};

	// (should be aligned to take up a very minimal 24 bytes)
	// (4 for MemberDataPointer + 4 for Kind // 8 for union pointers // 8 for StringName)
	// VarStructPtrType::MemberDataPointer address; // (used as base class to simplify code; is positioned here)
	Kind kind;
	union {
		VPointer<AbstractTypeInfo> type_info;
		StringName class_name;
	};
	StringName name;

// public:
	_ALWAYS_INLINE_ StringName get_name() const {
		return name;
	}
	_FORCE_INLINE_ Variant::Type get_variant_type() const {
		switch (kind) {
			case NATIVE_BUILTIN:
				return type_info->get_variant_type();
			case NATIVE_OBJECT:
				return Variant::Type::OBJECT;
			default:
				ERR_FAIL_V(Variant::Type::NIL);
		}
	}

	StructPropertyInfo(const StringName &p_name, const MemberPtrType &p_address, const VPointer<AbstractTypeInfo> &p_type) :
			kind(NATIVE_BUILTIN), type_info(p_type), name(p_name), MemberPtrType(p_address) {}

	StructPropertyInfo(const StructPropertyInfo &p_other) :
			kind(p_other.kind), name(p_other.name), MemberPtrType(p_other) {
		switch (kind) {
			case NATIVE_BUILTIN:
				type_info = p_other.type_info;
				break;
			case NATIVE_OBJECT:
				new (&class_name) StringName(p_other.class_name);
				break;
			default:
				ERR_FAIL();
		}
	}

	~StructPropertyInfo() {
		switch (kind) {
			case NATIVE_BUILTIN:
				type_info.~VPointer<AbstractTypeInfo>();
				break;
			case NATIVE_OBJECT:
				class_name.~StringName();
				break;
			default:
				ERR_FAIL();
				break;
		}
	}
};

template <typename ST, typename MT>
_ALWAYS_INLINE_ static StructPropertyInfo build_native_property(StringName const &p_name, const MT ST::*const &p_member_pointer) {
	return StructPropertyInfo(p_name, p_member_pointer, NativeTypeInfo<MT>());
}

} //namespace Internal

///////////////////////////////

namespace Internal {

// Represents a pairing between the pointer to an instance and a pointer to the definition that defines how it is structured and accessed
// Handles the internal logic of memory management and distinct behaviours in an as-invisible way as possible
// Is never on it's own considered an actual reference to the struct, so does not affect refcount on construction or destruction

// It is a trivial type (which should allow for easy and quick operations)
// This also endeavers to ensure that there is no way a pointer to an instance can get lost from its definition

// This doesn't count as a "true reference" to the given struct unless it exists within either a Record<> or VariantStruct

class VarStructRef {
	using InstanceMetaData = Internal::InstanceMetaData;

	// Needs to fit on Variant, so should not have any other properties
	VarHeapPointer<InstanceMetaData> instance; // 8 bytes
	const StructDefinition *definition; // 8 bytes

public:
	_ALWAYS_INLINE_ operator bool() const {
		return instance;
	}
	_ALWAYS_INLINE_ void *operator->*(const MemberPtrType &p_ptr) {
		return instance->*p_ptr;
	}
	_ALWAYS_INLINE_ const void *operator->*(const MemberPtrType &p_ptr) const {
		return instance->*p_ptr;
	}

	_ALWAYS_INLINE_ bool is_refcounted() {
		return instance->store == InstanceMetaData::IN_HEAP;
	}
	_ALWAYS_INLINE_ bool is_cow() {
		return instance->store == InstanceMetaData::IN_HEAP && instance->type == InstanceMetaData::PASS_BY_VALUE_COW;
	}
	_ALWAYS_INLINE_ bool check_cow() {
		return instance->store == InstanceMetaData::IN_HEAP && instance->type == InstanceMetaData::PASS_BY_VALUE_COW && instance->refcount.get() > 1;
	}
	_ALWAYS_INLINE_ void init() {
		DEV_ASSERT(is_refcounted());
		// if (!is_refcounted()) {
		// 	ERR_FAIL_MSG("Cannot init structs not stored on the heap");
		// }
		instance->refcount.init();
	}
	_ALWAYS_INLINE_ void ref() {
		DEV_ASSERT(is_refcounted());
		// if (!is_refcounted()) {
		// 	ERR_FAIL_MSG("Cannot ref structs not stored on the heap");
		// }
		if (!instance->refcount.ref()) {
			instance->refcount.init();
		}
	}
	_ALWAYS_INLINE_ void unref();
	// _ALWAYS_INLINE_ void unref() {
	// 	DEV_ASSERT(is_refcounted());
	// 	// if (!is_refcounted()) {
	// 	// 	ERR_FAIL_MSG("Cannot unref structs not stored on the heap");
	// 	// }
	// 	if (instance->refcount.unref()) {
	// 		// Call the destructor for the type, then free the memory
	// 		if (definition->destructor != StructDefinition::trivial_destructor) {
	// 			definition->destructor(*this);
	// 		}
	// 		instance.free(); // sets instance to nullptr
	// 	} else {
	// 		instance = nullptr;
	// 	}
	// }
	// _ALWAYS_INLINE_ void *struct_ptr() {
	// 	return instance.get_heap_();
	// }
	_ALWAYS_INLINE_ void *get_struct_ptr() {
		return instance.get_heap_();
	}
	template <class T>
	_ALWAYS_INLINE_ T &get_struct_ref() {
		return *(T *)(get_struct_ptr());
	}
	_ALWAYS_INLINE_ const StructDefinition *get_definition() const {
		return definition;
	}
	// static VarStructRef allocate(const StructDefinition *p_definition);
	// // static VarStructRef allocate(const StructDefinition *p_definition) {
	// // 	VarStructRef ret;
	// // 	ret.instance.allocate_(p_definition->size);
	// // 	ret.definition = p_definition;
	// // 	return ret;
	// // }
	// static VarStructRef duplicate(VarStructRef p_copy_from);
	// // static VarStructRef duplicate(VarStructRef p_copy_from) {
	// // 	VarStructRef ret = allocate(p_copy_from.definition);
	// // 	ret.definition->copy_constructor(ret, p_copy_from.instance);
	// // }

	VarStructRef() = default;
	VarStructRef(nullptr_t, nullptr_t) :
			instance(nullptr), definition(nullptr) {}
	VarStructRef(VarHeapPointer<InstanceMetaData> p_instance, const StructDefinition *p_definition) :
			instance(p_instance), definition(p_definition) {}
};

} //namespace Internal

/////////////////////////////////////

class StructDefinition : public VarHeapObject {
	using StructPropertyInfo = Internal::StructPropertyInfo;
	using VarStructRef = Internal::VarStructRef;

public:
	typedef void (*StructConstructor)(VarStructRef);
	typedef void (*StructCopyConstructor)(VarStructRef, const VarStructRef);
	typedef void (*StructDestructor)(VarStructRef);
	// StructDefinition Properties
	StringName qualified_name;
	StructConstructor constructor;
	StructCopyConstructor copy_constructor;
	StructDestructor destructor;
	size_t size;
	VarHeapData<StructPropertyInfo> properties;

	const StructPropertyInfo *get_property_info(const int &p_property_index) const;
	const StructPropertyInfo *get_property_info(const StringName &p_property) const;
	const StructPropertyInfo *get_property_info(const String &p_property) const;
	const StructPropertyInfo *get_property_info(const Variant &p_property) const;

	// Generic struct constructors; for non-native structs, or for safely creating native structs that don't have constructors
	static void generic_constructor(VarStructRef p_struct);
	static void generic_copy_constructor(VarStructRef p_struct, const VarStructRef);
	// Generic struct destructors; for non-native structs
	static void generic_destructor(VarStructRef p_struct);
	static void trivial_destructor(VarStructRef p_struct) {}

	//

	static const StructDefinition *get_native(const StringName &p_name);
	static void _register_native_definition(StructDefinition **p_definition);
	static void unregister_native_types();
	static void _register_struct_definition(StructDefinition *p_definition, bool to_clear = true);
	static void clean_struct_definitions();

private:
	StructDefinition(StringName p_qualified_name, size_t p_size, StructConstructor p_constructor, StructCopyConstructor p_copy_constructor, StructDestructor p_destructor) :
			qualified_name{ p_qualified_name }, size{ p_size }, constructor{ p_constructor }, copy_constructor{ p_copy_constructor }, destructor{ p_destructor } {}

public:
	static StructDefinition *create(std::initializer_list<StructPropertyInfo> p_properties, StringName p_qualified_name, size_t p_size, StructConstructor p_constructor, StructCopyConstructor p_copy_constructor, StructDestructor p_destructor) {
		void *ptr = heap_allocate(&StructDefinition::properties, p_properties);
		new (ptr) StructDefinition(p_qualified_name, p_size, p_constructor, p_copy_constructor, p_destructor);
		return (StructDefinition *)ptr;
	}
};

void Internal::VarStructRef::unref() {
	DEV_ASSERT(is_refcounted());
	if (instance->refcount.unref()) {
		// Call the destructor for the type, then free the memory
		if (definition->destructor != StructDefinition::trivial_destructor) {
			definition->destructor(*this);
		}
		instance.free(); // sets instance to nullptr
	} else {
		instance = nullptr;
	}
}

///////////////////////////////

template <class T>
struct NativeStructDefinition {
	friend class VariantStruct;
	// using VarStructPtrType = Internal::VarStructPtrType;
	using VarStructRef = Internal::VarStructRef;

private:
	static StructDefinition *sdef;
	static StructDefinition *build_definition();

	// If there are any non-trivial properties, C++ should be able to handle these operations faster than the generic variants above
	static void init_struct(VarStructRef p_struct) {
		if constexpr (!std::is_trivially_constructible_v<T>) {
			new (p_struct.get_struct_ptr()) T();
		}
	}
	static void copy_struct(VarStructRef p_struct, const VarStructRef p_other) {
		if constexpr (!std::is_trivially_copy_constructible_v<T>) {
			new (p_struct.get_struct_ptr()) T(p_other.get_struct_ref<T>());
		}
	}
	static void deinit_struct(VarStructRef p_struct) {
		if constexpr (!std::is_trivially_destructible_v<T>) {
			p_struct.get_struct_ref<T>()->~T();
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

///////////////////////////////

namespace Internal {

static VarStructRef allocate(const StructDefinition *p_definition) {
	VarHeapPointer<InstanceMetaData> ref(p_definition->size);
	ref->refcount.init(0);
	ref->type = InstanceMetaData::PASS_BY_VALUE_COW;
	ref->store = InstanceMetaData::IN_HEAP;
	ref->__padding__ = 0;
	return VarStructRef(ref, p_definition);
}

static VarStructRef duplicate(VarStructRef p_target) {
	const StructDefinition *def = p_target.get_definition();
	VarStructRef dupe = allocate(def);
	def->copy_constructor(dupe, p_target);
	return dupe;
}

} //namespace Internal

///////////////////////////////

template <class T>
class Record {
	Internal::VarStructRef _p;

public:
	_ALWAYS_INLINE_ Record() {
		_p = Internal::allocate(NativeStructDefinition<T>::get_definition());
		_p.init();
		new (_p.get_struct_ptr()) T();
	}
	_ALWAYS_INLINE_ Record(const Record<T> &p_other) {
		_p = p_other._p;
		_p.ref();
	}
	template <typename... VarArgs>
	_ALWAYS_INLINE_ Record(VarArgs... p_args) {
		_p = Internal::allocate(NativeStructDefinition<T>::get_definition());
		_p.init();
		new (_p.get_struct_ptr()) T(p_args...);
	}
	_ALWAYS_INLINE_ Record<T> &operator=(const Record<T> &p_other) {
		if (_p) {
			_p.unref();
		}
		_p = p_other._p;
		if (_p) {
			_p.ref();
		}
		return *this;
	}
	_ALWAYS_INLINE_ ~Record() {
		if (_p) {
			_p.unref();
		}
	}
	_ALWAYS_INLINE_ T *operator->() {
		return (T *)(_p.get_struct_ptr());
	}
};

template <class T>
class Inplace : private Internal::InstanceMetaData, public T {
public:
	_ALWAYS_INLINE_ Inplace() {
		store = IN_PLACE;
	}
	// template <typename... VarArgs>
	// _ALWAYS_INLINE_ Inplace(VarArgs... p_args) :
	// 		T(p_args...) {
	// 	store = IN_PLACE;
	// }
	operator Internal::VarStructRef() const {
		return VarStructRef(VarHeapPointer<InstanceMetaData>(this), NativeStructDefinition<T>::get_definition());
	}
};

///////////////////////////////

class VariantStruct {
	// using VarStructPtrType = Internal::VarStructPtrType;
	using VarStructRef = Internal::VarStructRef;
	using MemberPtrType = Internal::MemberPtrType;
	friend class StructDefinition;

protected:
	// Needs to fit on Variant, so should not have any other properties
	// const StructDefinition *definition;
	// VarStructPtrType instance;
	VarStructRef _p; // 16 bytes

	// // Assigns memory for the given struct definition and initialises the meta-data
	// // NOTE: does not construct it; it is presumed that the caller is either going to construct or copy-construct
	// void new_instance(const StructDefinition *p_definition);

	// _ALWAYS_INLINE_ void _init(VarStructRef p_p) {
	// 	_p = p_p;
	// 	if (_p && _p.is_refcounted()) {
	// 		_p.init();
	// 	}
	// }
	_ALWAYS_INLINE_ void _ref(VarStructRef p_p) {
		_p = p_p;
		// if (_p && _p.is_refcounted()) {
		// 	_p.ref();
		// }
		if (_p) {
			_p.ref();
		}
	}
	_ALWAYS_INLINE_ void _unref() {
		// if (_p && _p.is_refcounted()) {
		// 	_p.unref();
		// }
		if (_p) {
			_p.unref();
		}
	}

	// #ifdef VSTRUCT_IS_REFERENCE_TYPE
	// 	// NOTE: member_ptr DOES NOT check if instance != nullptr
	// 	// It is presumed that anywhere these are called has already done that
	// 	_ALWAYS_INLINE_ void *member_ptr(const MemberPtrType &address) {
	// 		return instance->*address;
	// 	}
	// 	template <typename T>
	// 	_ALWAYS_INLINE_ T &member_data(const MemberPtrType &address) {
	// 		return instance.get_data<T>(address);
	// 	}
	// 	_ALWAYS_INLINE_ const void *member_ptr(const MemberPtrType &address) const {
	// 		return instance->*address;
	// 	}
	// 	template <typename T>
	// 	_ALWAYS_INLINE_ const T &member_data(const MemberPtrType &address) const {
	// 		return instance.get_data<T>(address);
	// 	}
	// #else
		void _copy_on_write();

	// 	// NOTE: member_ptr and member_ptrw DO NOT check if instance != nullptr
	// 	// It is presumed that anywhere these are called has already done that
	// 	_ALWAYS_INLINE_ void *member_ptrw(const MemberPtrType &address) {
	// 		_copy_on_write();
	// 		return instance->*address;
	// 	}
	// 	template <typename T>
	// 	_ALWAYS_INLINE_ T &member_rw(const MemberPtrType &address) {
	// 		_copy_on_write();
	// 		return instance.get_data<T>(address);
	// 	}
	// 	_ALWAYS_INLINE_ const void *member_ptr(const MemberPtrType &address) const {
	// 		return instance->*address;
	// 	}
	// 	template <typename T>
	// 	_ALWAYS_INLINE_ const T &member_r(const MemberPtrType &address) const {
	// 		return instance.get_data<T>(address);
	// 	}
	// #endif

public:
	// #ifdef VSTRUCT_IS_REFERENCE_TYPE
	// 	VariantStruct duplicate();
	// #endif
	void set(const StringName &p_name, const Variant &p_value, bool &r_valid);
	Variant get(const StringName &p_name, bool &r_valid) const;

	void clear();
	bool is_empty() const;

	template <class T>
	_ALWAYS_INLINE_ T &get_struct() {
		if (NativeStructDefinition<T>::get_definition() != _p.get_definition()) {
			ERR_FAIL_MSG("Type Mis-match");
		}

		return *(T *)(_p.get_struct_ptr());
	}

	// // // // Creates a new instance of the given StructDefinition, without initialising any properties
	// // // // This is the only way to do this, and relies on the caller to be doing so intentionally (or risk undefined behaviour)
	// // // // (any other way of creating a VariantStruct instance either: allocates no additional memory; uses a given struct; or uses constructors)
	// // // _FORCE_INLINE_ static VariantStruct allocate_struct(const StructDefinition *p_definition) {
	// // // 	VariantStruct ret;
	// // // 	ret.new_instance(p_definition);
	// // // 	return ret;
	// // // }
	// // // template <class T>
	// // // _ALWAYS_INLINE_ static VariantStruct allocate_struct() {
	// // // 	return allocate_struct(NativeStructDefinition<T>::get_definition());
	// // // }

	// // // _FORCE_INLINE_ static VariantStruct construct_new(const StructDefinition *p_definition) {
	// // // 	VariantStruct ret;
	// // // 	ret.new_instance(p_definition);
	// // // 	p_definition->constructor(ret.instance, p_definition);
	// // // 	return ret;
	// // // }
	// // // template <class T>
	// // // _ALWAYS_INLINE_ static VariantStruct construct_new() {
	// // // 	return construct_new(NativeStructDefinition<T>::get_definition());
	// // // }

	VariantStruct() :
			_p{ nullptr, nullptr } {}

	VariantStruct(VarStructRef p_ref) {
		// _p = p_ref;
		// _p.ref();
		_ref(p_ref);
	}
	VariantStruct(const VariantStruct &p_struct) {
		// _p = p_struct._p;
		// if (_p) {
		// 	_p.ref();
		// }
		_ref(p_struct._p);
	}
	VariantStruct(VariantStruct &&p_struct) {
		_p = p_struct._p;
		p_struct._p = VarStructRef(nullptr, nullptr);
	}

	_FORCE_INLINE_ void operator=(const VariantStruct &p_struct) {
		_unref();
		_ref(p_struct._p);
		// _p = p_struct._p;
		// if (_p) {
		// 	_p.ref();
		// }
		// definition = p_struct.definition;
		// if (p_struct.instance) {
		// 	instance = p_struct.instance;
		// 	instance->refcount.ref();
		// }
	}

	_FORCE_INLINE_ void operator=(VariantStruct &&p_struct) {
		_unref();
		std::swap(_p, p_struct._p);
		// _p = p_struct._p;
		// p_struct._p = VarStructRef(nullptr, nullptr);
		// definition = p_struct.definition;
		// if (p_struct.instance) {
		// 	instance = p_struct.instance;
		// 	p_struct.instance = nullptr;
		// }
	}

	~VariantStruct() {
		static_assert(std::is_trivial_v<VarStructRef>);
		_unref();
	}
};

// (I couldn't figure out how to get random things to stop trying to use the template constructor for VariantStruct)
template <class T>
class NativeVariantStruct : public VariantStruct {
public:
	_ALWAYS_INLINE_ T &get_struct() {
		return *(T *)(_p.get_struct_ptr());
	}

	_ALWAYS_INLINE_ explicit NativeVariantStruct(bool p_should_construct) {
		// new_instance(NativeStructDefinition<T>::get_definition());
		_ref(Internal::allocate(NativeStructDefinition<T>::get_definition()));
		if (p_should_construct) {
			new (_p.get_struct_ptr()) T;
		}
	}

	_ALWAYS_INLINE_ NativeVariantStruct(const VariantStruct &p_struct) :
			VariantStruct(p_struct) {
		DEV_ASSERT(_p.get_definition() == NativeStructDefinition<T>::get_definition());
	}
	_ALWAYS_INLINE_ NativeVariantStruct(VariantStruct &&p_struct) :
			VariantStruct(std::move(p_struct)) {
		DEV_ASSERT(_p.get_definition() == NativeStructDefinition<T>::get_definition());
	}

	_ALWAYS_INLINE_ NativeVariantStruct(const T &p_struct) {
		// new_instance(NativeStructDefinition<T>::get_definition());
		_ref(Internal::allocate(NativeStructDefinition<T>::get_definition()));
		new (_p.get_struct_ptr()) T(p_struct);
	}

	_ALWAYS_INLINE_ NativeVariantStruct(T &&p_struct) {
		// new_instance(NativeStructDefinition<T>::get_definition());
		_ref(Internal::allocate(NativeStructDefinition<T>::get_definition()));
		new (_p.get_struct_ptr()) T(std::move(p_struct));
	}
};

// class VariantStruct;
// class StructDefinition;
// class VarStructRef;

// // namespace Internal {
// // struct VarStructMetaData {
// // 	// It is important that this is sized to a multiple of 8 bytes
// // 	// This ensures that we know the struct data will start with a 8-byte alignment
// // 	SafeRefCount refcount; // 4 bytes
// // 	enum StructType : uint32_t {
// // 		IN_HEAP,
// // 		IN_PLACE,
// // 	};
// // 	StructType type; // 4 bytes
// // };
// // using VarStructPtrType = VarHeapPointer<VarStructMetaData>;
// // } //namespace Internal

// namespace Internal {
// // It is important that the meta data is sized to a multiple of 8 bytes
// // This ensures that we know the struct data will start with a 8-byte alignment
// struct InstanceMetaData {
// 	SafeRefCount refcount; // 4 bytes
// 	enum StructType : uint8_t {
// 		PASS_BY_REFERENCE,
// 		PASS_BY_VALUE_COW,
// 	};
// 	StructType type; // 1 byte
// 	enum StoreType : uint8_t {
// 		IN_PLACE,
// 		IN_HEAP,
// 		IN_PLACE_IN_HEAP, // <-- should only be possible for PASS_BY_VALUE_COW
// 	};
// 	StoreType store; // 1 byte
// 	uint16_t __padding__; // 2 bytes
// };
// // using VarStructPtrType = VarHeapPointer<InstanceMetaData>;
// // using MemberPtrType = VarStructPtrType::MemberDataPointer;
// using MemberPtrType = VarHeapPointer<InstanceMetaData>::MemberDataPointer;

// struct AbstractTypeInfo {
// 	// IMPORTANT: *NEVER* add any properties to this!

// 	// info
// 	virtual size_t size() const = 0;
// 	virtual size_t align() const = 0;
// 	virtual Variant::Type get_variant_type() const = 0;
// 	virtual const StringName get_class_name() const = 0;

// 	// operations
// 	virtual void construct(void *p_target) const = 0;
// 	virtual void copy_construct(void *p_target, const void *p_value) const = 0;
// 	virtual void destruct(void *p_target) const = 0;

// 	virtual Variant read(const void *p_target) const = 0;
// 	virtual void ptr_get(const void *p_target, void *p_into) const = 0;

// 	virtual void write(void *p_target, const Variant &p_value) const = 0;
// 	virtual void ptr_set(void *p_target, const void *p_value) const = 0;
// };

// template <class T>
// struct NativeTypeInfo : public AbstractTypeInfo {
// 	// IMPORTANT: *NEVER* add any properties to this!

// 	// info
// 	size_t size() const override;
// 	size_t align() const override;
// 	Variant::Type get_variant_type() const override;
// 	const StringName get_class_name() const override;

// 	// operations
// 	void construct(void *p_target) const override;
// 	void copy_construct(void *p_target, const void *p_value) const override;
// 	void destruct(void *p_target) const override;

// 	Variant read(const void *p_target) const override;
// 	void ptr_get(const void *p_target, void *p_into) const override;

// 	void write(void *p_target, const Variant &p_value) const override;
// 	void ptr_set(void *p_target, const void *p_value) const override;
// };
// } //namespace Internal

/////////////////////////////////////

// class StructDefinition : public VarHeapObject {
// 	/////////////////////////////////////

// public:
// 	// typedef void (*StructConstructor)(VarStructPtrType, const StructDefinition *);
// 	// typedef void (*StructCopyConstructor)(VarStructPtrType, const StructDefinition *, const VarStructPtrType);
// 	// typedef void (*StructDestructor)(VarStructPtrType, const StructDefinition *);
// 	typedef void (*StructConstructor)(VarStructRef);
// 	typedef void (*StructCopyConstructor)(VarStructRef, const VarStructRef);
// 	typedef void (*StructDestructor)(VarStructRef);

// 	// -- StructDefinition definition starts here
// public:
// 	// StructDefinition Properties
// 	StringName qualified_name;
// 	StructConstructor constructor;
// 	StructCopyConstructor copy_constructor;
// 	StructDestructor destructor;
// 	size_t size;
// 	VarHeapData<StructPropertyInfo> properties;

// 	const StructPropertyInfo *get_property_info(const int &p_property_index) const;
// 	const StructPropertyInfo *get_property_info(const StringName &p_property) const;
// 	const StructPropertyInfo *get_property_info(const String &p_property) const;
// 	const StructPropertyInfo *get_property_info(const Variant &p_property) const;

// 	// Generic struct constructors; for non-native structs, or for safely creating native structs that don't have constructors
// 	static void generic_constructor(VarStructRef p_struct);
// 	static void generic_copy_constructor(VarStructRef p_struct, const VarStructRef);
// 	// Generic struct destructors; for non-native structs
// 	static void generic_destructor(VarStructRef p_struct);
// 	static void trivial_destructor(VarStructRef p_struct) {}

// 	//

// 	static const StructDefinition *get_native(const StringName &p_name);
// 	static void _register_native_definition(StructDefinition **p_definition);
// 	static void unregister_native_types();
// 	static void _register_struct_definition(StructDefinition *p_definition, bool to_clear = true);
// 	static void clean_struct_definitions();

// private:
// 	StructDefinition(StringName p_qualified_name, size_t p_size, StructConstructor p_constructor, StructCopyConstructor p_copy_constructor, StructDestructor p_destructor) :
// 			qualified_name{ p_qualified_name }, size{ p_size }, constructor{ p_constructor }, copy_constructor{ p_copy_constructor }, destructor{ p_destructor } {}

// public:
// 	static StructDefinition *create(std::initializer_list<StructPropertyInfo> p_properties, StringName p_qualified_name, size_t p_size, StructConstructor p_constructor, StructCopyConstructor p_copy_constructor, StructDestructor p_destructor) {
// 		void *ptr = heap_allocate(&StructDefinition::properties, p_properties);
// 		new (ptr) StructDefinition(p_qualified_name, p_size, p_constructor, p_copy_constructor, p_destructor);
// 		return (StructDefinition *)ptr;
// 	}
// };
