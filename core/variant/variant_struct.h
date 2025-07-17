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

/////////////////////////////////////

namespace Internal {

// It is important that the meta data is sized to a multiple of 8 bytes
// This ensures that we know the struct data will start with a 8-byte alignment
struct InstanceMetaData {
	SafeRefCount refcount; // 4 bytes

	enum StoreType : uint8_t {
		IN_PLACE,
		IN_HEAP_ROOT,
#ifndef VSTRUCT_IS_REFERENCE_TYPE
		IN_PLACE_IN_HEAP,
#endif
	};
	StoreType store; // 1 byte

	enum DataState : uint8_t {
		STATE_UNINITIALISED,
		STATE_VALID,
		STATE_CLEARED,
	};
	DataState state; // 1 byte

	uint16_t __padding__; // 2 bytes
};
using StructPtrType = VarHeapPointer<InstanceMetaData>;
using MemberPtrType = VarHeapPointer<InstanceMetaData>::MemberDataPointer;

// Represents a pairing between the pointer to an instance and a pointer to the definition for that instance
// It is a trivial type (which should allow for quick and easy easy transfer between functions)
// Primary purpose is to ensure that there is no way a pointer to an instance can be separated from its definition
// This doesn't count as a "true reference" (as in, it should have no bearing on refcount)
struct VarStructRef {
	StructPtrType instance; // 8 bytes
	const StructDefinition *definition; // 8 bytes
};

/////////////////////////////////////

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

/////////////////////////////////////

struct StructPropertyInfo : public MemberPtrType {
	enum Kind : uint32_t {
		NATIVE_BUILTIN,
		NATIVE_OBJECT,
		KIND_MAX,
	};

	// (should be aligned to take up a very minimal 24 bytes)
	// (4 for MemberDataPointer + 4 for Kind // 8 for union pointers // 8 for StringName)
	// MemberPtrType address; // (used as base class to simplify code; is positioned here)
	Kind kind;
	union {
		VPointer<AbstractTypeInfo> type_info;
		StringName class_name;
	};
	StringName name;

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
			kind(NATIVE_BUILTIN), MemberPtrType(p_address), name(p_name), type_info(p_type) {}

	StructPropertyInfo(const StructPropertyInfo &p_other) :
			kind(p_other.kind), MemberPtrType(p_other), name(p_other.name) {
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

/////////////////////////////////////

class StructDefinition : public VarHeapObject {
	using StructPropertyInfo = Internal::StructPropertyInfo;
	using VarStructRef = Internal::VarStructRef;
	using StructPtrType = Internal::StructPtrType;

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

	//

	static const StructDefinition *get_native(const StringName &p_name);
	static void _register_native_definition(StructDefinition **p_definition);
	static void unregister_native_types();
	static void _register_struct_definition(StructDefinition *p_definition, bool to_clear = true);
	static void clean_struct_definitions();

	// Generic struct constructors; for non-native structs, or for safely creating native structs that don't have constructors
	static void generic_constructor(VarStructRef p_struct);
	static void generic_copy_constructor(VarStructRef p_struct, const VarStructRef);
	// Generic struct destructors; for non-native structs
	static void generic_destructor(VarStructRef p_struct);
	static void trivial_destructor(VarStructRef p_struct) {}

private:
	StructDefinition(StringName p_qualified_name, size_t p_size, StructConstructor p_constructor, StructCopyConstructor p_copy_constructor, StructDestructor p_destructor) :
			qualified_name{ p_qualified_name }, size{ p_size }, constructor{ p_constructor }, copy_constructor{ p_copy_constructor }, destructor{ p_destructor } {}

public:
	static StructDefinition *create(std::initializer_list<StructPropertyInfo> p_properties, StringName p_qualified_name, size_t p_size, StructConstructor p_constructor, StructCopyConstructor p_copy_constructor, StructDestructor p_destructor) {
		void *ptr = heap_allocate(&StructDefinition::properties, p_properties);
		new (ptr) StructDefinition(p_qualified_name, p_size, p_constructor, p_copy_constructor, p_destructor);
		return (StructDefinition *)ptr;
	}

	_FORCE_INLINE_ VarStructRef allocate_instance() const {
		StructPtrType instance(size);
		instance->store = Internal::InstanceMetaData::IN_HEAP_ROOT;
		instance->state = Internal::InstanceMetaData::STATE_UNINITIALISED;
		return VarStructRef{ instance, this };
	}
	_FORCE_INLINE_ VarStructRef construct_instance() const {
		VarStructRef ret = allocate_instance();
		constructor(ret);
		ret.instance->state = Internal::InstanceMetaData::STATE_VALID;
		return ret;
	}
	_FORCE_INLINE_ VarStructRef copy_instance(VarStructRef p_struct) const {
		DEV_ASSERT(p_struct.instance->state == Internal::InstanceMetaData::STATE_VALID); // should be checked before getting here
		DEV_ASSERT(this == p_struct.definition); // should be checked before getting here
		VarStructRef ret = allocate_instance();
		copy_constructor(ret, p_struct);
		return ret;
	}
};

///////////////////////////////

class VariantStruct {
	using StructPtrType = Internal::StructPtrType;
	using VarStructRef = Internal::VarStructRef;

protected:
	// Needs to fit on Variant, so should not have any other properties
	StructPtrType instance;
	const StructDefinition *definition;

	void _ref(VarStructRef p_struct);
	void _unref();

#ifndef VSTRUCT_IS_REFERENCE_TYPE
	_ALWAYS_INLINE_ _cow_check() {
		DEV_ASSERT()
	}
	void _copy_on_write();
#endif

public:
	void set(const StringName &p_name, const Variant &p_value, bool &r_valid);
	Variant get(const StringName &p_name, bool &r_valid) const;

	void clear();
	bool is_empty() const;

	//

	VariantStruct() :
			instance(nullptr), definition(nullptr) {}
	VariantStruct(const VariantStruct &p_struct) :
			instance(p_struct.instance), definition(p_struct.definition) {
		if (instance->store == Internal::InstanceMetaData::IN_HEAP_ROOT) {
			instance->refcount.ref();
		}
	}
	VariantStruct(VariantStruct &&p_struct) :
			instance(p_struct.instance), definition(p_struct.definition) {
		p_struct.instance = nullptr;
	}

	explicit VariantStruct(VarStructRef p_ref) :
			instance(p_ref.instance), definition(p_ref.definition) {
		if (instance->store == Internal::InstanceMetaData::IN_HEAP_ROOT) {
			instance->refcount.ref();
		}
	}

	_FORCE_INLINE_ VariantStruct &operator=(const VariantStruct &p_other) {
		_ref({ p_other.instance, p_other.definition });
		return *this;
	}

	_FORCE_INLINE_ VariantStruct &operator=(VariantStruct &&p_other) {
		std::swap(instance, p_other.instance);
		std::swap(definition, p_other.definition);
		return *this;
	}

	~VariantStruct() {
		if (instance) {
			_unref();
		}
	}

	template <class T>
	_ALWAYS_INLINE_ T &get_struct();
};
