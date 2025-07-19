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

	enum DataState : uint16_t {
		STATE_VALID,
		STATE_CLEARED,
	};
	DataState state; // 2 bytes

	uint16_t __padding__; // 2 bytes
};
using StructPtrType = VarHeapPointer<InstanceMetaData>;
using MemberPtrType = VarHeapPointer<InstanceMetaData>::MemberDataPointer;

/////////////////////////////////////

struct AbstractTypeInfo {
	// IMPORTANT: *NEVER* add any properties to this!

	// info
	virtual size_t size() const = 0;
	virtual size_t align() const = 0;
	virtual Variant::Type get_variant_type() const = 0;

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

class StructPropertyInfo : public MemberPtrType {
	friend class VariantStruct;
	friend class StructDefinition;

public:
	enum Kind : uint32_t {
		NATIVE_BUILTIN,
		NATIVE_OBJECT_WEAK,
		NATIVE_OBJECT_REF,
		KIND_MAX,
	};

private:
	// (should be aligned to take up a very minimal 24 bytes)
	// (4 for MemberDataPointer + 4 for Kind // 8 for union pointers // 8 for StringName)

	// MemberPtrType address; // 4 bytes - (used as base class to simplify code; is positioned here)
	Kind kind;
	union {
		VPointer<AbstractTypeInfo> type_info;
		StringName class_name;
	};
	StringName name;

public:
	_ALWAYS_INLINE_ StringName get_name() const {
		return name;
	}

	StructPropertyInfo(const StringName &p_name, const MemberPtrType &p_address, const VPointer<AbstractTypeInfo> &p_type) :
			kind(NATIVE_BUILTIN), name(p_name), MemberPtrType(p_address), type_info(p_type) {}

	StructPropertyInfo(Kind p_kind, const StringName &p_name, const MemberPtrType &p_address, const StringName &p_class) :
			kind(p_kind), name(p_name), MemberPtrType(p_address), class_name(p_class) {}

	StructPropertyInfo(const StructPropertyInfo &p_other) :
			kind(p_other.kind), name(p_other.name), MemberPtrType(p_other) {
		switch (kind) {
			case NATIVE_BUILTIN:
				type_info = p_other.type_info;
				break;
			case NATIVE_OBJECT_WEAK:
			case NATIVE_OBJECT_REF:
				memnew_placement(&class_name, StringName(p_other.class_name));
				break;
			default:
				DEV_ASSERT(false);
		}
	}

	~StructPropertyInfo() {
		switch (kind) {
			case NATIVE_BUILTIN:
				type_info.~VPointer<AbstractTypeInfo>();
				break;
			case NATIVE_OBJECT_WEAK:
			case NATIVE_OBJECT_REF:
				class_name.~StringName();
				break;
			default:
				DEV_ASSERT(false);
		}
	}
};

} //namespace Internal

/////////////////////////////////////

class StructDefinition : public VarHeapObject {
	using StructPropertyInfo = Internal::StructPropertyInfo;
	using StructPtrType = Internal::StructPtrType;

public:
	typedef void (*StructConstructor)(const StructDefinition *, StructPtrType);
	typedef void (*StructCopyConstructor)(const StructDefinition *, StructPtrType, const StructPtrType);
	typedef void (*StructDestructor)(const StructDefinition *, StructPtrType);

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
	static void generic_constructor(const StructDefinition *p_definition, StructPtrType p_target);
	static void generic_copy_constructor(const StructDefinition *p_definition, StructPtrType p_target, const StructPtrType p_other);
	// Generic struct destructors; for non-native structs
	static void generic_destructor(const StructDefinition *p_definition, StructPtrType p_target);
	static void trivial_destructor(const StructDefinition *p_definition, StructPtrType p_target) {}

private:
	StructDefinition(StringName p_qualified_name, size_t p_size, StructConstructor p_constructor, StructCopyConstructor p_copy_constructor, StructDestructor p_destructor) :
			qualified_name{ p_qualified_name }, size{ p_size }, constructor{ p_constructor }, copy_constructor{ p_copy_constructor }, destructor{ p_destructor } {}

public:
	static StructDefinition *create(std::initializer_list<StructPropertyInfo> p_properties, StringName p_qualified_name, size_t p_size, StructConstructor p_constructor, StructCopyConstructor p_copy_constructor, StructDestructor p_destructor) {
		void *ptr = heap_allocate(&StructDefinition::properties, p_properties);
		new (ptr) StructDefinition(p_qualified_name, p_size, p_constructor, p_copy_constructor, p_destructor);
		return (StructDefinition *)ptr;
	}

	VariantStruct create_instance() const;
};

///////////////////////////////

class VariantStruct {
	using StructPtrType = Internal::StructPtrType;
	friend class StructDefinition;

protected:
	// Needs to fit on Variant, so should not have any other properties
	StructPtrType instance = nullptr;
	const StructDefinition *definition = nullptr;

	void _init(const StructDefinition *p_definition);
	void _ref(StructPtrType p_ptr, const StructDefinition *p_definition);
	void _unref();

public:
	void set(const StringName &p_name, const Variant &p_value, bool &r_valid);
	Variant get(const StringName &p_name, bool &r_valid) const;
	void clear();
	bool is_empty() const;
	uint32_t hash() const;
	uint32_t recursive_hash(int recursion_count) const;
	VariantStruct duplicate(bool deep = false) const;
	VariantStruct recursive_duplicate(bool p_deep, ResourceDeepDuplicateMode p_deep_subresources_mode, int recursion_count) const;

	// Basic Constructors

	constexpr VariantStruct() {}
	VariantStruct(const VariantStruct &p_struct) :
			instance(p_struct.instance), definition(p_struct.definition) {
		if (instance) {
			instance->refcount.ref();
		}
	}
	VariantStruct(VariantStruct &&p_struct) :
			instance(p_struct.instance), definition(p_struct.definition) {
		p_struct.instance = nullptr;
	}

	// Basic assignment

	_FORCE_INLINE_ VariantStruct &operator=(const VariantStruct &p_other) {
		_ref(p_other.instance, p_other.definition);
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
	_ALWAYS_INLINE_ T &is_struct();
	template <class T>
	_ALWAYS_INLINE_ T &get_struct();

protected:
#ifndef VSTRUCT_IS_REFERENCE_TYPE
	_ALWAYS_INLINE_ void _cow_check() {
		DEV_ASSERT(instance && definition); // should be checked before getting here
		DEV_ASSERT(instance->state == Internal::InstanceMetaData::STATE_VALID); // should be checked before getting here
		if (instance->refcount.get() > 1) {
			*this = duplicate(false);
		}
	}
#endif
};
